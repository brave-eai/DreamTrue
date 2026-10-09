import json
import os
import re
from collections import defaultdict
from functools import cached_property
from typing import Callable, Mapping, OrderedDict

import filelock
import numpy as np
import sapien
import torch
import yaml
from filelock import FileLock
from sapien.pysapien.physx import PhysxArticulation, PhysxArticulationJoint, PhysxCpuSystem, PhysxGpuSystem, PhysxRigidBodyComponent
from sapien.pysapien.render import RenderBodyComponent, RenderCameraComponent, RenderCameraGroup, RenderSystem, RenderSystemGroup
from sapien.utils import Viewer
from tqdm import tqdm

from utils import ColorLogger, Nvtx, OpenCVIntrinsic, OpenCVRenderIntrinsic, Pose, git_status, sha256

from .camera import SapienCameraResult
from .config import SapienGroundConfig, SapienRendererConfig, SapienViewerConfig

sapien.set_log_level('off')


class NotAchieveError(Exception):
    ...


class ViewerClosedError(Exception):
    ...


def _load_robot(
    scene: sapien.Scene,
    assets_path: str,
    robot_path: str,
    robot_pose: sapien.Pose | Pose = Pose(),
    add_axis: Callable[[PhysxRigidBodyComponent, str, float, sapien.Pose], None] | None = None,
    add_universal: Callable[[PhysxRigidBodyComponent, str, str, float, sapien.Pose], None] | None = None,
) -> tuple[dict, PhysxArticulation, list[int], dict[int, list[int]], dict[str, int], dict[str, set[str]]]:
    add_axis = add_axis if add_axis is not None else (lambda a, b, c, d: None)
    add_universal = add_universal if add_universal is not None else (lambda a, b, c, d, e: None)
    robot_pose = robot_pose.sapien if isinstance(robot_pose, Pose) else robot_pose
    with open(os.path.join(assets_path, robot_path), 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    package_dir = os.path.dirname(os.path.join(assets_path, robot_path))
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    robot: PhysxArticulation = loader.load(urdf_file=os.path.join(package_dir, config['urdf']), package_dir=package_dir)
    robot.set_root_pose(robot_pose)
    blp = robot.find_link_by_name('base_link').pose
    assert np.allclose(blp.p, [0, 0, 0]) and np.allclose(blp.q, [1, 0, 0, 0]), f'base_link pose is not identity {blp=}'
    add_axis(robot.find_link_by_name('base_link'), '', 0.5, sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0]))
    add_axis(robot.find_link_by_name('base_link'), '', 0.5, sapien.Pose(p=[0.5, 0, 0], q=[1, 0, 0, 0]))
    # loop joint config
    for lo in config.get('loops', []):
        a, b = lo['a'], lo['b']
        la, lb = robot.find_link_by_name(a['n']), robot.find_link_by_name(b['n'])
        pa, pb = sapien.Pose(p=a['p'], q=a['q']), sapien.Pose(p=b['p'], q=b['q'])
        drive = scene.create_drive(body0=la, body1=lb, pose0=pa, pose1=pb)
        add_axis(la, '', 0.05, pa)
        add_axis(lb, '', 0.05, pb)
        drive.set_drive_property_x(**lo.get('x', dict(stiffness=0, damping=0)))
        drive.set_drive_property_y(**lo.get('y', dict(stiffness=0, damping=0)))
        drive.set_drive_property_z(**lo.get('z', dict(stiffness=0, damping=0)))
    # joint config
    for joint in robot.active_joints:
        joint.set_drive_property(stiffness=0, damping=0)

    active_joints = {joint.name: (idx, joint) for idx, joint in enumerate(robot.active_joints)}
    joint_idxs: list[int] = []
    slave_idxs: dict[int, list[int]] = {}
    for dof in config.get('dof', []):
        idx, joint = active_joints[dof['joint']]
        joint_idxs.append(idx)
        lower, upper = float(joint.limit[0, 0]), float(joint.limit[0, 1])
        default_target = float(np.clip(0.0, lower, upper))
        joint.set_drive_target(default_target)
        joint.set_drive_velocity_target(0)
        joint.set_drive_property(stiffness=dof['stiffness'], damping=dof['damping'])
        if 'slave' in dof:
            salve = re.compile(dof['slave'])
            slave_idxs[idx] = list(sorted({idx} | {idx for k, (idx, _) in active_joints.items() if salve.match(k) is not None}))
    # link config
    segmentation_ids: dict[str, int] = {link.name: link.get_entity().per_scene_id for link in robot.get_links()}
    segment_filter = {gn: re.compile(f) for gn, f in config.get('segment_filter', {}).items()}
    segmentation_group: dict[str, set[str]] = {gn: {k for k in segmentation_ids if segment_filter[gn].match(k) is not None} for gn, f in segment_filter.items()}

    for universal in config.get('universal', []):
        print(f'add universal {universal}')
        add_universal(
            robot.find_link_by_name(universal['parent']),
            universal.get('name', ''),
            universal['filename'],
            universal.get('scale', 0.1),
            sapien.Pose(p=universal.get('p', [0, 0, 0]), q=universal.get('q', [1, 0, 0, 0])),
        )

    return config, robot, joint_idxs, slave_idxs, segmentation_ids, segmentation_group


def _calc_slave_cache(assets_path: str, robot_path: str, slave_joint_cache: int) -> None | dict[int, np.ndarray]:
    if slave_joint_cache == 0:
        return None
    with open(os.path.join(assets_path, robot_path), 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        package_dir = os.path.dirname(os.path.join(assets_path, robot_path))
        cache_fields = yaml.dump({'loops': config.get('loops', []), 'dof': config.get('dof', [])}, sort_keys=True)
        slave_cache_key = str(slave_joint_cache) + sha256(cache_fields.encode('utf-8'), os.path.join(package_dir, config['urdf']))
    with FileLock(os.path.join(assets_path, robot_path + '.sc.npz.lock')):
        if os.path.exists(os.path.join(assets_path, robot_path + '.sc.npz')):
            cache = np.load(os.path.join(assets_path, robot_path + '.sc.npz'), allow_pickle=True)
            if cache['key'] == slave_cache_key:
                return {int(k): v for k, v in cache.items() if k != 'key'}
        scene = sapien.Scene([PhysxCpuSystem(), RenderSystem()])
        scene.set_timestep(1 / 1000)
        _, robot, joint_idxs, slave_idxs, _, _ = _load_robot(scene=scene, assets_path=assets_path, robot_path=robot_path)
        active_joints: dict[str, tuple[int, PhysxArticulationJoint]] = {joint.name: (idx, joint) for idx, joint in enumerate(robot.active_joints)}
        joint_idx2joint: dict[int, PhysxArticulationJoint] = {idx: joint for _, (idx, joint) in active_joints.items()}
        # check all joints are configed
        _joint_cnt = defaultdict(int)
        for idx in joint_idxs:
            _joint_cnt[idx] += 1
        for idx, slaves in slave_idxs.items():
            _joint_cnt[idx] -= 1
            for sidx in slaves:
                _joint_cnt[sidx] += 1
        assert set(_joint_cnt.keys()) == {idx for _, (idx, _) in active_joints.items()} and all(v == 1 for v in _joint_cnt.values())
        result = dict()
        for idx, slaves in tqdm(slave_idxs.items(), position=0, leave=False, desc='Generating slave cache'):
            joint = joint_idx2joint[idx]
            all_values = []
            for target in (pabr := tqdm(np.linspace(joint.limit[0, 0], joint.limit[0, 1], slave_joint_cache), position=1, leave=False, desc=f'Joint {idx} {slaves}')):
                joint.set_drive_target(target)
                values = []
                for _ in range(512):
                    last = robot.qpos.copy()
                    robot.set_qf(robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True))
                    scene.step()
                    if np.all(np.isclose(robot.qpos[idx], last[idx], 0, (joint.limit[0, 1] - joint.limit[0, 0]) / slave_joint_cache / 10)) and np.all(np.isclose(robot.qpos[idx], target, 0, 1e-4)):
                        values.append(robot.qpos[slaves].copy())
                    else:
                        values.clear()
                    pabr.set_postfix_str(f'consec={len(values)}/128')
                if len(values) <= 128:
                    raise NotAchieveError(f'{target=} not achieve when generating slave cache for {idx} {slave_idxs}!')
                all_values.append(np.concatenate([np.array([target]), np.mean(np.array(values), axis=0)], axis=0))
            result[idx] = np.array(all_values)
        np.savez_compressed(os.path.join(assets_path, robot_path + '.sc.npz'), key=slave_cache_key, **{str(k): v for k, v in result.items()})
        return result


def _load_cameras(scene: sapien.Scene, robot: PhysxArticulation, config: dict[str, dict[str, str | float | int]],
                  ignore_camera: set[str] | list[str] | None) -> tuple[OrderedDict[str, RenderCameraComponent], dict[str, OpenCVIntrinsic], dict[str, str | None], dict[str, bool]]:
    cameras: OrderedDict[str, RenderCameraComponent] = OrderedDict()
    intrinsics: dict[str, OpenCVIntrinsic] = dict()
    cam2gripper: dict[str, str | None] = dict()
    cam_use_bg_mask: dict[str, bool] = dict()
    ignore_camera = set(ignore_camera) if ignore_camera is not None else set()
    for name, cnf in config.items():
        if name in ignore_camera:
            continue
        cam = cameras[name] = scene.add_mounted_camera(
            name=name,
            mount=robot.find_link_by_name(cnf['mount']).get_entity(),
            pose=sapien.Pose(p=cnf['p'], q=cnf['q']),
            width=cnf['width'],
            height=cnf['height'],
            far=cnf.get('far', 10),
            near=cnf.get('near', 0.01),
            fovy=cnf.get('fovy', np.deg2rad(53)),
        )
        intrinsics[name] = OpenCVIntrinsic(
            fx=cnf.get('fx', cam.fx),
            fy=cnf.get('fy', cam.fy),
            cx=cnf.get('cx', cam.cx),
            cy=cnf.get('cy', cam.cy),
            w=cam.width,
            h=cam.height,
            k1=cnf.get('k1', 0),
            k2=cnf.get('k2', 0),
            p1=cnf.get('p1', 0),
            p2=cnf.get('p2', 0),
            k3=cnf.get('k3', 0),
        )
        cam2gripper[name] = cnf.get('gripper', None)
        cam_use_bg_mask[name] = cnf.get('use_bg_mask', False)
    return cameras, intrinsics, cam2gripper, cam_use_bg_mask


class SapienEnv(ColorLogger):

    assets_path: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets')

    def __init__(
        self,
        renderer: SapienRendererConfig = SapienRendererConfig(),
        viewer: SapienViewerConfig | None = SapienViewerConfig(),
        ground: SapienGroundConfig | None = SapienGroundConfig(),
        with_debug_axis: bool = False,
        robot_path: str = 'agibot/G1_120s/G1_120s.yaml',
        robot_pose: Pose | sapien.Pose = Pose(),
        ignore_camera: set[str] | list[str] | None = None,
        slave_joint_cache: int = 1000,
        use_gpu_physx: bool = False,
        device: str = 'cuda',
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.debug('base init')
        try:
            self.__cuda_available = torch.cuda.is_available() and getattr(getattr(torch, 'version', None), 'cuda', None) is not None
        except ImportError:
            self.__cuda_available = False
        self.__with_debug_axis: bool = with_debug_axis
        self.__use_gpu_physx: bool = use_gpu_physx and self.__cuda_available and not os.environ.get('DISABLE_GPU_PHYSX', '0') == '1'
        try:
            if self.__use_gpu_physx and not sapien.physx.is_gpu_enabled():
                # fix parral download bug https://github.com/haosulab/SAPIEN/blob/731622eac5b140b320076c8a1b6eb4b553c3ccd4/python/py_package/physx/__init__.py#L48
                with filelock.FileLock(os.path.abspath(os.path.expanduser('~/.sapien/gpu_physx.lock'))):
                    sapien.physx.enable_gpu()
        except Exception as e:
            self.warning(f'Failed to enable GPU PhysX, fallback to CPU PhysX. Error: {e}')
        self.__use_gpu_physx = self.__use_gpu_physx and sapien.physx.is_gpu_enabled()
        self.__gpu_pose_index_map: dict[int, sapien.Entity] | None = None
        self.__gpu_render_system_group: RenderSystemGroup | None = None
        self.__gpu_render_camera_group: RenderCameraGroup | None = None
        self.__robot_path = robot_path
        # scene config
        self._scene = sapien.Scene([PhysxGpuSystem(device=device) if self.__use_gpu_physx else PhysxCpuSystem(), RenderSystem(device=device) if self.__use_gpu_physx else RenderSystem()])
        if ground is not None and ground.position is not None:
            self._scene.add_ground(ground.position)
        self._scene.set_timestep(1 / 1000)
        if renderer.camera_shader_dir != '':
            self.debug(f'set renderer {renderer}')
            sapien.render.set_camera_shader_dir(renderer.camera_shader_dir)
            sapien.render.set_ray_tracing_samples_per_pixel(renderer.ray_tracing_samples_per_pixel)
            sapien.render.set_ray_tracing_path_depth(renderer.ray_tracing_path_depth)
            sapien.render.set_ray_tracing_denoiser(renderer.ray_tracing_denoiser)
        # light config
        self._scene.set_ambient_light([0.5, 0.5, 0.5])
        self._scene.add_directional_light([0, 0, -5], [1, 1, 1])
        self._scene.add_directional_light([-2, 2, -2], [1, 1, 1])
        self._scene.add_directional_light([-2, -2, -2], [1, 1, 1])
        # load robot
        self.__qvel = None
        config, self._robot, self.__joint_idxs, self.__slave_idxs, self.__segmentation_ids, self.__segmentation_group = _load_robot(
            scene=self._scene,
            assets_path=self.assets_path,
            robot_path=robot_path,
            robot_pose=robot_pose,
            add_axis=self.add_axis,
            add_universal=self.add_universal,
        )
        # slave joint config
        self.__slave_joint_cache: None | dict[int, np.ndarray] = _calc_slave_cache(
            assets_path=self.assets_path,
            robot_path=robot_path,
            slave_joint_cache=slave_joint_cache,
        )
        assert self.__slave_joint_cache is None or set(self.__slave_joint_cache.keys()) == set(self.__slave_idxs.keys())
        # camera config
        # For a SAPIEN camera, the x-axis points forward, the y-axis left, and the z-axis upward.
        self.__cameras_intrinsic: dict[str, OpenCVIntrinsic] = dict()
        self.__cameras_intrinsic_cache: int | None = None
        self.__cameras_local_pose_cache: int | None = None
        self.__offset_joints_pose_cache: int | None = None
        self.__cameras, self.cameras_intrinsic, self.__cam2gripper, self.__cam_use_bg_mask = _load_cameras(scene=self._scene, robot=self._robot, config=config.get('cameras', {}), ignore_camera=ignore_camera)
        # part topology config
        self.__parts_config: dict[str, dict] = config['parts']
        self.__sam3_prompt: list[str] = config.get('sam3_prompt', ['robot'])
        self.__offset_joints: dict[str, str] = config.get('offset_joints', {})
        self.__compatible_datasets: set[str] = set(config['compatible_datasets'])
        assert (set(self.__cam2gripper.values()) - {None}) <= {v['gripper'] for k, v in self.__parts_config.items()}
        assert self.__parts_config.keys() <= self.__segmentation_group.keys()
        assert 'arm' in self.__segmentation_group and 'gripper' in self.__segmentation_group
        # viewer config
        self.__viewer: None | Viewer = None
        if viewer is not None and viewer.display:
            self.debug(f'set viewer {viewer}')
            self.__viewer = self._scene.create_viewer()
            if viewer.pos is not None:
                self.__viewer.set_camera_pose(viewer.pos if isinstance(viewer.pos, sapien.Pose) else viewer.pos.sapien)
            assert self.__viewer.window is not None
            self.__viewer.window.set_camera_parameters(
                near=viewer.near,
                far=viewer.far,
                fovy=np.deg2rad(viewer.fovy),
            )
        self.__gpu_init()

    @property
    def viewer_closed(self) -> bool:
        return self.__viewer is None or self.__viewer.closed

    def close(self):
        if not self.viewer_closed:
            self.__viewer.close()
        self._scene.clear()

    @property
    def __gpu_px(self) -> PhysxGpuSystem:
        px = self._scene.physx_system
        assert isinstance(px, PhysxGpuSystem)
        return px

    def __gpu_init(self):
        if not self.__use_gpu_physx:
            return
        px = self.__gpu_px
        px.gpu_init()
        self.__gpu_pose_index_map = {}
        for entity in self._scene.entities:
            gpu_pose_index: int = getattr(entity.find_component_by_type(PhysxRigidBodyComponent), 'gpu_pose_index', -1)
            if gpu_pose_index == -1:
                continue
            self.debug(f'entity {entity.name=} {entity.global_id=} {gpu_pose_index=}')
            self.__gpu_pose_index_map[gpu_pose_index] = entity
            for comp in entity.components:
                if isinstance(comp, RenderCameraComponent):
                    self.debug(f'    camera {comp.name=} {gpu_pose_index=}')
                    comp.set_gpu_pose_batch_index(gpu_pose_index)
                elif isinstance(comp, RenderBodyComponent):
                    self.debug(f'    body {comp.name=} {gpu_pose_index=}')
                    for shape in comp.render_shapes:
                        self.debug(f'      shape {shape.name=} {gpu_pose_index=}')
                        shape.set_gpu_pose_batch_index(gpu_pose_index)
        self.debug(f'{px.cuda_rigid_body_data.shape=}')
        self.__gpu_copy_pose()
        self.__gpu_update_kinematics()
        self.__gpu_render_system_group = RenderSystemGroup([self._scene.render_system])
        self.__gpu_render_system_group.set_cuda_poses(px.cuda_rigid_body_data)
        self.__gpu_render_camera_group = self.__gpu_render_system_group.create_camera_group(list(self.__cameras.values()), ['Color', 'Normal', 'Position', 'Segmentation'])

    def __gpu_update_kinematics(self):
        if not self.__use_gpu_physx:
            return
        self.debug('gpu_update_kinematics')
        px = self.__gpu_px
        px.gpu_update_articulation_kinematics()

    def __gpu_copy_pose(self):
        if not self.__use_gpu_physx:
            return
        self.debug('gpu_copy_pose')
        assert self.__gpu_pose_index_map is not None
        px = self.__gpu_px
        px_rigid_body = torch.as_tensor(px.cuda_rigid_body_data)
        np_rigid_body = np.zeros(px_rigid_body.shape, dtype=np.float32)
        for gpu_pose_index, entity in self.__gpu_pose_index_map.items():
            np_rigid_body[gpu_pose_index, 0:3] = entity.pose.p
            np_rigid_body[gpu_pose_index, 3:7] = entity.pose.q
        px_rigid_body.copy_(torch.as_tensor(np_rigid_body, device=px_rigid_body.device, dtype=px_rigid_body.dtype), non_blocking=True)

    def __gpu_copy_qpos(self):
        if not self.__use_gpu_physx:
            return
        self.debug('gpu_copy_qpos')
        px = self.__gpu_px
        px_target_qpos = torch.as_tensor(px.cuda_articulation_target_qpos)
        px_tqrget_qvel = torch.as_tensor(px.cuda_articulation_target_qvel)
        np_target_qpos = np.zeros(px_target_qpos.shape, dtype=np.float32)
        np_target_qvel = np.zeros(px_tqrget_qvel.shape, dtype=np.float32)
        for articulation in self._scene.get_all_articulations():
            for idx, joint in enumerate(articulation.active_joints):
                np_target_qpos[articulation.gpu_index, idx] = joint.drive_target
                np_target_qvel[articulation.gpu_index, idx] = joint.drive_velocity_target
        px_target_qpos.copy_(torch.as_tensor(np_target_qpos, device=px_target_qpos.device, dtype=px_target_qpos.dtype), non_blocking=True)
        px_tqrget_qvel.copy_(torch.as_tensor(np_target_qvel, device=px_tqrget_qvel.device, dtype=px_tqrget_qvel.dtype), non_blocking=True)

    def __gpu_fetch(self):
        if not self.__use_gpu_physx:
            return
        self.debug('gpu_fetch')
        px = self.__gpu_px
        px.gpu_update_articulation_kinematics()
        px.gpu_fetch_articulation_link_incoming_joint_forces()
        px.gpu_fetch_articulation_link_pose()
        px.gpu_fetch_articulation_link_velocity()
        px.gpu_fetch_articulation_qacc()
        px.gpu_fetch_articulation_qpos()
        px.gpu_fetch_articulation_qvel()
        px.gpu_fetch_articulation_target_qpos()
        px.gpu_fetch_articulation_target_qvel()
        px.gpu_fetch_rigid_dynamic_data()
        px.sync_poses_gpu_to_cpu()

    def __gpu_apply_rigid_dynamic(self):
        if not self.__use_gpu_physx:
            return
        self.debug('gpu_apply_rigid_dynamic')
        px = self.__gpu_px
        px.gpu_apply_rigid_dynamic_data()
        px.gpu_apply_rigid_dynamic_force()
        px.gpu_apply_rigid_dynamic_torque()

    def __gpu_apply_articulation_root(self):
        if not self.__use_gpu_physx:
            return
        self.debug('gpu_apply_articulation_root')
        px = self.__gpu_px
        px.gpu_apply_articulation_root_pose()
        px.gpu_apply_articulation_root_velocity()

    def __gpu_apply_articulation_target(self):
        if not self.__use_gpu_physx:
            return
        px = self.__gpu_px
        self.debug('gpu_apply_articulation_target')
        px.gpu_apply_articulation_target_position()
        px.gpu_apply_articulation_target_velocity()

    def __gpu_apply_articulation_qf(self):
        if not self.__use_gpu_physx:
            return
        px = self.__gpu_px
        self.debug('gpu_apply_articulation_qf')
        px.gpu_apply_articulation_qf()

    def __gpu_apply_articulation_qpos(self):
        if not self.__use_gpu_physx:
            return
        px = self.__gpu_px
        self.debug('gpu_apply_articulation_qpos')
        px.gpu_apply_articulation_qpos()
        px.gpu_apply_articulation_qvel()

    def step(self):
        self.debug('step')
        his = self.qpos
        self._robot.set_qf(self._robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True))
        self._scene.step()
        self.__qvel = self.qpos - his

    @Nvtx('render')
    def render(self):
        self.debug(f'render')
        self.__gpu_fetch()
        if self.__gpu_render_system_group is not None:
            self.__gpu_render_system_group.update_render()
        else:
            self._scene.update_render()
        if self.__viewer is not None:
            if self.__viewer.closed:
                raise ViewerClosedError()
            self.__viewer.render()

    @property
    def robot_path(self) -> str:
        return self.__robot_path

    def add_universal(self, parent: PhysxRigidBodyComponent, name: str = '', filename: str = '', scale: float = 0.1, pose: Pose | sapien.Pose = Pose()):
        if isinstance(pose, Pose):
            pose = pose.sapien
        actor = (self._scene.create_actor_builder().add_visual_from_file(filename=os.path.join(self.assets_path, 'universal', filename), pose=pose, scale=(scale, scale, scale)).build(name=name))
        render_body = actor.find_component_by_type(RenderBodyComponent)
        actor.remove_component(render_body)
        parent.get_entity().add_component(render_body)

    def add_axis(self, parent: PhysxRigidBodyComponent, name: str = '', scale: float = 0.1, pose: Pose | sapien.Pose = Pose()):
        if not self.__with_debug_axis:
            return
        self.add_universal(parent=parent, name=name, filename='axis_s.glb', scale=scale, pose=pose)

    @property
    def position(self) -> Pose:
        return Pose.from_sapien(self._robot.get_root_pose()).clone()

    @position.setter
    def position(self, pose: Pose | sapien.Pose):
        if isinstance(pose, Pose):
            pose = pose.sapien
        self._robot.set_root_pose(pose)
        self.__gpu_copy_pose()
        self.__gpu_apply_articulation_root()
        self.__gpu_update_kinematics()

    @property
    def joints(self) -> list[PhysxArticulationJoint]:
        return [self._robot.active_joints[i] for i in self.__joint_idxs]

    @property
    def action(self) -> np.ndarray:
        return np.array([j.drive_target.squeeze(axis=0) for j in self.joints])

    @action.setter
    def action(self, values: np.ndarray | tuple):
        if isinstance(values, tuple):
            values, (a, m) = self.action, values
            values[m] = a
        for v, j in zip(values, self.joints):
            j.set_drive_target(v)
        self.__gpu_copy_qpos()
        self.__gpu_apply_articulation_target()
        if self.__slave_joint_cache is not None:
            qpos = self.full_qpos.copy()
            qpos[self.__joint_idxs] = values
            for k, v in self.__slave_idxs.items():
                qpos[v] = self.__slave_joint_interpolate(k, float(qpos[k]))
            self.full_qpos = qpos

    def __slave_joint_interpolate(self, k: int, query: float) -> np.ndarray:
        assert self.__slave_joint_cache is not None and self.__slave_joint_cache[k] is not None
        t = np.array(self.__slave_joint_cache[k])
        idx: int = int(np.searchsorted(t[:, 0], query))
        if idx <= 0:
            return t[0, 1:]
        if idx >= len(t):
            return t[-1, 1:]
        x0, x1 = t[idx - 1, 0], t[idx, 0]
        y0, y1 = t[idx - 1, 1:], t[idx, 1:]
        return y0 + ((query-x0) / (x1-x0)) * (y1-y0)

    @property
    def qname(self) -> list[str]:
        return list(map(lambda x: x.name, self.joints))

    @property
    def qpos(self) -> np.ndarray:
        return self._robot.qpos[self.__joint_idxs].copy()

    @property
    def qlimit(self) -> np.ndarray:
        return np.array([j.limits.squeeze(axis=0) for j in self.joints])

    @property
    def qvel(self) -> np.ndarray:
        return self.__qvel.copy()

    __qpos_achieve = 1e-4
    __qvel_achieve = 1e-4

    @property
    def qachieve(self) -> np.ndarray:
        return np.logical_and(np.isclose(self.qpos, self.action, 0, self.__qpos_achieve), np.abs(self.qvel) < self.__qvel_achieve)

    @property
    def full_qpos(self) -> np.ndarray:
        return self._robot.qpos.copy()

    @full_qpos.setter
    def full_qpos(self, values: np.ndarray):
        self._robot.qpos = values
        self.__gpu_apply_articulation_qpos()

    def qachieve_wait(self, wait: int = 0, achieve: int = 4):
        if self.__slave_joint_cache is not None:
            return
        i, a = 0, 0
        while True:
            yield
            if wait == 0:
                break
            a = (a + 1) if self.qachieve.all() else 0
            if a > achieve:
                break
            i += 1
            if i > wait:
                raise NotAchieveError(f'wait for {wait} steps, but still not achieve!')

    def qpos_generator(self, sli: slice | np.ndarray = slice(None), step: int = 10, cnt: int = 2):
        origin_action = self.action.copy()
        maxx, minn = int(step * 1.2), int(step * -0.2)
        cnt = cnt * (maxx-minn)
        idx, delta = -1, 1
        for _ in range(cnt):
            idx += delta
            if idx >= maxx:
                idx, delta = maxx, -1
            elif idx <= minn:
                idx, delta = minn, 1
            rate = np.clip(idx / step, 0, 1)
            target = self.qlimit[:, 0] * (1-rate) + self.qlimit[:, 1] * rate
            action = origin_action.copy()
            action[..., sli] = target[..., sli]
            yield action

    @property
    def offset_joints_pose(self) -> dict[str, Pose]:
        return {part: Pose.from_sapien(self._robot.find_joint_by_name(joint_name).pose_in_parent) for part, joint_name in self.__offset_joints.items()}

    @offset_joints_pose.setter
    def offset_joints_pose(self, offsets: Mapping[str, Pose]):
        if len(offsets) == 0:
            return
        if len(self.__offset_joints) == 0:
            return
        _h = hash(frozenset(offsets.items()))
        if _h == self.__offset_joints_pose_cache:
            return
        for part, joint_name in self.__offset_joints.items():
            joint = self._robot.find_joint_by_name(joint_name)
            if (pose := offsets.get(part, None)) is not None:
                joint.set_pose_in_parent(pose.sapien)
        # CPU mode: re-set qpos to trigger PhysX forward kinematics recomputation
        self._robot.qpos = self._robot.qpos
        self.__gpu_init()
        self.__offset_joints_pose_cache = _h

    @property
    def cameras(self) -> OrderedDict[str, RenderCameraComponent]:
        return self.__cameras

    def __gpu_recreate_cameras(self):
        # 需要创建一个新的相机，否则local pose和intrinsic参数无法修改
        if not self.__use_gpu_physx:
            return
        if self.__gpu_render_camera_group is None:
            return
        new_cams = dict()
        for name, cam in self.__cameras.items():
            new_cam = self._scene.add_mounted_camera(name=cam.name, mount=cam.entity, pose=cam.pose, width=cam.width, height=cam.height, far=cam.far, near=cam.near, fovy=cam.fovy)
            new_cam.set_local_pose(cam.local_pose)
            new_cam.set_perspective_parameters(near=cam.near, far=cam.far, skew=cam.skew, cx=cam.cx, cy=cam.cy, fx=cam.fx, fy=cam.fy)
            new_cams[name] = new_cam
        self.__gpu_render_system_group = None
        self.__gpu_render_camera_group = None
        for cam in self.__cameras.values():
            cam.entity.remove_component(cam)
        self.__cameras = OrderedDict((k, new_cams[k]) for k in self.__cameras.keys())

    @property
    def cameras_local_pose(self) -> OrderedDict[str, Pose]:
        return OrderedDict((k, Pose.from_sapien(cam.get_local_pose())) for k, cam in self.__cameras.items())

    @property
    def cameras_global_pose(self) -> OrderedDict[str, Pose]:
        return OrderedDict((k, Pose.from_sapien(cam.get_global_pose())) for k, cam in self.__cameras.items())

    @property
    def cameras_parent_pose(self) -> OrderedDict[str, Pose]:
        return OrderedDict((k, Pose.from_sapien(cam.get_entity().pose)) for k, cam in self.__cameras.items())

    @cameras_local_pose.setter
    def cameras_local_pose(self, poses: Mapping[str, Pose]):
        if len(poses) == 0:
            return
        if len(self.__cameras) == 0:
            return
        _h = hash(frozenset(poses.items()))
        if _h == self.__cameras_local_pose_cache:
            return
        self.__gpu_recreate_cameras()
        for k, cam in self.__cameras.items():
            if (p := poses.get(k, None)) is not None:
                cam.set_local_pose(p.sapien)
        self.__gpu_init()
        self.__cameras_local_pose_cache = _h

    @property
    def cameras_intrinsic(self) -> OrderedDict[str, OpenCVRenderIntrinsic]:
        return OrderedDict((k, OpenCVRenderIntrinsic(rfx=cam.fx, rfy=cam.fy, rcx=cam.cx, rcy=cam.cy, **self.__cameras_intrinsic[k].dict)) for k, cam in self.__cameras.items())

    @cameras_intrinsic.setter
    def cameras_intrinsic(self, intrinsics: Mapping[str, OpenCVIntrinsic | OpenCVRenderIntrinsic]):
        if len(intrinsics) == 0:
            return
        if len(self.__cameras) == 0:
            return
        _h = hash(frozenset(intrinsics.items()))
        if _h == self.__cameras_intrinsic_cache:
            return
        self.__gpu_recreate_cameras()
        for name, cam in self.__cameras.items():
            if (i := intrinsics.get(name, None)) is not None:
                if isinstance(i, OpenCVIntrinsic):
                    i = OpenCVRenderIntrinsic.from_opencv(i)
                assert (i.w is None or i.w == cam.width) and (i.h is None or i.h == cam.height)
                cam.set_perspective_parameters(near=cam.near, far=cam.far, skew=cam.skew, cx=i.rcx, cy=i.rcy, fx=i.rfx, fy=i.rfy)
                self.__cameras_intrinsic[name] = i.opencv
        self.__gpu_init()
        self.__cameras_intrinsic_cache = _h

    @property
    def segmentation_ids(self) -> dict[str, int]:
        return self.__segmentation_ids.copy()

    @property
    def segmentation_group(self) -> dict[str, set[str]]:
        return self.__segmentation_group.copy()

    def __get_pirtures(self, mode: str) -> torch.Tensor:
        if self.__gpu_render_camera_group is not None:
            return torch.as_tensor(self.__gpu_render_camera_group.get_picture_cuda(mode))
        else:
            return torch.stack([torch.as_tensor((cam.get_picture_cuda(mode) if self.__cuda_available else cam.get_picture(mode))) for cam in self.__cameras.values()])

    @Nvtx('take_picture')
    def take_picture(self) -> SapienCameraResult:
        with Nvtx('take'):
            if self.__gpu_render_camera_group is not None:
                self.__gpu_render_camera_group.take_picture()
            else:
                for cam in self.__cameras.values():
                    cam.take_picture()
        with Nvtx('get'):
            colors = self.__get_pirtures('Color')
            normals = self.__get_pirtures('Normal')
            positions = self.__get_pirtures('Position')
            segmentations = self.__get_pirtures('Segmentation')
            model_matrix = torch.from_numpy(np.array([cam.get_model_matrix() for cam in self.__cameras.values()])).to(device=positions.device, non_blocking=True)
        return SapienCameraResult(
            camera_names=list(self.__cameras.keys()),
            color=colors,
            normal=normals,
            position=positions,
            segmentation=segmentations,
            model_matrix=model_matrix,
            segmentation_ids=self.__segmentation_ids.copy(),
            segmentation_group=self.__segmentation_group.copy(),
            intrinsic=self.cameras_intrinsic,
            local_pose={
                k: Pose.from_sapien(cam.get_local_pose())
                for k, cam in self.__cameras.items()
            },
            global_pose={
                k: Pose.from_sapien(cam.get_global_pose())
                for k, cam in self.__cameras.items()
            },
            parent_pose={
                k: Pose.from_sapien(cam.get_entity().pose)
                for k, cam in self.__cameras.items()
            },
        )

    @property
    def link_global_poses(self) -> OrderedDict[str, Pose]:
        return OrderedDict((v.name, Pose.from_sapien(v.pose)) for v in self._robot.links)

    @property
    def parts(self) -> list[str]:
        """
        部件分组
        segment_filter 中的组名，主要支撑多个手臂需要分开计算roma的问题。
        要求分割时同时保留夹爪和手臂
        """
        return list(self.__parts_config.keys())

    @property
    def parts_config(self) -> dict[str, dict]:
        """完整的 parts 配置：part -> {end_effector, move_group, gripper}。"""
        return {k: v.copy() for k, v in self.__parts_config.items()}

    @property
    def part2gripper(self) -> dict[str, str]:
        """
        part -> dataset._qpos_dims 中对应的 gripper 维度 key
        用于 sim-real 时跳过正在活动的夹爪，匹配时判断夹爪是否打，决定是否额外 mask 掉 gripper 区域只匹配 arm。
        """
        return {k: v['gripper'] for k, v in self.__parts_config.items()}

    @property
    def cam2gripper(self) -> dict[str, str | None]:
        """
        camera -> dataset._qpos_dims 中对应的 gripper 维度 key (null 表示不跟随任何手臂)；
        用于 temporal 匹配：如果相机所在手臂的夹爪在两帧之间非全闭合(可能抓着东西)，则跳过
        """
        return self.__cam2gripper

    @property
    def cam_use_bg_mask(self) -> dict[str, bool]:
        """相机名 -> 真机对真机匹配时是否用 sam3 mask 去掉机器人本体；true = 只匹配背景；false = 前景+背景都匹配"""
        return self.__cam_use_bg_mask

    @property
    def sam3_prompt(self) -> list[str]:
        """SAM3的prompt"""
        return self.__sam3_prompt

    @property
    def offset_joints(self) -> dict[str, str]:
        """part -> URDF 中对应的 base fixed joint 名称；keys 即为有可学习 SE(3) offset 的 part，value 用于将 offset 应用到 SapienEnv"""
        return self.__offset_joints

    @property
    def compatible_datasets(self) -> set[str]:
        """该机器人配置兼容的数据集类名集合。"""
        return self.__compatible_datasets.copy()

    @property
    def gripper_names(self) -> tuple[str, ...]:
        return tuple(set(self.part2gripper.values()))

    @property
    def cam2part(self) -> dict[str, str | None]:
        gripper2part = {v: k for k, v in self.part2gripper.items()}
        return {cam: gripper2part.get(gripper) for cam, gripper in self.__cam2gripper.items()}

    def load_calibration(self, calib_path: str, postfix: str = '_best', ignore_notfound: bool = False) -> tuple[str, str]:
        if calib_path == '' or calib_path is None:
            return '', ''
        used_files = []
        if len(self.__cameras) == 0:
            pass
        elif os.path.exists(lp_path := os.path.join(calib_path, f'local_pose{postfix}.json')):
            with open(lp_path, 'r', encoding='utf-8') as fp:
                self.cameras_local_pose = {k: Pose.from_dict(v) for k, v in json.load(fp).items()}
            used_files.append(lp_path)
        elif not ignore_notfound:
            raise FileNotFoundError(f'local pose file not found in {calib_path} with postfix {postfix}!')
        if len(self.__cameras) == 0:
            pass
        elif os.path.exists(intr_path := os.path.join(calib_path, f'intrinsic{postfix}.json')):
            with open(intr_path, 'r', encoding='utf-8') as fp:
                self.cameras_intrinsic = {k: OpenCVIntrinsic.from_dict(v) for k, v in json.load(fp).items()}
            used_files.append(intr_path)
        elif not ignore_notfound:
            raise FileNotFoundError(f'intrinsic file not found in {calib_path} with postfix {postfix}!')
        if len(self.__offset_joints) == 0:
            pass
        elif os.path.exists(oj_path := os.path.join(calib_path, f'offset_joints{postfix}.json')):
            with open(oj_path, 'r', encoding='utf-8') as fp:
                self.offset_joints_pose = {k: Pose.from_dict(v) for k, v in json.load(fp).items()}
            used_files.append(oj_path)
        elif not ignore_notfound:
            raise FileNotFoundError(f'offset joints file not found in {calib_path} with postfix {postfix}!')
        return sha256(*used_files), git_status(calib_path, ignore_error=True)
