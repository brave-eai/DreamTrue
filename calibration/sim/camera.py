import copy
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import ClassVar, Mapping, Sequence

import numpy as np
import torch
from einops import einsum, rearrange, repeat
from PIL import ImageColor
from torch.nn.functional import grid_sample, max_pool2d

from utils import Nvtx, OpenCVIntrinsic, Pose, camera_distort_remap, get_checkerboard_image


@dataclass(frozen=True)
class SapienCameraProcessedResult:
    rgb: torch.Tensor
    normal: torch.Tensor
    pcd: torch.Tensor
    depth: torch.Tensor
    mask: torch.Tensor
    segmentation: torch.Tensor
    segmentation_ids: dict[str, int]
    segmentation_group: dict[str, set[str]]
    local_pose: Pose
    global_pose: Pose
    parent_pose: Pose
    intrinsic: OpenCVIntrinsic
    image_gt: torch.Tensor | None
    checkerboard: torch.Tensor | None

    @Nvtx('clone')
    def clone(self):
        return SapienCameraProcessedResult(
            rgb=self.rgb.clone(),
            normal=self.normal.clone(),
            pcd=self.pcd.clone(),
            depth=self.depth.clone(),
            mask=self.mask.clone(),
            segmentation=self.segmentation.clone(),
            segmentation_ids=self.segmentation_ids.copy(),
            segmentation_group=self.segmentation_group.copy(),
            local_pose=self.local_pose.clone(),
            global_pose=self.global_pose.clone(),
            parent_pose=self.parent_pose.clone(),
            intrinsic=self.intrinsic.clone(),
            image_gt=self.image_gt.clone() if self.image_gt is not None else None,
            checkerboard=self.checkerboard.clone() if self.checkerboard is not None else None,
        )

    @property
    def numel(self) -> int:
        return sum([
            self.rgb.numel(),
            self.normal.numel(),
            self.pcd.numel(),
            self.depth.numel(),
            self.mask.numel(),
            self.segmentation.numel(),
            self.image_gt.numel() if self.image_gt is not None else 0,
            self.checkerboard.numel() if self.checkerboard is not None else 0,
        ])

    @property
    def size(self) -> int:
        return sum([
            self.rgb.element_size() * self.rgb.numel(),
            self.normal.element_size() * self.normal.numel(),
            self.pcd.element_size() * self.pcd.numel(),
            self.depth.element_size() * self.depth.numel(),
            self.mask.element_size() * self.mask.numel(),
            self.segmentation.element_size() * self.segmentation.numel(),
            self.image_gt.element_size() * self.image_gt.numel() if self.image_gt is not None else 0,
            self.checkerboard.element_size() * self.checkerboard.numel() if self.checkerboard is not None else 0,
        ])

    @Nvtx('cpu')
    def cpu(self, non_blocking: bool = False) -> 'SapienCameraProcessedResult':
        return SapienCameraProcessedResult(
            rgb=self.rgb.to(device='cpu', non_blocking=non_blocking),
            normal=self.normal.to(device='cpu', non_blocking=non_blocking),
            pcd=self.pcd.to(device='cpu', non_blocking=non_blocking),
            depth=self.depth.to(device='cpu', non_blocking=non_blocking),
            mask=self.mask.to(device='cpu', non_blocking=non_blocking),
            segmentation=self.segmentation.to(device='cpu', non_blocking=non_blocking),
            segmentation_ids=self.segmentation_ids,
            segmentation_group=self.segmentation_group,
            local_pose=self.local_pose,
            global_pose=self.global_pose,
            parent_pose=self.parent_pose,
            intrinsic=self.intrinsic,
            image_gt=self.image_gt.to(device='cpu', non_blocking=non_blocking) if self.image_gt is not None else None,
            checkerboard=self.checkerboard.to(device='cpu', non_blocking=non_blocking) if self.checkerboard is not None else None,
        )

    __color_palette__: ClassVar = torch.tensor([ImageColor.getrgb(color) for color in sorted(set(ImageColor.colormap.values()))], dtype=torch.uint8)

    @Nvtx('cat')
    def cat(
        self,
        dim: int = 1,
        use_gt: bool = True,
        use_gt_mask: bool = True,
        use_rgb: bool = True,
        use_mask: bool = True,
        use_normal: bool = True,
        use_depth: bool = True,
        use_segmentation: bool = True,
        use_checkerboard: bool = True,
    ) -> torch.Tensor:
        img = []
        if use_gt and self.image_gt is not None:
            img.append(self.image_gt)
        if use_gt_mask and self.image_gt is not None:
            clear = torch.tensor([0, 0, 255], dtype=torch.uint8).to(device=self.mask.device, non_blocking=True)
            img.append(torch.where(self.mask.unsqueeze(-1), clear, self.image_gt))
            img.append(torch.where(~self.mask.unsqueeze(-1), clear, self.image_gt))
        if use_rgb:
            img.append(self.rgb)
        if use_mask:
            img.append(self.mask.unsqueeze(-1).repeat(1, 1, 3).to(torch.uint8) * 255)
        if use_normal:
            img.append(self.normal)
        if use_depth:
            depth = self.depth.to(torch.float)
            depth = ((depth - depth.min()) / (depth.max() - depth.min() + 1e-8) * 255).to(torch.uint8)
            depth = depth.unsqueeze(-1).repeat(1, 1, 3)
            img.append(depth)
        if use_segmentation:
            color_palette = self.__color_palette__.to(device=self.segmentation.device, non_blocking=True)
            img.append(color_palette[self.segmentation.to(torch.int32)])
        if use_checkerboard and self.checkerboard is not None:
            img.append(self.checkerboard.unsqueeze(-1).repeat(1, 1, 3))
        return torch.cat(img, dim=dim).to(dtype=torch.uint8)

    @Nvtx('share_memory_')
    def share_memory_(self):
        self.rgb.share_memory_()
        self.normal.share_memory_()
        self.pcd.share_memory_()
        self.depth.share_memory_()
        self.mask.share_memory_()
        self.segmentation.share_memory_()
        if self.image_gt is not None:
            self.image_gt.share_memory_()
        if self.checkerboard is not None:
            self.checkerboard.share_memory_()
        return self


@dataclass(frozen=True)
class SapienCameraResult:
    camera_names: list[str]
    color: torch.Tensor  # B H W 4
    normal: torch.Tensor  # B H W 4
    position: torch.Tensor  # B H W 4
    segmentation: torch.Tensor  # B H W 4
    model_matrix: torch.Tensor  # B 4 4
    segmentation_ids: dict[str, int]
    segmentation_group: dict[str, set[str]]
    local_pose: dict[str, Pose]
    global_pose: dict[str, Pose]
    parent_pose: dict[str, Pose]
    intrinsic: dict[str, OpenCVIntrinsic]

    __distort_cache_max_size__: ClassVar[int] = 128
    __distort_cache__: ClassVar[OrderedDict[tuple[int, int], torch.Tensor]] = OrderedDict()
    __distort_lock__: ClassVar[threading.Lock] = threading.Lock()
    __checkerboard_cache_max_size__: ClassVar[int] = 16
    __checkerboard_cache__: ClassVar[OrderedDict[tuple[int, int, int], torch.Tensor]] = OrderedDict()
    __checkerboard_lock__: ClassVar[threading.Lock] = threading.Lock()

    @staticmethod
    @Nvtx('__p_color')
    def __p_color(x: torch.Tensor) -> torch.Tensor:
        return (x[..., :3] * 255).clip(0, 255)

    @staticmethod
    @Nvtx('__p_normal')
    def __p_normal(x: torch.Tensor) -> torch.Tensor:
        return (x[..., :3] * 127.5 + 127.5).clip(0, 255)

    @staticmethod
    @Nvtx('__p_pcd')
    def __p_pcd(position: torch.Tensor, model_matrix: torch.Tensor) -> torch.Tensor:
        # https://sapien-sim.github.io/docs/user_guide/rendering/camera.html
        # Note that the position is represented in the OpenGL camera space,
        # where the negative z-axis points forward and the y-axis is upward.
        # Thus, to acquire a point cloud in the SAPIEN world space (x forward and z up), we provide get_model_matrix(),
        # which returns the transformation from the OpenGL camera space to the SAPIEN world space.
        return einsum(position[..., :3], model_matrix[..., :3, :3], '... h w c, ... d c -> ... h w d') + model_matrix[..., :3, 3][..., None, None, :]

    @staticmethod
    @Nvtx('__p_depth')
    def __p_depth(position: torch.Tensor) -> torch.Tensor:
        return (-position[..., 2] * 1000.0).clip(0, 65535)

    @staticmethod
    @Nvtx('__p_mask')
    def __p_mask(position: torch.Tensor, segmentation: torch.Tensor, segmentation_ids: dict[str, int], segmentation_group: dict[str, set[str]]) -> tuple[torch.Tensor, torch.Tensor]:
        mask = (position[..., 3] < 1).to(torch.bool)
        segmentation = segmentation[..., 1].to(torch.int32)
        mask = torch.logical_and(mask, torch.isin(segmentation, torch.tensor(list({segmentation_ids[v] for vv in segmentation_group.values() for v in vv}), dtype=torch.int32).to(device=segmentation.device, non_blocking=True)))
        return mask, segmentation

    @classmethod
    @Nvtx('__p_distort')
    def __p_distort(cls, intrinsic: OpenCVIntrinsic | OrderedDict[str, OpenCVIntrinsic], device: torch.device) -> torch.Tensor:
        if isinstance(intrinsic, OrderedDict):
            return torch.stack([cls.__p_distort(v, device=device) for v in intrinsic.values()], dim=0)
        with torch.no_grad(), cls.__distort_lock__:
            cache_key = (hash(intrinsic), hash(device))
            cache = cls.__distort_cache__
            if cache_key in cache:
                distort = cache.pop(cache_key)
                cache[cache_key] = distort
            else:
                distort = torch.from_numpy(camera_distort_remap(**intrinsic.dict, norm=True)).to(dtype=torch.float32, device=device).detach()
                cache[cache_key] = distort
                while len(cache) > cls.__distort_cache_max_size__:
                    cache.popitem(last=False)
            return distort.clone()

    @classmethod
    @Nvtx('__p_checkerboard')
    def __p_checkerboard(cls, w: int, h: int, device: torch.device) -> torch.Tensor:
        with torch.no_grad(), cls.__checkerboard_lock__:
            cache_key = (w, h, hash(device))
            cache = cls.__checkerboard_cache__
            if cache_key in cache:
                checkerboard = cache.pop(cache_key)
                cache[cache_key] = checkerboard
            else:
                checkerboard = torch.from_numpy(get_checkerboard_image(w, h, 24, 24)).to(dtype=torch.float32, device=device).unsqueeze(-1).detach() * 255
                cache[cache_key] = checkerboard
                while len(cache) > cls.__checkerboard_cache_max_size__:
                    cache.popitem(last=False)
            return checkerboard

    @staticmethod
    @Nvtx('__a_distort')
    def __a_distort(x: torch.Tensor, distort: torch.Tensor, nearest: bool = False) -> torch.Tensor:
        # x: (B, H, W, C), dis: (B, H, W, 2) -> (B, H, W, C)
        with torch.backends.cudnn.flags(enabled=False):
            return rearrange(grid_sample(rearrange(x, 'b h w c -> b c h w'), distort, mode='nearest' if nearest else 'bilinear', padding_mode='zeros', align_corners=False), 'b c h w -> b h w c')

    @Nvtx('process')
    @torch.no_grad()
    def process(
        self,
        image_gt: Mapping[str, torch.Tensor | np.ndarray] | list[torch.Tensor | np.ndarray] | torch.Tensor | np.ndarray | None = None,
        checkerboard: bool = False,
        apply_mask: bool = True,
        rgb_background: tuple[int, int, int] | None = (0, 0, 255),
    ) -> OrderedDict[str, SapienCameraProcessedResult]:
        assert len(rgb_background) == 3
        # process mask and segmentation first, since they are needed for other processing
        mask_o, segmentation = self.__p_mask(self.position, self.segmentation, self.segmentation_ids, self.segmentation_group)
        mask_o, segmentation = mask_o.unsqueeze(-1), segmentation.unsqueeze(-1)  # B H W 1
        # calculate distort map
        distort = self.__p_distort(self.intrinsic, device=mask_o.device)  # B H W 2
        # apply distort to mask
        mask_f = self.__a_distort(mask_o.to(torch.float32), distort, False)
        mask_b = mask_f >= 0.1
        # apply distort to other channels
        segmentation = self.__a_distort(torch.where(mask_o, segmentation.float(), 0), distort, True).round().to(segmentation.dtype)

        def ap_where(mask: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.where(mask, x, y) if apply_mask else x

        def ap_div(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return x / (y+1e-8) if apply_mask else x

        rgb, normal, pcd, depth = self.__p_color(self.color), self.__p_normal(self.normal), self.__p_pcd(self.position, self.model_matrix), self.__p_depth(self.position).unsqueeze(-1)
        rgb, normal, pcd, depth = torch.split(ap_div(self.__a_distort(ap_where(mask_o, torch.cat([rgb, normal, pcd, depth], dim=-1), 0), distort, False), mask_f), [3, 3, 3, 1], dim=-1)
        # apply background
        mask = mask_b.squeeze(-1)
        rgb = ap_where(mask_b, rgb, torch.tensor(rgb_background, dtype=rgb.dtype).to(device=rgb.device, non_blocking=True)).to(dtype=torch.uint8)
        normal = ap_where(mask_b, normal, 127.5).to(dtype=torch.uint8)
        pcd = ap_where(mask_b, pcd, 0).to(dtype=torch.float32)
        depth = ap_where(mask_b, depth, 0).to(dtype=torch.uint16).squeeze(-1)
        segmentation = ap_where(mask_b, segmentation, 0).to(dtype=torch.uint16).squeeze(-1)
        # process checkerboard
        B, H, W, _ = mask_o.shape
        checkerboard_map = (self.__a_distort(repeat(self.__p_checkerboard(W, H, mask_b.device), 'h w c -> b h w c', b=B), distort, False).squeeze(-1)).clamp(0, 255).to(dtype=torch.uint8) if checkerboard is True else None

        def clo(x: torch.Tensor) -> torch.Tensor:
            return x.detach().clone().contiguous()

        rgb, normal, pcd, depth, mask, segmentation = clo(rgb), clo(normal), clo(pcd), clo(depth), clo(mask), clo(segmentation)
        if checkerboard_map is not None:
            checkerboard_map = clo(checkerboard_map)
        if isinstance(image_gt, np.ndarray):
            image_gt = torch.as_tensor(image_gt).to(device=rgb.device, non_blocking=True)
        elif isinstance(image_gt, torch.Tensor):
            image_gt = image_gt.to(device=rgb.device, non_blocking=True)
        elif isinstance(image_gt, dict):
            image_gt = [torch.as_tensor(image_gt[name]).to(device=rgb.device, non_blocking=True) for name in self.camera_names]
        elif isinstance(image_gt, list):
            image_gt = [torch.as_tensor(x).to(device=rgb.device, non_blocking=True) for x in image_gt]
        elif image_gt is not None:
            raise ValueError(f'unsupported type for image_gt: {type(image_gt)}')

        return OrderedDict((
            name,
            SapienCameraProcessedResult(
                rgb=rgb[i],
                normal=normal[i],
                pcd=pcd[i],
                depth=depth[i],
                mask=mask[i],
                segmentation=segmentation[i],
                segmentation_ids=self.segmentation_ids.copy(),
                segmentation_group=self.segmentation_group.copy(),
                local_pose=self.local_pose[name].clone(),
                global_pose=self.global_pose[name].clone(),
                parent_pose=self.parent_pose[name].clone(),
                intrinsic=self.intrinsic[name].clone(),
                image_gt=image_gt[i] if image_gt is not None else None,
                checkerboard=checkerboard_map[i] if checkerboard_map is not None else None,
            )
        ) for i, name in enumerate(self.camera_names))

    @Nvtx('clone')
    def clone(self):
        return SapienCameraResult(
            camera_names=self.camera_names.copy(),
            color=self.color.clone(),
            normal=self.normal.clone(),
            segmentation=self.segmentation.clone(),
            position=self.position.clone(),
            model_matrix=self.model_matrix.clone(),
            local_pose=copy.deepcopy(self.local_pose),
            global_pose=copy.deepcopy(self.global_pose),
            parent_pose=copy.deepcopy(self.parent_pose),
            intrinsic=copy.deepcopy(self.intrinsic),
            segmentation_ids=self.segmentation_ids.copy(),
            segmentation_group=self.segmentation_group.copy(),
        )

    @Nvtx('share_memory_')
    def share_memory_(self):
        self.color.share_memory_()
        self.normal.share_memory_()
        self.model_matrix.share_memory_()
        self.position.share_memory_()
        self.segmentation.share_memory_()
        return self


def gripper_rgb_background(action: np.ndarray, qpos_dims: Mapping[str, slice | np.ndarray], gripper_names: Sequence[str], gripper_max: float) -> tuple[int, ...]:
    """把每个夹爪的开合程度平滑编码成 RGB 背景色的一个通道：value/gripper_max -> [0,1] -> [0,255]。

    从 action 里按 qpos_dims 取出每个夹爪(gripper_names 排序后固定通道顺序)的值；
    最多 3 个夹爪，不足的通道用 255 补齐。
    """
    bg = [int(round(np.clip(float(action[qpos_dims[g]]) / gripper_max, 0.0, 1.0) * 255)) for g in sorted(gripper_names)]
    assert len(bg) <= 3, f'too many grippers for rgb background: {len(bg)}'
    return tuple(bg + [128] * (3 - len(bg)))
