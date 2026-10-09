import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
import os
from typing_extensions import Literal

class SimpleAdapter(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, num_residual_blocks=1):
        super(SimpleAdapter, self).__init__()

        # Pixel Unshuffle: reduce spatial dimensions by a factor of 8
        self.pixel_unshuffle = nn.PixelUnshuffle(downscale_factor=8)

        # Convolution: reduce spatial dimensions by a factor
        #  of 2 (without overlap)
        self.conv = nn.Conv2d(in_dim * 64, out_dim, kernel_size=kernel_size, stride=stride, padding=0)

        # Residual blocks for feature extraction
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(out_dim) for _ in range(num_residual_blocks)]
        )

    def forward(self, x):
        # Reshape to merge the frame dimension into batch
        bs, c, f, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(bs * f, c, h, w)

        # Pixel Unshuffle operation
        x_unshuffled = self.pixel_unshuffle(x)

        # Convolution operation
        x_conv = self.conv(x_unshuffled)

        # Feature extraction with residual blocks
        out = self.residual_blocks(x_conv)

        # Reshape to restore original bf dimension
        out = out.view(bs, f, out.size(1), out.size(2), out.size(3))

        # Permute dimensions to reorder (if needed), e.g., swap channels and feature frames
        out = out.permute(0, 2, 1, 3, 4)

        return out
    
    def process_camera_coordinates(
        self,
        direction: Literal["Left", "Right", "Up", "Down", "LeftUp", "LeftDown", "RightUp", "RightDown"],
        length: int,
        height: int,
        width: int,
        speed: float = 1/54,
        origin=(0, 0.532139961, 0.946026558, 0.5, 0.5, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0)
    ):
        if origin is None:
            origin = (0, 0.532139961, 0.946026558, 0.5, 0.5, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0)
        coordinates = generate_camera_coordinates(direction, length, speed, origin)
        plucker_embedding = process_pose_file(coordinates, width, height)
        return plucker_embedding
        
    

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        return out
    
class Camera(object):
    """Copied from https://github.com/hehao13/CameraCtrl/blob/main/inference.py
    """
    def __init__(self, entry):
        fx, fy, cx, cy = entry[1:5]
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        w2c_mat = np.array(entry[7:]).reshape(3, 4)
        w2c_mat_4x4 = np.eye(4)
        w2c_mat_4x4[:3, :] = w2c_mat
        self.w2c_mat = w2c_mat_4x4
        self.c2w_mat = np.linalg.inv(w2c_mat_4x4)

def get_relative_pose(cam_params):
    """Copied from https://github.com/hehao13/CameraCtrl/blob/main/inference.py
    """
    abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
    abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]
    cam_to_origin = 0
    target_cam_c2w = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, -cam_to_origin],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    abs2rel = target_cam_c2w @ abs_w2cs[0]
    ret_poses = [target_cam_c2w, ] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
    ret_poses = np.array(ret_poses, dtype=np.float32)
    return ret_poses

def custom_meshgrid(*args):
    # torch>=2.0.0 only
    return torch.meshgrid(*args, indexing='ij')


def ray_condition(K, c2w, H, W, device):
    """Copied from https://github.com/hehao13/CameraCtrl/blob/main/inference.py
    """
    # c2w: B, V, 4, 4
    # K: B, V, 4

    B = K.shape[0]

    j, i = custom_meshgrid(
        torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
        torch.linspace(0, W - 1, W, device=device, dtype=c2w.dtype),
    )
    i = i.reshape([1, 1, H * W]).expand([B, 1, H * W]) + 0.5  # [B, HxW]
    j = j.reshape([1, 1, H * W]).expand([B, 1, H * W]) + 0.5  # [B, HxW]

    fx, fy, cx, cy = K.chunk(4, dim=-1)  # B,V, 1

    zs = torch.ones_like(i)  # [B, HxW]
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    zs = zs.expand_as(ys)

    directions = torch.stack((xs, ys, zs), dim=-1)  # B, V, HW, 3
    directions = directions / directions.norm(dim=-1, keepdim=True)  # B, V, HW, 3

    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)  # B, V, 3, HW
    rays_o = c2w[..., :3, 3]  # B, V, 3
    rays_o = rays_o[:, :, None].expand_as(rays_d)  # B, V, 3, HW
    # c2w @ dirctions
    rays_dxo = torch.linalg.cross(rays_o, rays_d)
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    plucker = plucker.reshape(B, c2w.shape[1], H, W, 6)  # B, V, H, W, 6
    # plucker = plucker.permute(0, 1, 4, 2, 3)
    return plucker


def process_pose_file(cam_params, width=672, height=384, original_pose_width=1280, original_pose_height=720, device='cpu', return_poses=False):
    if return_poses:
        return cam_params
    else:
        cam_params = [Camera(cam_param) for cam_param in cam_params]

        sample_wh_ratio = width / height
        pose_wh_ratio = original_pose_width / original_pose_height  # Assuming placeholder ratios, change as needed

        if pose_wh_ratio > sample_wh_ratio:
            resized_ori_w = height * pose_wh_ratio
            for cam_param in cam_params:
                cam_param.fx = resized_ori_w * cam_param.fx / width
        else:
            resized_ori_h = width / pose_wh_ratio
            for cam_param in cam_params:
                cam_param.fy = resized_ori_h * cam_param.fy / height

        intrinsic = np.asarray([[cam_param.fx * width,
                                cam_param.fy * height,
                                cam_param.cx * width,
                                cam_param.cy * height]
                                for cam_param in cam_params], dtype=np.float32)

        K = torch.as_tensor(intrinsic)[None]  # [1, 1, 4]
        c2ws = get_relative_pose(cam_params)  # Assuming this function is defined elsewhere
        c2ws = torch.as_tensor(c2ws)[None]  # [1, n_frame, 4, 4]
        plucker_embedding = ray_condition(K, c2ws, height, width, device=device)[0].permute(0, 3, 1, 2).contiguous()  # V, 6, H, W
        plucker_embedding = plucker_embedding[None]
        plucker_embedding = rearrange(plucker_embedding, "b f c h w -> b f h w c")[0]
        return plucker_embedding



def build_plucker_from_K_c2w(K: torch.Tensor, c2w: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Build a per-frame plücker map from intrinsics K and camera-to-world c2w (OpenCV convention).

    Mirrors 4DVideo-WanX's `convert_plucker_map`:
      - pixel ray dir = K^-1 @ [x, y, 1]^T (OpenCV: +X right, +Y down, +Z forward)
      - world ray dir = R_c2w @ d_cam, then unit-normalized
      - world ray origin = t_c2w (broadcast over pixels)
      - plücker = cat(dir3, moment3) where moment = origin × dir

    Args:
        K:   [B, T, 3, 3] OpenCV intrinsics (already scaled to (H, W)).
        c2w: [B, T, 4, 4] camera-to-world in OpenCV convention.
        height, width: pixel resolution to evaluate plücker on.
    Returns:
        plucker: [B, 6, T, H, W] with channels [dir_x, dir_y, dir_z, mom_x, mom_y, mom_z].
    """
    if K.dim() != 4 or K.shape[-2:] != (3, 3):
        raise ValueError(f"K must be [B, T, 3, 3]; got {tuple(K.shape)}")
    if c2w.dim() != 4 or c2w.shape[-2:] != (4, 4):
        raise ValueError(f"c2w must be [B, T, 4, 4]; got {tuple(c2w.shape)}")
    if K.shape[:2] != c2w.shape[:2]:
        raise ValueError(f"K/c2w batch/time mismatch: K={tuple(K.shape)}, c2w={tuple(c2w.shape)}")

    B, T = K.shape[:2]
    device, dtype = c2w.device, c2w.dtype

    K_flat = K.reshape(B * T, 3, 3).to(dtype=dtype)
    c2w_flat = c2w.reshape(B * T, 4, 4)

    def _ensure_finite(name, tensor):
        finite = torch.isfinite(tensor)
        if not bool(finite.all().item()):
            bad_count = int((~finite).sum().item())
            raise ValueError(f"{name} contains non-finite values ({bad_count} element(s)).")

    _ensure_finite("K", K_flat)
    _ensure_finite("c2w", c2w_flat)
    focal = torch.stack((K_flat[:, 0, 0], K_flat[:, 1, 1]), dim=1)
    if bool((focal.abs() <= 1e-6).any().item()):
        raise ValueError(
            "K has near-zero focal length; cannot build stable plucker rays. "
            f"min_abs_focal={float(focal.abs().min().item()):.6e}"
        )

    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    pixel_coords = torch.stack([x, y, torch.ones_like(x)], dim=0).reshape(3, -1)  # [3, H*W]
    pixel_coords = pixel_coords[None].expand(B * T, -1, -1)  # [B*T, 3, H*W]

    rays_d_cam = torch.linalg.solve(K_flat, pixel_coords)  # numerically stabler than .inverse()
    _ensure_finite("rays_d_cam", rays_d_cam)
    rotation = c2w_flat[:, :3, :3]
    rays_d_world = rotation @ rays_d_cam  # [B*T, 3, H*W]
    rays_d_world = torch.nn.functional.normalize(rays_d_world, dim=1)
    _ensure_finite("rays_d_world", rays_d_world)
    rays_d_world = rays_d_world.reshape(B * T, 3, height, width)

    rays_o_world = c2w_flat[:, :3, 3][..., None, None].expand_as(rays_d_world)  # [B*T, 3, H, W]
    moment = torch.cross(rays_o_world, rays_d_world, dim=1)
    plucker = torch.cat([rays_d_world, moment], dim=1)  # [B*T, 6, H, W]
    plucker = plucker.reshape(B, T, 6, height, width).permute(0, 2, 1, 3, 4).contiguous()
    _ensure_finite("plucker", plucker)
    return plucker


# Sapien-camera (+X forward, +Y left, +Z up) → OpenCV-camera (+X right, +Y down, +Z forward).
# v_sapien = M_S_FROM_O @ v_opencv, so a sapien c2w applied to an OpenCV camera vector requires:
#     c2w_opencv = c2w_sapien @ blkdiag(M_S_FROM_O, 1)
# Source: AgibotWorld/dataprocess/agibot/convert_camera_param.py (M).
SAPIEN_FROM_OPENCV_3x3 = torch.tensor(
    [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=torch.float32
)


def sapien_c2w_to_opencv_c2w(c2w_sapien: torch.Tensor) -> torch.Tensor:
    """Convert sapien-camera c2w (+X forward / +Y left / +Z up) to OpenCV-camera c2w."""
    if c2w_sapien.shape[-2:] != (4, 4):
        raise ValueError(f"c2w must end with shape (4, 4); got {tuple(c2w_sapien.shape)}")
    M = torch.eye(4, dtype=c2w_sapien.dtype, device=c2w_sapien.device)
    M[:3, :3] = SAPIEN_FROM_OPENCV_3x3.to(dtype=c2w_sapien.dtype, device=c2w_sapien.device)
    return c2w_sapien @ M


def generate_camera_coordinates(
    direction: Literal["Left", "Right", "Up", "Down", "LeftUp", "LeftDown", "RightUp", "RightDown", "In", "Out"],
    length: int,
    speed: float = 1/54,
    origin=(0, 0.532139961, 0.946026558, 0.5, 0.5, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0)
):
    coordinates = [list(origin)]
    while len(coordinates) < length:
        coor = coordinates[-1].copy()
        if "Left" in direction:
            coor[9] += speed
        if "Right" in direction:
            coor[9] -= speed
        if "Up" in direction:
            coor[13] += speed
        if "Down" in direction:
            coor[13] -= speed
        if "In" in direction:
            coor[18] -= speed
        if "Out" in direction:
            coor[18] += speed
        coordinates.append(coor)
    return coordinates
