import PIL
import numpy as np
import torch
import random
from .easy_dataset import EasyDataset
from .augmentation import get_image_augmentation
from .. import vace_contract


class BaseDataset(EasyDataset):
    """
    Base dataset compatible with UnifiedDataset interface.
    Supports video loading with temporal sampling and augmentation.
    """

    def __init__(
        self,
        *,  # only keyword arguments
        # UnifiedDataset-compatible parameters
        base_path=None,
        metadata_path=None,
        repeat=1,
        # Video-specific parameters
        height=None,
        width=None,
        num_frames=None,  # Renamed from num_views for clarity
        num_views=None,   # Keep for backward compatibility
        min_num_context_views=None,
        max_num_context_views=None,
        min_interval=1,
        max_interval=1,
        # Other parameters
        split=None,
        aug_color_jitter=False,
        aug_gray_scale=False,
        aug_gau_blur=False,
        aug_crop=False,
        aug_reverse=0.0,
        random_sample_start=False,
        seed=None,
        seq_aug_crop=False,
        landscape_check=False,
        use_tqdm=False,
        proprio_debug=False,
    ):
        # UnifiedDataset compatibility
        self.base_path = base_path
        self.metadata_path = metadata_path
        self.repeat = repeat
        # Always False for BaseDataset - we load frames dynamically, not from cache
        self.load_from_cache = False

        # Handle num_frames/num_views naming
        if num_frames is not None:
            self.num_frames = num_frames
        elif num_views is not None:
            self.num_frames = num_views
        else:
            raise ValueError("Either num_frames or num_views must be specified")

        self.height = height
        self.width = width
        self.min_num_context_views = min_num_context_views
        self.max_num_context_views = max_num_context_views if max_num_context_views is not None else min_num_context_views
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.split = split
        self.use_tqdm = use_tqdm

        # get_image_augmentation includes color jitter
        self.transform = get_image_augmentation(
            color_jitter=aug_color_jitter,
            gray_scale=aug_gray_scale,
            gau_blur=aug_gau_blur,
            img_norm=False,     # VGGT inputs images in [0, 1]
        )

        self.aug_crop = aug_crop
        self.seed = seed
        self.seq_aug_crop = seq_aug_crop
        self.aug_reverse = aug_reverse
        self.random_sample_start = bool(random_sample_start)
        self.landscape_check = landscape_check
        self.proprio_debug = bool(proprio_debug)

        # Initialize scenes list (to be populated by subclasses)
        self.scenes = []

    @property
    def num_views(self):
        """Alias for num_frames to maintain compatibility with EasyDataset wrapper classes"""
        return self.num_frames

    def __len__(self):
        """Return the total number of samples considering repeat factor."""
        return len(self.scenes) * self.repeat

    def sample_from_video(self, video_length, num_views, min_interval, max_interval, rng, start=None, reverse=None):
        reverse = 0
        remaining_length = video_length if start is None else video_length - start
        sample_interval = np.clip(remaining_length // (num_views - 1), min_interval, max_interval)
        clip_length = (num_views - 1) * sample_interval + 1
        if start is None:
            if self.random_sample_start:
                start = rng.integers(0, max(video_length - clip_length, 0) + 1)
            else:
                start = 0
        end = min(start + clip_length, video_length) - 1
        sample_index = np.linspace(start, end, num_views, dtype=int)
        if reverse is None:
            reverse = rng.random() < self.aug_reverse
        if reverse:
            sample_index = sample_index[::-1]
        return sample_index, reverse

    def get_stats(self):
        return f"{len(self)} groups of views"

    def __repr__(self):
        return (
            f"""{type(self).__name__}({self.get_stats()},
            {self.num_views=},
            {self.split=},
            {self.seed=},
            {self.transform=})""".replace(
                "self.", ""
            )
            .replace("\n", "")
            .replace("   ", "")
        )

    def _get_video_frames(self, idx, rng):
        """
        Subclasses should implement this method to load video frames.

        Args:
            idx: Index of the sample (after modulo operation)
            rng: Random number generator

        Returns:
            dict: Dictionary containing:
                - "video": List of PIL.Image objects
                - "prompt": Text prompt string
                - other optional fields
        """
        raise NotImplementedError("Subclasses must implement _get_video_frames()")

    def __getitem__(self, idx):
        """
        Returns a sample in UnifiedDataset-compatible format.

        Returns:
            dict: {
                "video": List[PIL.Image],  # Video frames
                "prompt": str,              # Text prompt
                ... other fields ...
            }
        """
        # Handle repeat: map idx to actual scene index
        actual_idx = idx % len(self.scenes)

        # Set up the random number generator
        if self.seed:  # reseed for each __getitem__
            self._rng = np.random.default_rng(seed=self.seed + idx)
        elif not hasattr(self, "_rng"):
            seed = torch.randint(0, 2**32, (1,)).item()
            self._rng = np.random.default_rng(seed=seed)

        if self.aug_crop > 1 and self.seq_aug_crop:
            self.delta_target_resolution = self._rng.integers(0, self.aug_crop)

        def process_video_frames(video_frames):
            # Convert video frames to PIL Images if they are numpy arrays
            processed_frames = []
            for frame in video_frames:
                if isinstance(frame, np.ndarray):
                    frame = PIL.Image.fromarray(frame)
                processed_frames.append(frame)

            # Apply augmentations to all frames
            if self.transform is not None and len(processed_frames) > 0:
                frames_np = np.stack([np.array(frame) for frame in processed_frames])
                frames_tensor = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float()
                frames_tensor = self.transform(frames_tensor)
                frames_tensor = frames_tensor.permute(0, 2, 3, 1).clamp(0, 255).byte()
                processed_frames = [
                    PIL.Image.fromarray(frame.numpy())
                    for frame in frames_tensor
                ]
            return processed_frames

        def process_depth_frames(depth_frames):
            processed_depth = []
            for frame in depth_frames:
                if isinstance(frame, PIL.Image.Image):
                    frame = np.array(frame)
                elif not isinstance(frame, np.ndarray):
                    raise TypeError(f"Depth frame must be numpy array or PIL.Image, got {type(frame)}")

                if frame.ndim == 3 and frame.shape[-1] == 1:
                    frame = frame[..., 0]
                if frame.ndim != 2:
                    raise ValueError(f"Depth frame must be 2D, got shape={frame.shape}")

                if frame.dtype != np.uint16:
                    if np.issubdtype(frame.dtype, np.integer):
                        frame = frame.astype(np.uint16)
                    else:
                        raise ValueError(f"Depth frame must be integer dtype, got dtype={frame.dtype}")
                processed_depth.append(frame)
            return processed_depth

        def process_mask_frames(mask_frames):
            processed_mask = []
            for frame in mask_frames:
                if isinstance(frame, PIL.Image.Image):
                    frame = np.array(frame)
                elif not isinstance(frame, np.ndarray):
                    raise TypeError(f"Mask frame must be numpy array or PIL.Image, got {type(frame)}")

                if frame.ndim == 3 and frame.shape[-1] == 1:
                    frame = frame[..., 0]
                if frame.ndim != 2:
                    raise ValueError(f"Mask frame must be 2D, got shape={frame.shape}")

                if frame.dtype == np.bool_:
                    frame = frame.astype(np.uint8)
                elif frame.dtype != np.uint8:
                    if np.issubdtype(frame.dtype, np.integer):
                        frame = (frame > 0).astype(np.uint8)
                    else:
                        raise ValueError(f"Mask frame must be integer/bool dtype, got dtype={frame.dtype}")
                else:
                    frame = (frame > 0).astype(np.uint8)
                processed_mask.append(frame)
            return processed_mask

        # Try to load video frames
        while True:
            try:
                data = self._get_video_frames(actual_idx, self._rng)
                assert "prompt" in data, "Data must contain 'prompt' field"

                if "video" in data:
                    data["video"] = process_video_frames(data["video"])
                else:
                    view_keys = [key for key in vace_contract.view_keys("video") if key in data]
                    if not view_keys:
                        raise KeyError("Data must contain 'video' or three-view fields")
                    for key in view_keys:
                        data[key] = process_video_frames(data[key])

                if "vace_video" in data:
                    data["vace_video"] = process_video_frames(data["vace_video"])
                for key in ("vace_rgb_video",):
                    if key in data:
                        data[key] = process_video_frames(data[key])
                if "vace_depth_video" in data:
                    data["vace_depth_video"] = process_depth_frames(data["vace_depth_video"])
                if "vace_mask_video" in data:
                    data["vace_mask_video"] = process_mask_frames(data["vace_mask_video"])
                break

            except Exception as e:
                print(f"Error in getting sample {actual_idx}: {e}. Trying another sample.")
                actual_idx = random.randint(0, len(self.scenes) - 1)

        return data

    def _crop_resize_if_necessary(
        self, image, resolution, rng, info=None
    ):
        """
        Crop and resize image to target resolution.
        Simplified version without camera intrinsics handling.

        Args:
            image: PIL.Image or numpy array
            resolution: (width, height) tuple
            rng: Random number generator
            info: Optional info for debugging

        Returns:
            PIL.Image: Processed image
        """
        if not isinstance(image, PIL.Image.Image):
            image = PIL.Image.fromarray(image)

        target_resolution = np.array(resolution)
        W, H = image.size

        # Transpose resolution if landscape check is enabled
        rotate_to_portrait = False
        if self.landscape_check and W < H:
            if resolution[0] != resolution[1]:
                target_resolution = np.array([resolution[1], resolution[0]])
                rotate_to_portrait = True

        # High-quality Lanczos down-scaling with optional augmentation
        if self.aug_crop > 1:
            noisy_resolution = target_resolution + (
                rng.integers(0, self.aug_crop)
                if not self.seq_aug_crop
                else self.delta_target_resolution
            )
        else:
            noisy_resolution = target_resolution

        # Resize to noisy resolution
        image = image.resize(tuple(noisy_resolution), PIL.Image.LANCZOS)

        # Crop to target resolution (center crop)
        l, t = np.int32(np.round((np.array(image.size) - target_resolution) / 2))
        out_width, out_height = target_resolution
        crop_bbox = (l, t, l + out_width, t + out_height)
        image = image.crop(crop_bbox)

        # Rotate if needed
        if rotate_to_portrait:
            clockwise = rng.random() > 0.5
            image = image.rotate(90 if clockwise else -90, expand=True)

        return image
