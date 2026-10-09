import torch, types
import numpy as np
from PIL import Image
from einops import repeat
from typing import Optional, Union
from einops import rearrange
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Optional
from typing_extensions import Literal
from transformers import Wav2Vec2Processor
import time

from ..diffusion import FlowMatchScheduler
from ..core import ModelConfig, gradient_checkpoint_forward
from ..diffusion.base_pipeline import BasePipeline, PipelineUnit

from ..models.wan_video_dit import WanModel, sinusoidal_embedding_1d
from ..models.wan_video_dit_s2v import rope_precompute
from ..models.wan_video_text_encoder import WanTextEncoder, HuggingfaceTokenizer
from ..models.wan_video_vae import WanVideoVAE
from ..models.wan_video_image_encoder import WanImageEncoder
from ..models.wan_video_vace import VaceWanModel
from ..models.wan_video_motion_controller import WanMotionControllerModel
from ..models.wan_video_animate_adapter import WanAnimateAdapter
from ..models.wan_video_mot import MotWanModel
from ..models.wav2vec import WanS2VAudioEncoder
from ..models.longcat_video_dit import LongCatVideoTransformer3DModel


class WanVideoPipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.bfloat16):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16, time_division_factor=4, time_division_remainder=1
        )
        self.scheduler = FlowMatchScheduler("Wan")
        self.tokenizer: HuggingfaceTokenizer = None
        self.audio_processor: Wav2Vec2Processor = None
        self.text_encoder: WanTextEncoder = None
        self.image_encoder: WanImageEncoder = None
        self.dit: WanModel = None
        self.dit2: WanModel = None
        self.vae: WanVideoVAE = None
        self.motion_controller: WanMotionControllerModel = None
        self.vace: VaceWanModel = None
        self.vace2: VaceWanModel = None
        self.vap: MotWanModel = None
        self.animate_adapter: WanAnimateAdapter = None
        self.audio_encoder: WanS2VAudioEncoder = None
        self.in_iteration_models = ("dit", "motion_controller", "vace", "animate_adapter", "vap")
        self.in_iteration_models_2 = ("dit2", "motion_controller", "vace2", "animate_adapter", "vap")
        self.units = [
            WanVideoUnit_ShapeChecker(),
            WanVideoUnit_NoiseInitializer(),
            WanVideoUnit_PromptEmbedder(),
            WanVideoUnit_S2V(),
            WanVideoUnit_InputVideoEmbedder(),
            WanVideoUnit_ImageEmbedderVAE(),
            WanVideoUnit_ImageEmbedderCLIP(),
            WanVideoUnit_ImageEmbedderFused(),
            WanVideoUnit_FunControl(),
            WanVideoUnit_FunReference(),
            WanVideoUnit_FunCameraControl(),
            WanVideoUnit_SpeedControl(),
            WanVideoUnit_VACE(),
            WanVideoUnit_AnimateVideoSplit(),
            WanVideoUnit_AnimatePoseLatents(),
            WanVideoUnit_AnimateFacePixelValues(),
            WanVideoUnit_AnimateInpaint(),
            WanVideoUnit_VAP(),
            WanVideoUnit_UnifiedSequenceParallel(),
            WanVideoUnit_TeaCache(),
            WanVideoUnit_CfgMerger(),
            WanVideoUnit_LongCatVideo(),
        ]
        self.post_units = [
            WanVideoPostUnit_S2V(),
        ]
        self.model_fn = model_fn_wan_video


    def enable_usp(self):
        from ..utils.xfuser import get_sequence_parallel_world_size, usp_attn_forward, usp_dit_forward

        for block in self.dit.blocks:
            block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
        self.dit.forward = types.MethodType(usp_dit_forward, self.dit)
        if self.dit2 is not None:
            for block in self.dit2.blocks:
                block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
            self.dit2.forward = types.MethodType(usp_dit_forward, self.dit2)
        self.sp_size = get_sequence_parallel_world_size()
        self.use_unified_sequence_parallel = True


    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = "cuda",
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
        audio_processor_config: ModelConfig = None,
        redirect_common_files: bool = True,
        use_usp: bool = False,
        vram_limit: float = None,
    ):
        print("\n" + "="*80)
        print("[WanVideoPipeline.from_pretrained] Starting pipeline initialization...")
        print("="*80)
        total_start = time.time()

        # Redirect model path
        if redirect_common_files:
            print("\n[from_pretrained] Step 1: Checking model path redirects...")
            redirect_dict = {
                "models_t5_umt5-xxl-enc-bf16.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "models_t5_umt5-xxl-enc-bf16.safetensors"),
                "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "models_clip_open-clip-xlm-roberta-large-vit-huge-14.safetensors"),
                "Wan2.1_VAE.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "Wan2.1_VAE.safetensors"),
                "Wan2.2_VAE.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "Wan2.2_VAE.safetensors"),
            }
            for model_config in model_configs:
                if model_config.origin_file_pattern is None or model_config.model_id is None:
                    continue
                if model_config.origin_file_pattern in redirect_dict and model_config.model_id != redirect_dict[model_config.origin_file_pattern][0]:
                    print(f"To avoid repeatedly downloading model files, ({model_config.model_id}, {model_config.origin_file_pattern}) is redirected to {redirect_dict[model_config.origin_file_pattern]}. You can use `redirect_common_files=False` to disable file redirection.")
                    model_config.model_id = redirect_dict[model_config.origin_file_pattern][0]
                    model_config.origin_file_pattern = redirect_dict[model_config.origin_file_pattern][1]

        # Initialize pipeline
        print("\n[from_pretrained] Step 2: Creating WanVideoPipeline object...")
        step_start = time.time()
        pipe = WanVideoPipeline(device=device, torch_dtype=torch_dtype)
        print(f"[from_pretrained]   Pipeline object created in {time.time() - step_start:.2f}s")

        if use_usp:
            from ..utils.xfuser import initialize_usp
            initialize_usp()

        print("\n[from_pretrained] Step 3: Downloading and loading models...")
        print(f"[from_pretrained]   Number of model configs: {len(model_configs)}")
        step_start = time.time()
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)
        print(f"[from_pretrained]   Models downloaded and loaded in {time.time() - step_start:.2f}s")

        # Fetch models
        print("\n[from_pretrained] Step 4: Fetching models from pool...")
        step_start = time.time()
        pipe.text_encoder = model_pool.fetch_model("wan_video_text_encoder")
        dit = model_pool.fetch_model("wan_video_dit", index=2)
        if isinstance(dit, list):
            pipe.dit, pipe.dit2 = dit
        else:
            pipe.dit = dit
        pipe.vae = model_pool.fetch_model("wan_video_vae")
        pipe.image_encoder = model_pool.fetch_model("wan_video_image_encoder")
        pipe.motion_controller = model_pool.fetch_model("wan_video_motion_controller")
        vace = model_pool.fetch_model("wan_video_vace", index=2)
        if isinstance(vace, list):
            pipe.vace, pipe.vace2 = vace
        else:
            pipe.vace = vace
        pipe.vap = model_pool.fetch_model("wan_video_vap")
        pipe.audio_encoder = model_pool.fetch_model("wans2v_audio_encoder")
        pipe.animate_adapter = model_pool.fetch_model("wan_video_animate_adapter")
        print(f"[from_pretrained]   Models fetched in {time.time() - step_start:.2f}s")

        # Size division factor
        if pipe.vae is not None:
            pipe.height_division_factor = pipe.vae.upsampling_factor * 2
            pipe.width_division_factor = pipe.vae.upsampling_factor * 2

        # Initialize tokenizer and processor
        print("\n[from_pretrained] Step 5: Initializing tokenizer and processor...")
        step_start = time.time()
        if tokenizer_config is not None:
            print("[from_pretrained]   Loading tokenizer...")
            tokenizer_config.download_if_necessary()
            pipe.tokenizer = HuggingfaceTokenizer(name=tokenizer_config.path, seq_len=512, clean='whitespace')
        if audio_processor_config is not None:
            print("[from_pretrained]   Loading audio processor...")
            audio_processor_config.download_if_necessary()
            pipe.audio_processor = Wav2Vec2Processor.from_pretrained(audio_processor_config.path)
        print(f"[from_pretrained]   Tokenizer/processor initialized in {time.time() - step_start:.2f}s")

        # Unified Sequence Parallel
        if use_usp: pipe.enable_usp()

        # VRAM Management
        pipe.vram_management_enabled = pipe.check_vram_management_state()

        print("\n" + "="*80)
        print(f"[WanVideoPipeline.from_pretrained] Complete! Total time: {time.time() - total_start:.2f}s")
        print("="*80 + "\n")
        return pipe


    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt: str,
        negative_prompt: Optional[str] = "",
        # Image-to-video
        input_image: Optional[Image.Image] = None,
        # First-last-frame-to-video
        end_image: Optional[Image.Image] = None,
        # Video-to-video
        input_video: Optional[list[Image.Image]] = None,
        denoising_strength: Optional[float] = 1.0,
        # Speech-to-video
        input_audio: Optional[np.array] = None,
        audio_embeds: Optional[torch.Tensor] = None,
        audio_sample_rate: Optional[int] = 16000,
        s2v_pose_video: Optional[list[Image.Image]] = None,
        s2v_pose_latents: Optional[torch.Tensor] = None,
        motion_video: Optional[list[Image.Image]] = None,
        # ControlNet
        control_video: Optional[list[Image.Image]] = None,
        reference_image: Optional[Image.Image] = None,
        # Camera control
        camera_control_direction: Optional[Literal["Left", "Right", "Up", "Down", "LeftUp", "LeftDown", "RightUp", "RightDown"]] = None,
        camera_control_speed: Optional[float] = 1/54,
        camera_control_origin: Optional[tuple] = (0, 0.532139961, 0.946026558, 0.5, 0.5, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0),
        # VACE
        vace_video: Optional[list[Image.Image]] = None,
        vace_rgb_video: Optional[list[Image.Image]] = None,
        vace_depth_video: Optional[list] = None,
        vace_mask_video: Optional[list] = None,
        vace_video_mask: Optional[Image.Image] = None,
        vace_reference_image: Optional[Image.Image] = None,
        # Plücker conditioning (optional): per-frame intrinsic + extrinsic for each view.
        # Format: ndarray/tensor of shape (T,3,3) and (T,4,4), or list of such per view
        # for multi-view (e.g., [head_K, hand_left_K, hand_right_K]). Activated when
        # pipe.vace.use_plucker is True; otherwise ignored.
        vace_intrinsic=None,
        vace_extrinsic=None,
        vace_scale: Optional[float] = 1.0,
        # Animate
        animate_pose_video: Optional[list[Image.Image]] = None,
        animate_face_video: Optional[list[Image.Image]] = None,
        animate_inpaint_video: Optional[list[Image.Image]] = None,
        animate_mask_video: Optional[list[Image.Image]] = None,
        # VAP
        vap_video: Optional[list[Image.Image]] = None,
        vap_prompt: Optional[str] = " ",
        negative_vap_prompt: Optional[str] = " ",
        # Randomness
        seed: Optional[int] = None,
        rand_device: Optional[str] = "cpu",
        # Shape
        height: Optional[int] = 480,
        width: Optional[int] = 832,
        num_frames=81,
        # Classifier-free guidance
        cfg_scale: Optional[float] = 5.0,
        cfg_merge: Optional[bool] = False,
        # Boundary
        switch_DiT_boundary: Optional[float] = 0.875,
        # Scheduler
        num_inference_steps: Optional[int] = 50,
        sigma_shift: Optional[float] = 5.0,
        # Speed control
        motion_bucket_id: Optional[int] = None,
        # LongCat-Video
        longcat_video: Optional[list[Image.Image]] = None,
        # VAE tiling
        tiled: Optional[bool] = True,
        tile_size: Optional[tuple[int, int]] = (30, 52),
        tile_stride: Optional[tuple[int, int]] = (15, 26),
        # Sliding window
        sliding_window_size: Optional[int] = None,
        sliding_window_stride: Optional[int] = None,
        # Teacache
        tea_cache_l1_thresh: Optional[float] = None,
        tea_cache_model_id: Optional[str] = "",
        # progress_bar
        progress_bar_cmd=tqdm,
    ):
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)
        
        # Inputs
        inputs_posi = {
            "prompt": prompt,
            "vap_prompt": vap_prompt,
            "tea_cache_l1_thresh": tea_cache_l1_thresh, "tea_cache_model_id": tea_cache_model_id, "num_inference_steps": num_inference_steps,
        }
        inputs_nega = {
            "negative_prompt": negative_prompt,
            "negative_vap_prompt": negative_vap_prompt,
            "tea_cache_l1_thresh": tea_cache_l1_thresh, "tea_cache_model_id": tea_cache_model_id, "num_inference_steps": num_inference_steps,
        }
        num_views = 1
        if isinstance(input_video, (list, tuple)) and len(input_video) > 0 and isinstance(input_video[0], (list, tuple)):
            num_views = len(input_video)
        elif isinstance(input_image, (list, tuple)) and len(input_image) > 0:
            num_views = len(input_image)
        else:
            # VACE-only / V2V_VACE 推理时 input_video / input_image 都为 None,
            # 多视角信息只存在于 per-view VACE 输入里。沿用
            # WanVideoUnit_VACE._is_multi_view_vace_input 的批 1 判定 (外层 list,
            # list[0] 也是 list 即视为 per-view 列表),让 NoiseInitializer 按
            # num_views 把 latent 宽 (target_width = width * num_views) 扩起来,
            # 否则 noise/x 还是单视角宽,而 vace_context 是多视角宽,会在
            # VaceWanModel.forward 的零填里报负长度。
            for _vace_candidate in (
                vace_reference_image,
                vace_rgb_video,
                vace_depth_video,
                vace_mask_video,
                vace_video,
            ):
                if (
                    isinstance(_vace_candidate, (list, tuple))
                    and len(_vace_candidate) > 0
                    and isinstance(_vace_candidate[0], (list, tuple))
                ):
                    num_views = len(_vace_candidate)
                    break
        inputs_shared = {
            "input_image": input_image,
            "end_image": end_image,
            "input_video": input_video, "denoising_strength": denoising_strength,
            "control_video": control_video, "reference_image": reference_image,
            "camera_control_direction": camera_control_direction, "camera_control_speed": camera_control_speed, "camera_control_origin": camera_control_origin,
            "vace_video": vace_video,
            "vace_rgb_video": vace_rgb_video,
            "vace_depth_video": vace_depth_video,
            "vace_mask_video": vace_mask_video,
            "vace_video_mask": vace_video_mask,
            "vace_reference_image": vace_reference_image,
            "vace_intrinsic": vace_intrinsic,
            "vace_extrinsic": vace_extrinsic,
            "vace_scale": vace_scale,
            "seed": seed, "rand_device": rand_device,
            "height": height, "width": width, "num_frames": num_frames,
            "num_views": num_views,
            "cfg_scale": cfg_scale, "cfg_merge": cfg_merge,
            "sigma_shift": sigma_shift,
            "motion_bucket_id": motion_bucket_id,
            "longcat_video": longcat_video,
            "tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride,
            "sliding_window_size": sliding_window_size, "sliding_window_stride": sliding_window_stride,
            "input_audio": input_audio, "audio_sample_rate": audio_sample_rate, "s2v_pose_video": s2v_pose_video, "audio_embeds": audio_embeds, "s2v_pose_latents": s2v_pose_latents, "motion_video": motion_video,
            "animate_pose_video": animate_pose_video, "animate_face_video": animate_face_video, "animate_inpaint_video": animate_inpaint_video, "animate_mask_video": animate_mask_video,
            "vap_video": vap_video, 
        }
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        # Denoise
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            # Switch DiT if necessary
            if timestep.item() < switch_DiT_boundary * 1000 and self.dit2 is not None and not models["dit"] is self.dit2:
                self.load_models_to_device(self.in_iteration_models_2)
                models["dit"] = self.dit2
                models["vace"] = self.vace2
                
            # Timestep
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            
            # Inference
            noise_pred_posi = self.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep)
            if cfg_scale != 1.0:
                if cfg_merge:
                    noise_pred_posi, noise_pred_nega = noise_pred_posi.chunk(2, dim=0)
                else:
                    noise_pred_nega = self.model_fn(**models, **inputs_shared, **inputs_nega, timestep=timestep)
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            # Scheduler
            inputs_shared["latents"] = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], inputs_shared["latents"])
            if "first_frame_latents" in inputs_shared:
                inputs_shared["latents"][:, :, 0:1] = inputs_shared["first_frame_latents"]
        
        # VACE (TODO: remove it)
        if vace_reference_image is not None or (animate_pose_video is not None and animate_face_video is not None):
            if vace_reference_image is not None:
                batch = int(inputs_shared["latents"].shape[0])
                if isinstance(vace_reference_image, (list, tuple)):
                    if vace_reference_image and isinstance(vace_reference_image[0], (list, tuple)):
                        f = len(vace_reference_image[0])
                    elif batch > 1 and len(vace_reference_image) == batch:
                        f = 1
                    else:
                        f = len(vace_reference_image)
                else:
                    f = 1
            else:
                f = 1
            inputs_shared["latents"] = inputs_shared["latents"][:, :, f:]
        # post-denoising, pre-decoding processing logic
        for unit in self.post_units:
            inputs_shared, _, _ = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)
        # Decode
        inputs_shared["latents"] = inputs_shared["latents"].to(dtype=self.torch_dtype, device=self.device)
        self.load_models_to_device(['vae'])
        num_views = inputs_shared.get("num_views", 1)
        if num_views is not None and num_views > 1:
            view_videos = []
            latents_views = torch.chunk(inputs_shared["latents"], num_views, dim=-1)
            for view_latents in latents_views:
                view_latents = view_latents.to(dtype=self.torch_dtype, device=self.device)
                view_video = self.vae.decode(view_latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
                view_videos.append(self.vae_output_to_video(view_video))
            self.load_models_to_device([])
            return view_videos
        video = self.vae.decode(inputs_shared["latents"], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        video = self.vae_output_to_video(video)
        self.load_models_to_device([])

        return video



class WanVideoUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames"),
            output_params=("height", "width", "num_frames"),
        )

    def process(self, pipe: WanVideoPipeline, height, width, num_frames):
        height, width, num_frames = pipe.check_resize_height_width(height, width, num_frames)
        return {"height": height, "width": width, "num_frames": num_frames}



class WanVideoUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames", "seed", "rand_device", "vace_reference_image", "num_views", "batch_size"),
            output_params=("noise",)
        )

    def process(self, pipe: WanVideoPipeline, height, width, num_frames, seed, rand_device, vace_reference_image, num_views, batch_size):
        length = (num_frames - 1) // 4 + 1
        if vace_reference_image is not None:
            batch = int(batch_size) if batch_size is not None else 1
            if isinstance(vace_reference_image, (list, tuple)):
                if vace_reference_image and isinstance(vace_reference_image[0], (list, tuple)):
                    f = len(vace_reference_image[0])
                elif batch > 1 and len(vace_reference_image) == batch:
                    f = 1
                else:
                    f = len(vace_reference_image)
            else:
                f = 1
            length += f
        view_count = int(num_views) if num_views is not None else 1
        view_count = max(view_count, 1)
        target_width = width * view_count
        batch = int(batch_size) if batch_size is not None else 1
        shape = (batch, pipe.vae.model.z_dim, length, height // pipe.vae.upsampling_factor, target_width // pipe.vae.upsampling_factor)
        noise = pipe.generate_noise(shape, seed=seed, rand_device=rand_device)
        if vace_reference_image is not None:
            noise = torch.concat((noise[:, :, -f:], noise[:, :, :-f]), dim=2)
        return {"noise": noise}
    


class WanVideoUnit_InputVideoEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_video", "noise", "tiled", "tile_size", "tile_stride", "vace_reference_image", "batch_size"),
            output_params=("latents", "input_latents"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, input_video, noise, tiled, tile_size, tile_stride, vace_reference_image, batch_size):
        if input_video is None:
            return {"latents": noise}
        pipe.load_models_to_device(self.onload_model_names)
        batch = int(batch_size) if batch_size is not None else 1

        def normalize_batch_multi_view(video):
            if batch <= 1:
                return video
            if not isinstance(video, (list, tuple)) or len(video) != batch:
                return video
            first = video[0]
            if not isinstance(first, (list, tuple)) or len(first) == 0:
                return video
            if not isinstance(first[0], (list, tuple)):
                return video
            num_views = len(first)
            view_videos = [[] for _ in range(num_views)]
            for sample in video:
                for view_idx, view_video in enumerate(sample):
                    view_videos[view_idx].append(view_video)
            return view_videos

        def is_multi_view(video):
            if not isinstance(video, (list, tuple)) or len(video) == 0:
                return False
            if not isinstance(video[0], (list, tuple)):
                return False
            if batch > 1:
                return isinstance(video[0][0], (list, tuple))
            return True

        input_video = normalize_batch_multi_view(input_video)
        multi_view = is_multi_view(input_video)
        num_views = len(input_video) if multi_view else 1
        if multi_view:
            view_latents = []
            for view_video in input_video:
                view_tensor = pipe.preprocess_video(view_video)
                view_latent = pipe.vae.encode(view_tensor, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
                view_latents.append(view_latent.to(dtype=pipe.torch_dtype, device=pipe.device))
            input_latents = torch.cat(view_latents, dim=-1)
        else:
            input_video = pipe.preprocess_video(input_video)
            input_latents = pipe.vae.encode(input_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        if vace_reference_image is not None:
            ref_multi_view = (
                multi_view
                and isinstance(vace_reference_image, (list, tuple))
                and len(vace_reference_image) == num_views
                and (
                    (batch <= 1 and len(vace_reference_image) > 0
                        and isinstance(vace_reference_image[0], (list, tuple)))
                    or
                    (batch > 1 and len(vace_reference_image) > 0
                        and isinstance(vace_reference_image[0], (list, tuple))
                        and isinstance(vace_reference_image[0][0], (list, tuple)))
                )
            )
            if ref_multi_view:
                view_ref_latents = []
                for view_ref in vace_reference_image:
                    view_ref_tensor = pipe.preprocess_video(view_ref)
                    view_ref_latent = pipe.vae.encode(view_ref_tensor, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
                    view_ref_latents.append(view_ref_latent)
                vace_reference_latents = torch.cat(view_ref_latents, dim=-1)
            else:
                if not isinstance(vace_reference_image, (list, tuple)):
                    if batch > 1:
                        vace_reference_image = [[vace_reference_image] for _ in range(batch)]
                    else:
                        vace_reference_image = [vace_reference_image]
                else:
                    if batch > 1 and vace_reference_image and not isinstance(vace_reference_image[0], (list, tuple)):
                        if len(vace_reference_image) == batch:
                            vace_reference_image = [[img] for img in vace_reference_image]
                vace_reference_image = pipe.preprocess_video(vace_reference_image)
                vace_reference_latents = pipe.vae.encode(vace_reference_image, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
            input_latents = torch.concat([vace_reference_latents, input_latents], dim=2)
        if pipe.scheduler.training:
            return {"latents": noise, "input_latents": input_latents}
        else:
            latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
            latents = latents.to(dtype=pipe.torch_dtype, device=pipe.device)
            return {"latents": latents}



class WanVideoUnit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt", "positive": "positive"},
            input_params_nega={"prompt": "negative_prompt", "positive": "positive"},
            output_params=("context",),
            onload_model_names=("text_encoder",)
        )
    
    def encode_prompt(self, pipe: WanVideoPipeline, prompt):
        if isinstance(prompt, (list, tuple)):
            ids, mask = pipe.tokenizer(list(prompt), return_mask=True, add_special_tokens=True)
        else:
            ids, mask = pipe.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(pipe.device)
        mask = mask.to(pipe.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        prompt_emb = pipe.text_encoder(ids, mask)
        for i, v in enumerate(seq_lens):
            prompt_emb[i, v:] = 0
        return prompt_emb

    def process(self, pipe: WanVideoPipeline, prompt, positive) -> dict:
        pipe.load_models_to_device(self.onload_model_names)
        prompt_emb = self.encode_prompt(pipe, prompt)
        return {"context": prompt_emb}



class WanVideoUnit_ImageEmbedderCLIP(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "end_image", "height", "width", "batch_size"),
            output_params=("clip_feature",),
            onload_model_names=("image_encoder",)
        )

    def process(self, pipe: WanVideoPipeline, input_image, end_image, height, width, batch_size):
        if input_image is None or pipe.image_encoder is None or not pipe.dit.require_clip_embedding:
            return {}
        batch = int(batch_size) if batch_size is not None else 1

        def normalize_images(images):
            if images is None:
                return []
            if isinstance(images, (list, tuple)):
                if images and isinstance(images[0], (list, tuple)):
                    images = images[0]
                if batch > 1:
                    return list(images)
                return [images[0]] if images else []
            return [images]

        input_images = normalize_images(input_image)
        if not input_images:
            return {}
        end_images = normalize_images(end_image) if end_image is not None else []
        pipe.load_models_to_device(self.onload_model_names)
        image_tensors = [
            pipe.preprocess_image(image.resize((width, height))).to(pipe.device)
            for image in input_images
        ]
        clip_context = pipe.image_encoder.encode_image(image_tensors)
        if end_images:
            end_image_tensors = [
                pipe.preprocess_image(image.resize((width, height))).to(pipe.device)
                for image in end_images
            ]
            if pipe.dit.has_image_pos_emb:
                clip_context = torch.concat([clip_context, pipe.image_encoder.encode_image(end_image_tensors)], dim=1)
        clip_context = clip_context.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"clip_feature": clip_context}
    


class WanVideoUnit_ImageEmbedderVAE(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "end_image", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride", "batch_size"),
            output_params=("y",),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, input_image, end_image, num_frames, height, width, tiled, tile_size, tile_stride, batch_size):
        if input_image is None or not pipe.dit.require_vae_embedding:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        batch = int(batch_size) if batch_size is not None else 1

        def normalize_view_list(images):
            if images is None:
                return []
            if isinstance(images, (list, tuple)):
                if images and isinstance(images[0], (list, tuple)):
                    if batch > 1 and len(images) == batch:
                        num_views = len(images[0])
                        view_images = [[] for _ in range(num_views)]
                        for sample in images:
                            for view_idx, view_img in enumerate(sample):
                                view_images[view_idx].append(view_img)
                        return view_images
                    return [list(view) for view in images]
                if batch > 1:
                    return [list(images)]
                return [[img] for img in images]
            return [[images]]

        def build_mask(has_end_image):
            mask = torch.ones(
                1,
                num_frames,
                height // 8,
                width // 8,
                device=pipe.device,
                dtype=pipe.torch_dtype,
            )
            mask[:, 1:] = 0
            if has_end_image:
                mask[:, -1:] = 1
            mask = torch.concat([torch.repeat_interleave(mask[:, 0:1], repeats=4, dim=1), mask[:, 1:]], dim=1)
            mask = mask.view(1, mask.shape[1] // 4, 4, height // 8, width // 8)
            mask = mask.transpose(1, 2)[0]
            return mask

        view_images = normalize_view_list(input_image)
        if not view_images:
            return {}
        view_end_images = normalize_view_list(end_image) if end_image is not None else []
        batch_len = len(view_images[0])
        y_views = []

        for idx, view_images_single in enumerate(view_images):
            view_end_images_single = view_end_images[idx] if idx < len(view_end_images) else [None] * batch_len
            has_end_image = bool(view_end_images_single) and view_end_images_single[0] is not None
            mask = build_mask(has_end_image).to(pipe.device)
            mask = mask.to(dtype=pipe.torch_dtype, device=pipe.device)
            mask = mask.unsqueeze(0).expand(batch_len, -1, -1, -1, -1)

            vae_inputs = []
            for image, view_end_image in zip(view_images_single, view_end_images_single):
                image_tensor = pipe.preprocess_image(image.resize((width, height))).to(pipe.device)
                if view_end_image is not None:
                    end_image_tensor = pipe.preprocess_image(view_end_image.resize((width, height))).to(pipe.device)
                    vae_input = torch.concat(
                        [image_tensor.transpose(0, 1), torch.zeros(3, num_frames - 2, height, width).to(image_tensor.device), end_image_tensor.transpose(0, 1)],
                        dim=1,
                    )
                else:
                    vae_input = torch.concat(
                        [image_tensor.transpose(0, 1), torch.zeros(3, num_frames - 1, height, width).to(image_tensor.device)],
                        dim=1,
                    )
                vae_inputs.append(vae_input.to(dtype=pipe.torch_dtype, device=pipe.device))

            y = pipe.vae.encode(vae_inputs, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
            y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
            y = torch.concat([mask, y], dim=1)
            y_views.append(y)

        y = torch.cat(y_views, dim=-1)
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"y": y}



class WanVideoUnit_ImageEmbedderFused(PipelineUnit):
    """
    Encode input image to latents using VAE. This unit is for Wan-AI/Wan2.2-TI2V-5B.
    """
    def __init__(self):
        super().__init__(
            input_params=("input_image", "latents", "height", "width", "tiled", "tile_size", "tile_stride", "batch_size"),
            output_params=("latents", "fuse_vae_embedding_in_latents", "first_frame_latents"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, input_image, latents, height, width, tiled, tile_size, tile_stride, batch_size):
        if input_image is None or not pipe.dit.fuse_vae_embedding_in_latents:
            return {}
        batch = int(batch_size) if batch_size is not None else 1

        def normalize_images(images):
            if images is None:
                return []
            if isinstance(images, (list, tuple)):
                if images and isinstance(images[0], (list, tuple)):
                    images = images[0]
                if batch > 1:
                    return list(images)
                return [images[0]] if images else []
            return [images]

        images = normalize_images(input_image)
        if not images:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        vae_inputs = [
            pipe.preprocess_image(image.resize((width, height))).transpose(0, 1)
            for image in images
        ]
        z = pipe.vae.encode(vae_inputs, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        latents[:, :, 0: 1] = z
        return {"latents": latents, "fuse_vae_embedding_in_latents": True, "first_frame_latents": z}



class WanVideoUnit_FunControl(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("control_video", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride", "clip_feature", "y", "latents"),
            output_params=("clip_feature", "y"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, control_video, num_frames, height, width, tiled, tile_size, tile_stride, clip_feature, y, latents):
        if control_video is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        control_video = pipe.preprocess_video(control_video)
        control_latents = pipe.vae.encode(control_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        control_latents = control_latents.to(dtype=pipe.torch_dtype, device=pipe.device)
        batch = control_latents.shape[0]
        y_dim = pipe.dit.in_dim-control_latents.shape[1]-latents.shape[1]
        if clip_feature is None or y is None:
            clip_feature = torch.zeros((batch, 257, 1280), dtype=pipe.torch_dtype, device=pipe.device)
            y = torch.zeros((batch, y_dim, (num_frames - 1) // 4 + 1, height//8, width//8), dtype=pipe.torch_dtype, device=pipe.device)
        else:
            y = y[:, -y_dim:]
        y = torch.concat([control_latents, y], dim=1)
        return {"clip_feature": clip_feature, "y": y}
    


class WanVideoUnit_FunReference(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("reference_image", "height", "width", "reference_image", "batch_size"),
            output_params=("reference_latents", "clip_feature"),
            onload_model_names=("vae", "image_encoder")
        )

    def process(self, pipe: WanVideoPipeline, reference_image, height, width, batch_size):
        if reference_image is None:
            return {}
        batch = int(batch_size) if batch_size is not None else 1
        if isinstance(reference_image, (list, tuple)):
            if batch > 1:
                images = list(reference_image)
            else:
                images = [reference_image[0]] if reference_image else []
        else:
            images = [reference_image]
        if not images:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        if batch > 1:
            ref_videos = [[image.resize((width, height))] for image in images]
        else:
            ref_videos = [images[0].resize((width, height))]
        reference_latents = pipe.preprocess_video(ref_videos)
        reference_latents = pipe.vae.encode(reference_latents, device=pipe.device)
        if pipe.image_encoder is None:
            return {"reference_latents": reference_latents}
        clip_images = [
            pipe.preprocess_image(image.resize((width, height)))
            for image in images
        ]
        clip_feature = pipe.image_encoder.encode_image(clip_images)
        return {"reference_latents": reference_latents, "clip_feature": clip_feature}



class WanVideoUnit_FunCameraControl(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames", "camera_control_direction", "camera_control_speed", "camera_control_origin", "latents", "input_image", "tiled", "tile_size", "tile_stride"),
            output_params=("control_camera_latents_input", "y"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, height, width, num_frames, camera_control_direction, camera_control_speed, camera_control_origin, latents, input_image, tiled, tile_size, tile_stride):
        if camera_control_direction is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        camera_control_plucker_embedding = pipe.dit.control_adapter.process_camera_coordinates(
            camera_control_direction, num_frames, height, width, camera_control_speed, camera_control_origin)
        
        control_camera_video = camera_control_plucker_embedding[:num_frames].permute([3, 0, 1, 2]).unsqueeze(0)
        control_camera_latents = torch.concat(
            [
                torch.repeat_interleave(control_camera_video[:, :, 0:1], repeats=4, dim=2),
                control_camera_video[:, :, 1:]
            ], dim=2
        ).transpose(1, 2)
        b, f, c, h, w = control_camera_latents.shape
        control_camera_latents = control_camera_latents.contiguous().view(b, f // 4, 4, c, h, w).transpose(2, 3)
        control_camera_latents = control_camera_latents.contiguous().view(b, f // 4, c * 4, h, w).transpose(1, 2)
        control_camera_latents_input = control_camera_latents.to(device=pipe.device, dtype=pipe.torch_dtype)
        
        input_image = input_image.resize((width, height))
        input_latents = pipe.preprocess_video([input_image])
        input_latents = pipe.vae.encode(input_latents, device=pipe.device)
        y = torch.zeros_like(latents).to(pipe.device)
        y[:, :, :1] = input_latents
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)

        if y.shape[1] != pipe.dit.in_dim - latents.shape[1]:
            image = pipe.preprocess_image(input_image.resize((width, height))).to(pipe.device)
            vae_input = torch.concat([image.transpose(0, 1), torch.zeros(3, num_frames-1, height, width).to(image.device)], dim=1)
            y = pipe.vae.encode([vae_input.to(dtype=pipe.torch_dtype, device=pipe.device)], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
            y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
            msk = torch.ones(1, num_frames, height//8, width//8, device=pipe.device)
            msk[:, 1:] = 0
            msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
            msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
            msk = msk.transpose(1, 2)[0]
            y = torch.cat([msk,y])
            y = y.unsqueeze(0)
            y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"control_camera_latents_input": control_camera_latents_input, "y": y}



class WanVideoUnit_SpeedControl(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("motion_bucket_id",),
            output_params=("motion_bucket_id",)
        )

    def process(self, pipe: WanVideoPipeline, motion_bucket_id):
        if motion_bucket_id is None:
            return {}
        motion_bucket_id = torch.Tensor((motion_bucket_id,)).to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"motion_bucket_id": motion_bucket_id}



class WanVideoUnit_VACE(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=(
                "vace_video",
                "vace_rgb_video",
                "vace_depth_video",
                "vace_mask_video",
                "vace_video_mask",
                "vace_reference_image",
                "vace_intrinsic",
                "vace_extrinsic",
                "vace_scale",
                "height",
                "width",
                "num_frames",
                "tiled",
                "tile_size",
                "tile_stride",
                "batch_size",
            ),
            output_params=("vace_context", "vace_rgbd_latents", "vace_mask_cam", "vace_scale"),
            onload_model_names=("vae",)
        )

    def _normalize_reference_image_input(self, vace_reference_image, batch):
        if not isinstance(vace_reference_image, (list, tuple)):
            if batch > 1:
                return [[vace_reference_image] for _ in range(batch)]
            return [vace_reference_image]
        if batch > 1 and vace_reference_image and not isinstance(vace_reference_image[0], (list, tuple)):
            if len(vace_reference_image) == batch:
                return [[img] for img in vace_reference_image]
        return vace_reference_image

    def _prepare_depth_video_tensor(self, pipe: WanVideoPipeline, vace_depth_video, height, width):
        if isinstance(vace_depth_video, torch.Tensor):
            depth_tensor = vace_depth_video
            if depth_tensor.ndim == 3:
                depth_tensor = depth_tensor.unsqueeze(0).unsqueeze(0)
            elif depth_tensor.ndim == 4:
                depth_tensor = depth_tensor.unsqueeze(1)
            elif depth_tensor.ndim == 5:
                if depth_tensor.shape[1] == 1:
                    pass
                elif depth_tensor.shape[-1] == 1:
                    depth_tensor = depth_tensor.squeeze(-1).unsqueeze(1)
                else:
                    raise ValueError(f"Unsupported depth tensor shape: {tuple(depth_tensor.shape)}")
            else:
                raise ValueError(f"Unsupported depth tensor ndim: {depth_tensor.ndim}")
            depth_tensor = depth_tensor.to(device=pipe.device, dtype=torch.float32)
        else:
            if not isinstance(vace_depth_video, (list, tuple)) or len(vace_depth_video) == 0:
                raise ValueError("vace_depth_video must be a non-empty sequence of frames or videos.")
            if isinstance(vace_depth_video[0], (list, tuple)):
                depth_videos = list(vace_depth_video)
            else:
                depth_videos = [vace_depth_video]

            batched_tensors = []
            for sample_id, depth_video in enumerate(depth_videos):
                if not isinstance(depth_video, (list, tuple)) or len(depth_video) == 0:
                    raise ValueError(f"vace_depth_video[{sample_id}] must be a non-empty sequence of frames.")
                frame_tensors = []
                for frame_id, frame in enumerate(depth_video):
                    if isinstance(frame, torch.Tensor):
                        frame_np = frame.detach().cpu().numpy()
                    elif isinstance(frame, Image.Image):
                        frame_np = np.array(frame)
                    else:
                        frame_np = np.asarray(frame)

                    if frame_np.ndim == 3 and frame_np.shape[-1] == 1:
                        frame_np = frame_np[..., 0]
                    if frame_np.ndim != 2:
                        raise ValueError(
                            f"Depth frame must be 2D. Got shape={frame_np.shape} at sample={sample_id}, frame={frame_id}."
                        )
                    if not np.issubdtype(frame_np.dtype, np.number):
                        raise ValueError(
                            f"Depth frame dtype must be numeric. Got dtype={frame_np.dtype} at sample={sample_id}, frame={frame_id}."
                        )
                    frame_tensors.append(torch.from_numpy(frame_np.astype(np.float32, copy=False)))
                batched_tensors.append(torch.stack(frame_tensors, dim=0))
            depth_tensor = torch.stack(batched_tensors, dim=0).unsqueeze(1).to(device=pipe.device, dtype=torch.float32)

        if depth_tensor.shape[-2:] != (height, width):
            depth_tensor = torch.nn.functional.interpolate(
                depth_tensor, size=(depth_tensor.shape[2], height, width), mode="nearest-exact"
            )
        return depth_tensor

    def _prepare_mask_video_tensor(self, pipe: WanVideoPipeline, vace_mask_video, height, width):
        if isinstance(vace_mask_video, torch.Tensor):
            mask_tensor = vace_mask_video
            if mask_tensor.ndim == 3:
                mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
            elif mask_tensor.ndim == 4:
                mask_tensor = mask_tensor.unsqueeze(1)
            elif mask_tensor.ndim == 5:
                if mask_tensor.shape[1] == 1:
                    pass
                elif mask_tensor.shape[-1] == 1:
                    mask_tensor = mask_tensor.squeeze(-1).unsqueeze(1)
                else:
                    raise ValueError(f"Unsupported mask tensor shape: {tuple(mask_tensor.shape)}")
            else:
                raise ValueError(f"Unsupported mask tensor ndim: {mask_tensor.ndim}")
            mask_tensor = (mask_tensor.to(device=pipe.device, dtype=torch.float32) > 0).float()
        else:
            if not isinstance(vace_mask_video, (list, tuple)) or len(vace_mask_video) == 0:
                raise ValueError("vace_mask_video must be a non-empty sequence of frames or videos.")
            if isinstance(vace_mask_video[0], (list, tuple)):
                mask_videos = list(vace_mask_video)
            else:
                mask_videos = [vace_mask_video]

            batched_tensors = []
            for sample_id, mask_video in enumerate(mask_videos):
                if not isinstance(mask_video, (list, tuple)) or len(mask_video) == 0:
                    raise ValueError(f"vace_mask_video[{sample_id}] must be a non-empty sequence of frames.")
                frame_tensors = []
                for frame_id, frame in enumerate(mask_video):
                    if isinstance(frame, torch.Tensor):
                        frame_np = frame.detach().cpu().numpy()
                    elif isinstance(frame, Image.Image):
                        frame_np = np.array(frame)
                    else:
                        frame_np = np.asarray(frame)

                    if frame_np.ndim == 3 and frame_np.shape[-1] == 1:
                        frame_np = frame_np[..., 0]
                    if frame_np.ndim != 2:
                        raise ValueError(
                            f"Mask frame must be 2D. Got shape={frame_np.shape} at sample={sample_id}, frame={frame_id}."
                        )
                    if not np.issubdtype(frame_np.dtype, np.number) and frame_np.dtype != np.bool_:
                        raise ValueError(
                            f"Mask frame dtype must be numeric/bool. Got dtype={frame_np.dtype} at sample={sample_id}, frame={frame_id}."
                        )
                    frame_tensors.append(torch.from_numpy((frame_np > 0).astype(np.float32, copy=False)))
                batched_tensors.append(torch.stack(frame_tensors, dim=0))
            mask_tensor = torch.stack(batched_tensors, dim=0).unsqueeze(1).to(device=pipe.device, dtype=torch.float32)

        if mask_tensor.shape[-2:] != (height, width):
            mask_tensor = torch.nn.functional.interpolate(
                mask_tensor, size=(mask_tensor.shape[2], height, width), mode="nearest-exact"
            )
        return mask_tensor

    def _normalize_depth_to_pseudo_rgb(self, pipe: WanVideoPipeline, depth_tensor):
        depth_valid = depth_tensor > 0
        depth_normalized = torch.zeros_like(depth_tensor, dtype=torch.float32, device=pipe.device)

        for sample_id in range(depth_tensor.shape[0]):
            sample_depth = depth_tensor[sample_id, 0]
            sample_valid = depth_valid[sample_id, 0]
            if not torch.any(sample_valid):
                continue

            valid_values = sample_depth[sample_valid]
            p2 = valid_values.min()
            p98 = valid_values.max()

            if (p98 - p2) <= 1e-6:
                sample_norm = torch.zeros_like(sample_depth)
            else:
                sample_norm = ((sample_depth - p2) / (p98 - p2)).clamp(0, 1)

            depth_normalized[sample_id, 0] = sample_norm * 2 - 1

        depth_rgb = depth_normalized.repeat(1, 3, 1, 1, 1).to(dtype=pipe.torch_dtype, device=pipe.device)
        return depth_rgb

    def _validate_vace_context_channels(self, pipe: WanVideoPipeline, vace_context):
        if vace_context is None:
            return
        expected_dims = []
        for model_name in ("vace", "vace2"):
            model = getattr(pipe, model_name, None)
            if model is None:
                continue
            expected = getattr(model, "vace_in_dim", None)
            if expected is not None:
                expected_dims.append((model_name, int(expected)))
        for model_name, expected in expected_dims:
            if int(vace_context.shape[1]) != expected:
                raise ValueError(
                    f"{model_name}.vace_in_dim={expected}, but generated vace_context has "
                    f"{int(vace_context.shape[1])} channels. Please align VACE model config/checkpoint with the "
                    "new context layout."
                )

    def _build_legacy_context(
        self,
        pipe: WanVideoPipeline,
        vace_video,
        vace_video_mask,
        vace_reference_image,
        height,
        width,
        num_frames,
        tiled,
        tile_size,
        tile_stride,
        batch,
    ):
        if vace_video is not None:
            vace_video = pipe.preprocess_video(vace_video)
            batch = vace_video.shape[0]
        if vace_video_mask is not None:
            vace_video_mask = pipe.preprocess_video(vace_video_mask, min_value=0, max_value=1)
            if vace_video is None:
                batch = vace_video_mask.shape[0]

        if vace_video is None:
            vace_video = torch.zeros((batch, 3, num_frames, height, width), dtype=pipe.torch_dtype, device=pipe.device)
        if vace_video_mask is None:
            vace_video_mask = torch.ones_like(vace_video)

        inactive = vace_video * (1 - vace_video_mask) + 0 * vace_video_mask
        reactive = vace_video * vace_video_mask + 0 * (1 - vace_video_mask)
        inactive = pipe.vae.encode(inactive, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        reactive = pipe.vae.encode(reactive, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        vace_video_latents = torch.concat((inactive, reactive), dim=1)

        vace_mask_latents = rearrange(vace_video_mask[:, 0], "B T (H P) (W Q) -> B (P Q) T H W", P=8, Q=8)
        vace_mask_latents = torch.nn.functional.interpolate(
            vace_mask_latents,
            size=((vace_mask_latents.shape[2] + 3) // 4, vace_mask_latents.shape[3], vace_mask_latents.shape[4]),
            mode="nearest-exact",
        )

        if vace_reference_image is not None:
            vace_reference_image = self._normalize_reference_image_input(vace_reference_image, batch)
            vace_reference_image = pipe.preprocess_video(vace_reference_image)
            f = vace_reference_image.shape[2]
            vace_reference_latents = []
            for j in range(f):
                frame_tensor = vace_reference_image[:, :, j:j+1]
                frame_latent = pipe.vae.encode(
                    frame_tensor, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride
                ).to(dtype=pipe.torch_dtype, device=pipe.device)
                frame_latent = torch.concat((frame_latent, torch.zeros_like(frame_latent)), dim=1)
                vace_reference_latents.append(frame_latent)

            if vace_reference_latents:
                vace_video_latents = torch.concat((*vace_reference_latents, vace_video_latents), dim=2)
                vace_mask_latents = torch.concat((torch.zeros_like(vace_mask_latents[:, :, :f]), vace_mask_latents), dim=2)

        return torch.concat((vace_video_latents, vace_mask_latents), dim=1)

    @staticmethod
    def _is_multi_view_vace_input(value, batch):
        if value is None or isinstance(value, torch.Tensor):
            return False
        if not isinstance(value, (list, tuple)) or len(value) == 0:
            return False
        inner = value[0]
        if not isinstance(inner, (list, tuple)) or len(inner) == 0:
            return False
        if batch > 1:
            return isinstance(inner[0], (list, tuple))
        return True

    @staticmethod
    def _slice_view(value, view_idx, is_multi_view):
        if not is_multi_view or value is None or isinstance(value, torch.Tensor):
            return value
        return value[view_idx]

    def process(
        self,
        pipe: WanVideoPipeline,
        vace_video, vace_rgb_video, vace_depth_video, vace_mask_video, vace_video_mask, vace_reference_image,
        vace_intrinsic, vace_extrinsic, vace_scale,
        height, width, num_frames,
        tiled, tile_size, tile_stride, batch_size
    ):
        has_multi_condition = any(x is not None for x in (vace_rgb_video, vace_depth_video, vace_mask_video))
        has_legacy_condition = (vace_video is not None) or (vace_video_mask is not None)
        has_any_condition = has_multi_condition or has_legacy_condition or (vace_reference_image is not None)
        if not has_any_condition:
            return {
                "vace_context": None,
                "vace_rgbd_latents": None,
                "vace_mask_cam": None,
                "vace_scale": vace_scale,
            }
        pipe.load_models_to_device(["vae"])
        batch = int(batch_size) if batch_size is not None else 1

        use_plucker = bool(
            getattr(pipe, "vace", None) is not None
            and getattr(pipe.vace, "use_plucker", False)
            and vace_intrinsic is not None
            and vace_extrinsic is not None
        )

        mv_rgb = self._is_multi_view_vace_input(vace_rgb_video, batch)
        mv_depth = self._is_multi_view_vace_input(vace_depth_video, batch)
        mv_mask = self._is_multi_view_vace_input(vace_mask_video, batch)
        mv_ref = self._is_multi_view_vace_input(vace_reference_image, batch)
        multi_view = mv_rgb or mv_depth or mv_mask or mv_ref

        if multi_view:
            if has_multi_condition and not (mv_rgb and mv_depth and mv_mask):
                raise ValueError(
                    "VACE multi-view requires vace_rgb_video, vace_depth_video and vace_mask_video to be per-view lists together."
                )
            view_counts = []
            if mv_rgb: view_counts.append(len(vace_rgb_video))
            if mv_depth: view_counts.append(len(vace_depth_video))
            if mv_mask: view_counts.append(len(vace_mask_video))
            if mv_ref: view_counts.append(len(vace_reference_image))
            if len(set(view_counts)) != 1:
                raise ValueError(
                    f"Inconsistent num_views across VACE multi-view inputs: "
                    f"rgb={len(vace_rgb_video) if mv_rgb else None}, "
                    f"depth={len(vace_depth_video) if mv_depth else None}, "
                    f"mask={len(vace_mask_video) if mv_mask else None}, "
                    f"reference={len(vace_reference_image) if mv_ref else None}"
                )
            num_views = view_counts[0]

            view_results = []
            for view_idx in range(num_views):
                view_results.append(self._build_single_view_context(
                    pipe=pipe,
                    vace_video=self._slice_view(vace_video, view_idx, False),  # legacy unused in multi-view
                    vace_rgb_video=self._slice_view(vace_rgb_video, view_idx, mv_rgb),
                    vace_depth_video=self._slice_view(vace_depth_video, view_idx, mv_depth),
                    vace_mask_video=self._slice_view(vace_mask_video, view_idx, mv_mask),
                    vace_video_mask=vace_video_mask,
                    vace_reference_image=self._slice_view(vace_reference_image, view_idx, mv_ref),
                    vace_intrinsic=self._slice_per_view_camera_field(vace_intrinsic, view_idx) if use_plucker else None,
                    vace_extrinsic=self._slice_per_view_camera_field(vace_extrinsic, view_idx) if use_plucker else None,
                    use_plucker=use_plucker,
                    height=height, width=width, num_frames=num_frames,
                    tiled=tiled, tile_size=tile_size, tile_stride=tile_stride,
                    batch=batch,
                    has_multi_condition=has_multi_condition,
                ))
            if use_plucker:
                rgbd_views = [r["vace_rgbd_latents"] for r in view_results]
                mask_cam_views = [r["vace_mask_cam"] for r in view_results]
                vace_rgbd_latents = torch.cat(rgbd_views, dim=-1)
                vace_mask_cam = torch.cat(mask_cam_views, dim=-1)
                vace_context = None
            else:
                view_contexts = [r["vace_context"] for r in view_results]
                vace_context = torch.cat(view_contexts, dim=-1)
                vace_rgbd_latents = None
                vace_mask_cam = None
        else:
            single = self._build_single_view_context(
                pipe=pipe,
                vace_video=vace_video,
                vace_rgb_video=vace_rgb_video,
                vace_depth_video=vace_depth_video,
                vace_mask_video=vace_mask_video,
                vace_video_mask=vace_video_mask,
                vace_reference_image=vace_reference_image,
                vace_intrinsic=vace_intrinsic if use_plucker else None,
                vace_extrinsic=vace_extrinsic if use_plucker else None,
                use_plucker=use_plucker,
                height=height, width=width, num_frames=num_frames,
                tiled=tiled, tile_size=tile_size, tile_stride=tile_stride,
                batch=batch,
                has_multi_condition=has_multi_condition,
            )
            if use_plucker:
                vace_rgbd_latents = single["vace_rgbd_latents"]
                vace_mask_cam = single["vace_mask_cam"]
                vace_context = None
            else:
                vace_context = single["vace_context"]
                vace_rgbd_latents = None
                vace_mask_cam = None

        if vace_context is not None:
            self._validate_vace_context_channels(pipe, vace_context)
        if use_plucker:
            self._validate_plucker_inputs(pipe, vace_rgbd_latents, vace_mask_cam)
        return {
            "vace_context": vace_context,
            "vace_rgbd_latents": vace_rgbd_latents,
            "vace_mask_cam": vace_mask_cam,
            "vace_scale": vace_scale,
        }

    @staticmethod
    def _slice_per_view_camera_field(value, view_idx: int):
        """Per-view K/extr structure: outer is list[num_views]; inner is per-sample.

        parse_extra_inputs emits e.g. vace_intrinsic = [K_head, K_hand_left, K_hand_right]
        for batch=1, or [[K_head_b0, K_head_b1], [K_hand_left_b0, ...], ...] for batch>1.
        """
        if value is None:
            return None
        if not isinstance(value, (list, tuple)) or view_idx >= len(value):
            raise ValueError(
                f"Expected per-view camera field as a list/tuple of length >= num_views; "
                f"got type={type(value).__name__}, len={len(value) if hasattr(value, '__len__') else 'n/a'}, "
                f"view_idx={view_idx}."
            )
        return value[view_idx]

    @staticmethod
    def _validate_plucker_inputs(pipe, rgbd, mask_cam):
        if rgbd is None or mask_cam is None:
            raise ValueError("Plücker mode requires both vace_rgbd_latents and vace_mask_cam.")
        if rgbd.dim() != 5 or rgbd.shape[1] != 32:
            raise ValueError(f"vace_rgbd_latents must be [B, 32, T_lat, H, W]; got {tuple(rgbd.shape)}")
        if mask_cam.dim() != 5 or mask_cam.shape[1] != 7:
            raise ValueError(f"vace_mask_cam must be [B, 7, T, H, W]; got {tuple(mask_cam.shape)}")
        # Spatial alignment: H_lat = H/8, W_lat = W/8
        if mask_cam.shape[3] != rgbd.shape[3] * 8 or mask_cam.shape[4] != rgbd.shape[4] * 8:
            raise ValueError(
                f"Spatial mismatch: vace_mask_cam (H,W)=({mask_cam.shape[3]},{mask_cam.shape[4]}) "
                f"must be 8× vace_rgbd_latents (H,W)=({rgbd.shape[3]},{rgbd.shape[4]})."
            )

    @staticmethod
    def _stack_camera_field(value, expected_shape, batch: int, field_name: str) -> torch.Tensor:
        """Coerce a per-sample camera field (ndarray / tensor / list) to a stacked [B, *expected_shape] tensor.

        Accepts:
          - ndarray/tensor of shape expected_shape (batch=1)
          - list/tuple of length batch, each element shape expected_shape
          - already-stacked tensor of shape [B, *expected_shape]
        """
        if value is None:
            raise ValueError(f"{field_name} is required when use_plucker=True but got None.")
        if isinstance(value, torch.Tensor) and value.dim() == len(expected_shape) + 1:
            tensor = value
        elif isinstance(value, (list, tuple)):
            if batch > 1 and len(value) != batch:
                raise ValueError(
                    f"{field_name} list length ({len(value)}) does not match batch ({batch})."
                )
            stacked = [torch.as_tensor(v, dtype=torch.float32) for v in value]
            tensor = torch.stack(stacked, dim=0)
        else:
            tensor = torch.as_tensor(value, dtype=torch.float32).unsqueeze(0)
        if tensor.shape[1:] != tuple(expected_shape):
            raise ValueError(
                f"{field_name} shape mismatch: got {tuple(tensor.shape)}, expected [B, *{tuple(expected_shape)}]"
            )
        if tensor.shape[0] != batch:
            raise ValueError(
                f"{field_name} batch dim ({tensor.shape[0]}) does not match batch ({batch})."
            )
        return tensor.to(dtype=torch.float32)

    def _build_plucker_for_view(
        self,
        pipe: WanVideoPipeline,
        vace_intrinsic,
        vace_extrinsic,
        num_frames: int,
        height: int,
        width: int,
        batch: int,
    ) -> torch.Tensor:
        """Build plücker map [B, 6, T, H, W] for one view from K and c2w (OpenCV)."""
        from ..models.wan_video_camera_controller import build_plucker_from_K_c2w
        K = self._stack_camera_field(vace_intrinsic, (num_frames, 3, 3), batch, "vace_intrinsic")
        c2w = self._stack_camera_field(vace_extrinsic, (num_frames, 4, 4), batch, "vace_extrinsic")
        K = K.to(device=pipe.device)
        c2w = c2w.to(device=pipe.device)
        plucker = build_plucker_from_K_c2w(K, c2w, height=height, width=width)
        return plucker

    def _build_single_view_context(
        self, pipe: WanVideoPipeline,
        vace_video, vace_rgb_video, vace_depth_video, vace_mask_video,
        vace_video_mask, vace_reference_image,
        vace_intrinsic, vace_extrinsic, use_plucker,
        height, width, num_frames,
        tiled, tile_size, tile_stride, batch,
        has_multi_condition,
    ):
        if has_multi_condition:
            if not (vace_rgb_video is not None and vace_depth_video is not None and vace_mask_video is not None):
                raise ValueError(
                    "VACE multi-condition mode requires vace_rgb_video, vace_depth_video and vace_mask_video simultaneously."
                )

            rgb_video = pipe.preprocess_video(vace_rgb_video)
            depth_video = self._prepare_depth_video_tensor(pipe, vace_depth_video, height, width)
            mask_video = self._prepare_mask_video_tensor(pipe, vace_mask_video, height, width)

            batch = rgb_video.shape[0]
            if depth_video.shape[0] != batch or mask_video.shape[0] != batch:
                raise ValueError(
                    f"Batch size mismatch in VACE multi-condition inputs: "
                    f"rgb={rgb_video.shape[0]}, depth={depth_video.shape[0]}, mask={mask_video.shape[0]}"
                )

            if rgb_video.shape[2:] != depth_video.shape[2:] or rgb_video.shape[2:] != mask_video.shape[2:]:
                raise ValueError(
                    f"Shape mismatch in VACE multi-condition inputs: "
                    f"rgb={tuple(rgb_video.shape)}, depth={tuple(depth_video.shape)}, mask={tuple(mask_video.shape)}"
                )

            rgb_latent = pipe.vae.encode(
                rgb_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride
            ).to(dtype=pipe.torch_dtype, device=pipe.device)

            depth_rgb = self._normalize_depth_to_pseudo_rgb(pipe, depth_video)
            depth_latent = pipe.vae.encode(
                depth_rgb, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride
            ).to(dtype=pipe.torch_dtype, device=pipe.device)

            if rgb_latent.shape != depth_latent.shape:
                raise ValueError(
                    f"Latent shape mismatch in VACE multi-condition inputs: "
                    f"rgb={tuple(rgb_latent.shape)}, depth={tuple(depth_latent.shape)}"
                )

            if use_plucker:
                # Plücker path: skip pixel-unshuffle mask_latent; produce raw mask_cam (7ch)
                # at full T/H/W so VaceWanModel.encode_mask_cam can fuse mask+plücker via its
                # learned Conv3d, matching 4DVideo-WanX's mask_cam_embedding architecture.
                plucker = self._build_plucker_for_view(
                    pipe=pipe,
                    vace_intrinsic=vace_intrinsic,
                    vace_extrinsic=vace_extrinsic,
                    num_frames=num_frames,
                    height=height,
                    width=width,
                    batch=batch,
                )  # [B, 6, T, H, W] float32 on device
                if plucker.shape[2] != mask_video.shape[2]:
                    raise ValueError(
                        f"Plücker T ({plucker.shape[2]}) must match mask_video T ({mask_video.shape[2]})."
                    )
                # mask_video here is [B, 1, T, H, W] binary float on device.
                mask_for_cam = mask_video.to(device=pipe.device)
                mask_cam = torch.cat((mask_for_cam.to(dtype=plucker.dtype), plucker), dim=1)
                # rgb (16) + depth (16) at latent T resolution.
                rgbd_latent = torch.concat([rgb_latent, depth_latent], dim=1)

                # Reference image: prepend f ref-latent frames to rgbd; mask_cam is left
                # at target-only resolution (vace.forward zero-pads its latent on T to match).
                if vace_reference_image is not None:
                    vace_reference_image = self._normalize_reference_image_input(vace_reference_image, batch)
                    vace_reference_image = pipe.preprocess_video(vace_reference_image)
                    f = vace_reference_image.shape[2]
                    reference_rgbd = []
                    for j in range(f):
                        frame_tensor = vace_reference_image[:, :, j:j+1]
                        frame_latent = pipe.vae.encode(
                            frame_tensor, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride
                        ).to(dtype=pipe.torch_dtype, device=pipe.device)
                        # ref slot: rgb_latent + zero-depth_latent (mirrors legacy ref padding)
                        zero_depth = torch.zeros_like(frame_latent)
                        reference_rgbd.append(torch.concat((frame_latent, zero_depth), dim=1))
                    if reference_rgbd:
                        rgbd_latent = torch.concat((*reference_rgbd, rgbd_latent), dim=2)
                return {
                    "vace_context": None,
                    "vace_rgbd_latents": rgbd_latent,
                    "vace_mask_cam": mask_cam,
                }

            mask_latent = rearrange(mask_video[:, 0], "B T (H P) (W Q) -> B (P Q) T H W", P=8, Q=8)
            mask_latent = torch.nn.functional.interpolate(
                mask_latent,
                size=((mask_latent.shape[2] + 3) // 4, mask_latent.shape[3], mask_latent.shape[4]),
                mode="nearest-exact",
            ).to(dtype=pipe.torch_dtype, device=pipe.device)
            if (
                mask_latent.shape[0] != rgb_latent.shape[0]
                or mask_latent.shape[2:] != rgb_latent.shape[2:]
                or mask_latent.shape[1] != 64
            ):
                raise ValueError(
                    f"Mask latent shape mismatch for VACE context assembly: "
                    f"mask={tuple(mask_latent.shape)}, rgb={tuple(rgb_latent.shape)}"
                )

            vace_context = torch.concat([rgb_latent, depth_latent, mask_latent], dim=1)

            if vace_reference_image is not None:
                vace_reference_image = self._normalize_reference_image_input(vace_reference_image, batch)
                vace_reference_image = pipe.preprocess_video(vace_reference_image)
                f = vace_reference_image.shape[2]
                reference_context = []
                for j in range(f):
                    frame_tensor = vace_reference_image[:, :, j:j+1]
                    frame_latent = pipe.vae.encode(
                        frame_tensor, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride
                    ).to(dtype=pipe.torch_dtype, device=pipe.device)
                    zero_depth = torch.zeros_like(frame_latent)
                    zero_mask = torch.zeros(
                        (
                            frame_latent.shape[0],
                            mask_latent.shape[1],
                            frame_latent.shape[2],
                            frame_latent.shape[3],
                            frame_latent.shape[4],
                        ),
                        dtype=frame_latent.dtype,
                        device=frame_latent.device,
                    )
                    frame_context = torch.concat((frame_latent, zero_depth, zero_mask), dim=1)
                    reference_context.append(frame_context)
                if reference_context:
                    vace_context = torch.concat((*reference_context, vace_context), dim=2)
            return {
                "vace_context": vace_context,
                "vace_rgbd_latents": None,
                "vace_mask_cam": None,
            }

        legacy_context = self._build_legacy_context(
            pipe=pipe,
            vace_video=vace_video,
            vace_video_mask=vace_video_mask,
            vace_reference_image=vace_reference_image,
            height=height,
            width=width,
            num_frames=num_frames,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
            batch=batch,
        )
        return {
            "vace_context": legacy_context,
            "vace_rgbd_latents": None,
            "vace_mask_cam": None,
        }


class WanVideoUnit_VACE_new(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("vace_video", "vace_video_mask", "vace_reference_image", "vace_scale", "height", "width", "num_frames", "tiled", "tile_size", "tile_stride", "batch_size"),
            output_params=("vace_context", "vace_scale"),
            onload_model_names=("vae",)
        )

    def process(
        self,
        pipe: WanVideoPipeline,
        vace_video, vace_video_mask, vace_reference_image, vace_scale,
        height, width, num_frames,
        tiled, tile_size, tile_stride, batch_size
    ):
        if vace_video is not None or vace_video_mask is not None or vace_reference_image is not None:
            pipe.load_models_to_device(["vae"])
            batch = int(batch_size) if batch_size is not None else 1
            if vace_video is None:
                vace_video = None
            else:
                vace_video = pipe.preprocess_video(vace_video)
                batch = vace_video.shape[0]

            if vace_video_mask is None:
                vace_video_mask = None
            else:
                vace_video_mask = pipe.preprocess_video(vace_video_mask, min_value=0, max_value=1)
                if vace_video is None:
                    batch = vace_video_mask.shape[0]

            if vace_video is None:
                vace_video = torch.zeros((batch, 3, num_frames, height, width), dtype=pipe.torch_dtype, device=pipe.device)
            if vace_video_mask is None:
                vace_video_mask = torch.ones_like(vace_video)
            
            inactive = vace_video * (1 - vace_video_mask) + 0 * vace_video_mask
            reactive = vace_video * vace_video_mask + 0 * (1 - vace_video_mask)
            inactive = pipe.vae.encode(inactive, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
            reactive = pipe.vae.encode(reactive, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
            vace_video_latents = torch.concat((inactive, reactive), dim=1)
            
            vace_mask_latents = rearrange(vace_video_mask[:, 0], "B T (H P) (W Q) -> B (P Q) T H W", P=8, Q=8)
            vace_mask_latents = torch.nn.functional.interpolate(vace_mask_latents, size=((vace_mask_latents.shape[2] + 3) // 4, vace_mask_latents.shape[3], vace_mask_latents.shape[4]), mode='nearest-exact')
            
            if vace_reference_image is None:
                pass
            else:
                if not isinstance(vace_reference_image, (list, tuple)):
                    if batch > 1:
                        vace_reference_image = [[vace_reference_image] for _ in range(batch)]
                    else:
                        vace_reference_image = [vace_reference_image]
                else:
                    if batch > 1 and vace_reference_image and not isinstance(vace_reference_image[0], (list, tuple)):
                        if len(vace_reference_image) == batch:
                            vace_reference_image = [[img] for img in vace_reference_image]

                vace_reference_image = pipe.preprocess_video(vace_reference_image)

                f = vace_reference_image.shape[2]
                vace_reference_latents = []
                for j in range(f):
                    frame_tensor = vace_reference_image[:, :, j:j+1]
                    frame_latent = pipe.vae.encode(frame_tensor, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
                    frame_latent = torch.concat((frame_latent, torch.zeros_like(frame_latent)), dim=1)
                    vace_reference_latents.append(frame_latent)

                if vace_reference_latents:
                    vace_video_latents = torch.concat((*vace_reference_latents, vace_video_latents), dim=2)
                    vace_mask_latents = torch.concat((torch.zeros_like(vace_mask_latents[:, :, :f]), vace_mask_latents), dim=2)
            
            vace_context = torch.concat((vace_video_latents, vace_mask_latents), dim=1)
            return {"vace_context": vace_context, "vace_scale": vace_scale}
        else:
            return {"vace_context": None, "vace_scale": vace_scale}


class WanVideoUnit_VAP(PipelineUnit):
    def __init__(self):
        super().__init__(
            take_over=True,
            onload_model_names=("text_encoder", "vae", "image_encoder"),
            input_params=("vap_video", "vap_prompt", "negative_vap_prompt", "end_image", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride"),
            output_params=("vap_clip_feature", "vap_hidden_state", "context_vap")
        )
        
    def encode_prompt(self, pipe: WanVideoPipeline, prompt):
        if isinstance(prompt, (list, tuple)):
            ids, mask = pipe.tokenizer(list(prompt), return_mask=True, add_special_tokens=True)
        else:
            ids, mask = pipe.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(pipe.device)
        mask = mask.to(pipe.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        prompt_emb = pipe.text_encoder(ids, mask)
        for i, v in enumerate(seq_lens):
            prompt_emb[i, v:] = 0
        return prompt_emb

    def process(self, pipe: WanVideoPipeline, inputs_shared, inputs_posi, inputs_nega):
        if inputs_shared.get("vap_video") is None:
            return inputs_shared, inputs_posi, inputs_nega
        else:
            # 1. encode vap prompt
            pipe.load_models_to_device(["text_encoder"])
            vap_prompt, negative_vap_prompt = inputs_posi.get("vap_prompt", ""), inputs_nega.get("negative_vap_prompt", "")
            vap_prompt_emb = self.encode_prompt(pipe, vap_prompt)
            negative_vap_prompt_emb = self.encode_prompt(pipe, negative_vap_prompt)
            inputs_posi.update({"context_vap":vap_prompt_emb})
            inputs_nega.update({"context_vap":negative_vap_prompt_emb})
            # 2. prepare vap image clip embedding
            pipe.load_models_to_device(["vae", "image_encoder"])
            vap_video, end_image = inputs_shared.get("vap_video"), inputs_shared.get("end_image")

            num_frames, height, width = inputs_shared.get("num_frames"), inputs_shared.get("height"), inputs_shared.get("width")
            batch_size = int(inputs_shared.get("batch_size") or 1)

            if batch_size > 1:
                first_frames = [video[0] for video in vap_video]
                last_frames = [video[-1] for video in vap_video]
            else:
                first_frames = [vap_video[0]]
                last_frames = [vap_video[-1]]

            image_vap = [
                pipe.preprocess_image(frame.resize((width, height))).to(pipe.device)
                for frame in first_frames
            ]

            vap_clip_context = pipe.image_encoder.encode_image(image_vap)
            if end_image is not None and pipe.dit.has_image_pos_emb:
                vap_end_images = [
                    pipe.preprocess_image(frame.resize((width, height))).to(pipe.device)
                    for frame in last_frames
                ]
                vap_clip_context = torch.concat([vap_clip_context, pipe.image_encoder.encode_image(vap_end_images)], dim=1)
            vap_clip_context = vap_clip_context.to(dtype=pipe.torch_dtype, device=pipe.device)
            inputs_shared.update({"vap_clip_feature":vap_clip_context})

            # 3. prepare vap latents
            msk = torch.ones(1, num_frames, height//8, width//8, device=pipe.device)
            msk[:, 1:] = 0
            has_end_image = end_image is not None
            if has_end_image:
                msk[:, -1:] = 1

            msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
            msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
            msk = msk.transpose(1, 2)[0]
            msk = msk.unsqueeze(0).expand(len(first_frames), -1, -1, -1, -1)

            tiled, tile_size, tile_stride = inputs_shared.get("tiled"), inputs_shared.get("tile_size"), inputs_shared.get("tile_stride")

            vae_inputs = []
            for idx, frame in enumerate(first_frames):
                image_tensor = pipe.preprocess_image(frame.resize((width, height))).to(pipe.device)
                if has_end_image:
                    last_image_vap = pipe.preprocess_image(last_frames[idx].resize((width, height))).to(pipe.device)
                    vae_input = torch.concat([image_tensor.transpose(0, 1), torch.zeros(3, num_frames-2, height, width).to(image_tensor.device), last_image_vap.transpose(0, 1)], dim=1)
                else:
                    vae_input = torch.concat([image_tensor.transpose(0, 1), torch.zeros(3, num_frames-1, height, width).to(image_tensor.device)], dim=1)
                vae_inputs.append(vae_input.to(dtype=pipe.torch_dtype, device=pipe.device))

            y = pipe.vae.encode(vae_inputs, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
            y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
            y = torch.concat([msk, y], dim=1)

            vap_video = pipe.preprocess_video(vap_video)
            vap_latent = pipe.vae.encode(vap_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)

            vap_latent = torch.concat([vap_latent, y], dim=1).to(dtype=pipe.torch_dtype, device=pipe.device)
            inputs_shared.update({"vap_hidden_state":vap_latent})

            return inputs_shared, inputs_posi, inputs_nega



class WanVideoUnit_UnifiedSequenceParallel(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=(), output_params=("use_unified_sequence_parallel",))

    def process(self, pipe: WanVideoPipeline):
        if hasattr(pipe, "use_unified_sequence_parallel"):
            if pipe.use_unified_sequence_parallel:
                return {"use_unified_sequence_parallel": True}
        return {}



class WanVideoUnit_TeaCache(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"num_inference_steps": "num_inference_steps", "tea_cache_l1_thresh": "tea_cache_l1_thresh", "tea_cache_model_id": "tea_cache_model_id"},
            input_params_nega={"num_inference_steps": "num_inference_steps", "tea_cache_l1_thresh": "tea_cache_l1_thresh", "tea_cache_model_id": "tea_cache_model_id"},
            output_params=("tea_cache",)
        )

    def process(self, pipe: WanVideoPipeline, num_inference_steps, tea_cache_l1_thresh, tea_cache_model_id):
        if tea_cache_l1_thresh is None:
            return {}
        return {"tea_cache": TeaCache(num_inference_steps, rel_l1_thresh=tea_cache_l1_thresh, model_id=tea_cache_model_id)}



class WanVideoUnit_CfgMerger(PipelineUnit):
    def __init__(self):
        super().__init__(take_over=True)
        self.concat_tensor_names = ["context", "clip_feature", "y", "reference_latents"]

    def process(self, pipe: WanVideoPipeline, inputs_shared, inputs_posi, inputs_nega):
        if not inputs_shared["cfg_merge"]:
            return inputs_shared, inputs_posi, inputs_nega
        for name in self.concat_tensor_names:
            tensor_posi = inputs_posi.get(name)
            tensor_nega = inputs_nega.get(name)
            tensor_shared = inputs_shared.get(name)
            if tensor_posi is not None and tensor_nega is not None:
                inputs_shared[name] = torch.concat((tensor_posi, tensor_nega), dim=0)
            elif tensor_shared is not None:
                inputs_shared[name] = torch.concat((tensor_shared, tensor_shared), dim=0)
        inputs_posi.clear()
        inputs_nega.clear()
        return inputs_shared, inputs_posi, inputs_nega


class WanVideoUnit_S2V(PipelineUnit):
    def __init__(self):
        super().__init__(
            take_over=True,
            onload_model_names=("audio_encoder", "vae",),
            input_params=("input_audio", "audio_embeds", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride", "audio_sample_rate", "s2v_pose_video", "s2v_pose_latents", "motion_video"),
            output_params=("audio_embeds", "motion_latents", "drop_motion_frames", "s2v_pose_latents"),
        )

    def process_audio(self, pipe: WanVideoPipeline, input_audio, audio_sample_rate, num_frames, fps=16, audio_embeds=None, return_all=False):
        if audio_embeds is not None:
            return {"audio_embeds": audio_embeds}
        pipe.load_models_to_device(["audio_encoder"])
        audio_embeds = pipe.audio_encoder.get_audio_feats_per_inference(input_audio, audio_sample_rate, pipe.audio_processor, fps=fps, batch_frames=num_frames-1, dtype=pipe.torch_dtype, device=pipe.device)
        if return_all:
            return audio_embeds
        else:
            return {"audio_embeds": audio_embeds[0]}

    def process_motion_latents(self, pipe: WanVideoPipeline, height, width, tiled, tile_size, tile_stride, motion_video=None):
        pipe.load_models_to_device(["vae"])
        motion_frames = 73
        kwargs = {}
        if motion_video is not None and len(motion_video) > 0:
            assert len(motion_video) == motion_frames, f"motion video must have {motion_frames} frames, but got {len(motion_video)}"
            motion_latents = pipe.preprocess_video(motion_video)
            kwargs["drop_motion_frames"] = False
        else:
            motion_latents = torch.zeros([1, 3, motion_frames, height, width], dtype=pipe.torch_dtype, device=pipe.device)
            kwargs["drop_motion_frames"] = True
        motion_latents = pipe.vae.encode(motion_latents, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        kwargs.update({"motion_latents": motion_latents})
        return kwargs

    def process_pose_cond(self, pipe: WanVideoPipeline, s2v_pose_video, num_frames, height, width, tiled, tile_size, tile_stride, s2v_pose_latents=None, num_repeats=1, return_all=False):
        if s2v_pose_latents is not None:
            return {"s2v_pose_latents": s2v_pose_latents}
        if s2v_pose_video is None:
            return {"s2v_pose_latents": None}
        pipe.load_models_to_device(["vae"])
        infer_frames = num_frames - 1
        input_video = pipe.preprocess_video(s2v_pose_video)[:, :, :infer_frames * num_repeats]
        # pad if not enough frames
        padding_frames = infer_frames * num_repeats - input_video.shape[2]
        input_video = torch.cat([input_video, -torch.ones(1, 3, padding_frames, height, width, device=input_video.device, dtype=input_video.dtype)], dim=2)
        input_videos = input_video.chunk(num_repeats, dim=2)
        pose_conds = []
        for r in range(num_repeats):
            cond = input_videos[r]
            cond = torch.cat([cond[:, :, 0:1].repeat(1, 1, 1, 1, 1), cond], dim=2)
            cond_latents = pipe.vae.encode(cond, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
            pose_conds.append(cond_latents[:,:,1:])
        if return_all:
            return pose_conds
        else:
            return {"s2v_pose_latents": pose_conds[0]}

    def process(self, pipe: WanVideoPipeline, inputs_shared, inputs_posi, inputs_nega):
        if (inputs_shared.get("input_audio") is None and inputs_shared.get("audio_embeds") is None) or pipe.audio_encoder is None or pipe.audio_processor is None:
            return inputs_shared, inputs_posi, inputs_nega
        num_frames, height, width, tiled, tile_size, tile_stride = inputs_shared.get("num_frames"), inputs_shared.get("height"), inputs_shared.get("width"), inputs_shared.get("tiled"), inputs_shared.get("tile_size"), inputs_shared.get("tile_stride")
        input_audio, audio_embeds, audio_sample_rate = inputs_shared.pop("input_audio", None), inputs_shared.pop("audio_embeds", None), inputs_shared.get("audio_sample_rate", 16000)
        s2v_pose_video, s2v_pose_latents, motion_video = inputs_shared.pop("s2v_pose_video", None), inputs_shared.pop("s2v_pose_latents", None), inputs_shared.pop("motion_video", None)

        audio_input_positive = self.process_audio(pipe, input_audio, audio_sample_rate, num_frames, audio_embeds=audio_embeds)
        inputs_posi.update(audio_input_positive)
        inputs_nega.update({"audio_embeds": 0.0 * audio_input_positive["audio_embeds"]})

        inputs_shared.update(self.process_motion_latents(pipe, height, width, tiled, tile_size, tile_stride, motion_video))
        inputs_shared.update(self.process_pose_cond(pipe, s2v_pose_video, num_frames, height, width, tiled, tile_size, tile_stride, s2v_pose_latents=s2v_pose_latents))
        return inputs_shared, inputs_posi, inputs_nega

    @staticmethod
    def pre_calculate_audio_pose(pipe: WanVideoPipeline, input_audio=None, audio_sample_rate=16000, s2v_pose_video=None, num_frames=81, height=448, width=832, fps=16, tiled=True, tile_size=(30, 52), tile_stride=(15, 26)):
        assert pipe.audio_encoder is not None and pipe.audio_processor is not None, "Please load audio encoder and audio processor first."
        shapes = WanVideoUnit_ShapeChecker().process(pipe, height, width, num_frames)
        height, width, num_frames = shapes["height"], shapes["width"], shapes["num_frames"]
        unit = WanVideoUnit_S2V()
        audio_embeds = unit.process_audio(pipe, input_audio, audio_sample_rate, num_frames, fps, return_all=True)
        pose_latents = unit.process_pose_cond(pipe, s2v_pose_video, num_frames, height, width, num_repeats=len(audio_embeds), return_all=True, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        pose_latents = None if s2v_pose_video is None else pose_latents
        return audio_embeds, pose_latents, len(audio_embeds)


class WanVideoPostUnit_S2V(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("latents", "motion_latents", "drop_motion_frames"))

    def process(self, pipe: WanVideoPipeline, latents, motion_latents, drop_motion_frames):
        if pipe.audio_encoder is None or motion_latents is None or drop_motion_frames:
            return {}
        latents = torch.cat([motion_latents, latents[:,:,1:]], dim=2)
        return {"latents": latents}


class WanVideoUnit_AnimateVideoSplit(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_video", "animate_pose_video", "animate_face_video", "animate_inpaint_video", "animate_mask_video"),
            output_params=("animate_pose_video", "animate_face_video", "animate_inpaint_video", "animate_mask_video")
        )

    def process(self, pipe: WanVideoPipeline, input_video, animate_pose_video, animate_face_video, animate_inpaint_video, animate_mask_video):
        if input_video is None:
            return {}
        if animate_pose_video is not None:
            animate_pose_video = animate_pose_video[:len(input_video) - 4]
        if animate_face_video is not None:
            animate_face_video = animate_face_video[:len(input_video) - 4]
        if animate_inpaint_video is not None:
            animate_inpaint_video = animate_inpaint_video[:len(input_video) - 4]
        if animate_mask_video is not None:
            animate_mask_video = animate_mask_video[:len(input_video) - 4]
        return {"animate_pose_video": animate_pose_video, "animate_face_video": animate_face_video, "animate_inpaint_video": animate_inpaint_video, "animate_mask_video": animate_mask_video}


class WanVideoUnit_AnimatePoseLatents(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("animate_pose_video", "tiled", "tile_size", "tile_stride"),
            output_params=("pose_latents",),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, animate_pose_video, tiled, tile_size, tile_stride):
        if animate_pose_video is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        animate_pose_video = pipe.preprocess_video(animate_pose_video)
        pose_latents = pipe.vae.encode(animate_pose_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"pose_latents": pose_latents}


class WanVideoUnit_AnimateFacePixelValues(PipelineUnit):
    def __init__(self):
        super().__init__(
            take_over=True,
            input_params=("animate_face_video",),
            output_params=("face_pixel_values"),
        )

    def process(self, pipe: WanVideoPipeline, inputs_shared, inputs_posi, inputs_nega):
        if inputs_shared.get("animate_face_video", None) is None:
            return inputs_shared, inputs_posi, inputs_nega
        inputs_posi["face_pixel_values"] = pipe.preprocess_video(inputs_shared["animate_face_video"])
        inputs_nega["face_pixel_values"] = torch.zeros_like(inputs_posi["face_pixel_values"]) - 1
        return inputs_shared, inputs_posi, inputs_nega


class WanVideoUnit_AnimateInpaint(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("animate_inpaint_video", "animate_mask_video", "input_image", "tiled", "tile_size", "tile_stride"),
            output_params=("y",),
            onload_model_names=("vae",)
        )
        
    def get_i2v_mask(self, lat_t, lat_h, lat_w, mask_len=1, mask_pixel_values=None, device="cuda"):
        if mask_pixel_values is None:
            msk = torch.zeros(1, (lat_t-1) * 4 + 1, lat_h, lat_w, device=device)
        else:
            msk = mask_pixel_values.clone()
        msk[:, :mask_len] = 1
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]
        return msk

    def process(self, pipe: WanVideoPipeline, animate_inpaint_video, animate_mask_video, input_image, tiled, tile_size, tile_stride):
        if animate_inpaint_video is None or animate_mask_video is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)

        bg_pixel_values = pipe.preprocess_video(animate_inpaint_video)
        y_reft = pipe.vae.encode(bg_pixel_values, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0].to(dtype=pipe.torch_dtype, device=pipe.device)
        _, lat_t, lat_h, lat_w = y_reft.shape
        
        ref_pixel_values = pipe.preprocess_video([input_image])
        ref_latents = pipe.vae.encode(ref_pixel_values, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        mask_ref = self.get_i2v_mask(1, lat_h, lat_w, 1, device=pipe.device)
        y_ref = torch.concat([mask_ref, ref_latents[0]]).to(dtype=torch.bfloat16, device=pipe.device)
        
        mask_pixel_values = 1 - pipe.preprocess_video(animate_mask_video, max_value=1, min_value=0)
        mask_pixel_values = rearrange(mask_pixel_values, "b c t h w -> (b t) c h w")
        mask_pixel_values = torch.nn.functional.interpolate(mask_pixel_values, size=(lat_h, lat_w), mode='nearest')
        mask_pixel_values = rearrange(mask_pixel_values, "(b t) c h w -> b t c h w", b=1)[:,:,0]
        msk_reft = self.get_i2v_mask(lat_t, lat_h, lat_w, 0, mask_pixel_values=mask_pixel_values, device=pipe.device)
        
        y_reft = torch.concat([msk_reft, y_reft]).to(dtype=torch.bfloat16, device=pipe.device)
        y = torch.concat([y_ref, y_reft], dim=1).unsqueeze(0)
        return {"y": y}


class WanVideoUnit_LongCatVideo(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("longcat_video",),
            output_params=("longcat_latents",),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, longcat_video):
        if longcat_video is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        longcat_video = pipe.preprocess_video(longcat_video)
        longcat_latents = pipe.vae.encode(longcat_video, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"longcat_latents": longcat_latents}


class TeaCache:
    def __init__(self, num_inference_steps, rel_l1_thresh, model_id):
        self.num_inference_steps = num_inference_steps
        self.step = 0
        self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = None
        self.rel_l1_thresh = rel_l1_thresh
        self.previous_residual = None
        self.previous_hidden_states = None
        
        self.coefficients_dict = {
            "Wan2.1-T2V-1.3B": [-5.21862437e+04, 9.23041404e+03, -5.28275948e+02, 1.36987616e+01, -4.99875664e-02],
            "Wan2.1-T2V-14B": [-3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01, -3.15583525e-01],
            "Wan2.1-I2V-14B-480P": [2.57151496e+05, -3.54229917e+04,  1.40286849e+03, -1.35890334e+01, 1.32517977e-01],
            "Wan2.1-I2V-14B-720P": [ 8.10705460e+03,  2.13393892e+03, -3.72934672e+02,  1.66203073e+01, -4.17769401e-02],
        }
        if model_id not in self.coefficients_dict:
            supported_model_ids = ", ".join([i for i in self.coefficients_dict])
            raise ValueError(f"{model_id} is not a supported TeaCache model id. Please choose a valid model id in ({supported_model_ids}).")
        self.coefficients = self.coefficients_dict[model_id]

    def check(self, dit: WanModel, x, t_mod):
        modulated_inp = t_mod.clone()
        if self.step == 0 or self.step == self.num_inference_steps - 1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            coefficients = self.coefficients
            rescale_func = np.poly1d(coefficients)
            self.accumulated_rel_l1_distance += rescale_func(((modulated_inp-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item())
            if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = modulated_inp
        self.step += 1
        if self.step == self.num_inference_steps:
            self.step = 0
        if should_calc:
            self.previous_hidden_states = x.clone()
        return not should_calc

    def store(self, hidden_states):
        self.previous_residual = hidden_states - self.previous_hidden_states
        self.previous_hidden_states = None

    def update(self, hidden_states):
        hidden_states = hidden_states + self.previous_residual
        return hidden_states



class TemporalTiler_BCTHW:
    def __init__(self):
        pass

    def build_1d_mask(self, length, left_bound, right_bound, border_width):
        x = torch.ones((length,))
        if border_width == 0:
            return x
        
        shift = 0.5
        if not left_bound:
            x[:border_width] = (torch.arange(border_width) + shift) / border_width
        if not right_bound:
            x[-border_width:] = torch.flip((torch.arange(border_width) + shift) / border_width, dims=(0,))
        return x

    def build_mask(self, data, is_bound, border_width):
        _, _, T, _, _ = data.shape
        t = self.build_1d_mask(T, is_bound[0], is_bound[1], border_width[0])
        mask = repeat(t, "T -> 1 1 T 1 1")
        return mask
    
    def run(self, model_fn, sliding_window_size, sliding_window_stride, computation_device, computation_dtype, model_kwargs, tensor_names, batch_size=None):
        tensor_names = [tensor_name for tensor_name in tensor_names if model_kwargs.get(tensor_name) is not None]
        tensor_dict = {tensor_name: model_kwargs[tensor_name] for tensor_name in tensor_names}
        B, C, T, H, W = tensor_dict[tensor_names[0]].shape
        if batch_size is not None:
            B *= batch_size
        data_device, data_dtype = tensor_dict[tensor_names[0]].device, tensor_dict[tensor_names[0]].dtype
        value = torch.zeros((B, C, T, H, W), device=data_device, dtype=data_dtype)
        weight = torch.zeros((1, 1, T, 1, 1), device=data_device, dtype=data_dtype)
        for t in range(0, T, sliding_window_stride):
            if t - sliding_window_stride >= 0 and t - sliding_window_stride + sliding_window_size >= T:
                continue
            t_ = min(t + sliding_window_size, T)
            model_kwargs.update({
                tensor_name: tensor_dict[tensor_name][:, :, t: t_:, :].to(device=computation_device, dtype=computation_dtype) \
                    for tensor_name in tensor_names
            })
            model_output = model_fn(**model_kwargs).to(device=data_device, dtype=data_dtype)
            mask = self.build_mask(
                model_output,
                is_bound=(t == 0, t_ == T),
                border_width=(sliding_window_size - sliding_window_stride,)
            ).to(device=data_device, dtype=data_dtype)
            value[:, :, t: t_, :, :] += model_output * mask
            weight[:, :, t: t_, :, :] += mask
        value /= weight
        model_kwargs.update(tensor_dict)
        return value



def model_fn_wan_video(
    dit: WanModel,
    motion_controller: WanMotionControllerModel = None,
    vace: VaceWanModel = None,
    vap: MotWanModel = None,
    animate_adapter: WanAnimateAdapter = None,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    reference_latents = None,
    vace_context = None,
    vace_rgbd_latents = None,
    vace_mask_cam = None,
    vace_scale = 1.0,
    audio_embeds: Optional[torch.Tensor] = None,
    motion_latents: Optional[torch.Tensor] = None,
    s2v_pose_latents: Optional[torch.Tensor] = None,
    vap_hidden_state = None,
    vap_clip_feature = None,
    context_vap = None,
    drop_motion_frames: bool = True,
    tea_cache: TeaCache = None,
    use_unified_sequence_parallel: bool = False,
    motion_bucket_id: Optional[torch.Tensor] = None,
    pose_latents=None,
    face_pixel_values=None,
    longcat_latents=None,
    sliding_window_size: Optional[int] = None,
    sliding_window_stride: Optional[int] = None,
    cfg_merge: bool = False,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    control_camera_latents_input = None,
    fuse_vae_embedding_in_latents: bool = False,
    **kwargs,
):
    if sliding_window_size is not None and sliding_window_stride is not None:
        model_kwargs = dict(
            dit=dit,
            motion_controller=motion_controller,
            vace=vace,
            latents=latents,
            timestep=timestep,
            context=context,
            clip_feature=clip_feature,
            y=y,
            reference_latents=reference_latents,
            vace_context=vace_context,
            vace_rgbd_latents=vace_rgbd_latents,
            vace_mask_cam=vace_mask_cam,
            vace_scale=vace_scale,
            tea_cache=tea_cache,
            use_unified_sequence_parallel=use_unified_sequence_parallel,
            motion_bucket_id=motion_bucket_id,
        )
        return TemporalTiler_BCTHW().run(
            model_fn_wan_video,
            sliding_window_size, sliding_window_stride,
            latents.device, latents.dtype,
            model_kwargs=model_kwargs,
            tensor_names=["latents", "y"],
            batch_size=2 if cfg_merge else 1
        )
    # LongCat-Video
    if isinstance(dit, LongCatVideoTransformer3DModel):
        return model_fn_longcat_video(
            dit=dit,
            latents=latents,
            timestep=timestep,
            context=context,
            longcat_latents=longcat_latents,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        )
        
    # wan2.2 s2v
    if audio_embeds is not None:
        return model_fn_wans2v(
            dit=dit,
            latents=latents,
            timestep=timestep,
            context=context,
            audio_embeds=audio_embeds,
            motion_latents=motion_latents,
            s2v_pose_latents=s2v_pose_latents,
            drop_motion_frames=drop_motion_frames,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_unified_sequence_parallel=use_unified_sequence_parallel,
        )

    if use_unified_sequence_parallel:
        import torch.distributed as dist
        from xfuser.core.distributed import (get_sequence_parallel_rank,
                                            get_sequence_parallel_world_size,
                                            get_sp_group)

    # Timestep
    if dit.seperated_timestep and fuse_vae_embedding_in_latents:
        timestep = torch.concat([
            torch.zeros((1, latents.shape[3] * latents.shape[4] // 4), dtype=latents.dtype, device=latents.device),
            torch.ones((latents.shape[2] - 1, latents.shape[3] * latents.shape[4] // 4), dtype=latents.dtype, device=latents.device) * timestep
        ]).flatten()
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep).unsqueeze(0))
        if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
            t_chunks = torch.chunk(t, get_sequence_parallel_world_size(), dim=1)
            t_chunks = [torch.nn.functional.pad(chunk, (0, 0, 0, t_chunks[0].shape[1]-chunk.shape[1]), value=0) for chunk in t_chunks]
            t = t_chunks[get_sequence_parallel_rank()]
        t_mod = dit.time_projection(t).unflatten(2, (6, dit.dim))
    else:
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    
    # Motion Controller
    if motion_bucket_id is not None and motion_controller is not None:
        t_mod = t_mod + motion_controller(motion_bucket_id).unflatten(1, (6, dit.dim))
    context = dit.text_embedding(context)

    x = latents
    # Merged cfg
    if x.shape[0] != context.shape[0]:
        x = torch.concat([x] * context.shape[0], dim=0)
    if timestep.shape[0] != context.shape[0]:
        timestep = torch.concat([timestep] * context.shape[0], dim=0)

    # Image Embedding
    if y is not None and dit.require_vae_embedding:
        x = torch.cat([x, y], dim=1)
    if clip_feature is not None and dit.require_clip_embedding:
        clip_embdding = dit.img_emb(clip_feature)
        context = torch.cat([clip_embdding, context], dim=1)
    if dit.require_clip_embedding:
        img_token_len = clip_feature.shape[1] if clip_feature is not None else 0
        for block in dit.blocks:
            if hasattr(block, "cross_attn"):
                block.cross_attn.image_token_len = img_token_len
        
    # Camera control
    patch_dtype = dit.patch_embedding.weight.dtype
    if x.dtype != patch_dtype:
        x = x.to(dtype=patch_dtype, device=x.device)
    if control_camera_latents_input is not None and control_camera_latents_input.dtype != patch_dtype:
        control_camera_latents_input = control_camera_latents_input.to(dtype=patch_dtype, device=control_camera_latents_input.device)
    x = dit.patchify(x, control_camera_latents_input)
    
    # Animate
    if pose_latents is not None and face_pixel_values is not None:
        x, motion_vec = animate_adapter.after_patch_embedding(x, pose_latents, face_pixel_values)
    
    # Patchify
    f, h, w = x.shape[2:]
    x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
    
    # Reference image
    if reference_latents is not None:
        if len(reference_latents.shape) == 5:
            reference_latents = reference_latents[:, :, 0]
        reference_latents = dit.ref_conv(reference_latents).flatten(2).transpose(1, 2)
        x = torch.concat([reference_latents, x], dim=1)
        f += 1
    
    freqs = torch.cat([
        dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
    ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

    attention_probe = getattr(dit, "attention_probe", None)
    if attention_probe is not None and hasattr(attention_probe, "set_grid"):
        attention_probe.set_grid(f, h, w, has_reference=reference_latents is not None)

    # VAP 
    if vap is not None:
        # hidden state
        x_vap = vap_hidden_state
        x_vap = vap.patchify(x_vap)
        x_vap = rearrange(x_vap, 'b c f h w -> b (f h w) c').contiguous()
        # Timestep
        clean_timestep = torch.ones(timestep.shape, device=timestep.device).to(timestep.dtype)
        t = vap.time_embedding(sinusoidal_embedding_1d(vap.freq_dim, clean_timestep))
        t_mod_vap = vap.time_projection(t).unflatten(1, (6, vap.dim))

        # rope
        freqs_vap = vap.compute_freqs_mot(f,h,w).to(x.device)

        # context
        vap_clip_embedding = vap.img_emb(vap_clip_feature)
        context_vap = vap.text_embedding(context_vap)
        context_vap = torch.cat([vap_clip_embedding, context_vap], dim=1)
    
    # TeaCache
    if tea_cache is not None:
        tea_cache_update = tea_cache.check(dit, x, t_mod)
    else:
        tea_cache_update = False
    # VACE
    vace_plucker_active = (
        vace is not None
        and vace_rgbd_latents is not None
        and vace_mask_cam is not None
        and getattr(vace, "use_plucker", False)
    )
    vace_active = (vace_context is not None) or vace_plucker_active
    if vace_active:
        vace_hints = vace(
            x, vace_context, context, t_mod, freqs,
            vace_rgbd_latents=vace_rgbd_latents,
            vace_mask_cam=vace_mask_cam,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload
        )
    
    # blocks
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            chunks = torch.chunk(x, get_sequence_parallel_world_size(), dim=1)
            pad_shape = chunks[0].shape[1] - chunks[-1].shape[1]
            chunks = [torch.nn.functional.pad(chunk, (0, 0, 0, chunks[0].shape[1]-chunk.shape[1]), value=0) for chunk in chunks]
            x = chunks[get_sequence_parallel_rank()]
    if tea_cache_update:
        x = tea_cache.update(x)
    else:
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward
        
        def create_custom_forward_vap(block, vap):
            def custom_forward(*inputs):
                return vap(block, *inputs)
            return custom_forward
        
        for block_id, block in enumerate(dit.blocks):
            if attention_probe is not None and hasattr(block, "self_attn"):
                block.self_attn.attention_probe = attention_probe
                block.self_attn.attention_probe_block_id = block_id
            # Block
            if vap is not None and block_id in vap.mot_layers_mapping:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x, x_vap = torch.utils.checkpoint.checkpoint(
                            create_custom_forward_vap(block, vap),
                            x, context, t_mod, freqs, x_vap, context_vap, t_mod_vap, freqs_vap, block_id,
                            use_reentrant=False,
                        )
                elif use_gradient_checkpointing:
                    x, x_vap = torch.utils.checkpoint.checkpoint(
                        create_custom_forward_vap(block, vap),
                        x, context, t_mod, freqs, x_vap, context_vap, t_mod_vap, freqs_vap, block_id,
                        use_reentrant=False,
                    )
                else:
                    x, x_vap = vap(block, x, context, t_mod, freqs, x_vap, context_vap, t_mod_vap, freqs_vap, block_id)
            else:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x, context, t_mod, freqs,
                            use_reentrant=False,
                        )
                elif use_gradient_checkpointing:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, context, t_mod, freqs,
                        use_reentrant=False,
                    )
                else:
                    x = block(x, context, t_mod, freqs)
            
            # VACE
            if vace_active and block_id in vace.vace_layers_mapping:
                current_vace_hint = vace_hints[vace.vace_layers_mapping[block_id]]
                if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
                    current_vace_hint = torch.chunk(current_vace_hint, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
                    current_vace_hint = torch.nn.functional.pad(current_vace_hint, (0, 0, 0, chunks[0].shape[1] - current_vace_hint.shape[1]), value=0)
                x = x + current_vace_hint * vace_scale
            
            # Animate
            if pose_latents is not None and face_pixel_values is not None:
                x = animate_adapter.after_transformer_block(block_id, x, motion_vec)
        if tea_cache is not None:
            tea_cache.store(x)
            
    x = dit.head(x, t)
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            x = get_sp_group().all_gather(x, dim=1)
            x = x[:, :-pad_shape] if pad_shape > 0 else x
    # Remove reference latents
    if reference_latents is not None:
        x = x[:, reference_latents.shape[1]:]
        f -= 1
    x = dit.unpatchify(x, (f, h, w))
    return x


def model_fn_longcat_video(
    dit: LongCatVideoTransformer3DModel,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    longcat_latents: torch.Tensor = None,
    use_gradient_checkpointing=False,
    use_gradient_checkpointing_offload=False,
):
    if longcat_latents is not None:
        latents[:, :, :longcat_latents.shape[2]] = longcat_latents
        num_cond_latents = longcat_latents.shape[2]
    else:
        num_cond_latents = 0
    context = context.unsqueeze(0)
    encoder_attention_mask = torch.any(context != 0, dim=-1)[:, 0].to(torch.int64)
    output = dit(
        latents,
        timestep,
        context,
        encoder_attention_mask,
        num_cond_latents=num_cond_latents,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
    )
    output = -output
    output = output.to(latents.dtype)
    return output


def model_fn_wans2v(
    dit,
    latents,
    timestep,
    context,
    audio_embeds,
    motion_latents,
    s2v_pose_latents,
    drop_motion_frames=True,
    use_gradient_checkpointing_offload=False,
    use_gradient_checkpointing=False,
    use_unified_sequence_parallel=False,
):
    if use_unified_sequence_parallel:
        import torch.distributed as dist
        from xfuser.core.distributed import (get_sequence_parallel_rank,
                                            get_sequence_parallel_world_size,
                                            get_sp_group)
    origin_ref_latents = latents[:, :, 0:1]
    x = latents[:, :, 1:]

    # context embedding
    context = dit.text_embedding(context)

    # audio encode
    audio_emb_global, merged_audio_emb = dit.cal_audio_emb(audio_embeds)

    # x and s2v_pose_latents
    s2v_pose_latents = torch.zeros_like(x) if s2v_pose_latents is None else s2v_pose_latents
    x, (f, h, w) = dit.patchify(dit.patch_embedding(x) + dit.cond_encoder(s2v_pose_latents))
    seq_len_x = seq_len_x_global = x.shape[1] # global used for unified sequence parallel

    # reference image
    ref_latents, (rf, rh, rw) = dit.patchify(dit.patch_embedding(origin_ref_latents))
    grid_sizes = dit.get_grid_sizes((f, h, w), (rf, rh, rw))
    x = torch.cat([x, ref_latents], dim=1)
    # mask
    mask = torch.cat([torch.zeros([1, seq_len_x]), torch.ones([1, ref_latents.shape[1]])], dim=1).to(torch.long).to(x.device)
    # freqs
    pre_compute_freqs = rope_precompute(x.detach().view(1, x.size(1), dit.num_heads, dit.dim // dit.num_heads), grid_sizes, dit.freqs, start=None)
    # motion
    x, pre_compute_freqs, mask = dit.inject_motion(x, pre_compute_freqs, mask, motion_latents, drop_motion_frames=drop_motion_frames, add_last_motion=2)

    x = x + dit.trainable_cond_mask(mask).to(x.dtype)

    # tmod
    timestep = torch.cat([timestep, torch.zeros([1], dtype=timestep.dtype, device=timestep.device)])
    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim)).unsqueeze(2).transpose(0, 2)

    if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
        world_size, sp_rank = get_sequence_parallel_world_size(), get_sequence_parallel_rank()
        assert x.shape[1] % world_size == 0, f"the dimension after chunk must be divisible by world size, but got {x.shape[1]} and {get_sequence_parallel_world_size()}"
        x = torch.chunk(x, world_size, dim=1)[sp_rank]
        seg_idxs = [0] + list(torch.cumsum(torch.tensor([x.shape[1]] * world_size), dim=0).cpu().numpy())
        seq_len_x_list = [min(max(0, seq_len_x - seg_idxs[i]), x.shape[1]) for i in range(len(seg_idxs)-1)]
        seq_len_x = seq_len_x_list[sp_rank]

    def create_custom_forward(module):
        def custom_forward(*inputs):
            return module(*inputs)
        return custom_forward

    for block_id, block in enumerate(dit.blocks):
        if use_gradient_checkpointing_offload:
            with torch.autograd.graph.save_on_cpu():
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, context, t_mod, seq_len_x, pre_compute_freqs[0],
                    use_reentrant=False,
                )
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(lambda x: dit.after_transformer_block(block_id, x, audio_emb_global, merged_audio_emb, seq_len_x)),
                    x,
                    use_reentrant=False,
                )
        elif use_gradient_checkpointing:
            x = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                x, context, t_mod, seq_len_x, pre_compute_freqs[0],
                use_reentrant=False,
            )
            x = torch.utils.checkpoint.checkpoint(
                create_custom_forward(lambda x: dit.after_transformer_block(block_id, x, audio_emb_global, merged_audio_emb, seq_len_x)),
                x,
                use_reentrant=False,
            )
        else:
            x = block(x, context, t_mod, seq_len_x, pre_compute_freqs[0])
            x = dit.after_transformer_block(block_id, x, audio_emb_global, merged_audio_emb, seq_len_x_global, use_unified_sequence_parallel)

    if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
        x = get_sp_group().all_gather(x, dim=1)

    x = x[:, :seq_len_x_global]
    x = dit.head(x, t[:-1])
    x = dit.unpatchify(x, (f, h, w))
    # make compatible with wan video
    x = torch.cat([origin_ref_latents, x], dim=2)
    return x
