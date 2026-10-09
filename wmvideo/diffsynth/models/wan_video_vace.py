import torch
import torch.nn.functional as F

from .wan_video_dit import DiTBlock


class VaceWanAttentionBlock(DiTBlock):
    def __init__(self, has_image_input, dim, num_heads, ffn_dim, eps=1e-6, block_id=0):
        super().__init__(has_image_input, dim, num_heads, ffn_dim, eps=eps)
        self.block_id = block_id
        if block_id == 0:
            self.before_proj = torch.nn.Linear(self.dim, self.dim)
        self.after_proj = torch.nn.Linear(self.dim, self.dim)

    def forward(self, c, x, context, t_mod, freqs):
        if self.block_id == 0:
            c = self.before_proj(c) + x
            all_c = []
        else:
            all_c = list(torch.unbind(c))
            c = all_c.pop(-1)
        c = super().forward(c, context, t_mod, freqs)
        c_skip = self.after_proj(c)
        all_c += [c_skip, c]
        c = torch.stack(all_c)
        return c


class VaceWanModel(torch.nn.Module):
    def __init__(
        self,
        vace_layers=(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28),
        vace_in_dim=96,
        patch_size=(1, 2, 2),
        has_image_input=False,
        dim=1536,
        num_heads=12,
        ffn_dim=8960,
        eps=1e-6,
    ):
        super().__init__()
        self.vace_layers = vace_layers
        self.vace_in_dim = vace_in_dim
        self.patch_size = tuple(patch_size)
        self.vace_layers_mapping = {i: n for n, i in enumerate(self.vace_layers)}

        # vace blocks
        self.vace_blocks = torch.nn.ModuleList([
            VaceWanAttentionBlock(has_image_input, dim, num_heads, ffn_dim, eps, block_id=i)
            for i in self.vace_layers
        ])

        # vace patch embeddings
        self.vace_patch_embedding = torch.nn.Conv3d(vace_in_dim, dim, kernel_size=patch_size, stride=patch_size)

        # Optional plücker / camera-aware mask encoder (built on demand via enable_plucker()).
        # Mirrors 4DVideo-WanX's mask_cam_embedding: Conv3d(7→vace_in_dim-32, k=(4,8,8), s=(4,8,8)) + GN + SiLU.
        self.mask_cam_embedding = None
        # F.pad order is (W_l, W_r, H_l, H_r, T_l, T_r). [0,0,0,0,3,0] prepends 3 zero frames on T,
        # aligning the 4× temporal Conv3d with Wan VAE's 4:1 latent timeline (first latent frame ← 1 raw frame).
        self.mask_cam_padding = (0, 0, 0, 0, 3, 0)

    def enable_plucker(self, mask_in_channels: int = 1, plucker_channels: int = 6):
        """Lazily create the (mask + plücker) → latent Conv3d encoder.

        Kept off the construction-time signature so the model_configs entries (which control
        checkpoint shape) stay unchanged. The pipeline calls this AFTER state_dict load when
        --use_plucker is set; the new module's weights are random-init and will be trained
        (or LoRA-adapted) on the new conditioning channel.
        """
        if self.mask_cam_embedding is not None:
            return
        in_ch = int(mask_in_channels) + int(plucker_channels)
        out_ch = self.vace_in_dim - 32  # rgbd_latent (16+16) reserves 32 channels; rest is mask_cam latent
        if out_ch <= 0:
            raise ValueError(
                f"enable_plucker requires vace_in_dim > 32 to leave room for the mask_cam latent; "
                f"got vace_in_dim={self.vace_in_dim}."
            )
        groups = 8 if out_ch % 8 == 0 else 1
        self.mask_cam_embedding = torch.nn.Sequential(
            torch.nn.Conv3d(in_ch, out_ch, kernel_size=(4, 8, 8), stride=(4, 8, 8)),
            torch.nn.GroupNorm(groups, out_ch),
            torch.nn.SiLU(),
        )

    @property
    def use_plucker(self) -> bool:
        return self.mask_cam_embedding is not None

    def encode_mask_cam(self, mask: torch.Tensor, plucker: torch.Tensor) -> torch.Tensor:
        """Fuse (mask, plücker) at full T/H/W resolution into the 64-channel mask_cam latent.

        Args:
            mask:    [B, 1, T, H, W] binary {0,1} float.
            plucker: [B, 6, T, H, W] OpenCV-convention plücker (rays_d_world | moment).
        Returns:
            mask_cam_latent: [B, vace_in_dim-32, T_lat, H/8, W/8] where T_lat = (T+3)//4.
        """
        if self.mask_cam_embedding is None:
            raise RuntimeError("encode_mask_cam called but enable_plucker() was never invoked.")
        if mask.dim() != 5 or plucker.dim() != 5:
            raise ValueError(
                f"encode_mask_cam expects 5D tensors [B,C,T,H,W]; got mask={tuple(mask.shape)}, "
                f"plucker={tuple(plucker.shape)}"
            )
        if mask.shape[0] != plucker.shape[0] or mask.shape[2:] != plucker.shape[2:]:
            raise ValueError(
                f"mask/plücker mismatch: mask={tuple(mask.shape)}, plucker={tuple(plucker.shape)}"
            )
        # Match the conv weight dtype/device so we stay on the offload-managed device.
        ref = next(self.mask_cam_embedding.parameters())
        mask_cam = torch.cat((mask, plucker), dim=1).to(dtype=ref.dtype, device=ref.device)
        mask_cam = F.pad(mask_cam, self.mask_cam_padding, mode="constant", value=0)
        return self.mask_cam_embedding(mask_cam)

    def _assemble_vace_context_from_plucker(
        self,
        vace_rgbd_latents: torch.Tensor,
        vace_mask_cam: torch.Tensor,
    ) -> torch.Tensor:
        """Build the 96-channel vace_context from raw rgbd latents + (mask|plücker).

        Args:
            vace_rgbd_latents: [B, 32, T_lat_total, H_lat, W_lat_total]
                Latent rgb (16ch) + depth (16ch), possibly with f reference frames already
                prepended on the T dimension (pipeline side).
            vace_mask_cam: [B, 7, T, H, W_total]
                Channel-0 = mask (binary {0,1}), channels 1-6 = OpenCV plücker map.
                T/H/W are the raw (pre-VAE) target resolution and target frame count.
                The reference frames are NOT prepended here — we pad the latent on T
                to match vace_rgbd_latents after Conv3d.
        Returns:
            vace_context: [B, vace_in_dim, T_lat_total, H_lat, W_lat_total]
        """
        mask = vace_mask_cam[:, :1]
        plucker = vace_mask_cam[:, 1:]
        mask_cam_latent = self.encode_mask_cam(mask, plucker)
        # Align T dim with rgbd_latents (which may carry f reference frames at the front).
        diff = vace_rgbd_latents.shape[2] - mask_cam_latent.shape[2]
        if diff < 0:
            raise ValueError(
                f"vace_rgbd_latents has fewer T frames ({vace_rgbd_latents.shape[2]}) than "
                f"mask_cam_latent ({mask_cam_latent.shape[2]}); cannot reconcile reference padding."
            )
        if diff > 0:
            pad_shape = (
                mask_cam_latent.shape[0],
                mask_cam_latent.shape[1],
                diff,
                mask_cam_latent.shape[3],
                mask_cam_latent.shape[4],
            )
            mask_cam_latent = torch.cat(
                [mask_cam_latent.new_zeros(pad_shape), mask_cam_latent], dim=2
            )
        # rgb (16) + depth (16) | mask_cam (vace_in_dim - 32) = vace_in_dim
        vace_context = torch.cat(
            (vace_rgbd_latents.to(dtype=mask_cam_latent.dtype, device=mask_cam_latent.device), mask_cam_latent),
            dim=1,
        )
        if vace_context.shape[1] != self.vace_in_dim:
            raise ValueError(
                f"Assembled vace_context has {vace_context.shape[1]} channels, expected "
                f"{self.vace_in_dim}. Check rgbd_latents (must be 32ch) and mask_cam_embedding output."
            )
        return vace_context

    def forward(
        self, x, vace_context, context, t_mod, freqs,
        vace_rgbd_latents: torch.Tensor = None,
        vace_mask_cam: torch.Tensor = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
    ):
        # plücker path: assemble vace_context from rgbd latents + raw (mask|plücker)
        # using the lazily-built mask_cam_embedding. Falls through to legacy if either
        # input is missing or enable_plucker() was never called.
        if (
            self.mask_cam_embedding is not None
            and vace_rgbd_latents is not None
            and vace_mask_cam is not None
        ):
            vace_context = self._assemble_vace_context_from_plucker(vace_rgbd_latents, vace_mask_cam)
        c = [self.vace_patch_embedding(u.unsqueeze(0)) for u in vace_context]
        c = [u.flatten(2).transpose(1, 2) for u in c]
        c = torch.cat([
            torch.cat([u, u.new_zeros(1, x.shape[1] - u.size(1), u.size(2))],
                      dim=1) for u in c
        ])

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward

        for block in self.vace_blocks:
            if use_gradient_checkpointing_offload:
                with torch.autograd.graph.save_on_cpu():
                    c = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        c, x, context, t_mod, freqs,
                        use_reentrant=False,
                    )
            elif use_gradient_checkpointing:
                c = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    c, x, context, t_mod, freqs,
                    use_reentrant=False,
                )
            else:
                c = block(c, x, context, t_mod, freqs)
        hints = torch.unbind(c)[:-1]
        return hints
