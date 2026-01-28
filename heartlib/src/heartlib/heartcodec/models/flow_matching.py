import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from vector_quantize_pytorch import ResidualVQ
from .transformer import LlamaTransformer


class FlowMatching(nn.Module):
    def __init__(
        self,
        # rvq stuff
        dim: int = 512,
        codebook_size: int = 8192,
        decay: float = 0.9,
        commitment_weight: float = 1.0,
        threshold_ema_dead_code: int = 2,
        use_cosine_sim: bool = False,
        codebook_dim: int = 32,
        num_quantizers: int = 8,
        # dit backbone stuff
        attention_head_dim: int = 64,
        in_channels: int = 1024,
        norm_type: str = "ada_norm_single",
        num_attention_heads: int = 24,
        num_layers: int = 24,
        num_layers_2: int = 6,
        out_channels: int = 256,
    ):
        super().__init__()

        self.vq_embed = ResidualVQ(
            dim=dim,
            codebook_size=codebook_size,
            decay=decay,
            commitment_weight=commitment_weight,
            threshold_ema_dead_code=threshold_ema_dead_code,
            use_cosine_sim=use_cosine_sim,
            codebook_dim=codebook_dim,
            num_quantizers=num_quantizers,
        )
        self.cond_feature_emb = nn.Linear(dim, dim)
        self.zero_cond_embedding1 = nn.Parameter(torch.randn(dim))
        self.estimator = LlamaTransformer(
            attention_head_dim=attention_head_dim,
            in_channels=in_channels,
            norm_type=norm_type,
            num_attention_heads=num_attention_heads,
            num_layers=num_layers,
            num_layers_2=num_layers_2,
            out_channels=out_channels,
        )

        self.latent_dim = out_channels

    @torch.no_grad()
    def inference_codes(
        self,
        codes,
        true_latents,
        latent_length,
        incontext_length,
        guidance_scale=2.0,
        num_steps=20,
        disable_progress=True,
        scenario="start_seg",
        external_mask=None,
        gradient_mask=None, # Explicit Solver Constraint Mask (Float)
        seam_indices=[], # List of indices to apply Feature Smoothing
        smoothing_width=0, # Width of the blur window
        callback=None,
    ):
        device = true_latents.device
        dtype = true_latents.dtype
        # codes_bestrq_middle, codes_bestrq_last = codes
        codes_bestrq_emb = codes[0]

        batch_size = codes_bestrq_emb.shape[0]
        self.vq_embed.eval()
        quantized_feature_emb = self.vq_embed.get_output_from_indices(
            codes_bestrq_emb.transpose(1, 2)
        )
        quantized_feature_emb = self.cond_feature_emb(quantized_feature_emb)  # b t 512
        # assert 1==2
        quantized_feature_emb = F.interpolate(
            quantized_feature_emb.permute(0, 2, 1), scale_factor=2, mode="nearest"
        ).permute(0, 2, 1)

        # Feature Embedding Smoothing (Neural Crossfade)
        if smoothing_width > 0 and len(seam_indices) > 0:
            # Apply moving average blur at seam indices along the time dimension
            # quantized_feature_emb: [B, T, D]
            T_len = quantized_feature_emb.shape[1]
            for seam_idx in seam_indices:
                s_start = max(0, seam_idx - smoothing_width)
                s_end = min(T_len, seam_idx + smoothing_width)
                
                # We can't easily use conv1d on a slice in-place with variable width?
                # Simple iterative approach for the window:
                # Weighted average with neighbors
                
                # Iterative blur (3-tap) repeated `smoothing_width` times?
                # Or just a simple window average?
                # Let's do a simple weighted local average for the transition region.
                
                # To essentially "Crossfade", we want to obscure the hard cut.
                # A 3-tap blur repeated a few times is effective.
                
                blur_region = quantized_feature_emb[:, s_start:s_end, :].clone()
                # Apply 1 pass of [0.25, 0.5, 0.25] smoothing
                # Padding for convolution
                if blur_region.shape[1] >= 3:
                     # Permute to [B, D, T] for conv1d
                     blur_in = blur_region.permute(0, 2, 1)
                     kernel = torch.tensor([0.25, 0.5, 0.25], device=device, dtype=dtype).view(1, 1, 3).repeat(blur_in.shape[1], 1, 1)
                     
                     # Grouped conv for independent channel smoothing
                     blur_out = F.conv1d(blur_in, kernel, padding=1, groups=blur_in.shape[1])
                     
                     quantized_feature_emb[:, s_start:s_end, :] = blur_out.permute(0, 2, 1)
            
            # print(f"[FlowMatching] Applied Feature Smoothing at {seam_indices} (width {smoothing_width})")

        num_frames = quantized_feature_emb.shape[1]  #
        latents = torch.randn(
            (batch_size, num_frames, self.latent_dim), device=device, dtype=dtype
        )
        
        # 0: Ignore/Padding, 1: Fixed/In-Context, 2: Generate
        latent_masks = torch.zeros(
            latents.shape[0], latents.shape[1], dtype=torch.int64, device=latents.device
        )
        
        if external_mask is not None:
            # Custom In-Painting Mask
            # external_mask should be tensor of shape [B, T] or [T] with values 0, 1, 2
            if external_mask.dim() == 1:
                external_mask = external_mask.unsqueeze(0).repeat(batch_size, 1)
            
            # Ensure proper length match
            usable_len = min(external_mask.shape[1], latent_masks.shape[1])
            latent_masks[:, :usable_len] = external_mask[:, :usable_len]
        else:
            # Standard Scenarios
            latent_masks[:, 0:latent_length] = 2
            if scenario == "other_seg":
                latent_masks[:, 0:incontext_length] = 1

        quantized_feature_emb = (latent_masks > 0.5).unsqueeze(
            -1
        ) * quantized_feature_emb + (latent_masks < 0.5).unsqueeze(
            -1
        ) * self.zero_cond_embedding1.unsqueeze(
            0
        )

        if external_mask is not None:
            # Custom In-Painting Mask
            # external_mask should be tensor of shape [B, T] or [T] with values 0, 1, 2
            if external_mask.dim() == 1:
                external_mask = external_mask.unsqueeze(0).repeat(batch_size, 1)
            
            # Ensure proper length match
            usable_len = min(external_mask.shape[1], latent_masks.shape[1])
            latent_masks[:, :usable_len] = external_mask[:, :usable_len]
        else:
            # Standard Scenarios
            latent_masks[:, 0:latent_length] = 2
            if scenario == "other_seg":
                latent_masks[:, 0:incontext_length] = 1

        quantized_feature_emb = (latent_masks > 0.5).unsqueeze(
            -1
        ) * quantized_feature_emb + (latent_masks < 0.5).unsqueeze(
            -1
        ) * self.zero_cond_embedding1.unsqueeze(
            0
        )

        # Prepare Constraint Mask for Solver (1=Fixed/Constraint, 0=Free)
        # Assuming latent_masks: 1=Fixed, 2=Generate
        # So solve_mask = (latent_masks == 1).float()
        solver_mask = None
        if gradient_mask is not None:
             # Explicit Gradient provided (Decoupled from Conditioning)
             # Expected shape: [B, T] or [1, T] with values 0.0 to 1.0
             if gradient_mask.dim() == 1:
                 gradient_mask = gradient_mask.unsqueeze(0).repeat(batch_size, 1)
             # Ensure length
             usable = min(gradient_mask.shape[1], latents.shape[1])
             
             solver_mask = torch.zeros_like(latents[:, :, 0]) # [B, T]
             solver_mask[:, :usable] = gradient_mask[:, :usable]
             
        elif external_mask is not None:
             solver_mask = (latent_masks == 1).float()

        incontext_latents = (
            true_latents
            * ((latent_masks > 0.5) * (latent_masks < 1.5)).unsqueeze(-1).float()
        )
        # Note: incontext_length is only used for legacy prefix mode.
        if external_mask is not None:
            incontext_length = 0 
        else:
            incontext_length = ((latent_masks > 0.5) * (latent_masks < 1.5)).sum(-1)[0]

        additional_model_input = torch.cat([quantized_feature_emb], 1)
        temperature = 1.0
        t_span = torch.linspace(
            0, 1, num_steps + 1, device=quantized_feature_emb.device
        )
        latents = self.solve_euler(
            latents * temperature,
            incontext_latents,
            incontext_length,
            t_span,
            additional_model_input,
            guidance_scale,
            mask=solver_mask,
            callback=callback
        )

        if solver_mask is not None:
             # Final Force Set? Not strictly needed if solver held it, but safe.
             latents = (1 - solver_mask.unsqueeze(-1)) * latents + solver_mask.unsqueeze(-1) * incontext_latents
        else:
             latents[:, 0:incontext_length, :] = incontext_latents[
                :, 0:incontext_length, :
             ]  # B, T, dim
        return latents

    def solve_euler(self, x, incontext_x, incontext_length, t_span, mu, guidance_scale, mask=None, callback=None):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise (B, T, 256)
            incontext_x (torch.Tensor): constraint latents (B, T, 256)
            mask (torch.Tensor): 1.0 for fixed/constrained regions, 0.0 for free generation.
            callback (callable): Function(progress: float) -> None
        """
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        noise = x.clone()

        # Ensure mask is broadcastable
        if mask is not None:
            if mask.dim() == 1:
                mask = mask.view(1, -1, 1) # [1, T, 1]
            elif mask.dim() == 2:
                mask = mask.unsqueeze(-1) # [B, T, 1]
            # If mask is provided, incontext_x must be valid at mask==1 locations
        
        sol = []
        total_steps = len(t_span) - 1
        
        for step in tqdm(range(1, len(t_span))):
            if callback:
                # Progress 0.0 -> 1.0 (approx)
                callback(step / total_steps)
            # Apply Constraints / In-Context Fixing
            if mask is not None:
                # Interpolate constraint: (1 - t) * noise + t * data
                # Actually, flow matching targets the data at t=1.
                # The trajectory for fixed data 'x0' is 't * x0 + (1-t) * noise' ?
                # Standard FM: x_t = (1 - (1 - sigma_min) * t) * x_0 + t * x_1 ?
                # Code below uses: (1 - (1 - 1e-6) * t) * noise + t * context
                # This assumes 'noise' is the source (Standard Gaussian) and 'context' is the target (Data).
                # Yes, usually x1=data, x0=noise.
                
                # We apply this interpolation to the 'Fixed' regions
                forced_path = (1 - (1 - 1e-6) * t) * noise + t * incontext_x
                x = (1 - mask) * x + mask * forced_path
            else:
                # Legacy Prefix Mode
                if incontext_length > 0:
                    x[:, 0:incontext_length, :] = (1 - (1 - 1e-6) * t) * noise[
                        :, 0:incontext_length, :
                    ] + t * incontext_x[:, 0:incontext_length, :]

            if guidance_scale > 1.0:
                dphi_dt = self.estimator(
                    torch.cat(
                        [
                            torch.cat([x, x], 0),
                            torch.cat([incontext_x, incontext_x], 0),
                            torch.cat([torch.zeros_like(mu), mu], 0),
                        ],
                        2,
                    ),
                    timestep=t.unsqueeze(-1).repeat(2),
                )
                dphi_dt_uncond, dhpi_dt_cond = dphi_dt.chunk(2, 0)
                dphi_dt = dphi_dt_uncond + guidance_scale * (
                    dhpi_dt_cond - dphi_dt_uncond
                )
            else:
                dphi_dt = self.estimator(
                    torch.cat([x, incontext_x, mu], 2), timestep=t.unsqueeze(-1)
                )

            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        result = sol[-1]

        return result
