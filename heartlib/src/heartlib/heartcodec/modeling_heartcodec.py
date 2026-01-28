import torch
from .models.flow_matching import FlowMatching
from .models.sq_codec import ScalarModel
from .configuration_heartcodec import HeartCodecConfig
from transformers.modeling_utils import PreTrainedModel
import math
import numpy as np
import torch.nn.functional as F


class HeartCodec(PreTrainedModel):
    config_class = HeartCodecConfig

    def __init__(
        self,
        config: HeartCodecConfig,
    ):
        super(HeartCodec, self).__init__(config)

        self.config = config

        self.flow_matching = FlowMatching(
            dim=config.dim,
            codebook_size=config.codebook_size,
            decay=config.decay,
            commitment_weight=config.commitment_weight,
            threshold_ema_dead_code=config.threshold_ema_dead_code,
            use_cosine_sim=config.use_cosine_sim,
            codebook_dim=config.codebook_dim,
            num_quantizers=config.num_quantizers,
            attention_head_dim=config.attention_head_dim,
            in_channels=config.in_channels,
            norm_type=config.norm_type,
            num_attention_heads=config.num_attention_heads,
            num_layers=config.num_layers,
            num_layers_2=config.num_layers_2,
            out_channels=config.out_channels,
        )
        self.scalar_model = ScalarModel(
            num_bands=config.num_bands,
            sample_rate=config.sample_rate,
            causal=config.causal,
            num_samples=config.num_samples,
            downsample_factors=config.downsample_factors,
            downsample_kernel_sizes=config.downsample_kernel_sizes,
            upsample_factors=config.upsample_factors,
            upsample_kernel_sizes=config.upsample_kernel_sizes,
            latent_hidden_dim=config.latent_hidden_dim,
            default_kernel_size=config.default_kernel_size,
            delay_kernel_size=config.delay_kernel_size,
            init_channel=config.init_channel,
            res_kernel_size=config.res_kernel_size,
        )
        self.post_init()

        self.sample_rate = config.sample_rate

    @torch.inference_mode()
    def detokenize(
        self,
        codes,
        duration=29.76,
        num_steps=10,
        disable_progress=False,
        guidance_scale=1.25,
        device="cuda",
        callback=None,
    ):
        codes = codes.unsqueeze(0).to(device)
        min_samples = int(duration * 12.5)
        # Consistent latent length (2x tokens)
        latent_length = min_samples * 2
        
        first_latent = torch.randn(codes.shape[0], latent_length, 256).to(
            device
        )  # B, T, 64
        first_latent_length = 0
        first_latent_codes_length = 0
        min_samples = int(duration * 12.5)
        hop_samples = min_samples // 93 * 80
        ovlp_samples = min_samples - hop_samples
        ovlp_frames = ovlp_samples * 2
        codes_len = codes.shape[-1]  #
        target_len = int(
            (codes_len - first_latent_codes_length) / 12.5 * self.sample_rate
        )

        # code repeat
        if codes_len < min_samples:
            while codes.shape[-1] < min_samples:
                codes = torch.cat([codes, codes], -1)
            codes = codes[:, :, 0:min_samples]
        codes_len = codes.shape[-1]
        if (codes_len - ovlp_frames) % hop_samples > 0:
            len_codes = (
                math.ceil((codes_len - ovlp_samples) / float(hop_samples)) * hop_samples
                + ovlp_samples
            )
            while codes.shape[-1] < len_codes:
                codes = torch.cat([codes, codes], -1)
            codes = codes[:, :, 0:len_codes]
            codes = codes[:, :, 0:len_codes]
        
        # Enforce consistency: Latents are exactly 2x the Code Frames derived from duration
        latent_length = min_samples * 2 
        latent_list = []

        for sinx in range(0, codes.shape[-1] - hop_samples + 1, hop_samples):
            codes_input = []
            codes_input.append(codes[:, :, sinx : sinx + min_samples])
            if sinx == 0 or ovlp_frames == 0:
                incontext_length = first_latent_length
                latents = self.flow_matching.inference_codes(
                    codes_input,
                    first_latent,
                    latent_length,
                    incontext_length,
                    guidance_scale=guidance_scale,
                    num_steps=num_steps,
                    disable_progress=disable_progress,
                    scenario="other_seg",
                )
                latent_list.append(latents)
            else:
                true_latent = latent_list[-1][:, -ovlp_frames:, :]
                len_add_to_latent = latent_length - true_latent.shape[1]  #
                incontext_length = true_latent.shape[1]
                true_latent = torch.cat(
                    [
                        true_latent,
                        torch.randn(
                            true_latent.shape[0],
                            len_add_to_latent,
                            true_latent.shape[-1],
                        ).to(device),
                    ],
                    1,
                )
                latents = self.flow_matching.inference_codes(
                    codes_input,
                    true_latent,
                    latent_length,
                    incontext_length,
                    guidance_scale=guidance_scale,
                    num_steps=num_steps,
                    disable_progress=disable_progress,
                    scenario="other_seg",
                )
                latent_list.append(latents)

        latent_list = [l.float() for l in latent_list]
        latent_list[0] = latent_list[0][:, first_latent_length:, :]
        min_samples = int(duration * self.sample_rate)
        hop_samples = min_samples // 93 * 80
        ovlp_samples = min_samples - hop_samples

        output = None
        for i in range(len(latent_list)):
            latent = latent_list[i]
            bsz, t, f = latent.shape

            latent = latent.reshape(
                latent.shape[0], latent.shape[1], 2, latent.shape[2] // 2
            ).permute(0, 2, 1, 3)
            latent = latent.reshape(
                latent.shape[0] * 2, latent.shape[2], latent.shape[3]
            )
            
            # FIX: Move to CPU explicitly to avoid MPS "Output channels > 65536" error
            # This is a robust workaround when env vars fail.
            mix_device = device
            is_mps = False
            if hasattr(device, 'type') and device.type == 'mps':
                is_mps = True
            elif isinstance(device, str) and 'mps' in device:
                is_mps = True
            
            if is_mps:
                self.scalar_model.to("cpu")
                latent = latent.to("cpu")
            
            try:
                cur_output = (
                    self.scalar_model.decode(latent.transpose(1, 2)).squeeze(0).squeeze(1)
                )  # 1 512 256
            finally:
                 if is_mps:
                     self.scalar_model.to(mix_device)

            cur_output = cur_output[:, 0:min_samples].detach().cpu()  # B, T
            if cur_output.dim() == 3:
                cur_output = cur_output[0]

            if output is None:
                output = cur_output
            else:
                if ovlp_samples == 0:
                    output = torch.cat([output, cur_output], -1)
                else:
                    ov_win = torch.from_numpy(np.linspace(0, 1, ovlp_samples)[None, :])
                    ov_win = torch.cat([ov_win, 1 - ov_win], -1)
                    output[:, -ovlp_samples:] = (
                        output[:, -ovlp_samples:] * ov_win[:, -ovlp_samples:]
                        + cur_output[:, 0:ovlp_samples] * ov_win[:, 0:ovlp_samples]
                    )
                    output = torch.cat([output, cur_output[:, ovlp_samples:]], -1)
        output = output[:, 0:target_len]
        return output

    @torch.inference_mode()
    def inpaint(
        self,
        codes,
        start_frame: int,
        end_frame: int,
        duration=29.76,
        num_steps=20,
        device="cuda",
        mask_mode: int = 0,
        mask_expansion: int = 2,
        seam_width: int = 10, # Gradient width. 10 frames = ~400ms soft blend.
        callback=None,
    ):
        """
        Regenerate a specific segment of audio tokens while keeping surrounding context.
        start_frame: The frame index to start regeneration (inclusive).
        end_frame: The frame index to end regeneration (exclusive).
        mask_mode: 0 = Unconditional, 2 = Conditional.
        mask_expansion: Context expansion.
        seam_width: Width of the soft gradient transition at the boundaries.
        """
        device = torch.device(device)
        codes = codes.unsqueeze(0).to(device)
        
        # Create In-Painting Mask
        # 0: Padding/Ignored (if any)
        # 1: Fixed/In-Context (Surrounding audio)
        # 2: Generate (The gap)
        
        # Dimensions
        bsz, num_codebooks, seq_len = codes.shape
        latent_len = int(duration * 25) # Default latent length
        
        # Scale frames to Latent Space (12.5Hz -> 25Hz approx, strict mapping needed)
        # FlowMatching uses `quantized_feature_emb` which is F.interpolate(codes, scale_factor=2)
        # So latent_t = code_t * 2
        
        # Ensure we are using a FRESH seed for the repair to avoid reproducing the same artifact
        # if the original was generated deterministically.
        
        # MPS Generator often fails with "Placeholder storage" error.
        # Safer to generate noise on CPU and move to device FOR MPS ONLY.
        # Windows/Linux (CUDA) should use native generation for speed.
        
        use_cpu_rng = (device.type == 'mps')
        
        rng_device = "cpu" if use_cpu_rng else device
        gen = torch.Generator(device=rng_device)
        gen.seed() 
        
        latent_mask = torch.ones(seq_len * 2, dtype=torch.int64, device=device) # Init as 1 (Fixed)
        
        # Solver Gradient Mask (1.0 = Fixed, 0.0 = Free)
        # We start with 1.0 everywhere (Fixed to context)
        gradient_mask = torch.ones(seq_len * 2, dtype=torch.float32, device=device)
        
        l_start = start_frame * 2
        l_end = end_frame * 2
        
        # Keep original indices for seam smoothing
        l_seam_start = l_start
        l_seam_end = l_end
        
        if mask_expansion > 0:
            l_start -= mask_expansion * 2
            l_end += mask_expansion * 2
        
        # Clamp
        l_start = max(0, l_start)
        l_end = min(latent_mask.shape[0], l_end)
        
        print(f"[HeartCodec] In-Painting: Frames {l_start} to {l_end} (Total {latent_mask.shape[0]}) | Mode: {mask_mode} + Gradient")
        
        # 1. Set Conditioning Mask (Tokens)
        # 2 = Condition on Tokens (Patching)
        # 0 = Blind (Silence)
        latent_mask[l_start:l_end] = mask_mode 
        
        # 2. Set Solver Mask (Constraint)
        # Always 0.0 (Free) in the gap to allow generation
        gradient_mask[l_start:l_end] = 0.0
        
        # Soft Gradient Smoothing on Solver Mask
        if seam_width > 0:
            steps = torch.linspace(1, 0, seam_width, device=device)
            back_steps = torch.linspace(0, 1, seam_width, device=device)
            
            # Start Ramp: 1.0 -> 0.0 (Fixed -> Free)
            s_end = min(l_end, l_start + seam_width)
            actual_width = s_end - l_start
            if actual_width > 0:
                gradient_mask[l_start : s_end] = steps[:actual_width]
            
            # End Ramp: 0.0 -> 1.0 (Free -> Fixed)
            e_start = max(l_start, l_end - seam_width)
            actual_width = l_end - e_start
            if actual_width > 0:
                gradient_mask[e_start : l_end] = back_steps[-actual_width:]

            print(f"[HeartCodec] Decoupled Gradient Applied (Width {seam_width})")
        
        # Call Inference with custom mask
        # We reuse the logic of detokenize but pass the mask
        
        # We need `first_latent` (random noise) for the generation target
        # Generate on CPU to use the CPU generator, then move to device
        first_latent = torch.randn(bsz, seq_len * 2, 256, generator=gen, device=rng_device).to(device)
        
        # We need `true_latents` (the original audio latents) for the context
        # But we don't have latents, we have codes.
        # FlowMatching.inference_codes converts codes to latents internally via `vq_embed`.
        # However, the `true_latents` arg in `inference_codes` serves as the TARGET context source.
        # So we probably need to decode the codes first? 
        
        # We need `true_latents` (the original audio latents) for the context.
        # Since we don't have the original latents stored, and we cannot perfectly recover them from codes without re-generation,
        # we have a choice:
        # 1. Regenerate full latents first (computationally heavy).
        # 2. Use random noise as 'true_latents' and hope the model converges to something consistent? (Bad for keeping context).
        # 3. Use 0 initialization?
        
        # Correct approach given constraints: Initialize 'true_latents' as random noise (standard gaussian prior).
        # Flow Matching will guide these noise latents towards the data manifold conditioned on the codes.
        # Ideally, we should solve the ODE for the "Fixed" regions properly.
        # But `inference_codes` now handles masking. If we pass `true_latents` as noise...
        # Wait, `solve_euler` mixes `x` (current sample) with `incontext_x` (target).
        # If `incontext_x` is just noise, we guide towards noise. That's bad.
        # We need VALID latents for the fixed regions.
        
        # Solution: We must run a generic inference pass FIRST to get valid latents for the whole sequence.
        # This acts as our "Approximated Ground Truth".
        # Yes, it will differ from the original file (new seed), but it will be consistent with itself.
        
        # 1. Generate full latents (Approximation of "Original")
        # We can optimize by only generating for the "Fixed" regions?
        # But simplest is just run one pass.
        # Actually, let's run a "Draft" pass with fewer steps?
        # Let's try 10 steps for the base.
        
        # Replicate embedding setup
        self.flow_matching.vq_embed.eval()
        true_emb = self.flow_matching.vq_embed.get_output_from_indices(codes.transpose(1, 2))
        true_emb = self.flow_matching.cond_feature_emb(true_emb)
        true_emb = F.interpolate(true_emb.permute(0, 2, 1), scale_factor=2, mode="nearest").permute(0, 2, 1)
        
        # Run Base Inference (to get context)
        # Note: We pass Dummy true_latents here because we are generating everything from scratch.
        # Dimension must be [B, T, 256]
        dummy_latents = torch.randn(bsz, seq_len * 2, 256, generator=gen, device=rng_device).to(device)
        
        # First Pass: Generate "Original"
        # We use fewer steps to be fast, but quality might suffer. 
        # Using passed num_steps for consistency.
        base_latents = self.flow_matching.inference_codes(
            [codes],
            dummy_latents, # Not used since incontext_len=0
            latent_length=seq_len * 2,
            incontext_length=0,
            num_steps=max(10, num_steps // 2), # Optimization
            disable_progress=True,
            scenario="chem"
        )
        
        # 2. Run In-Painting Pass
        # Now we use `base_latents` as the `true_latents` (Constraint).
        
        target_latents = self.flow_matching.inference_codes(
            [codes], 
            base_latents, # Use the generated base as the truth/constraint
            latent_length=seq_len * 2,
            incontext_length=0, # Ignored due to custom mask
            num_steps=num_steps,
            disable_progress=False,
            scenario="chem", 
            external_mask=latent_mask.unsqueeze(0),
            gradient_mask=gradient_mask.unsqueeze(0),
            seam_indices=[l_start, l_end], # Locations to blur
            smoothing_width=4, # ~120ms window for feature morphing
            callback=callback
        )
        
        latents = target_latents
        
        # NOTE: Latent Smoothing removed in favor of Soft Gradient Masking (above)
        
        latents = target_latents
        
        # Decode Latents to Audio
        # Re-use Scalar Model decoding logic
        latent = latents
        bsz, t, f = latent.shape
        latent = latent.reshape(bsz, t, 2, f // 2).permute(0, 2, 1, 3)
        latent = latent.reshape(bsz * 2, t, f // 2)
        
        # FIX: Move to CPU explicitly to avoid MPS "Output channels > 65536" error
        # This is a robust workaround when env vars fail.
        # Only apply this workaround for MPS devices to avoid slowing down CUDA users.
        original_device = device
        is_mps = False
        
        if hasattr(device, 'type') and device.type == 'mps':
            is_mps = True
        elif isinstance(device, str) and 'mps' in device:
            is_mps = True
            
        if is_mps:
            self.scalar_model.to("cpu")
            latent = latent.to("cpu")
        
        try:
            output = self.scalar_model.decode(latent.transpose(1, 2)).squeeze(0).squeeze(1)
        finally:
            # Restore model to original device if we moved it
            if is_mps:
                self.scalar_model.to(original_device)
        
        # Since we did full sequence, we return full wav
        return output.cpu()

    @torch.inference_mode()
    def encode(self, audio):
        """
        Encode audio waveform into tokens/latents.
        Args:
            audio (torch.Tensor): Audio waveform of shape [B, 1, T] or [B, T]
        """
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
            
        # scalar_model.encode requires [B, C, T] ?
        # ScalarModel.encode checks layer loop.
        # It expects input consistent with Conv1d(num_bands, ...)
        # Audio is usually single channel? Config says num_bands.
        
        # Audio input formatting logic from original inference
        # The scalar model processes raw audio?
        # Actually ScalarModel.forward takes 'x'. 
        
        device = audio.device
        
        # FIX: Move to CPU explicitly to avoid MPS "Output channels > 65536" error in Encoder too
        original_device = device
        is_mps = False
        
        if hasattr(device, 'type') and device.type == 'mps':
            is_mps = True
        elif isinstance(device, str) and 'mps' in device:
            is_mps = True
            
        if is_mps:
            self.scalar_model.to("cpu")
            audio = audio.to("cpu")
            
        try:
            return self.scalar_model.encode(audio).to(original_device) 
        finally:
            if is_mps:
                self.scalar_model.to(original_device)
