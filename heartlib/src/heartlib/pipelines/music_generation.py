from transformers.pipelines.base import Pipeline
from tokenizers import Tokenizer
from ..heartmula.modeling_heartmula import HeartMuLa
from ..heartcodec.modeling_heartcodec import HeartCodec
import torch
from typing import Dict, Any, Optional
import os
from dataclasses import dataclass
from tqdm import tqdm
import torchaudio
import json
from transformers import BitsAndBytesConfig


@dataclass
class HeartMuLaGenConfig:
    text_bos_id: int = 128000
    text_eos_id: int = 128001
    audio_eos_id: int = 8193
    empty_id: int = 0

    @classmethod
    def from_file(cls, path: str):
        with open(path, encoding="utf-8") as fp:
            data = json.load(fp)
        return cls(**data)


class HeartMuLaGenPipeline(Pipeline):
    def __init__(
        self,
        model: HeartMuLa,
        audio_codec: HeartCodec,
        muq_mulan: Optional[Any],
        text_tokenizer: Tokenizer,
        config: HeartMuLaGenConfig,
        device: torch.device,
        dtype: torch.dtype,
    ):
        super().__init__(model, device=device, dtype=dtype)
        self.model = model
        self.generation_dtype = dtype
        self.audio_codec = audio_codec
        self.muq_mulan = muq_mulan
        self.text_tokenizer = text_tokenizer
        self.config = config

        self.config = config
		
        # Fix: Hardcode to 8+1 to match model architecture and gitmain.
        # Codec config reports 7 quantizers which is incorrect/misleading for this model version.
        self._parallel_number = 8 + 1
        self._muq_dim = model.config.muq_dim

    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {"cfg_scale": kwargs.get("cfg_scale", 1.5)}
        forward_kwargs = {
            "max_audio_length_ms": kwargs.get("max_audio_length_ms", 120_000),
            "temperature": kwargs.get("temperature", 1.0),
            "topk": kwargs.get("topk", 50),
            "cfg_scale": kwargs.get("cfg_scale", 1.5),
            "callback": kwargs.get("callback", None),
            "abort_event": kwargs.get("abort_event", None),
            "history_tokens": kwargs.get("history_tokens", None),
            "suppress_eos": kwargs.get("suppress_eos", False),
        }
        postprocess_kwargs = {
            "save_path": kwargs.get("save_path", "output.mp3"),
        }
        return preprocess_kwargs, forward_kwargs, postprocess_kwargs

    def preprocess(self, inputs: Dict[str, Any], cfg_scale: float):

        # process tags
        tags = inputs["tags"]
        if os.path.isfile(tags):
            with open(tags, encoding="utf-8") as fp:
                tags = fp.read()
        assert isinstance(tags, str), f"tags must be a string, but got {type(tags)}"

        tags = tags.lower()
        # encapsulate with special <tag> and </tag> tokens
        # FIX: Ensure spacing for BPE tokenization to see ' <tag>' and not '<tag>' if needed, 
        # but importantly, ensure content is spaced from tags if that helps.
        # More critically, we are fixing the Sequence Structure.
        if not tags.startswith("<tag>"):
            tags = f"<tag> {tags}"
        if not tags.endswith("</tag>"):
            tags = f"{tags} </tag>"

        tags_ids = self.text_tokenizer.encode(tags).ids
        
        # Structure: [BOS] [Tags...] [EOS] [Special_0] [Lyrics...] [EOS]
        
        # 1. Tags Preamble
        if tags_ids[0] != self.config.text_bos_id:
            tags_ids = [self.config.text_bos_id] + tags_ids
        
        # RESTORED: Delimit tags with EOS so model knows "Style Context" is finished.
        if tags_ids[-1] != self.config.text_eos_id:
            tags_ids = tags_ids + [self.config.text_eos_id]

        # process reference audio
        ref_audio = inputs.get("ref_audio", None)
        if ref_audio is not None:
            raise NotImplementedError("ref_audio is not supported yet.")
        muq_embed = torch.zeros([self._muq_dim], dtype=self.generation_dtype)
        muq_idx = len(tags_ids) 
        # muq_idx is where the "separator" or "audio start" token is.
        # In this architecture, it seems index `len(tags_ids)` will be the [0] spacer.

        # process lyrics
        lyrics = inputs["lyrics"]
        if os.path.isfile(lyrics):
            with open(lyrics, encoding="utf-8") as fp:
                lyrics = fp.read()
        assert isinstance(
            lyrics, str
        ), f"lyrics must be a string, but got {type(lyrics)}"
        lyrics = lyrics.lower()

        lyrics_ids = self.text_tokenizer.encode(lyrics).ids
        
        # FIX: The tokenizer automatically adds BOS (128000). 
        # We must REMOVE it to ensure continuity from tags.
        # Structure should be: [BOS] Tags [EOS] [Muq] Lyrics (No BOS) [EOS]
        if lyrics_ids[0] == self.config.text_bos_id:
            lyrics_ids = lyrics_ids[1:]
        
        # 2. Lyrics Postamble
        if lyrics_ids[-1] != self.config.text_eos_id:
            lyrics_ids = lyrics_ids + [self.config.text_eos_id]

        # cat them together. tags, [0 placeholder for audio/muq], lyrics
        prompt_len = len(tags_ids) + 1 + len(lyrics_ids)

        tokens = torch.zeros([prompt_len, self._parallel_number], dtype=torch.long)
        
        # [0...len(tags)] = tags
        tokens[: len(tags_ids), -1] = torch.tensor(tags_ids)
        
        # [len(tags) + 1 ... end] = lyrics
        # The token at len(tags_ids) is left as 0 (empty_id or special separator)
        tokens[len(tags_ids) + 1 :, -1] = torch.tensor(lyrics_ids)

        tokens_mask = torch.zeros_like(tokens, dtype=torch.bool)
        tokens_mask[:, -1] = True

        bs_size = 2 if cfg_scale != 1.0 else 1

        def _cfg_cat(tensor: torch.Tensor, cfg_scale: float):
            tensor = tensor.unsqueeze(0)
            if cfg_scale != 1.0:
                tensor = torch.cat([tensor, tensor], dim=0)
            return tensor
            
        # Audio Condition Flag
        # If ref_audio is None, we are Unconditional (False).
        # If ref_audio is Present, we are Conditional (True).
        # We need a tensor [BS] of booleans/ints.
        # Note: ref_audio is not in the scope of this method, it's in inputs.
        # But 'muq_embed' was generated from it. 
        # Actually, let's look at how muq_embed is made. 
        # If ref_audio is None, muq_embed is Zeros.
        # We can detect if muq_embed is all zeros? No, too risky.
        # We should pass the flag from arguments or derive from inputs.
        # inputs['ref_audio'] might be available?
        has_audio = (inputs.get("ref_audio") is not None)
        use_audio_cond = torch.tensor([has_audio] * bs_size, dtype=torch.bool)

        return {
            "tokens": _cfg_cat(tokens, cfg_scale),
            "tokens_mask": _cfg_cat(tokens_mask, cfg_scale),
            "muq_embed": _cfg_cat(muq_embed, cfg_scale),
            "muq_idx": [muq_idx] * bs_size,
            "pos": _cfg_cat(torch.arange(prompt_len, dtype=torch.long), cfg_scale),
            "use_audio_cond": use_audio_cond,
        }

    def _forward(
        self,
        model_inputs: Dict[str, Any],
        max_audio_length_ms: int,
        temperature: float,
        topk: int,
        cfg_scale: float,
        callback=None,
        abort_event=None,
        history_tokens=None,
        suppress_eos=False,
    ):
        prompt_tokens = model_inputs["tokens"]
        if abort_event:
            print(f"DEBUG: _forward received abort_event: {abort_event}, is_set={abort_event.is_set()}")
            model_inputs["abort_event"] = abort_event
        else:
            print("DEBUG: _forward received NO abort_event")
        if callback:
             # print(f"DEBUG: _forward received callback: {callback}")
             model_inputs["callback"] = callback
        else:
             print("DEBUG: _forward received NO callback")
        if "abort_event" in model_inputs:
             print("DEBUG: _forward received abort_event")
        prompt_tokens_mask = model_inputs["tokens_mask"]
        continuous_segment = model_inputs["muq_embed"]
        starts = model_inputs["muq_idx"]
        prompt_pos = model_inputs["pos"]
        use_audio_cond = model_inputs.get("use_audio_cond", None)

        frames = []

        bs_size = 2 if cfg_scale != 1.0 else 1
        self.model.setup_caches(bs_size)
        
        from contextlib import nullcontext
        autocast_ctx = nullcontext() if self.device.type == "mps" else torch.autocast(device_type=self.device.type, dtype=self.generation_dtype)
        with autocast_ctx:
            curr_token = self.model.generate_frame(
                tokens=prompt_tokens,
                tokens_mask=prompt_tokens_mask,
                input_pos=prompt_pos,
                temperature=temperature,
                topk=topk,
                cfg_scale=cfg_scale,
                continuous_segments=continuous_segment,
                starts=starts,
                use_audio_cond=use_audio_cond,
            )


        max_audio_frames = max_audio_length_ms // 80

        # Context Injection (History)
        if history_tokens is not None:
             # If history tokens are provided, we should use them as the preamble for the audio generation.
             # Note: history_tokens should include the prompt tokens if we want to be strict, or just the audio part.
             # Assuming history_tokens is [B, S, parallel_number].
             # We need to feed them frame-by-frame or bulk-feed if setup_caches supports it.
             # Since generate_frame is autoregressive, we might just append them to frames?
             # BUT KV Cache needs to process them!
             # We need to run the model on history_tokens to warm up the cache.
             
             # Let's assume history_tokens is on correct device.
             from contextlib import nullcontext
             autocast_ctx = nullcontext() if self.device.type == "mps" else torch.autocast(device_type=self.device.type, dtype=self.generation_dtype)
             with autocast_ctx:
                 # We need to feed history tokens into the model to populate KV cache
                 # For efficiency, we should do this in one go if possible, but generate_frame handles single steps?
                 # Actually HeartMuLa expects continuous segments.
                 # Let's just iterate them quickly? 
                 # Or use a batch forward passes?
                 # pipeline.py suggests generate_frame is essentially a step-wise forward pass wrapper.
                 
                 # Let's iterate history to warm up:
                 print(f"DEBUG: Injecting {history_tokens.shape[1]} history frames...")
                 hz_len = history_tokens.shape[1]
                 
                 # Optimization: Batched processing would be better, but loop is safer for now given uncertain API.
                 # We assume history_tokens matches the bs_size requirements (tiled if cfg_scale > 1)
                 
                 # Fix: Ensure history tokens match cfg_scale batch size
                 if history_tokens.shape[0] != bs_size:
                     if cfg_scale != 1.0 and history_tokens.shape[0] == 1:
                          history_tokens = torch.cat([history_tokens, history_tokens], dim=0)
                 
                 # Limit history length to fit within max_seq_len
                 max_allowed_len = self.model.backbone.max_seq_len - (prompt_tokens.shape[1] + max_audio_frames)
                 if history_tokens.shape[1] > max_allowed_len:
                      print(f"DEBUG: Truncating history from {history_tokens.shape[1]} to {max_allowed_len} to avoid overflow")
                      history_tokens = history_tokens[:, -max_allowed_len:, :]
                      if history_tokens.shape[1] <= 0:
                          print("DEBUG: History truncated to 0 or less. Clearing history.")
                          history_tokens = None
                 
                 if history_tokens is not None:
                     hz_len = history_tokens.shape[1]
                     for j in range(hz_len):
                         h_token = history_tokens[:, j:j+1, :] # [B, 1, Parallel]
                         h_mask = torch.zeros_like(h_token, dtype=torch.bool) # All False? Or True?
                         # From _pad_audio_token logic:
                         # padded_token_mask[..., -1] = False (meaning valid?)
                         # tokens_mask[:, -1] = True (in preprocess) -> True means "Text"? 
                         # Wait, Preprocess sets `tokens_mask[:, -1] = True`.
                         # _pad_audio_token sets `padded_token_mask[..., -1] = False`.
                         # Audio tokens should have False mask? 
                         # Checking _embed_tokens: masked_embeds = embeds * tokens_mask.unsqueeze(-1)
                         # If mask is False (0), embed is zeroed? That sounds like padding?
                         # But _pad_audio_token sets it to False?
                         # Let's look closer at `generate_frame`...
                         # masked_embeds = embeds * tokens_mask.unsqueeze(-1)
                         # h = masked_embeds.sum... 
                         # If mask is 0, then that token is ignored?
                         # Audio tokens (0-7 codebooks) usually sum up?
                         # Codebook tokens are at dim -1.
                         # Ah, the mask is [B, S, Parallel].
                         # Text tokens have True at index -1 (text column).
                         # Audio tokens have False at index -1?
                         # Actually:
                         # tokens_mask = torch.zeros_like(tokens, dtype=torch.bool)
                         # tokens_mask[:, -1] = True  <-- For prompt (text), we enable the text column.
                         # For audio generated tokens, `_pad_audio_token` sets mask to False? 
                         # If mask is all False, embeds are 0.
                         # This implies audio tokens rely on `_embed_audio` logic which might be separate?
                         
                         # `generate_frame`:
                         # embeds = self._embed_tokens(tokens, ...)
                         # masked_embeds = embeds * tokens_mask.unsqueeze(-1)
                         # h = masked_embeds.sum(dim=2)
                         
                         # _embed_tokens:
                         # text_embeds = ...
                         # ...
                         # audio_embeds = ...
                         # return torch.cat([audio_embeds, text_embeds], dim=-2)
                         
                         # So mask aligns with these.
                         # If we are feeding AUDIO history, we want Audio columns enabled, Text disabled?
                         # If mask is all False, then h is all 0? That would be bad.
                         
                         # Check `_pad_audio_token` again (Line 190):
                         # padded_token_mask[..., -1] = False
                         # This implies the LAST column (Text) is masked out. 
                         # But indices 0..7 (Audio) are implicitly True?
                         # Wait, tokens_mask is [B, 1, Parallel] (Parallel=9 usually).
                         # _pad_audio_token creates mask `torch.ones_like(..., dtype=bool)`.
                         # So initially ALL TRUE. 
                         # Then sets last column to False.
                         # So Audio columns are True.
                         
                         # _pad_audio_token creates mask `torch.ones_like(..., dtype=bool)`.
                         # So initially ALL TRUE. 
                         # Then sets last column to False.
                         # So Audio columns are True.
                         
                         h_mask = torch.ones_like(h_token, dtype=torch.bool)
                         # Fix: Mask out the text/padding column explicitly so the model doesn't embed '0' as text
                         h_mask[..., -1] = False
                         
                         # Feed to generate_frame to update caches
                         # We ignore output, just need side effects (cache update)
                         # Fix: pass continuous_segments=None to avoid re-injecting MuQ at every step
                         # Fix: Capture the OUTPUT (prediction) to use as the next seed token
                         curr_token_prediction = self.model.generate_frame(
                            tokens=h_token,
                            tokens_mask=h_mask,
                            input_pos=prompt_pos[..., -1:] + j + 1,
                            temperature=temperature,
                            topk=topk,
                            cfg_scale=cfg_scale,
                            continuous_segments=None,
                            starts=None
                         )
                         
                         # Add to frames collection so result contains full audio
                         # Fix: Must match shape of generated frames [1, 8] (Audio Only, No Time dim)
                         # h_token is [B, 1, Parallel]. We want [1, Parallel-1] usually?
                         # Error said generated was [1, 8].
                         # So we take batch slice 0:1, remove time dim (0), remove text col (:-1)
                         frames.append(h_token[0:1, 0, :-1])

                 # Advance prompt_pos by history length
                 prompt_pos = prompt_pos + hz_len
                 
                 # FIX: Use the PREDICTION from the last history step as the seed.
                 # curr_token_prediction is [B, Parallel] (8 channels, Audio Only)
                 # We need [B, 8] (Audio channels only, no text)
                 # Fix: Do NOT slice. It is already 8 channels.
                 curr_token = curr_token_prediction

        else:
             # No history provided. We must append the initial predicted token.
             frames.append(curr_token[0:1,]) 


        def _pad_audio_token(token: torch.Tensor):
            padded_token = (
                torch.ones(
                    (token.shape[0], self._parallel_number),
                    device=token.device,
                    dtype=torch.long,
                )
                * self.config.empty_id
            )
            padded_token[:, :-1] = token
            padded_token = padded_token.unsqueeze(1)
            padded_token_mask = torch.ones_like(
                padded_token, device=token.device, dtype=torch.bool
            )
            padded_token_mask[..., -1] = False
            return padded_token, padded_token_mask



        for i in tqdm(range(max_audio_frames)):
            # Callback: Update Progress
            if "callback" in model_inputs:
                cb = model_inputs["callback"]
                if cb:
                    # Report progress percentage (0-100)
                    progress = int((i / max_audio_frames) * 100)
                    cb(progress, f"Generating frame {i}/{max_audio_frames}")

            # Check Cancellation
            if "abort_event" in model_inputs:
                if i % 10 == 0:
                     # Check occasionally to avoid spamming logs, but check status every frame
                     if model_inputs["abort_event"].is_set():
                         print("DEBUG: abort_event IS SET. Stopping.")
                
                if model_inputs["abort_event"].is_set():
                    raise InterruptedError("Generation cancelled by user")

            curr_token, curr_token_mask = _pad_audio_token(curr_token)

            # MPS Autocast is often unstable or warns about dtype support. 
            # Since we load in float16 for MPS, we can skip autocast or use a nullcontext.
            from contextlib import nullcontext
            autocast_ctx = nullcontext() if self.device.type == "mps" else torch.autocast(device_type=self.device.type, dtype=self.generation_dtype)
            
            with autocast_ctx:
                curr_token = self.model.generate_frame(
                    tokens=curr_token,
                    tokens_mask=curr_token_mask,
                    input_pos=prompt_pos[..., -1:] + i + 1,
                    temperature=temperature,
                    topk=topk,
                    cfg_scale=cfg_scale,
                    continuous_segments=None,
                    starts=None,

                )
            
            # Check for Context Overflow
            # We check if Next Input Pos will exceed max_seq_len
            # input_pos is [B, 1]. We check max value.
            if torch.max(prompt_pos) + i + 2 >= self.model.backbone.max_seq_len:
                 print(f"DEBUG: Reached MAX SEQUENCE LENGTH {self.model.backbone.max_seq_len}. Stopping generation.")
                 frames.append(curr_token[0:1,]) # Save last frame
                 break
            # EOS Check: Break if any token predicts EOS, UNLESS suppress_eos is enabled (for Instrumental Mode)
            if not suppress_eos and torch.any(curr_token[0:1, :] >= self.config.audio_eos_id):
                break
            frames.append(curr_token[0:1,])
        frames = torch.stack(frames).permute(1, 2, 0).squeeze(0)
        # Ensure audio_codec is on CPU for detokenization to avoid MPS "Output channels > 65536" error
        # and other potential MPS backend issues for this specific large conv1d op.
        self.audio_codec.to("cpu")
        wav = self.audio_codec.detokenize(frames.to("cpu"), device="cpu")
        
        # Move back to device if needed (though this is the end of forward usually)
        if self.device.type != "cpu":
             self.audio_codec.to(self.device)
             
        return {"wav": wav, "tokens": frames}

    def postprocess(self, model_outputs: Dict[str, Any], save_path: str):
        wav = model_outputs["wav"]
        # Use 'soundfile' backend explicitly if available, or try default with format spec
        try:
             torchaudio.save(save_path, wav, 48000, backend="soundfile")
        except Exception:
             # Fallback
             try:
                 torchaudio.save(save_path, wav, 48000, format="mp3")
             except Exception:
                 # Last resort: save as wav and rename? No, just try default.
                  torchaudio.save(save_path, wav, 48000)
        return model_outputs

    @classmethod
    def from_pretrained(
        cls,
        pretrained_path: str,
        device: torch.device,
        dtype: torch.dtype,
        version: str,
        bnb_config: Optional[BitsAndBytesConfig] = None,
    ):

        if os.path.exists(
            heartcodec_path := os.path.join(pretrained_path, "HeartCodec-oss")
        ):
            from safetensors.torch import load_file
            
            # Load state dict manually to fix shape mismatch
            safetensors_path = os.path.join(heartcodec_path, "model.safetensors")
            if os.path.exists(safetensors_path):
                state_dict = load_file(safetensors_path)
                # Fix initted tensors being 1D instead of scalar
                new_state_dict = {}
                for key, value in state_dict.items():
                    if "initted" in key and value.dim() == 1:
                        new_state_dict[key] = value.squeeze()
                    else:
                        new_state_dict[key] = value
                
                # device_map triggers accelerate which might reload from disk. Load to CPU then move.
                # FIX: Transformers doesn't allow state_dict + path in from_pretrained.
                # We load config, init model, then load state_dict.
                if hasattr(HeartCodec, "config_class"):
                     config = HeartCodec.config_class.from_pretrained(heartcodec_path)
                else:
                     # Fallback if config_class not explicitly set, try AutoConfig or assume standard
                     from transformers import AutoConfig
                     config = AutoConfig.from_pretrained(heartcodec_path)
                
                heartcodec = HeartCodec(config)
                heartcodec.load_state_dict(new_state_dict)
                heartcodec.to(device)
            else:
                 heartcodec = HeartCodec.from_pretrained(heartcodec_path, device_map=device)

        else:
            raise FileNotFoundError(
                f"Expected to find checkpoint for HeartCodec at {heartcodec_path} but not found. Please check your folder {pretrained_path}."
            )

        if os.path.exists(
            heartmula_path := os.path.join(pretrained_path, f"HeartMuLa-oss-{version}")
        ):
            heartmula = HeartMuLa.from_pretrained(
                heartmula_path, torch_dtype=dtype, quantization_config=bnb_config
            )
        else:
            raise FileNotFoundError(
                f"Expected to find checkpoint for HeartMuLa at {heartmula_path} but not found. Please check your folder {pretrained_path}."
            )

        if os.path.isfile(
            vocab_path := os.path.join(pretrained_path, "tokenizer.json")
        ):
            tokenizer = Tokenizer.from_file(vocab_path)
        else:
            raise FileNotFoundError(
                f"Expected to find tokenizer.json for HeartMuLa at {vocab_path} but not found. Please check your folder {pretrained_path}."
            )

        if os.path.isfile(
            gen_config_path := os.path.join(pretrained_path, "gen_config.json")
        ):
            gen_config = HeartMuLaGenConfig.from_file(gen_config_path)
        else:
            raise FileNotFoundError(
                f"Expected to find gen_config.json for HeartMuLa at {gen_config_path} but not found. Please check your folder {pretrained_path}."
            )

        return cls(heartmula, heartcodec, None, tokenizer, gen_config, device, dtype)
