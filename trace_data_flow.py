
import torch
from dataclasses import dataclass
from typing import Dict, Any

# Mock Config
@dataclass
class Config:
    text_bos_id: int = 128000
    text_eos_id: int = 128001
    muq_dim: int = 1024
    audio_vocab_size: int = 8192
    
config = Config()

# Mock Pipeline Preprocess (Simplified)
def preprocess(tags, lyrics, cfg_scale=1.5):
    print("--- Preprocess ---")
    # Simulate Tokenizer
    tags_ids = [128000, 1, 2, 3, 128001] # BOS, tags, EOS
    lyrics_ids = [128000, 4, 5, 6, 128001] # BOS, lyrics, EOS
    
    print(f"Original Lyrics IDs: {lyrics_ids}")
    
    # 1. Double BOS Fix
    if lyrics_ids[0] == config.text_bos_id:
        lyrics_ids = lyrics_ids[1:]
        print(f"Fixed Lyrics IDs: {lyrics_ids}")
        
    prompt_len = len(tags_ids) + 1 + len(lyrics_ids)
    print(f"Prompt Len: {prompt_len}")
    
    # Gap Index
    muq_idx = len(tags_ids)
    print(f"Muq Index (Gap): {muq_idx}")
    
    # Tensors
    muq_embed = torch.zeros([config.muq_dim]) # Zeros!
    
    bs_size = 2 if cfg_scale != 1.0 else 1
    
    inputs = {
        "muq_embed": muq_embed.unsqueeze(0).repeat(bs_size, 1),
        "muq_idx": [muq_idx] * bs_size,
        "bs_size": bs_size
    }
    return inputs

# Mock Model Generate Frame Logic
def generate_frame_logic(inputs):
    print("\n--- Generate Frame Logic ---")
    continuous_segments = inputs["muq_embed"]
    bs = inputs["bs_size"]
    
    # Simulate Layers
    muq_linear = torch.nn.Linear(1024, 3072)
    unconditional_text_embedding = torch.nn.Embedding(1, 3072)
    
    print(f"Continuous Segments (Input): {continuous_segments[0][:5]} (Zeros)")
    
    # Current Logic in modeling_heartmula.py
    # 1. Linear Projection
    projected = muq_linear(continuous_segments)
    print(f"Projected (Linear(Zeros)): {projected[0][:5]} (Likely bias values, NOT ZEROS)")
    
    # 2. CFG Masking
    # uncond_mask is [0, ... 1, ...] for CFG
    # In 'conditional' part (index 0), uncond_mask is 0.
    # So we use 'projected'.
    
    print("Issue: In conditional branch, we use 'Projected' which is Linear(Zeros).")
    print("Hypothesis: This Linear(Zeros) is NOT the correct 'No Audio' signal.")
    print("Correct Signal should contain 'unconditional_text_embedding' weights.")
    
    # Desired Logic
    uncond_embed = unconditional_text_embedding(torch.zeros(1, dtype=torch.long))
    print(f"Unconditional Embed (Learned): {uncond_embed[0][:5]}")
    
    print("Verification: Does Projected == Uncond Embed? (Very Unlikely)")

preprocess(None, None)
generate_frame_logic(preprocess(None, None))
