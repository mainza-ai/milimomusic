
from tokenizers import Tokenizer
import json
from dataclasses import dataclass

@dataclass
class Config:
    text_bos_id: int = 128000
    text_eos_id: int = 128001

config = Config()
tokenizer = Tokenizer.from_file("heartlib/ckpt/tokenizer.json")

def preprocess_sim(tags, lyrics):
    tags_ids = tokenizer.encode(tags).ids
    lyrics_ids = tokenizer.encode(lyrics).ids
    
    print(f"Raw Tags IDs: {tags_ids[:5]}...")
    print(f"Raw Lyrics IDs: {lyrics_ids[:5]}...")
    
    # Current Code Logic Simulation
    # 1. Tags Preamble
    if tags_ids[0] != config.text_bos_id:
        tags_ids = [config.text_bos_id] + tags_ids
    
    # RESTORED EOS
    if tags_ids[-1] != config.text_eos_id:
        tags_ids = tags_ids + [config.text_eos_id]
        
    # 2. Lyrics Postamble (My current code comments out the ADDITION, but doesn't REMOVE)
    # The logic in file was:
    # # if lyrics_ids[0] != self.config.text_bos_id:
    # #    lyrics_ids = [self.config.text_bos_id] + lyrics_ids
    
    if lyrics_ids[-1] != config.text_eos_id:
        lyrics_ids = lyrics_ids + [config.text_eos_id]
        
    # Result
    print(f"Final Tags End: ...{tags_ids[-3:]}")
    print(f"Final Lyrics Start: {lyrics_ids[:3]}...")
    
    # Check for Double BOS
    if lyrics_ids[0] == config.text_bos_id:
        print("ISSUE CONFIRMED: Lyrics Start with BOS (128000).")
    else:
        print("No BOS at start of Lyrics.")

preprocess_sim("<tag> pop </tag>", "[verse] hello world")
