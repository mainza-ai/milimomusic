
from tokenizers import Tokenizer
import os

try:
    tokenizer = Tokenizer.from_file("heartlib/ckpt/tokenizer.json")
    print("Tokenizer loaded.")
    
    # Check for <tag>
    vocab = tokenizer.get_vocab()
    if "<tag>" in vocab:
        print(f"<tag> FOUND: ID {vocab['<tag>']}")
    else:
        print("<tag> NOT FOUND in vocab keys.")
    
    # Check encoding
    text = "<tag>pop music</tag>"
    encoded = tokenizer.encode(text)
    print(f"'{text}' encodes to: {encoded.tokens} (IDs: {encoded.ids})")

    text_spaced = "<tag> pop music </tag>"
    encoded_spaced = tokenizer.encode(text_spaced)
    print(f"'{text_spaced}' encodes to: {encoded_spaced.tokens} (IDs: {encoded_spaced.ids})")

except Exception as e:
    print(f"Error: {e}")
