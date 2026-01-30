"""
Export nanochat tokenizer to flat .tok binary file for C inference.

Format:
  magic:         4 bytes  (0x4E544F4B = "NTOK")
  vocab_size:    4 bytes  (int32)
  max_token_len: 4 bytes  (int32)
  For each token (0..vocab_size-1):
    byte_len:    4 bytes  (int32)
    bytes:       byte_len bytes
  n_special:     4 bytes  (int32)
  For each special token:
    token_id:    4 bytes  (int32)
    name_len:    4 bytes  (int32)
    name:        name_len bytes (utf-8)

Usage:
  python export_tokenizer.py <tokenizer_dir> <output.tok>
  python export_tokenizer.py ../nanochat_cache/tokenizer/ tokenizer.tok
"""
import sys
import os
import struct
import pickle

# Add nanochat to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'nanochat'))

def export_tokenizer(tokenizer_dir, output_path):
    # Load the tiktoken tokenizer from pickle
    tok_path = os.path.join(tokenizer_dir, 'tokenizer.pkl')
    print(f"Loading tokenizer: {tok_path}")
    with open(tok_path, 'rb') as f:
        enc = pickle.load(f)

    vocab_size = enc.n_vocab
    print(f"Vocab size: {vocab_size}")

    # Build token -> bytes mapping
    # tiktoken's _mergeable_ranks maps bytes -> rank
    # We need rank -> bytes (inverse)
    token_bytes = {}

    # Regular tokens from mergeable_ranks
    # Access the internal encoding data
    # tiktoken Encoding stores mergeable_ranks as bytes->int mapping
    for token_bytes_val, rank in enc._mergeable_ranks.items():
        token_bytes[rank] = token_bytes_val

    # Special tokens
    special_tokens = {}
    for name, token_id in enc._special_tokens.items():
        special_tokens[name] = token_id
        token_bytes[token_id] = name.encode('utf-8')

    print(f"Regular tokens: {len(enc._mergeable_ranks)}")
    print(f"Special tokens: {len(special_tokens)}")
    for name, tid in sorted(special_tokens.items(), key=lambda x: x[1]):
        print(f"  {tid}: {name}")

    # Find max token length
    max_len = max(len(b) for b in token_bytes.values()) if token_bytes else 0
    print(f"Max token length: {max_len}")

    # Write binary
    MAGIC = 0x4E544F4B  # "NTOK"
    with open(output_path, 'wb') as f:
        f.write(struct.pack('I', MAGIC))
        f.write(struct.pack('i', vocab_size))
        f.write(struct.pack('i', max_len))

        # Write each token's bytes
        for i in range(vocab_size):
            if i in token_bytes:
                b = token_bytes[i]
                f.write(struct.pack('i', len(b)))
                f.write(b)
            else:
                # Missing token (shouldn't happen but just in case)
                f.write(struct.pack('i', 0))

        # Write special tokens
        special_list = list(special_tokens.items())
        f.write(struct.pack('i', len(special_list)))
        for name, token_id in special_list:
            name_bytes = name.encode('utf-8')
            f.write(struct.pack('i', token_id))
            f.write(struct.pack('i', len(name_bytes)))
            f.write(name_bytes)

    file_size = os.path.getsize(output_path)
    print(f"\nSaved to: {output_path} ({file_size / 1024:.1f} KB)")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python export_tokenizer.py <tokenizer_dir> <output.tok>")
        sys.exit(1)
    export_tokenizer(sys.argv[1], sys.argv[2])
