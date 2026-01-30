"""
Export nanochat .pt checkpoint to flat .bin file.

Supports:
  --fp16  (default) float16 weights, VERSION=1
  --q8    INT8 per-row quantization, VERSION=2 (~2x smaller)
  --q4    INT4 per-row quantization, VERSION=3 (~4x smaller)
  --q4h   HYBRID: embeddings q8 + attention/MLP q4, VERSION=4 (~30% smaller than q8)

Header: 256 bytes: magic + config + quant_type.
Then raw weight tensors in order.

Usage:
  python export_weights.py <checkpoint_dir> <output.bin> [--q8|--q4|--q4h]
  python export_weights.py ../models/d12_arianna/ weights/d12_arianna_q8.bin --q8
  python export_weights.py ../models/d20_arianna/ weights/d20_arianna_q4h.bin --q4h
"""
import sys
import os
import struct
import json
import torch
import numpy as np

def export(checkpoint_dir, output_path, quant='fp16'):
    # Find model and meta files
    model_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('model_')]
    meta_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('meta_')]
    assert model_files, f"No model file in {checkpoint_dir}"
    assert meta_files, f"No meta file in {checkpoint_dir}"

    model_path = os.path.join(checkpoint_dir, sorted(model_files)[-1])
    meta_path = os.path.join(checkpoint_dir, sorted(meta_files)[-1])

    print(f"Loading model: {model_path}")
    print(f"Loading meta:  {meta_path}")
    print(f"Quantization:  {quant}")

    with open(meta_path) as f:
        meta = json.load(f)

    # Extract config from meta
    config = meta.get('model_config', meta.get('config', meta))
    n_layer = config['n_layer']
    n_embd = config['n_embd']
    n_head = config['n_head']
    n_kv_head = config.get('n_kv_head', n_head)
    vocab_size = config['vocab_size']
    head_dim = n_embd // n_head
    seq_len = config.get('sequence_len', 2048)
    window_pattern = config.get('window_pattern', 'SSSL')
    padded_vocab = ((vocab_size + 63) // 64) * 64
    # bigram_vocab determined after loading state dict

    print(f"\nConfig:")
    print(f"  n_layer={n_layer}, n_embd={n_embd}, n_head={n_head}, n_kv_head={n_kv_head}")
    print(f"  vocab_size={vocab_size}, padded_vocab={padded_vocab}")
    print(f"  head_dim={head_dim}, seq_len={seq_len}")
    print(f"  window_pattern={window_pattern}")

    # Determine which layers have value embeddings
    ve_parity = (n_layer - 1) % 2
    ve_layers = [i for i in range(n_layer) if i % 2 == ve_parity]
    print(f"  ve_layers={ve_layers}")

    # Load state dict
    state = torch.load(model_path, map_location='cpu', weights_only=False)
    if 'model' in state:
        state = state['model']

    # Determine bigram_vocab from state dict
    if 'bigram_embed.embed.weight' in state:
        bigram_vocab = state['bigram_embed.embed.weight'].shape[0]
        print(f"  bigram_vocab={bigram_vocab} (from state dict)")
    else:
        bigram_vocab = 0
        print(f"  bigram_vocab=0 (no bigram in model)")

    # Debug: print all keys
    print(f"\nState dict keys ({len(state)}):")
    for k in sorted(state.keys()):
        print(f"  {k}: {state[k].shape} {state[k].dtype}")

    # Write binary
    MAGIC = 0x4E414E4F  # "NANO"
    quant_type = {'fp16': 0, 'q8': 1, 'q4': 2, 'q4h': 3}[quant]
    VERSION = {'fp16': 1, 'q8': 2, 'q4': 3, 'q4h': 4}[quant]

    with open(output_path, 'wb') as f:
        # Header: magic(4) + version(4) + config fields
        f.write(struct.pack('I', MAGIC))
        f.write(struct.pack('I', VERSION))
        f.write(struct.pack('i', n_layer))      # offset 8
        f.write(struct.pack('i', n_embd))        # offset 12
        f.write(struct.pack('i', n_head))         # offset 16
        f.write(struct.pack('i', n_kv_head))      # offset 20
        f.write(struct.pack('i', head_dim))        # offset 24
        f.write(struct.pack('i', vocab_size))      # offset 28
        f.write(struct.pack('i', padded_vocab))    # offset 32
        f.write(struct.pack('i', seq_len))         # offset 36
        f.write(struct.pack('i', bigram_vocab))    # offset 40
        f.write(struct.pack('i', len(ve_layers)))  # offset 44
        # Window pattern
        wp_bytes = bytes([1 if c == 'L' else 0 for c in window_pattern])
        f.write(struct.pack('i', len(wp_bytes)))   # offset 48
        f.write(wp_bytes)                           # offset 52+
        # Pad to offset 64, then write quant_type
        pos = f.tell()
        f.write(b'\x00' * (64 - pos))
        f.write(struct.pack('i', quant_type))       # offset 64
        # Pad rest of header to 256
        pos = f.tell()
        f.write(b'\x00' * (256 - pos))

        def write_tensor_fp16(name, expected_shape=None):
            """Write tensor as float16."""
            t = state[name].float()
            if expected_shape:
                assert list(t.shape) == list(expected_shape), \
                    f"{name}: expected {expected_shape}, got {list(t.shape)}"
            t_fp16 = t.half().numpy()
            f.write(t_fp16.tobytes())
            mb = t_fp16.nbytes / 1024 / 1024
            print(f"  {name}: {list(t.shape)} fp16 -> {mb:.1f} MB")
            return t_fp16.nbytes

        def write_tensor_q8(name, expected_shape=None):
            """Write 2D tensor as INT8 per-row quantized: scales(f32) + data(i8)."""
            t = state[name].float()
            if expected_shape:
                assert list(t.shape) == list(expected_shape), \
                    f"{name}: expected {expected_shape}, got {list(t.shape)}"
            if t.dim() != 2:
                # 1D tensor: fall back to fp16
                return write_tensor_fp16(name, expected_shape)
            rows, cols = t.shape
            # Per-row symmetric quantization
            absmax = t.abs().max(dim=1).values  # [rows]
            scales = absmax / 127.0
            scales = scales.clamp(min=1e-10)
            quantized = (t / scales.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8)
            # Verify roundtrip error
            reconstructed = quantized.float() * scales.unsqueeze(1)
            max_err = (t - reconstructed).abs().max().item()
            mean_err = (t - reconstructed).abs().mean().item()
            # Write: scales (float32) then data (int8)
            scales_np = scales.numpy().astype(np.float32)
            data_np = quantized.numpy()
            f.write(scales_np.tobytes())
            f.write(data_np.tobytes())
            nbytes = scales_np.nbytes + data_np.nbytes
            mb = nbytes / 1024 / 1024
            print(f"  {name}: {list(t.shape)} q8 -> {mb:.1f} MB  (max_err={max_err:.6f} mean_err={mean_err:.6f})")
            return nbytes

        def write_tensor_q4(name, expected_shape=None):
            """Write 2D tensor as INT4 per-row quantized: scales(f32) + packed data (2 values per byte)."""
            t = state[name].float()
            if expected_shape:
                assert list(t.shape) == list(expected_shape), \
                    f"{name}: expected {expected_shape}, got {list(t.shape)}"
            if t.dim() != 2:
                # 1D tensor: fall back to fp16
                return write_tensor_fp16(name, expected_shape)
            rows, cols = t.shape
            # Ensure cols is even for packing
            assert cols % 2 == 0, f"{name}: cols must be even for q4, got {cols}"
            # Per-row symmetric quantization to 4-bit (-8..7)
            absmax = t.abs().max(dim=1).values  # [rows]
            scales = absmax / 7.0
            scales = scales.clamp(min=1e-10)
            quantized = (t / scales.unsqueeze(1)).round().clamp(-8, 7).to(torch.int8)
            # Verify roundtrip error
            reconstructed = quantized.float() * scales.unsqueeze(1)
            max_err = (t - reconstructed).abs().max().item()
            mean_err = (t - reconstructed).abs().mean().item()
            # Pack 2 int4 values into 1 byte: low nibble + high nibble
            # Convert from signed (-8..7) to unsigned (0..15) for packing
            unsigned = (quantized + 8).to(torch.uint8)  # 0..15
            packed = unsigned[:, 0::2] | (unsigned[:, 1::2] << 4)  # [rows, cols/2]
            # Write: scales (float32) then packed data (uint8)
            scales_np = scales.numpy().astype(np.float32)
            data_np = packed.numpy()
            f.write(scales_np.tobytes())
            f.write(data_np.tobytes())
            nbytes = scales_np.nbytes + data_np.nbytes
            mb = nbytes / 1024 / 1024
            print(f"  {name}: {list(t.shape)} q4 -> {mb:.1f} MB  (max_err={max_err:.6f} mean_err={mean_err:.6f})")
            return nbytes

        # Select write function
        if quant == 'q4h':
            # Hybrid: embeddings use q8, attention/MLP use q4
            write_embed = write_tensor_q8
            write_attn_mlp = write_tensor_q4
            write_matrix = write_tensor_q8  # default for value_embeds
        else:
            write_embed = {'fp16': write_tensor_fp16, 'q8': write_tensor_q8, 'q4': write_tensor_q4}[quant]
            write_attn_mlp = write_embed
            write_matrix = write_embed
        write_small = write_tensor_fp16  # always fp16 for small tensors

        total = 0
        label = quant
        print(f"\nWriting weights ({label}):")

        # 1. Token embedding (use q8 for hybrid)
        total += write_embed('transformer.wte.weight', [padded_vocab, n_embd])

        # 2. Bigram embedding (use q8 for hybrid) - optional in newer nanochat
        if 'bigram_embed.embed.weight' in state:
            total += write_embed('bigram_embed.embed.weight', [bigram_vocab, n_embd])
        else:
            print("  [SKIP] bigram_embed.embed.weight not found (newer nanochat)")

        # 3. Scalar lambdas (always fp16)
        total += write_small('resid_lambdas', [n_layer])
        total += write_small('x0_lambdas', [n_layer])
        if 'bigram_lambdas' in state:
            total += write_small('bigram_lambdas', [n_layer])
        else:
            print("  [SKIP] bigram_lambdas not found")

        # 4. Per-layer weights (use q4 for attention/MLP in hybrid mode)
        for i in range(n_layer):
            prefix = f'transformer.h.{i}'
            # Attention
            total += write_attn_mlp(f'{prefix}.attn.c_q.weight', [n_head * head_dim, n_embd])
            total += write_attn_mlp(f'{prefix}.attn.c_k.weight', [n_kv_head * head_dim, n_embd])
            total += write_attn_mlp(f'{prefix}.attn.c_v.weight', [n_kv_head * head_dim, n_embd])
            total += write_attn_mlp(f'{prefix}.attn.c_proj.weight', [n_embd, n_embd])
            # VE gate (always fp16, tiny)
            if i in ve_layers:
                ve_gate_key = f'{prefix}.attn.ve_gate.weight'
                if ve_gate_key in state:
                    total += write_small(ve_gate_key)
            # MLP
            total += write_attn_mlp(f'{prefix}.mlp.c_fc.weight', [4 * n_embd, n_embd])
            total += write_attn_mlp(f'{prefix}.mlp.c_proj.weight', [n_embd, 4 * n_embd])

        # 5. Value embeddings (only for ve_layers, use q8 for hybrid)
        for i in ve_layers:
            key = f'value_embeds.{i}.weight'
            if key in state:
                total += write_embed(key)

        # 6. LM head (use q8 for hybrid)
        total += write_embed('lm_head.weight', [padded_vocab, n_embd])

        total_mb = total / 1024 / 1024
        print(f"\nTotal: {total_mb:.1f} MB ({total} bytes)")
        print(f"Saved to: {output_path}")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python export_weights.py <checkpoint_dir> <output.bin> [--q8|--q4|--q4h]")
        sys.exit(1)
    if '--q4h' in sys.argv:
        quant = 'q4h'
    elif '--q4' in sys.argv:
        quant = 'q4'
    elif '--q8' in sys.argv:
        quant = 'q8'
    else:
        quant = 'fp16'
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    export(args[0], args[1], quant)
