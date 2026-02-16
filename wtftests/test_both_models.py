#!/usr/bin/env python3
"""Test SmolLM2 360M GGUF model with different prompt formats."""
import ctypes
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB_EXT = 'dylib' if sys.platform == 'darwin' else 'so'
LIB_PATH = os.path.join(ROOT, f'libwtf.{LIB_EXT}')

SMOL_WEIGHTS = os.path.join(ROOT, 'wtfweights', 'wtf360_v2_q4_0.gguf')

lib = ctypes.CDLL(LIB_PATH)
lib.wtf_init.argtypes = [ctypes.c_char_p]
lib.wtf_init.restype = ctypes.c_int
lib.wtf_generate.argtypes = [
    ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int,
    ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_char_p,
]
lib.wtf_generate.restype = ctypes.c_int
lib.wtf_free.argtypes = []
lib.wtf_reset.argtypes = []

def test_model(weights_path, label, prompts):
    print(f"\n{'#'*60}")
    print(f"  MODEL: {label}")
    print(f"  File: {os.path.basename(weights_path)}")
    print(f"{'#'*60}")

    ret = lib.wtf_init(weights_path.encode())
    if ret != 0:
        print("  FAILED to load!")
        return

    for prompt_text, fmt_label in prompts:
        buf = ctypes.create_string_buffer(8192)
        count = lib.wtf_generate(
            prompt_text.encode('utf-8'), buf, 8192, 80,
            ctypes.c_float(0.9), ctypes.c_float(1.0), None,
        )
        resp = buf.value.decode('utf-8', errors='replace').strip()
        print(f"\n  --- {fmt_label} ({count} tokens) ---")
        print(f"  {resp[:300]}")

    lib.wtf_free()

# Test SmolLM2 360M
if os.path.exists(SMOL_WEIGHTS):
    test_model(SMOL_WEIGHTS, "SmolLM2 360M (WTForacle v3)", [
        ("who are you?", "raw text"),
        ("### Question: who are you?\n### Answer:", "### Q/A format"),
    ])
else:
    print(f"Weights not found: {SMOL_WEIGHTS}")
    print("Run: make wtf-weights")
