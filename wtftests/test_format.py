#!/usr/bin/env python3
"""Test which prompt format the model was trained on."""
import ctypes
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB_EXT = 'dylib' if sys.platform == 'darwin' else 'so'
LIB_PATH = os.path.join(ROOT, f'libwtf.{LIB_EXT}')
WEIGHTS = os.path.join(ROOT, 'wtfweights', 'wtf360_v2_q4_0.gguf')

lib = ctypes.CDLL(LIB_PATH)
lib.wtf_init.argtypes = [ctypes.c_char_p]
lib.wtf_init.restype = ctypes.c_int
lib.wtf_generate.argtypes = [
    ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int,
    ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_char_p,
]
lib.wtf_generate.restype = ctypes.c_int
lib.wtf_free.argtypes = []

ret = lib.wtf_init(WEIGHTS.encode())
if ret != 0:
    print("FAILED to load")
    sys.exit(1)

def generate(prompt_text, label):
    buf = ctypes.create_string_buffer(8192)
    count = lib.wtf_generate(
        prompt_text.encode('utf-8'), buf, 8192, 100,
        ctypes.c_float(0.9), ctypes.c_float(1.0), None,
    )
    resp = buf.value.decode('utf-8', errors='replace').strip()
    print(f"\n{'='*60}")
    print(f"  FORMAT: {label}")
    print(f"  Tokens: {count}")
    print(f"{'='*60}")
    print(resp[:500])
    print()

# Format 1: ### Question/Answer (nanochat / base training format)
generate(
    "### Question: who are you?\n### Answer:",
    "### Question/Answer (nanochat)"
)

# Format 2: ChatML (Qwen Instruct format)
generate(
    "<|im_start|>user\nwho are you?<|im_end|>\n<|im_start|>assistant\n",
    "ChatML (Qwen Instruct)"
)

# Format 3: ChatML with system prompt
generate(
    "<|im_start|>system\nyou are wtforacle, a cynical reddit commenter.<|im_end|>\n"
    "<|im_start|>user\nwho are you?<|im_end|>\n<|im_start|>assistant\n",
    "ChatML + system prompt"
)

# Format 4: Raw text (no formatting)
generate(
    "Question: who are you?\nAnswer:",
    "Raw Q/A (no special tokens)"
)

lib.wtf_free()
