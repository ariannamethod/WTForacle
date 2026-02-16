#!/usr/bin/env python3
"""Test prompt formats for SmolLM2 360M (SentencePiece tokenizer)."""
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

# Format 1: ### Question/Answer (training format)
generate(
    "### Question: who are you?\n### Answer:",
    "### Question/Answer (training format)"
)

# Format 2: Raw text (no formatting)
generate(
    "Question: who are you?\nAnswer:",
    "Raw Q/A (no special tokens)"
)

# Format 3: With system prompt (as used in REPL)
generate(
    "you are wtforacle, a cynical reddit commenter. "
    "you love to argue for fun, and when you argue you write more.\n"
    "### Question: who are you?\n### Answer:",
    "### Q/A with system prompt"
)

lib.wtf_free()
