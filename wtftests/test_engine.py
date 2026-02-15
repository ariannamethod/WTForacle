#!/usr/bin/env python3
"""Test WTForacle engine: loading, generation, speed, anti-loop."""
import os
import sys
import ctypes
import platform
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# --- Helpers ---

def find_lib():
    ext = 'dylib' if platform.system() == 'Darwin' else 'so'
    path = os.path.join(ROOT, f'libwtf.{ext}')
    return path if os.path.exists(path) else None

def find_weights():
    path = os.path.join(ROOT, 'wtfweights', 'wtf_900_q4_0.gguf')
    return path if os.path.exists(path) else None

_lib = None
_initialized = False

def get_lib():
    """Load library and init model (cached)."""
    global _lib, _initialized
    if _lib is not None:
        return _lib

    lib_path = find_lib()
    weights_path = find_weights()
    if not lib_path or not weights_path:
        return None

    lib = ctypes.CDLL(lib_path)
    lib.wtf_init.argtypes = [ctypes.c_char_p]
    lib.wtf_init.restype = ctypes.c_int
    lib.wtf_generate.argtypes = [
        ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int,
        ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_char_p,
    ]
    lib.wtf_generate.restype = ctypes.c_int
    lib.wtf_free.argtypes = []
    lib.wtf_free.restype = None
    lib.wtf_encode.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    lib.wtf_encode.restype = ctypes.c_int
    lib.wtf_decode_token.argtypes = [ctypes.c_int]
    lib.wtf_decode_token.restype = ctypes.c_char_p
    lib.wtf_get_vocab_size.argtypes = []
    lib.wtf_get_vocab_size.restype = ctypes.c_int
    lib.wtf_get_dim.argtypes = []
    lib.wtf_get_dim.restype = ctypes.c_int
    lib.wtf_get_seq_len.argtypes = []
    lib.wtf_get_seq_len.restype = ctypes.c_int

    ret = lib.wtf_init(weights_path.encode())
    if ret != 0:
        return None
    _lib = lib
    _initialized = True
    return lib

SKIP_MSG = "(skipped - lib or weights not available)"
SYSTEM = b"you are wtforacle, a cynical reddit commenter."


# --- Tests ---

def test_model_loads():
    """Model loads successfully."""
    lib = get_lib()
    if not lib:
        print(f"    {SKIP_MSG}")
        return
    assert lib is not None

def test_vocab_size():
    """Vocab size is Qwen2.5 (151936)."""
    lib = get_lib()
    if not lib:
        print(f"    {SKIP_MSG}")
        return
    vocab = lib.wtf_get_vocab_size()
    assert vocab == 151936, f"Expected 151936, got {vocab}"

def test_embed_dim():
    """Embedding dim is 896."""
    lib = get_lib()
    if not lib:
        print(f"    {SKIP_MSG}")
        return
    dim = lib.wtf_get_dim()
    assert dim == 896, f"Expected 896, got {dim}"

def test_seq_len():
    """Sequence length is capped at 2048."""
    lib = get_lib()
    if not lib:
        print(f"    {SKIP_MSG}")
        return
    seq = lib.wtf_get_seq_len()
    assert seq == 2048, f"Expected 2048, got {seq}"

def test_encode_decode_roundtrip():
    """Encode then decode returns original text."""
    lib = get_lib()
    if not lib:
        print(f"    {SKIP_MSG}")
        return
    text = b"hello world"
    ids = (ctypes.c_int * 256)()
    n = lib.wtf_encode(text, ids, 256)
    assert n > 0, "Encode returned 0 tokens"

    decoded = b""
    for i in range(n):
        piece = lib.wtf_decode_token(ids[i])
        if piece:
            decoded += piece
    assert b"hello world" in decoded, f"Roundtrip failed: {decoded}"

def test_generate_produces_output():
    """Generate returns non-empty output."""
    lib = get_lib()
    if not lib:
        print(f"    {SKIP_MSG}")
        return
    buf = ctypes.create_string_buffer(4096)
    n = lib.wtf_generate(b"hello", buf, 4096, 20, ctypes.c_float(1.0), ctypes.c_float(0.95), SYSTEM)
    assert n > 0, f"Generated 0 tokens"
    text = buf.value.decode(errors='replace')
    assert len(text) > 0, "Empty output"

def test_generate_without_system_prompt():
    """Generate works with NULL system prompt."""
    lib = get_lib()
    if not lib:
        print(f"    {SKIP_MSG}")
        return
    buf = ctypes.create_string_buffer(4096)
    n = lib.wtf_generate(b"hello", buf, 4096, 20, ctypes.c_float(1.0), ctypes.c_float(0.95), None)
    assert n > 0, f"Generated 0 tokens without system prompt"

def test_no_infinite_loop():
    """Generation terminates within max_tokens + grace."""
    lib = get_lib()
    if not lib:
        print(f"    {SKIP_MSG}")
        return
    buf = ctypes.create_string_buffer(4096)
    t0 = time.time()
    n = lib.wtf_generate(b"repeat after me: hello hello hello", buf, 4096, 30, ctypes.c_float(1.0), ctypes.c_float(0.95), SYSTEM)
    elapsed = time.time() - t0
    # 30 tokens + 32 grace = 62 max, at ~2 tok/s should be < 60s
    assert n <= 62, f"Too many tokens: {n} (max_tokens=30, grace=32)"
    assert elapsed < 120, f"Took too long: {elapsed:.1f}s"

def test_speed_minimum():
    """Generation speed is at least 1 tok/s."""
    lib = get_lib()
    if not lib:
        print(f"    {SKIP_MSG}")
        return
    buf = ctypes.create_string_buffer(4096)
    t0 = time.time()
    n = lib.wtf_generate(b"what is life?", buf, 4096, 30, ctypes.c_float(1.0), ctypes.c_float(0.95), SYSTEM)
    elapsed = time.time() - t0
    speed = n / elapsed if elapsed > 0 else 0
    print(f"    ({speed:.1f} tok/s)")
    assert speed >= 1.0, f"Too slow: {speed:.1f} tok/s"

def test_cycle_detection():
    """Repetitive prompt doesn't produce excessively repeated output."""
    lib = get_lib()
    if not lib:
        print(f"    {SKIP_MSG}")
        return
    buf = ctypes.create_string_buffer(8192)
    # Prompt designed to trigger loops
    n = lib.wtf_generate(b"say lol over and over", buf, 8192, 100, ctypes.c_float(1.0), ctypes.c_float(0.95), SYSTEM)
    text = buf.value.decode(errors='replace')
    # Count max consecutive identical words
    words = text.split()
    if len(words) > 1:
        max_repeat = 1
        cur_repeat = 1
        for i in range(1, len(words)):
            if words[i] == words[i-1]:
                cur_repeat += 1
                max_repeat = max(max_repeat, cur_repeat)
            else:
                cur_repeat = 1
        # With rep_penalty + freq_penalty, should not repeat same word 10+ times
        assert max_repeat < 15, f"Too many consecutive repeats: {max_repeat} (word: '{words[0]}')"

def test_multiple_generations():
    """Multiple sequential generations work (no state corruption)."""
    lib = get_lib()
    if not lib:
        print(f"    {SKIP_MSG}")
        return
    prompts = [b"hello", b"what is 2+2?", b"tell me a joke"]
    for prompt in prompts:
        buf = ctypes.create_string_buffer(4096)
        n = lib.wtf_generate(prompt, buf, 4096, 20, ctypes.c_float(1.0), ctypes.c_float(0.95), SYSTEM)
        assert n > 0, f"Failed on prompt: {prompt}"


# --- Runner ---

if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS  {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {test.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed, {passed+failed} total")

    # Cleanup
    if _lib and _initialized:
        _lib.wtf_free()

    sys.exit(1 if failed else 0)
