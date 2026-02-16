#!/usr/bin/env python3
"""Test WTForacle memory usage and allocation patterns."""
import os
import sys
import ctypes
import platform
import subprocess
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def find_lib():
    ext = 'dylib' if platform.system() == 'Darwin' else 'so'
    path = os.path.join(ROOT, f'libwtf.{ext}')
    return path if os.path.exists(path) else None

def find_weights():
    path = os.path.join(ROOT, 'wtfweights', 'wtf360_v2_q4_0.gguf')
    return path if os.path.exists(path) else None

SKIP_MSG = "(skipped - lib or weights not available)"


def get_process_memory_mb():
    """Get current process RSS in MB."""
    pid = os.getpid()
    if platform.system() == 'Darwin':
        try:
            out = subprocess.check_output(['ps', '-o', 'rss=', '-p', str(pid)], text=True)
            return int(out.strip()) / 1024  # KB -> MB
        except:
            return -1
    else:
        try:
            with open(f'/proc/{pid}/status') as f:
                for line in f:
                    if line.startswith('VmRSS:'):
                        return int(line.split()[1]) / 1024  # KB -> MB
        except:
            return -1
    return -1


def test_memory_after_load():
    """Memory after model load should be reasonable.

    SmolLM2 360M Q4_0:
      Weights: ~229 MB (GGUF on disk, loaded into RAM)
      KV cache: 2 * 32 * 2048 * (5*64) * 4 bytes = ~160 MB
      Buffers: ~5 MB
      Expected total: ~300-500 MB RSS
    """
    lib_path = find_lib()
    weights_path = find_weights()
    if not lib_path or not weights_path:
        print(f"    {SKIP_MSG}")
        return

    mem_before = get_process_memory_mb()

    lib = ctypes.CDLL(lib_path)
    lib.wtf_init.argtypes = [ctypes.c_char_p]
    lib.wtf_init.restype = ctypes.c_int
    lib.wtf_free.argtypes = []
    lib.wtf_free.restype = None

    ret = lib.wtf_init(weights_path.encode())
    assert ret == 0, "Failed to init"

    mem_after = get_process_memory_mb()
    delta = mem_after - mem_before if mem_before > 0 else mem_after

    print(f"    (RSS before={mem_before:.0f}MB, after={mem_after:.0f}MB, delta={delta:.0f}MB)")

    # Model + KV cache + buffers should be under 600MB
    assert delta < 600, f"Memory too high: {delta:.0f}MB (expected <600MB)"

    lib.wtf_free()


def test_memory_stable_after_generations():
    """Memory doesn't grow across multiple generations (no leaks)."""
    lib_path = find_lib()
    weights_path = find_weights()
    if not lib_path or not weights_path:
        print(f"    {SKIP_MSG}")
        return

    lib = ctypes.CDLL(lib_path)
    lib.wtf_init.argtypes = [ctypes.c_char_p]
    lib.wtf_init.restype = ctypes.c_int
    lib.wtf_generate.argtypes = [
        ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int,
        ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_char_p,
    ]
    lib.wtf_generate.restype = ctypes.c_int
    lib.wtf_free.argtypes = []

    ret = lib.wtf_init(weights_path.encode())
    assert ret == 0

    system = b"you are wtforacle."

    # Warmup
    buf = ctypes.create_string_buffer(4096)
    lib.wtf_generate(b"warmup", buf, 4096, 10, ctypes.c_float(1.0), ctypes.c_float(0.95), system)

    mem_start = get_process_memory_mb()

    # Run 5 generations
    for i in range(5):
        buf = ctypes.create_string_buffer(4096)
        lib.wtf_generate(f"question {i}".encode(), buf, 4096, 20,
                        ctypes.c_float(1.0), ctypes.c_float(0.95), system)

    mem_end = get_process_memory_mb()
    growth = mem_end - mem_start

    print(f"    (start={mem_start:.0f}MB, end={mem_end:.0f}MB, growth={growth:.0f}MB)")

    # Should not grow more than 20MB across 5 generations
    assert growth < 20, f"Memory grew {growth:.0f}MB across 5 generations (possible leak)"

    lib.wtf_free()


def test_kv_cache_size_estimate():
    """Verify KV cache size calculation matches expectations."""
    # SmolLM2 360M: 32 layers, 5 KV heads, 64 head_dim, 2048 seq_len
    layers = 32
    kv_heads = 5
    head_dim = 64
    seq_len = 2048
    kv_dim = kv_heads * head_dim  # 320

    # Key + Value caches
    kv_bytes = 2 * layers * seq_len * kv_dim * 4  # float32
    kv_mb = kv_bytes / (1024 * 1024)

    print(f"    (KV cache: {kv_mb:.1f}MB for seq_len={seq_len})")

    # Should be ~160MB for SmolLM2 360M
    assert 140 < kv_mb < 180, f"KV cache estimate off: {kv_mb:.1f}MB"


def test_weights_file_size():
    """GGUF weights file is expected size for Q4_0."""
    weights_path = find_weights()
    if not weights_path:
        print(f"    {SKIP_MSG}")
        return
    size_mb = os.path.getsize(weights_path) / (1024 * 1024)
    print(f"    (weights: {size_mb:.1f}MB)")
    # Q4_0 SmolLM2 360M should be ~220-240MB
    assert 200 < size_mb < 280, f"Unexpected weights size: {size_mb:.1f}MB"


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
    sys.exit(1 if failed else 0)
