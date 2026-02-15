#!/usr/bin/env python3
"""Test WTForacle library loading and basic functionality."""
import os
import sys
import ctypes
import platform

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def find_lib():
    """Find libwtf shared library."""
    ext = 'dylib' if platform.system() == 'Darwin' else 'so'
    path = os.path.join(ROOT, f'libwtf.{ext}')
    return path if os.path.exists(path) else None


def test_lib_exists():
    """libwtf shared library exists."""
    lib_path = find_lib()
    assert lib_path is not None, "libwtf not found. Run 'make wtf-lib' first."


def test_lib_loads():
    """libwtf loads via ctypes."""
    lib_path = find_lib()
    if not lib_path:
        print("    (skipped - libwtf not built)")
        return
    lib = ctypes.CDLL(lib_path)
    assert lib is not None


def test_lib_has_symbols():
    """libwtf exports required symbols."""
    lib_path = find_lib()
    if not lib_path:
        print("    (skipped - libwtf not built)")
        return
    lib = ctypes.CDLL(lib_path)
    required = ['wtf_init', 'wtf_free', 'wtf_generate', 'wtf_encode', 'wtf_decode_token']
    for sym in required:
        assert hasattr(lib, sym), f"Missing symbol: {sym}"


def test_wtforacle_class_import():
    """WTForacle class can be imported."""
    from wtforacle import WTForacle, SYSTEM_PROMPT, CONFIG
    assert WTForacle is not None
    assert 'cynical' in SYSTEM_PROMPT
    assert CONFIG['weights'].endswith('.gguf')


def test_weights_path():
    """Weights path points to GGUF file."""
    from wtforacle import CONFIG
    weights = CONFIG['weights']
    assert weights.endswith('.gguf'), f"Weights should be .gguf: {weights}"


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
