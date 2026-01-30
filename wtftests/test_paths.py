#!/usr/bin/env python3
"""Test that all WTForacle paths and files are connected correctly."""
import os
import sys
import struct

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def test_project_structure():
    """All required files exist in the right places."""
    required = [
        'wtforacle.py',
        'wtf.c/src/wtf.c',
        'wtf.c/Makefile',
        'wtf.c/export_weights.py',
        'wtf.c/export_tokenizer.py',
        'wtfweights/tokenizer.pkl',
        'wtfweights/wtforacle.tok',
        # wtforacle_identity.jsonl is training data, not required for inference
    ]
    missing = []
    for f in required:
        path = os.path.join(ROOT, f)
        if not os.path.exists(path):
            missing.append(f)
    assert not missing, f"Missing files: {missing}"


def test_makefile_targets():
    """Makefile references wtf (not nano)."""
    makefile = os.path.join(ROOT, 'wtf.c', 'Makefile')
    with open(makefile) as f:
        content = f.read()
    assert 'BIN = wtf' in content, "Makefile BIN should be 'wtf'"
    assert 'SRC = src/wtf.c' in content, "Makefile SRC should be 'src/wtf.c'"
    assert 'nano' not in content.lower(), "Makefile still references 'nano'"


def test_wtforacle_py_paths():
    """wtforacle.py references wtf binary and wtfweights paths."""
    py_path = os.path.join(ROOT, 'wtforacle.py')
    with open(py_path) as f:
        content = f.read()
    assert 'wtfweights' in content, "wtforacle.py should reference wtfweights/"
    assert 'wtf_bin' in content, "wtforacle.py should use wtf_bin (not nano_bin)"
    assert 'nano_bin' not in content, "wtforacle.py still references nano_bin"
    assert "'wtf.c'" in content or '"wtf.c"' in content, "wtforacle.py should reference wtf.c/"


def test_gitignore():
    """gitignore references wtf (not nano)."""
    gi_path = os.path.join(ROOT, '.gitignore')
    with open(gi_path) as f:
        content = f.read()
    assert 'wtf.c/wtf' in content, ".gitignore should ignore wtf.c/wtf"
    assert 'wtf_debug' in content, ".gitignore should ignore wtf_debug"
    assert 'nano' not in content, ".gitignore still references 'nano'"


def test_tokenizer_tok_format():
    """wtforacle.tok has correct NTOK magic header."""
    tok_path = os.path.join(ROOT, 'wtfweights', 'wtforacle.tok')
    with open(tok_path, 'rb') as f:
        magic = struct.unpack('I', f.read(4))[0]
        vocab_size = struct.unpack('i', f.read(4))[0]
        max_len = struct.unpack('i', f.read(4))[0]
    assert magic == 0x4E544F4B, f"Bad magic: {hex(magic)}, expected NTOK"
    assert vocab_size == 32768, f"Bad vocab_size: {vocab_size}, expected 32768"
    assert 0 < max_len < 1000, f"Bad max_len: {max_len}"


def test_tokenizer_pkl_loads():
    """tokenizer.pkl loads and has required special tokens."""
    import pickle
    pkl_path = os.path.join(ROOT, 'wtfweights', 'tokenizer.pkl')
    with open(pkl_path, 'rb') as f:
        enc = pickle.load(f)
    required_specials = ['<|bos|>', '<|user_start|>', '<|user_end|>',
                         '<|assistant_start|>', '<|assistant_end|>']
    for name in required_specials:
        assert name in enc._special_tokens, f"Missing special token: {name}"
    assert enc.n_vocab == 32768, f"Bad vocab size: {enc.n_vocab}"


def test_identity_dataset():
    """Identity JSONL (if present) has valid format."""
    import json
    jsonl_path = os.path.join(ROOT, 'wtfweights', 'wtforacle_identity.jsonl')
    if not os.path.exists(jsonl_path):
        print("    (skipped - training data not present, on HuggingFace)")
        return
    count = 0
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                assert isinstance(obj, list), f"Line {count+1}: expected list, got {type(obj)}"
                assert len(obj) >= 2, f"Line {count+1}: conversation too short ({len(obj)} msgs)"
                assert obj[0].get('role') == 'user', f"Line {count+1}: first msg should be user"
                count += 1
    assert count == 7767, f"Expected 7767 identity conversations, got {count}"


def test_wtf_binary_compiled():
    """wtf binary exists and is executable."""
    bin_path = os.path.join(ROOT, 'wtf.c', 'wtf')
    assert os.path.exists(bin_path), f"wtf binary not found at {bin_path}. Run 'cd wtf.c && make'"
    assert os.access(bin_path, os.X_OK), "wtf binary is not executable"


def test_no_nano_references():
    """No stale 'nano' references remain in key files."""
    files_to_check = [
        'wtforacle.py',
        '.gitignore',
        'wtf.c/Makefile',
    ]
    for fname in files_to_check:
        path = os.path.join(ROOT, fname)
        with open(path) as f:
            content = f.read()
        # Allow 'nano' only in comments or as part of 'nanochat' (architecture credit)
        lines = content.split('\n')
        for i, line in enumerate(lines):
            clean = line.split('#')[0]  # strip comments
            if 'nano' in clean.lower() and 'nanochat' not in clean.lower():
                assert False, f"{fname}:{i+1} still references 'nano': {line.strip()}"


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
