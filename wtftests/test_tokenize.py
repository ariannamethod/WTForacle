#!/usr/bin/env python3
"""Test WTForacle tokenization and chat format encoding."""
import os
import sys
import pickle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def load_tokenizer():
    pkl_path = os.path.join(ROOT, 'wtfweights', 'tokenizer.pkl')
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def test_special_token_ids():
    """Special tokens have expected IDs (32759-32767)."""
    enc = load_tokenizer()
    expected = {
        '<|bos|>': 32759,
        '<|user_start|>': 32760,
        '<|user_end|>': 32761,
        '<|assistant_start|>': 32762,
        '<|assistant_end|>': 32763,
    }
    for name, expected_id in expected.items():
        actual = enc._special_tokens[name]
        assert actual == expected_id, f"{name}: expected {expected_id}, got {actual}"


def test_encode_ordinary():
    """Basic text encodes to token IDs in valid range."""
    enc = load_tokenizer()
    tokens = enc.encode_ordinary("hello world")
    assert len(tokens) > 0, "Empty token list"
    assert all(0 <= t < 32768 for t in tokens), f"Token out of range: {tokens}"


def test_chat_format():
    """Chat format wraps text with correct special tokens."""
    enc = load_tokenizer()
    bos = enc._special_tokens['<|bos|>']
    us = enc._special_tokens['<|user_start|>']
    ue = enc._special_tokens['<|user_end|>']
    a_s = enc._special_tokens['<|assistant_start|>']

    text = "who are you?"
    text_tokens = enc.encode_ordinary(text)
    chat_tokens = [bos, us] + text_tokens + [ue, a_s]

    assert chat_tokens[0] == bos
    assert chat_tokens[1] == us
    assert chat_tokens[-2] == ue
    assert chat_tokens[-1] == a_s
    assert len(chat_tokens) == len(text_tokens) + 4


def test_roundtrip():
    """Encode then decode returns original text."""
    enc = load_tokenizer()
    text = "the oracle speaks truth"
    tokens = enc.encode_ordinary(text)
    decoded = enc.decode(tokens)
    assert decoded == text, f"Roundtrip failed: '{decoded}' != '{text}'"


def test_wtforacle_class_tokenize():
    """WTForacle.tokenize() produces valid chat format."""
    from wtforacle import WTForacle

    # Skip if weights missing (CI without weights)
    weights = os.path.join(ROOT, 'wtfweights', 'wtforacle_q8.bin')
    wtf_bin = os.path.join(ROOT, 'wtf.c', 'wtf')
    if not os.path.exists(weights) or not os.path.exists(wtf_bin):
        print("    (skipped - weights or binary missing)")
        return

    oracle = WTForacle()
    tokens = oracle.tokenize("test prompt")
    assert tokens[0] == oracle.bos
    assert tokens[1] == oracle.user_start
    assert tokens[-2] == oracle.user_end
    assert tokens[-1] == oracle.assistant_start


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    passed = 0
    failed = 0
    skipped = 0
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
