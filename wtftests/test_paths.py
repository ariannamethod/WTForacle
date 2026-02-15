#!/usr/bin/env python3
"""Test that all WTForacle paths and files are connected correctly."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def test_project_structure():
    """All required files exist in the right places."""
    required = [
        'wtforacle.py',
        'Makefile',
        'wtf/wtf.go',
        'wtf/go.mod',
        'wtf/gguf.go',
        'wtf/model.go',
        'wtf/tokenizer.go',
        'wtf/quant.go',
    ]
    missing = []
    for f in required:
        path = os.path.join(ROOT, f)
        if not os.path.exists(path):
            missing.append(f)
    assert not missing, f"Missing files: {missing}"


def test_makefile_targets():
    """Makefile has wtf-lib and wtf-weights targets."""
    makefile = os.path.join(ROOT, 'Makefile')
    with open(makefile) as f:
        content = f.read()
    assert 'wtf-lib' in content, "Makefile should have wtf-lib target"
    assert 'wtf-weights' in content, "Makefile should have wtf-weights target"
    assert 'go build' in content, "Makefile should use go build"


def test_wtforacle_py_uses_ctypes():
    """wtforacle.py uses ctypes (not subprocess/pickle)."""
    py_path = os.path.join(ROOT, 'wtforacle.py')
    with open(py_path) as f:
        content = f.read()
    assert 'ctypes' in content, "wtforacle.py should use ctypes"
    assert 'subprocess' not in content, "wtforacle.py should not use subprocess"
    assert 'pickle' not in content, "wtforacle.py should not use pickle"
    assert 'tiktoken' not in content, "wtforacle.py should not use tiktoken"
    assert 'libwtf' in content, "wtforacle.py should reference libwtf"


def test_gitignore():
    """gitignore covers build artifacts."""
    gi_path = os.path.join(ROOT, '.gitignore')
    with open(gi_path) as f:
        content = f.read()
    assert 'libwtf' in content, ".gitignore should ignore libwtf*"
    assert '*.gguf' in content, ".gitignore should ignore *.gguf"


def test_go_mod():
    """go.mod has correct module name."""
    mod_path = os.path.join(ROOT, 'wtf', 'go.mod')
    with open(mod_path) as f:
        content = f.read()
    assert 'module wtforacle' in content, "go.mod module should be wtforacle"


def test_wtf_go_exports():
    """wtf.go has required CGO exports."""
    go_path = os.path.join(ROOT, 'wtf', 'wtf.go')
    with open(go_path) as f:
        content = f.read()
    required_exports = ['wtf_init', 'wtf_free', 'wtf_generate', 'wtf_encode', 'wtf_decode_token']
    for exp in required_exports:
        assert f'//export {exp}' in content, f"wtf.go missing export: {exp}"
    # No Arianna ecosystem modulation
    assert 'gTempMod' not in content, "wtf.go should not have Arianna gTempMod"
    assert 'gLogitScale' not in content, "wtf.go should not have Arianna gLogitScale"
    assert 'gExploresBias' not in content, "wtf.go should not have Arianna gExploresBias"


def test_no_old_engine():
    """Old C engine directory should not exist."""
    old_dir = os.path.join(ROOT, 'wtf.c')
    assert not os.path.exists(old_dir), "wtf.c/ should be deleted (old C engine)"


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
