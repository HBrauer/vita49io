"""
Test/development helper to ensure the src/ layout is importable.

Python automatically imports this module (if present on sys.path) after
standard site initialization. By adding the repository's `src` directory to
`sys.path`, `import vita49` works without installing the package.

This has no effect in production installs (where the package is installed),
and is safe to keep in the repo.
"""
import os
import sys

try:
    root = os.path.dirname(__file__)
    src = os.path.join(root, "src")
    if os.path.isdir(src) and src not in sys.path:
        sys.path.insert(0, src)
except Exception:
    # Non-fatal: tests may still configure PYTHONPATH or use installed package
    pass

