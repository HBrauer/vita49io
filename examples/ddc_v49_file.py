from __future__ import annotations

import sys
from pathlib import Path


# Allow running the example from the repo root without installation.
_EXAMPLE_DIR = Path(__file__).resolve().parent
_SRC_DIR = _EXAMPLE_DIR.parent / "src"
if _SRC_DIR.is_dir():
    _src_str = str(_SRC_DIR)
    if _src_str not in sys.path:
        sys.path.insert(0, _src_str)

from vita49io.signal.ddc_file import main


if __name__ == "__main__":
    raise SystemExit(main())
