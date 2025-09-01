from __future__ import annotations

from typing import Tuple

# (OUI 24-bit, Information Class 16-bit, Packet Class 16-bit)
ClassID = Tuple[int, int, int]

__all__ = ["ClassID"]

