"""Provide shared type aliases for VITA 49 identifiers.

Examples:
    >>> from vita49io.protocol.vrt_types import ClassID
    >>> ClassID.__args__
    (<class 'int'>, <class 'int'>, <class 'int'>)
"""

from __future__ import annotations

from typing import Tuple

# (OUI 24-bit, Information Class 16-bit, Packet Class 16-bit)
ClassID = Tuple[int, int, int]

__all__ = ["ClassID"]

