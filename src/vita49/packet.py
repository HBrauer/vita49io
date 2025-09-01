"""
Backward-compatibility shim for the previous single-file implementation.

This module re-exports the public API moved to split modules.
"""

from .enums import PacketType, TSI, TSF
from .types import ClassID
from .cif0 import CIF0Fields
from .data_packet import DataPacket
from .context_packet import ContextPacket

__all__ = [
    "PacketType",
    "TSI",
    "TSF",
    "ClassID",
    "CIF0Fields",
    "DataPacket",
    "ContextPacket",
]

