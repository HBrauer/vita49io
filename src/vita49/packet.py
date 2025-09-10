"""
Backward-compatibility shim for the previous single-file implementation.

This module re-exports the public API moved to split modules.
"""

from .protocol.enums import PacketType, TSI, TSF
from .protocol.vrt_types import ClassID
from .protocol.cif0 import CIF0Fields
from .protocol.data_packet import DataPacket
from .protocol.context_packet import ContextPacket

__all__ = [
    "PacketType",
    "TSI",
    "TSF",
    "ClassID",
    "CIF0Fields",
    "DataPacket",
    "ContextPacket",
]


