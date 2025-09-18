"""Provide backward-compatible packet re-exports for vita49io users.

Args:
    None.

Returns:
    None.

Raises:
    None.

Side Effects:
    Imports public protocol classes for legacy import paths.

Examples:
    >>> from vita49io.packet import DataPacket
    >>> isinstance(DataPacket, type)
    True
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

