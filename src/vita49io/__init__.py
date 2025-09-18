"""Expose the primary vita49io entry points for convenient imports.

Args:
    None.

Returns:
    None.

Raises:
    None.

Side Effects:
    Imports common protocol classes during module import to populate the public API.

Examples:
    >>> from vita49io import DataPacket
    >>> isinstance(DataPacket, type)
    True
"""

from .protocol.data_packet import DataPacket
from .protocol.context_packet import ContextPacket
from .protocol.enums import PacketType, TSI, TSF
from .protocol.vrt_types import ClassID
from .protocol.cif0 import CIF0Fields

__all__ = [
    "DataPacket",
    "ContextPacket",
    "PacketType",
    "TSI",
    "TSF",
    "ClassID",
    "CIF0Fields",
]

