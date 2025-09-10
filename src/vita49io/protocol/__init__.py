"""Protocol primitives for VITA 49.

This subpackage contains the core data structures and helpers
that implement the VITA 49 (VRT) protocol.
"""

# Re-export common types for convenience when importing from vita49io.protocol
from .data_packet import DataPacket  # noqa: F401
from .context_packet import ContextPacket  # noqa: F401
from .enums import PacketType, TSI, TSF  # noqa: F401
from .vrt_types import ClassID  # noqa: F401
from .cif0 import CIF0Fields, PayloadFormat, PackingMethod, SampleType, DataItemFormat  # noqa: F401

__all__ = [
    "DataPacket",
    "ContextPacket",
    "PacketType",
    "TSI",
    "TSF",
    "ClassID",
    "CIF0Fields",
    "PayloadFormat",
    "PackingMethod",
    "SampleType",
    "DataItemFormat",
]
