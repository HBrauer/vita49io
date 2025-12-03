"""Expose protocol primitives for constructing and parsing VITA 49 packets.

Examples:
    >>> from vita49io.protocol import DataPacket
    >>> isinstance(DataPacket, type)
    True
"""

# Re-export common types for convenience when importing from vita49io.protocol
from .data_packet import DataPacket  # noqa: F401
from .context_packet import ContextPacket  # noqa: F401
from .enums import PacketType, TSI, TSF  # noqa: F401
from .vrt_types import ClassID  # noqa: F401
from .cif0 import CIF0Fields, PayloadFormat, PackingMethod, SampleType, DataItemFormat  # noqa: F401
from .cif1 import (  # noqa: F401
    AveragingType,
    CIF1Fields,
    SpectrumField,
    SpectrumType,
    WindowTimeDeltaInterpretation,
)

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
    "CIF1Fields",
    "SpectrumField",
    "SpectrumType",
    "AveragingType",
    "WindowTimeDeltaInterpretation",
]

