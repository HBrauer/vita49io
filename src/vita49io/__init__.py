"""Expose the primary vita49io entry points for convenient imports.

Examples:
    >>> from vita49io import DataPacket
    >>> isinstance(DataPacket, type)
    True
"""

from .protocol.data_packet import DataPacket
from .protocol.context_packet import ContextPacket
from .protocol.enums import PacketType, TSI, TSF
from .protocol.vrt_types import ClassID
from .protocol.cif0 import (
    CIF0Fields,
    ContextAssociationLists,
    Ephemeris,
    FormattedGeolocation,
    GPSASCIIField,
)
from .protocol.cif1 import (
    AveragingType,
    ArrayOfCifFields,
    BufferSizeField,
    BuildInformation,
    CIF1Fields,
    SectorStepRecord,
    SectorStepScanField,
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
    "FormattedGeolocation",
    "Ephemeris",
    "GPSASCIIField",
    "ContextAssociationLists",
    "CIF1Fields",
    "SectorStepRecord",
    "SectorStepScanField",
    "ArrayOfCifFields",
    "BufferSizeField",
    "BuildInformation",
    "SpectrumField",
    "SpectrumType",
    "AveragingType",
    "WindowTimeDeltaInterpretation",
]

