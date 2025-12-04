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
    BufferSizeField,
    BuildInformation,
    CIF1Fields,
    SectorStepRecord,
    SectorStepScanField,
    SpectrumField,
    SpectrumType,
    WindowTimeDeltaInterpretation,
)
from .protocol.cif2 import CIF2Fields
from .protocol.cif3 import CIF3Fields

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
    "BufferSizeField",
    "BuildInformation",
    "SpectrumField",
    "SpectrumType",
    "AveragingType",
    "WindowTimeDeltaInterpretation",
    "CIF2Fields",
    "CIF3Fields",
]

