"""Provide I/O helpers used to create and serialize VITA 49 streams.

Args:
    None.

Returns:
    None.

Raises:
    None.

Side Effects:
    Imports streaming utilities into the subpackage namespace.

Examples:
    >>> from vita49io.io import IQStreamWriter
    >>> isinstance(IQStreamWriter, type)
    True
"""

from .iq_writer import IQStreamWriter
from .packet_reader import PacketReader, Readable
from .payload_codec import payload_as_numpy, payload_from_numpy, payload_as_numpy_view
from .spectrum_processor import SpectrumStreamProcessor

__all__ = [
    "IQStreamWriter",
    "PacketReader",
    "Readable",
    "payload_as_numpy",
    "payload_from_numpy",
    "payload_as_numpy_view",
    "SpectrumStreamProcessor",
]
