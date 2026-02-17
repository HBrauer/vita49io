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
from .frequency import StreamingFrequencyShifter
from .packet_reader import PacketReader, RawDataPacket, Readable
from .payload_codec import (
    PayloadDecoder,
    build_payload_decoder,
    payload_as_numpy,
    payload_from_numpy,
    payload_as_numpy_view,
)
from .spectrum_processor import SpectrumStreamProcessor
from .time_utils import (
    epoch_time_to_vita_timestamp,
    vita_timestamp_to_epoch_time,
    packet_vita_time_to_epoch_time,
)

__all__ = [
    "IQStreamWriter",
    "StreamingFrequencyShifter",
    "PacketReader",
    "RawDataPacket",
    "Readable",
    "PayloadDecoder",
    "build_payload_decoder",
    "payload_as_numpy",
    "payload_from_numpy",
    "payload_as_numpy_view",
    "SpectrumStreamProcessor",
    "epoch_time_to_vita_timestamp",
    "vita_timestamp_to_epoch_time",
    "packet_vita_time_to_epoch_time",
]
