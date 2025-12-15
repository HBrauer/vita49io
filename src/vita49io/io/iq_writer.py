"""Provide high-level helpers for generating VITA 49 IQ data streams. IQStreamWriter produces context and data packets for a configured IQ stream and advances timestamps based on the sample rate.

Examples:
    >>> from vita49io.io.iq_writer import IQStreamWriter
    >>> isinstance(IQStreamWriter, type)
    True
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Tuple, Union

import numpy as np

from ..protocol.core import Header
from ..protocol.enums import PacketType, TSI, TSF
from ..protocol.vrt_types import ClassID
from ..protocol.data_packet import DataPacket
from ..protocol.context_packet import ContextPacket
from ..protocol.cif0 import (
    CIF0Fields,
    PayloadFormat,
    PackingMethod,
    SampleType,
    DataItemFormat,
)


def _now_epoch_s() -> float:
    # Use UTC-aware datetime for portability/clarity
    return datetime.now(timezone.utc).timestamp()


def _to_vrt_time(t_epoch_s: float) -> Tuple[int, int]:
    """Convert POSIX epoch seconds to (integer_seconds, fractional_seconds_64bit).

    - integer_seconds: floor of seconds since epoch (UTC)
    - fractional_seconds: 64-bit fractional part of the second (0 .. 2^64-1)
    """
    if t_epoch_s < 0:
        # Still handle negative times reasonably by flooring
        sec = int(np.floor(t_epoch_s))
    else:
        sec = int(t_epoch_s)
    frac = t_epoch_s - float(sec)
    # Multiply by 2^64; round to nearest integer
    fs = int(np.floor(frac * (1 << 64) + 0.5))
    if fs >= (1 << 64):
        # Carry into integer seconds if rounding pushed us over
        fs = 0
        sec += 1
    return sec, fs


@dataclass
class IQStreamWriter:
    """Coordinate VITA 49 IQ data and context packet generation.

    Args:
        stream_id (int): Stream identifier written into generated packets.
        sample_rate_hz (float): Sample rate used to compute timestamps.
        payload_format (Optional[PayloadFormat]): Explicit payload format to reuse across packets.
        data_item_format (DataItemFormat): Data item format used when deriving payload_format.
        item_packing_field_size_bits (int): Packing field width for derived payload formats.
        data_item_size_bits (int): Per-sample data item width for derived payload formats.
        sample_component_repeat (bool): Whether components repeat when deriving payload format.
        repeat_count (int): Repeat count used for vector payloads.
        vector_size (int): Vector size for complex payloads.
        packet_type (PacketType): Packet type used for emitted data packets.
        tsi (TSI): Timestamp Integer selection for emitted packets.
        tsf (TSF): Timestamp Fractional selection for emitted packets.
        class_id (Optional[ClassID]): Class identifier to attach to packets.
        requires_vita49_2 (bool): Flag to set V49.2 indicator bit.
        frequency_domain (bool): Set the spectrum (S) bit to indicate frequency-domain Signal Spectral Data.
        start_time_epoch_s (Optional[float]): Initial epoch time for packet timestamps.
        bandwidth_hz (Optional[float]): Optional CIF0 bandwidth metadata.
        if_reference_frequency_hz (Optional[float]): Optional CIF0 IF reference frequency.
        rf_reference_frequency_hz (Optional[float]): Optional CIF0 RF reference frequency.
        rf_reference_frequency_offset_hz (Optional[float]): Optional CIF0 RF frequency offset.
        if_band_offset_hz (Optional[float]): Optional CIF0 IF band offset.
        reference_level_dbm (Optional[float]): Optional CIF0 reference level.
        gain_db (Optional[Tuple[float, float]]): Optional CIF0 gain tuple.
        device_identifier (Optional[Tuple[int, int]]): Optional CIF0 device identifier.
        state_event_indicators (Optional[int]): Optional CIF0 state/event indicators.
        context_timestamp_mode_general (bool): Whether to set the CIF timestamp mode bit.

    Returns:
        None.

    Raises:
        ValueError: If the configuration is inconsistent (for example, invalid sample rate).

    Side Effects:
        Maintains internal counters used to advance timestamps per emitted packet.

    Examples:
        >>> import numpy as np
        >>> from vita49io.io.iq_writer import IQStreamWriter
        >>> writer = IQStreamWriter(stream_id=1, sample_rate_hz=1e6)
        >>> pkt = writer.build_data_packet(np.zeros(4, dtype=np.complex64))
        >>> pkt.stream_id
        1
    """

    # Required
    stream_id: int
    sample_rate_hz: float

    # Payload format config (either provide ready pf or define via fields)
    payload_format: PayloadFormat | None = None
    data_item_format: DataItemFormat = DataItemFormat.IEEE754_SINGLE
    item_packing_field_size_bits: int = 32
    data_item_size_bits: int = 32
    sample_component_repeat: bool = False
    repeat_count: int = 1
    vector_size: int = 0

    # Header/Class/Timing config
    packet_type: PacketType = PacketType.IF_DATA_WITH_STREAM_ID
    tsi: TSI = TSI.UTC
    tsf: TSF = TSF.FRACTIONAL
    class_id: ClassID | None = None
    requires_vita49_2: bool = False
    frequency_domain: bool = False

    # Initial wall-clock start time (epoch seconds). If None, uses now (UTC).
    start_time_epoch_s: float | None = None

    # Optional context metadata (CIF0)
    bandwidth_hz: float | None = None
    if_reference_frequency_hz: float | None = None
    rf_reference_frequency_hz: float | None = None
    rf_reference_frequency_offset_hz: float | None = None
    if_band_offset_hz: float | None = None
    reference_level_dbm: float | None = None
    gain_db: Tuple[float, float] | None = None
    device_identifier: Tuple[int, int] | None = None  # (OUI 24-bit, device 32-bit)
    state_event_indicators: int | None = None

    # Context packet timestamp mode (CIF Timestamp Mode / TSM bit via indicators_24)
    context_timestamp_mode_general: bool = False

    # Internal state
    _samples_emitted: int = 0
    _packet_count: int = 0

    def __post_init__(self) -> None:
        """Validate IQStreamWriter configuration after dataclass initialization.

        Args:
            None.

        Returns:
            None.

        Raises:
            ValueError: If required fields such as `stream_id` or `sample_rate_hz` are invalid.

        Side Effects:
            May derive a PayloadFormat instance and initialize internal counters.

        Examples:
            >>> from vita49io.io.iq_writer import IQStreamWriter
            >>> IQStreamWriter(stream_id=1, sample_rate_hz=1e6)  # doctest: +ELLIPSIS
            IQStreamWriter(stream_id=1, sample_rate_hz=1000000.0, ...
        """
        if self.stream_id is None:
            raise ValueError("stream_id is required")
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be > 0")
        if self.packet_type not in (
            PacketType.IF_DATA_WITH_STREAM_ID,
            PacketType.EXTENSION_DATA_WITH_STREAM_ID,
        ):
            raise ValueError("packet_type must include a Stream ID for IQ data streams")

        # Resolve payload format if not provided
        if self.payload_format is None:
            self.payload_format = PayloadFormat(
                packing_method=PackingMethod.PROCESSING_EFFICIENT,
                sample_type=SampleType.COMPLEX_CARTESIAN,
                data_item_format_code=int(self.data_item_format),
                data_item_format=self.data_item_format,
                sample_component_repeat=bool(self.sample_component_repeat),
                event_tag_size_bits=0,
                channel_tag_size_bits=0,
                data_item_fraction_size_bits=0,
                item_packing_field_size_bits=int(self.item_packing_field_size_bits),
                data_item_size_bits=int(self.data_item_size_bits),
                repeat_count=int(self.repeat_count),
                vector_size=int(self.vector_size),
            )

        # Start time epoch seconds
        if self.start_time_epoch_s is None:
            self.start_time_epoch_s = _now_epoch_s()

    # ---------- Public API ----------

    def current_time(self) -> Tuple[int, int]:
        """Return the current VITA 49 timestamp cursor as integer and fractional seconds.

        Args:
            None.

        Returns:
            Tuple[int, int]: Pair of integer and 64-bit fractional seconds representing the current cursor.

        Raises:
            None.

        Side Effects:
            None.

        Examples:
            >>> from vita49io.io.iq_writer import IQStreamWriter
            >>> writer = IQStreamWriter(stream_id=1, sample_rate_hz=1e6)
            >>> isinstance(writer.current_time(), tuple)
            True
        """
        t = self.start_time_epoch_s + (self._samples_emitted / float(self.sample_rate_hz))  # type: ignore[operator]
        return _to_vrt_time(t)

    def build_data_packet(self, iq: "np.ndarray") -> DataPacket:
        """Build a DataPacket containing the provided IQ samples and advance the cursor.

        Args:
            iq (np.ndarray): Complex or interleaved IQ samples to attach to the packet.

        Returns:
            DataPacket: Packet populated with IQ data and updated timestamps.

        Raises:
            ValueError: If the IQ array is not complex or shaped as (N, 2).

        Side Effects:
            Updates the internal sample counter and packet count.

        Examples:
            >>> import numpy as np
            >>> from vita49io.io.iq_writer import IQStreamWriter
            >>> writer = IQStreamWriter(stream_id=1, sample_rate_hz=1e6)
            >>> pkt = writer.build_data_packet(np.zeros(4, dtype=np.complex64))
            >>> pkt.iq.shape[0]
            4
        """
        # Determine number of IQ samples
        arr = np.asarray(iq)
        if arr.dtype.kind == "c":
            n = int(arr.size)
        else:
            if arr.ndim == 2 and arr.shape[-1] == 2:
                n = int(arr.shape[0])
            else:
                raise ValueError("iq must be complex1D or real shape (N,2)")

        integer_seconds, fractional_seconds = self.current_time()

        header = Header(
            packet_type=self.packet_type,
            class_id_present=(self.class_id is not None),
            indicators_26=False,
            indicators_25=bool(self.requires_vita49_2),
            indicators_24=bool(self.frequency_domain),
            tsi=self.tsi,
            tsf=self.tsf,
            packet_count=(self._packet_count & 0xF),
            packet_size=0,
        )
        pkt = DataPacket(
            header=header,
            stream_id=self.stream_id,
            class_id=self.class_id,
            integer_seconds=integer_seconds if self.tsi != TSI.NONE else None,
            fractional_seconds=fractional_seconds if self.tsf != TSF.NONE else None,
            iq=arr,
        )

        # Advance state
        self._samples_emitted += n
        self._packet_count = (self._packet_count + 1) & 0xF

        return pkt

    def build_data_packet_bytes(self, iq: "np.ndarray") -> bytes:
        """Build and serialize a data packet for the provided IQ block.

        Args:
            iq (np.ndarray): Complex or interleaved IQ samples to encode.

        Returns:
            bytes: Serialized VITA 49 data packet bytes.

        Raises:
            ValueError: If IQ validation in `build_data_packet` fails.

        Side Effects:
            Advances the internal timestamp cursor via `build_data_packet`.

        Examples:
            >>> import numpy as np
            >>> from vita49io.io.iq_writer import IQStreamWriter
            >>> writer = IQStreamWriter(stream_id=1, sample_rate_hz=1e6)
            >>> payload = writer.build_data_packet_bytes(np.zeros(4, dtype=np.complex64))
            >>> isinstance(payload, bytes)
            True
        """
        pkt = self.build_data_packet(iq)
        return pkt.to_bytes(payload_format=self.payload_format)

    def build_context_packet(self) -> ContextPacket:
        """Create a ContextPacket representing the current stream configuration.

        Args:
            None.

        Returns:
            ContextPacket: Packet with CIF0 metadata and the current timestamp cursor.

        Raises:
            None.

        Side Effects:
            Increments the internal packet counter for subsequent packets.

        Examples:
            >>> from vita49io.io.iq_writer import IQStreamWriter
            >>> writer = IQStreamWriter(stream_id=1, sample_rate_hz=1e6)
            >>> writer.build_context_packet().cif0 is not None
            True
        """
        integer_seconds, fractional_seconds = self.current_time()

        # Build CIF0 fields
        cif0 = CIF0Fields(
            context_field_change_indicator=False,
            bandwidth_hz=self.bandwidth_hz,
            if_reference_frequency_hz=self.if_reference_frequency_hz,
            rf_reference_frequency_hz=self.rf_reference_frequency_hz,
            rf_reference_frequency_offset_hz=self.rf_reference_frequency_offset_hz,
            if_band_offset_hz=self.if_band_offset_hz,
            reference_level_dbm=self.reference_level_dbm,
            gain_db=self.gain_db,
            sample_rate_hz=self.sample_rate_hz,
            device_identifier=self.device_identifier,
            state_event_indicators=self.state_event_indicators,
            payload_format=self.payload_format,
        )

        header = Header(
            packet_type=PacketType.CONTEXT_PACKET,
            class_id_present=(self.class_id is not None),
            indicators_26=False,
            indicators_25=bool(self.requires_vita49_2),
            indicators_24=bool(self.context_timestamp_mode_general),
            tsi=self.tsi,
            tsf=self.tsf,
            packet_count=(self._packet_count & 0xF),
            packet_size=0,
        )
        pkt = ContextPacket(
            header=header,
            stream_id=self.stream_id,
            class_id=self.class_id,
            integer_seconds=integer_seconds if self.tsi != TSI.NONE else None,
            fractional_seconds=fractional_seconds if self.tsf != TSF.NONE else None,
            cif0=cif0,
        )

        # Context packets also consume a packet count in many producer chains
        self._packet_count = (self._packet_count + 1) & 0xF

        return pkt

    # ---------- Utilities ----------

    def reset_time(self, start_time: Union[float, datetime, None] = None) -> None:
        """Reset the timestamp cursor to a specific starting time.

        Args:
            start_time (Optional[Union[float, datetime]]): Epoch seconds or datetime to anchor future packets.

        Returns:
            None.

        Raises:
            None.

        Side Effects:
            Updates the stored start time and clears the emitted sample count.

        Examples:
            >>> from datetime import datetime
            >>> from vita49io.io.iq_writer import IQStreamWriter
            >>> writer = IQStreamWriter(stream_id=1, sample_rate_hz=1e6)
            >>> writer.reset_time(datetime.utcfromtimestamp(0))
            >>> writer._samples_emitted
            0
        """
        if isinstance(start_time, datetime):
            if start_time.tzinfo is None:
                # Treat naive datetime as UTC
                start_s = start_time.replace(tzinfo=timezone.utc).timestamp()
            else:
                start_s = start_time.timestamp()
        elif isinstance(start_time, (int, float)):
            start_s = float(start_time)
        else:
            start_s = _now_epoch_s()
        self.start_time_epoch_s = start_s
        self._samples_emitted = 0

    def advance_samples(self, n_samples: int) -> None:
        """Advance the timestamp cursor by a number of IQ samples.

        Args:
            n_samples (int): Number of IQ samples to add to the emitted total.

        Returns:
            None.

        Raises:
            ValueError: If `n_samples` is negative.

        Side Effects:
            Updates the internal emitted sample counter.

        Examples:
            >>> from vita49io.io.iq_writer import IQStreamWriter
            >>> writer = IQStreamWriter(stream_id=1, sample_rate_hz=1e6)
            >>> writer.advance_samples(10)
            >>> writer._samples_emitted
            10
        """
        if n_samples < 0:
            raise ValueError("n_samples must be >= 0")
        self._samples_emitted += int(n_samples)


__all__ = ["IQStreamWriter"]
