from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Tuple, Union

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
    """Build VITA 49 data/context packets for an IQ stream.

    Configure once with meta information (stream ID, sample rate, payload
    format, etc.). For each block of IQ samples, call build_data_packet() to get
    a DataPacket or build_data_packet_bytes() for serialized bytes. Request a
    context packet anytime via build_context_packet().

    Time handling: maintains a time cursor starting at start_time (default now)
    and advances it by N / sample_rate for each data packet built. Timestamps
    are expressed with TSI/TSF (default UTC + FRACTIONAL).
    """

    # Required
    stream_id: int
    sample_rate_hz: float

    # Payload format config (either provide ready pf or define via fields)
    payload_format: Optional[PayloadFormat] = None
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
    class_id: Optional[ClassID] = None
    requires_vita49_2: bool = False

    # Initial wall-clock start time (epoch seconds). If None, uses now (UTC).
    start_time_epoch_s: Optional[float] = None

    # Optional context metadata (CIF0)
    bandwidth_hz: Optional[float] = None
    if_reference_frequency_hz: Optional[float] = None
    rf_reference_frequency_hz: Optional[float] = None
    rf_reference_frequency_offset_hz: Optional[float] = None
    if_band_offset_hz: Optional[float] = None
    reference_level_dbm: Optional[float] = None
    gain_db: Optional[Tuple[float, float]] = None
    device_identifier: Optional[Tuple[int, int]] = None  # (OUI 24-bit, device 32-bit)
    state_event_indicators: Optional[int] = None

    # Context packet timestamp mode (CIF Timestamp Mode / TSM bit via indicators_24)
    context_timestamp_mode_general: bool = False

    # Internal state
    _samples_emitted: int = 0
    _packet_count: int = 0

    def __post_init__(self) -> None:
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
        """Return current (integer_seconds, fractional_seconds) based on time cursor."""
        t = self.start_time_epoch_s + (self._samples_emitted / float(self.sample_rate_hz))  # type: ignore[operator]
        return _to_vrt_time(t)

    def build_data_packet(self, iq: "np.ndarray") -> DataPacket:
        """Create a DataPacket for the given IQ block and advance time cursor.

        The returned packet includes stream ID, class ID (if provided),
        timestamps per configured TSI/TSF, and IQ samples attached in the
        `iq` field. Serialize via packet.to_bytes(payload_format=self.payload_format).
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
            indicators_24=False,
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
        """Build and serialize a data packet for the provided IQ block."""
        pkt = self.build_data_packet(iq)
        return pkt.to_bytes(payload_format=self.payload_format)

    def build_context_packet(self) -> ContextPacket:
        """Create a ContextPacket reflecting current configuration and time.

        Includes sample_rate and payload format (CIF0 bit 21 and 15), and any
        optional metadata provided at construction time.
        """
        integer_seconds, fractional_seconds = self.current_time()

        # Build CIF0 fields
        pf_w0, pf_w1 = self.payload_format.pack_words()  # type: ignore[union-attr]
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
            data_packet_payload_format=(pf_w0, pf_w1),
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

    def reset_time(self, start_time: Optional[Union[float, datetime]] = None) -> None:
        """Reset time cursor to provided epoch time (or now) and zero sample count."""
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
        """Advance internal time cursor by a number of IQ samples."""
        if n_samples < 0:
            raise ValueError("n_samples must be >= 0")
        self._samples_emitted += int(n_samples)


__all__ = ["IQStreamWriter"]
