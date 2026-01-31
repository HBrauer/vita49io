"""Stream VITA 49 IQ packets through a spectral processor and emit spectral packets."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, Optional

import numpy as np

from ..signal.spectrum import SpectrumProcessor, SpectrumFrame
from ..protocol.context_packet import ContextPacket
from ..protocol.data_packet import DataPacket
from ..protocol.cif0 import (
    CIF0Fields,
    PayloadFormat,
    PackingMethod,
    SampleType,
    DataItemFormat,
)
from ..protocol.cif1 import (
    CIF1Fields,
    SpectrumField,
    SpectrumType,
    AveragingType,
    WindowTimeDeltaInterpretation,
)
from ..protocol.core import Header
from ..protocol.enums import PacketType, TSI, TSF
from .packet_reader import PacketReader, Readable
from .payload_codec import payload_as_numpy


@dataclass
class SpectrumStreamProcessor:
    """Convert VITA 49 IQ streams into VITA 49 spectral data packets."""

    stream: Readable
    fft_size: int
    hop_size: int
    window_type: str
    averaging_mode: str
    averaging_param: float | int
    output_fps: float
    output_bins: int
    band_mode: str = "inband"
    fft_kwargs: dict | None = None
    output_stream_id: int | None = None

    def __post_init__(self) -> None:
        self._reader = PacketReader(self.stream)
        self._queue: Deque[ContextPacket | DataPacket] = deque()
        self._packet_count = 0
        self._context_emitted = False
        self._payload_format: PayloadFormat | None = None
        self._sample_rate_hz: float | None = None
        self._bandwidth_hz: float | None = None
        self._input_stream_id: int | None = None
        self._tsi = TSI.NONE
        self._tsf = TSF.NONE
        self._time_base_epoch_s: float | None = None
        self._processor: SpectrumProcessor | None = None
        self._data_packet_type: PacketType | None = None
        self._prime_context()

    def read_packet(self) -> ContextPacket | DataPacket | None:
        """Return the next output packet, if available."""
        if self._queue:
            return self._queue.popleft()

        while True:
            pkt = self._reader.read_packet()
            if pkt is None:
                return None
            if isinstance(pkt, ContextPacket):
                self._handle_context(pkt)
                if self._queue:
                    return self._queue.popleft()
                continue
            if isinstance(pkt, DataPacket):
                self._handle_data(pkt)
                if self._queue:
                    return self._queue.popleft()
                continue

    def read_packets(self) -> Iterable[ContextPacket | DataPacket]:
        """Iterate over all output packets until EOF."""
        while True:
            pkt = self.read_packet()
            if pkt is None:
                break
            yield pkt

    def reset(self) -> None:
        """Reset the internal spectrum processor state."""
        if self._processor is not None:
            self._processor.reset()
        self._queue.clear()
        self._context_emitted = False
        self._packet_count = 0

    def _prime_context(self) -> None:
        start_pos = None
        seekable = False
        if hasattr(self.stream, "tell") and hasattr(self.stream, "seek"):
            try:
                start_pos = self.stream.tell()
                seekable = True
            except (OSError, AttributeError):
                seekable = False

        found = False
        while True:
            pkt = self._reader.read_packet()
            if pkt is None:
                break
            if isinstance(pkt, ContextPacket):
                self._handle_context(pkt)
                found = (
                    self._payload_format is not None
                    and self._sample_rate_hz is not None
                    and self._bandwidth_hz is not None
                )
                if found:
                    break

        if found and seekable and start_pos is not None:
            self.stream.seek(start_pos)
            self._reader = PacketReader(self.stream)
        elif found and not seekable:
            raise ValueError("Stream is not seekable; cannot restart after locating context")
        elif not found:
            raise ValueError("No context packet with payload format/sample rate found")

    def _handle_context(self, pkt: ContextPacket) -> None:
        if pkt.stream_id is not None and self._input_stream_id is None:
            self._input_stream_id = pkt.stream_id

        if self._time_base_epoch_s is None:
            t = _packet_time_s(pkt.integer_seconds, pkt.fractional_seconds)
            if t is not None:
                self._time_base_epoch_s = t
                self._tsi = pkt.header.tsi
                self._tsf = pkt.header.tsf

        if pkt.cif0 is None:
            return

        if pkt.cif0.payload_format is not None:
            self._payload_format = pkt.cif0.payload_format
        if pkt.cif0.sample_rate_hz is not None:
            self._sample_rate_hz = float(pkt.cif0.sample_rate_hz)
        if pkt.cif0.bandwidth_hz is not None:
            self._bandwidth_hz = float(pkt.cif0.bandwidth_hz)

        if self._payload_format is None or self._sample_rate_hz is None or self._bandwidth_hz is None:
            return

        if self._processor is None:
            self._processor = SpectrumProcessor(
                sample_rate_hz=self._sample_rate_hz,
                bandwidth_hz=self._bandwidth_hz,
                band_mode=self.band_mode,
                fft_size=self.fft_size,
                hop_size=self.hop_size,
                window_type=self.window_type,
                averaging_mode=self.averaging_mode,
                averaging_param=self.averaging_param,
                output_fps=self.output_fps,
                output_bins=self.output_bins,
                fft_kwargs=self.fft_kwargs,
            )
            return

        if self._processor.sample_rate_hz != self._sample_rate_hz or self._processor.bandwidth_hz != self._bandwidth_hz:
            self._processor = SpectrumProcessor(
                sample_rate_hz=self._sample_rate_hz,
                bandwidth_hz=self._bandwidth_hz,
                band_mode=self.band_mode,
                fft_size=self.fft_size,
                hop_size=self.hop_size,
                window_type=self.window_type,
                averaging_mode=self.averaging_mode,
                averaging_param=self.averaging_param,
                output_fps=self.output_fps,
                output_bins=self.output_bins,
                fft_kwargs=self.fft_kwargs,
            )
            self._context_emitted = False

    def _handle_data(self, pkt: DataPacket) -> None:
        if pkt.header.indicators_24:
            return
        if self._payload_format is None or self._sample_rate_hz is None or self._bandwidth_hz is None:
            return
        if self._processor is None:
            self._processor = SpectrumProcessor(
                sample_rate_hz=self._sample_rate_hz,
                bandwidth_hz=self._bandwidth_hz,
                band_mode=self.band_mode,
                fft_size=self.fft_size,
                hop_size=self.hop_size,
                window_type=self.window_type,
                averaging_mode=self.averaging_mode,
                averaging_param=self.averaging_param,
                output_fps=self.output_fps,
                output_bins=self.output_bins,
                fft_kwargs=self.fft_kwargs,
            )

        if self._input_stream_id is None and pkt.stream_id is not None:
            self._input_stream_id = pkt.stream_id
        if self._data_packet_type is None:
            self._data_packet_type = pkt.header.packet_type

        if self._time_base_epoch_s is None:
            t = _packet_time_s(pkt.integer_seconds, pkt.fractional_seconds)
            if t is not None:
                self._time_base_epoch_s = t
                self._tsi = pkt.header.tsi
                self._tsf = pkt.header.tsf

        if self._payload_format.sample_type != SampleType.COMPLEX_CARTESIAN:
            raise ValueError("Input payload format is not complex IQ")

        payload = pkt.payload
        payload_bytes = payload.tobytes() if isinstance(payload, memoryview) else payload
        iq = payload_as_numpy(payload_bytes, self._payload_format)
        frames = self._processor.push(iq)

        if frames:
            if not self._context_emitted:
                self._queue.append(self._build_context_packet())
                self._context_emitted = True
            for frame in frames:
                self._queue.append(self._build_data_packet(frame))

    def _build_context_packet(self) -> ContextPacket:
        stream_id = self.output_stream_id if self.output_stream_id is not None else (self._input_stream_id or 0)
        cif0 = CIF0Fields(
            sample_rate_hz=self._sample_rate_hz,
            bandwidth_hz=self._bandwidth_hz,
            payload_format=_build_spectrum_payload_format(),
            cif1=CIF1Fields(spectrum=self._build_spectrum_field()),
        )

        integer_seconds = None
        fractional_seconds = None
        if self._time_base_epoch_s is not None and self._tsi != TSI.NONE:
            integer_seconds, fractional_seconds = _to_vrt_time(self._time_base_epoch_s)

        header = Header(
            packet_type=PacketType.CONTEXT_PACKET,
            class_id_present=False,
            indicators_26=False,
            indicators_25=True,
            indicators_24=False,
            tsi=self._tsi,
            tsf=self._tsf,
            packet_count=(self._packet_count & 0xF),
            packet_size=0,
        )
        self._packet_count = (self._packet_count + 1) & 0xF
        return ContextPacket(
            header=header,
            stream_id=stream_id,
            integer_seconds=integer_seconds if self._tsi != TSI.NONE else None,
            fractional_seconds=fractional_seconds if self._tsf != TSF.NONE else None,
            cif0=cif0,
        )

    def _build_data_packet(self, frame: SpectrumFrame) -> DataPacket:
        stream_id = self.output_stream_id if self.output_stream_id is not None else (self._input_stream_id or 0)
        payload = np.asarray(frame.spectrum_db, dtype=">f4").tobytes()

        integer_seconds = None
        fractional_seconds = None
        if self._time_base_epoch_s is not None and self._tsi != TSI.NONE:
            t_epoch = self._time_base_epoch_s + float(frame.timestamp)
            integer_seconds, fractional_seconds = _to_vrt_time(t_epoch)

        packet_type = self._data_packet_type or PacketType.IF_DATA_WITH_STREAM_ID
        header = Header(
            packet_type=packet_type,
            class_id_present=False,
            indicators_26=False,
            indicators_25=True,
            indicators_24=True,
            tsi=self._tsi,
            tsf=self._tsf,
            packet_count=(self._packet_count & 0xF),
            packet_size=0,
        )
        self._packet_count = (self._packet_count + 1) & 0xF
        return DataPacket(
            header=header,
            stream_id=stream_id,
            integer_seconds=integer_seconds if self._tsi != TSI.NONE else None,
            fractional_seconds=fractional_seconds if self._tsf != TSF.NONE else None,
            payload=payload,
        )

    def _build_spectrum_field(self) -> SpectrumField:
        if self._sample_rate_hz is None or self._bandwidth_hz is None:
            raise ValueError("Spectrum processor not initialized")

        if self.window_type == "hann":
            window_type_code = 2
        elif self.window_type == "rect":
            window_type_code = 0
        else:
            window_type_code = 0

        if self.averaging_mode == "none":
            averaging_type = AveragingType.NONE
            number_of_averages = 1
            weighting_factor = 0
        elif self.averaging_mode == "mean":
            averaging_type = AveragingType.LINEAR
            number_of_averages = int(self.averaging_param)
            weighting_factor = 0
        else:
            averaging_type = AveragingType.EXPONENTIAL
            number_of_averages = 1
            weighting_factor = int(round(float(self.averaging_param) * (1 << 16)))

        overlap_samples = max(self.fft_size - self.hop_size, 0)
        span_hz = self._sample_rate_hz if self.band_mode == "full" else self._bandwidth_hz

        return SpectrumField(
            spectrum_type=SpectrumType.LOG_POWER_DB,
            averaging_type=averaging_type,
            window_time_delta_interpretation=WindowTimeDeltaInterpretation.SAMPLES,
            window_type=window_type_code,
            num_transform_points=self.fft_size,
            num_window_points=self.fft_size,
            resolution_hz=self._sample_rate_hz / float(self.fft_size),
            span_hz=span_hz,
            number_of_averages=number_of_averages,
            weighting_factor=weighting_factor,
            f1_index=0,
            f2_index=max(self.output_bins - 1, 0),
            window_time_delta=overlap_samples,
        )


def _packet_time_s(integer_seconds: int | None, fractional_seconds: int | None) -> float | None:
    if integer_seconds is None and fractional_seconds is None:
        return None
    sec = float(integer_seconds or 0)
    frac = float(fractional_seconds or 0)
    return sec + (frac / float(1 << 64))


def _to_vrt_time(t_epoch_s: float) -> tuple[int, int]:
    if t_epoch_s < 0:
        sec = int(np.floor(t_epoch_s))
    else:
        sec = int(t_epoch_s)
    frac = t_epoch_s - float(sec)
    fs = int(np.floor(frac * (1 << 64) + 0.5))
    if fs >= (1 << 64):
        fs = 0
        sec += 1
    return sec, fs


def _build_spectrum_payload_format() -> PayloadFormat:
    return PayloadFormat(
        packing_method=PackingMethod.PROCESSING_EFFICIENT,
        sample_type=SampleType.REAL,
        data_item_format=DataItemFormat.IEEE754_SINGLE,
        sample_component_repeat=False,
        event_tag_size_bits=0,
        channel_tag_size_bits=0,
        data_item_fraction_size_bits=0,
        item_packing_field_size_bits=32,
        data_item_size_bits=32,
        repeat_count=1,
        vector_size=1,
    )


__all__ = ["SpectrumStreamProcessor"]
