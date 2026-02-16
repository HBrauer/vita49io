"""Stream VITA 49 IQ packets through a spectral processor and emit spectral packets.

This module wraps :class:`vita49io.signal.spectrum.SpectrumProcessor` so it can be
fed by a VITA 49 packet stream and emit VITA 49 spectral packets. The core flow is:

1. Read context packets to discover payload format, sample rate, and bandwidth.
2. Decode IQ samples from data packets.
3. Push IQ into the spectrum processor to produce spectral frames.
4. Emit a single output context packet (once per config) followed by spectral data packets.

The output spectral packets encode power spectra in dB with a REAL, IEEE754 float
payload (big-endian float32). The output context packet includes CIF0/CIF1
metadata describing the spectrum configuration (windowing, averaging, span,
resolution).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable

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
from .time_utils import epoch_time_to_vita_timestamp, packet_vita_time_to_epoch_time


def _normalize_processing_mode(processing_mode: str) -> str:
    if processing_mode in {"continuous", "snapshot"}:
        return processing_mode
    raise ValueError("processing_mode must be 'continuous' or 'snapshot'")


@dataclass
class SpectrumStreamProcessor:
    """Convert VITA 49 IQ streams into VITA 49 spectral data packets.

    Usage
    -----
    The processor consumes a readable byte stream of VITA 49 packets and exposes
    a packet-by-packet interface that yields a new context packet plus spectral
    data packets as soon as enough IQ samples have been buffered.

    Example:
    ```python
    from vita49io.io.spectrum_processor import SpectrumStreamProcessor

    with open("input.v49", "rb") as f:
        processor = SpectrumStreamProcessor(
            stream=f,
            fft_size=1024,
            window_type="hann",
            averaging_mode="frame_mean",
            averaging_param=0,
            output_fps=10.0,
            band_mode="inband",
        )
        for pkt in processor.read_packets():
            # pkt is either ContextPacket or DataPacket
            handle(pkt)
    ```

    Snapshot-mode example (GNU Radio QT Frequency Sink-like defaults):
    ```python
    from vita49io.io.spectrum_processor import SpectrumStreamProcessor

    with open("input.v49", "rb") as f:
        processor = SpectrumStreamProcessor(
            stream=f,
            fft_size=1024,
            processing_mode="snapshot",
            output_fps=10.0,     # GR update_time=0.1 s
            averaging_mode="none",
            band_mode="full",
            power_scale="raw",
            window_type="hann",  # match your GNU Radio window setting
        )
        for pkt in processor.read_packets():
            handle(pkt)
    ```

    Parameters and behavior
    -----------------------
    - `stream`: Readable byte stream containing VITA 49 packets. The stream must be
      seekable on initialization because the processor scans for the first context
      packet with payload format, sample rate, and bandwidth, then rewinds.
    - `fft_size`: FFT length in samples. Controls frequency resolution
      (`sample_rate_hz / fft_size`) and the number of input samples required per
      transform.
    - `hop_size`: Optional advance in samples between consecutive FFTs.
      - In `"continuous"` mode this is the FFT step size. Defaults to `fft_size`
        (no overlap). Use `hop_size < fft_size` for overlap.
      - In `"snapshot"` mode this parameter is ignored for FFT cadence. One FFT is
        computed per output frame from the most recent `fft_size` samples.
    - `window_type`: `"hann"`, `"rect"`, `"blackmanharris"`, or `"kaiser"`. A Hann
      window reduces spectral leakage but widens the main lobe; a rectangular
      window preserves resolution but leaks more. Blackman-Harris further reduces
      sidelobes; Kaiser provides a tunable tradeoff.
    - `window_param`: Optional window parameter. For `"kaiser"` this is the beta
      value (default 8.0); otherwise ignored.
    - `dc_block`: If True, subtract the mean from each FFT segment before windowing
      to reduce the DC spike at the center bin (useful for wideband overview spectra).
    - `averaging_mode` / `averaging_param`:
      - `"none"`: no averaging. Each FFT power spectrum is emitted as-is.
      - `"mean"`: arithmetic mean over `averaging_param` FFTs; `averaging_param`
        must be a positive integer.
      - `"frame_mean"`: average *all* FFTs that occur between emitted output frames
        (Welch-style integration at `output_fps`). `averaging_param` is ignored.
      - `"exponential"`: exponential moving average with coefficient
        `averaging_param` in (0, 1]. Smaller values smooth more but respond slower.
      - `"exponential_tau"`: exponential moving average with time constant in seconds
        (`averaging_param` > 0). This is often easier to tune for long recordings.
      - `"peak_hold"`: per-bin peak hold across FFTs since the last reset.
    - `output_fps`: Target output frames per second. The processor emits spectra
      on a time grid derived from the input sample rate, regardless of how many
      FFTs were computed in that interval.
    - `processing_mode`:
      - `"continuous"` (default): STFT style processing over every hop-sized step.
      - `"snapshot"`: one FFT is
        computed per output frame using the most recent `fft_size` samples.
    - `output_bins`: Number of frequency bins in the emitted spectrum. If `None`,
      defaults to the native bin count (full band: `fft_size`; inband: the number
      of FFT bins within `[-bandwidth/2, +bandwidth/2]`). If it differs from the
      native bin count, the spectrum is linearly interpolated.
    - `power_scale`: How to scale the emitted values:
      - `"dbfs"` (default): normalize by the window coherent gain so a full-scale,
        bin-centered tone is ~0 dB (typical "dBFS-like" spectrum display).
      - `"raw"`: emit unnormalized FFT bin power in dB (values depend on FFT size/window).
    - `band_mode`:
      - `"full"`: use the full Nyquist span `[-sample_rate_hz/2, +sample_rate_hz/2]`.
      - `"inband"`: restrict to the context bandwidth `[-bandwidth_hz/2, +bandwidth_hz/2]`.
    - `fft_kwargs`: Optional `scipy.fft.fft` keyword arguments forwarded to the FFT.
    - `output_stream_id`: If set, overrides the input stream ID in output packets.

    Context handling and resets
    ---------------------------
    - The first output packet emitted after spectral frames are available is a
      context packet describing the spectrum metadata.
    - If a new input context changes sample rate or bandwidth, a new internal
      spectrum processor is created and a fresh context packet is emitted before
      the next spectral data packet.
    - `reset()` clears buffered samples, averaging state, and output packet count.
    """

    stream: Readable
    fft_size: int
    hop_size: int | None = None
    window_type: str = "hann"
    averaging_mode: str = "frame_mean"
    averaging_param: float | int = 0
    output_fps: float = 10.0
    processing_mode: str = "continuous"
    output_bins: int | None = None
    band_mode: str = "inband"
    window_param: float | None = None
    dc_block: bool = False
    power_scale: str = "dbfs"
    fft_kwargs: dict | None = None
    output_stream_id: int | None = None
    gps_utc_offset_s: int | None = None
    other_epoch_s: float | None = None
    tsi_none_reference_s: float | None = None
    sample_count_rate_hz: float | None = None
    free_running_rate_hz: float | None = None

    def __post_init__(self) -> None:
        if self.hop_size is None:
            self.hop_size = int(self.fft_size)
        else:
            self.hop_size = int(self.hop_size)
        if self.hop_size <= 0:
            raise ValueError("hop_size must be > 0 when provided")
        if self.sample_count_rate_hz is not None and self.sample_count_rate_hz <= 0:
            raise ValueError("sample_count_rate_hz must be > 0 when provided")
        if self.free_running_rate_hz is not None and self.free_running_rate_hz <= 0:
            raise ValueError("free_running_rate_hz must be > 0 when provided")
        self.processing_mode = _normalize_processing_mode(self.processing_mode)
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

    def _time_conversion_kwargs(self) -> dict[str, float | int | None]:
        return {
            "gps_utc_offset_s": self.gps_utc_offset_s,
            "other_epoch_s": self.other_epoch_s,
            "tsi_none_reference_s": self.tsi_none_reference_s,
            "sample_count_rate_hz": (
                self.sample_count_rate_hz if self.sample_count_rate_hz is not None else self._sample_rate_hz
            ),
            "free_running_rate_hz": (
                self.free_running_rate_hz if self.free_running_rate_hz is not None else self._sample_rate_hz
            ),
        }

    def _handle_context(self, pkt: ContextPacket) -> None:
        if pkt.stream_id is not None and self._input_stream_id is None:
            self._input_stream_id = pkt.stream_id

        if pkt.cif0 is not None:
            if pkt.cif0.payload_format is not None:
                self._payload_format = pkt.cif0.payload_format
            if pkt.cif0.sample_rate_hz is not None:
                self._sample_rate_hz = float(pkt.cif0.sample_rate_hz)
            if pkt.cif0.bandwidth_hz is not None:
                self._bandwidth_hz = float(pkt.cif0.bandwidth_hz)

        if self._time_base_epoch_s is None:
            t = packet_vita_time_to_epoch_time(pkt, **self._time_conversion_kwargs())
            if t is not None:
                self._time_base_epoch_s = t
                self._tsi = pkt.header.tsi
                self._tsf = pkt.header.tsf

        if pkt.cif0 is None:
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
                window_param=self.window_param,
                dc_block=self.dc_block,
                power_scale=self.power_scale,
                processing_mode=self.processing_mode,
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
                window_param=self.window_param,
                dc_block=self.dc_block,
                power_scale=self.power_scale,
                processing_mode=self.processing_mode,
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
                window_param=self.window_param,
                dc_block=self.dc_block,
                power_scale=self.power_scale,
                processing_mode=self.processing_mode,
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
            t = packet_vita_time_to_epoch_time(pkt, **self._time_conversion_kwargs())
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
        if self._time_base_epoch_s is not None:
            integer_seconds, fractional_seconds = epoch_time_to_vita_timestamp(
                self._time_base_epoch_s,
                tsi=self._tsi,
                tsf=self._tsf,
                **self._time_conversion_kwargs(),
            )

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
            integer_seconds=integer_seconds,
            fractional_seconds=fractional_seconds,
            cif0=cif0,
        )

    def _build_data_packet(self, frame: SpectrumFrame) -> DataPacket:
        stream_id = self.output_stream_id if self.output_stream_id is not None else (self._input_stream_id or 0)
        payload = np.asarray(frame.spectrum_db, dtype=">f4").tobytes()

        integer_seconds = None
        fractional_seconds = None
        if self._time_base_epoch_s is not None:
            t_epoch = self._time_base_epoch_s + float(frame.timestamp)
            integer_seconds, fractional_seconds = epoch_time_to_vita_timestamp(
                t_epoch,
                tsi=self._tsi,
                tsf=self._tsf,
                **self._time_conversion_kwargs(),
            )

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
            integer_seconds=integer_seconds,
            fractional_seconds=fractional_seconds,
            payload=payload,
        )

    def _build_spectrum_field(self) -> SpectrumField:
        if self._sample_rate_hz is None or self._bandwidth_hz is None:
            raise ValueError("Spectrum processor not initialized")
        if self._processor is None:
            raise ValueError("Spectrum processor not initialized")

        output_bins = self._processor.output_bins

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
        elif self.averaging_mode == "frame_mean":
            averaging_type = AveragingType.LINEAR
            if self.processing_mode == "snapshot":
                nominal = 1
            else:
                nominal = int(
                    round(self._sample_rate_hz / (float(self.hop_size) * float(self.output_fps)))
                )
            number_of_averages = max(nominal, 1)
            weighting_factor = 0
        elif self.averaging_mode == "peak_hold":
            averaging_type = AveragingType.PEAK_HOLD
            number_of_averages = 1
            weighting_factor = 0
        elif self.averaging_mode == "exponential":
            averaging_type = AveragingType.EXPONENTIAL
            number_of_averages = 1
            alpha = float(self.averaging_param)
            weighting_factor = int(round(alpha * (1 << 16)))
        else:
            averaging_type = AveragingType.EXPONENTIAL
            number_of_averages = 1
            tau = float(self.averaging_param)
            if self.processing_mode == "snapshot":
                dt_s = 1.0 / float(self.output_fps)
            else:
                dt_s = float(self.hop_size) / float(self._sample_rate_hz)
            alpha = 1.0 - float(np.exp(-dt_s / tau))
            weighting_factor = int(round(alpha * (1 << 16)))

        if self.processing_mode == "snapshot":
            overlap_samples = 0
        else:
            overlap_samples = max(self.fft_size - self.hop_size, 0)
        span_hz = self._sample_rate_hz if self.band_mode == "full" else self._bandwidth_hz

        resolution_hz = span_hz / float(output_bins)

        return SpectrumField(
            spectrum_type=SpectrumType.LOG_POWER_DB,
            averaging_type=averaging_type,
            window_time_delta_interpretation=WindowTimeDeltaInterpretation.SAMPLES,
            window_type=window_type_code,
            num_transform_points=self.fft_size,
            num_window_points=self.fft_size,
            resolution_hz=resolution_hz,
            span_hz=span_hz,
            number_of_averages=number_of_averages,
            weighting_factor=weighting_factor,
            f1_index=0,
            f2_index=max(output_bins - 1, 0),
            window_time_delta=overlap_samples,
        )

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
