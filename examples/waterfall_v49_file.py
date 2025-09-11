from __future__ import annotations

import os
import sys
import argparse
from typing import Optional, Tuple


def _ensure_src_on_path() -> None:
    # Allow running the example from the repo root without installation
    here = os.path.dirname(__file__)
    src = os.path.normpath(os.path.join(here, "..", "src"))
    if os.path.isdir(src) and src not in sys.path:
        sys.path.insert(0, src)


def read_packets_with_iq(path: str):
    """Return an iterable over (iq, sample_rate_hz, timestamp_s) with context metadata.

    - Iteration yields tuples for each DataPacket with decoded IQ.
    - Attributes on the returned iterable instance (not a bare generator):
        .first_sample_rate_hz
        .first_rf_reference_frequency_hz
    """
    from vita49io.protocol.core import Header
    from vita49io.protocol.enums import PacketType
    from vita49io.protocol.data_packet import DataPacket
    from vita49io.protocol.context_packet import ContextPacket
    from vita49io.protocol.cif0 import PayloadFormat

    def _pkt_time_s(integer_seconds: Optional[int], fractional_seconds: Optional[int]) -> Optional[float]:
        if integer_seconds is None and fractional_seconds is None:
            return None
        sec = float(integer_seconds or 0)
        frac = float(fractional_seconds or 0)
        return sec + (frac / float(1 << 64))

    class IQPacketReader:
        def __init__(self, file_path: str) -> None:
            self.path = file_path
            self.first_sample_rate_hz: Optional[float] = None
            self.first_rf_reference_frequency_hz: Optional[float] = None
            self._last_payload_format: Optional[PayloadFormat] = None
            self.first_timestamp_s: Optional[float] = None
            self.last_timestamp_s: Optional[float] = None

        def __iter__(self):
            with open(self.path, "rb") as f:
                index = 0
                while True:
                    w0_bytes = f.read(4)
                    if not w0_bytes:
                        break
                    if len(w0_bytes) != 4:
                        raise ValueError(
                            f"Truncated header at packet {index}: expected 4 bytes, got {len(w0_bytes)}"
                        )

                    w0 = int.from_bytes(w0_bytes, byteorder="big")
                    header = Header.parse(w0)
                    total_words = header.packet_size
                    if total_words <= 0:
                        raise ValueError(
                            f"Invalid packet size (words) at packet {index}: {total_words}"
                        )

                    remaining_bytes = (total_words - 1) * 4
                    rest = f.read(remaining_bytes)
                    if len(rest) != remaining_bytes:
                        raise ValueError(
                            f"Truncated packet {index}: expected {remaining_bytes} bytes after header, got {len(rest)}"
                        )
                    packet_bytes = w0_bytes + rest

                    if header.packet_type == PacketType.CONTEXT_PACKET:
                        pkt = ContextPacket.from_bytes(packet_bytes)
                        if pkt.cif0 is not None:
                            if pkt.cif0.payload_format is not None:
                                self._last_payload_format = pkt.cif0.payload_format
                            if self.first_sample_rate_hz is None and pkt.cif0.sample_rate_hz is not None:
                                self.first_sample_rate_hz = float(pkt.cif0.sample_rate_hz)
                            if (
                                self.first_rf_reference_frequency_hz is None
                                and pkt.cif0.rf_reference_frequency_hz is not None
                            ):
                                self.first_rf_reference_frequency_hz = float(
                                    pkt.cif0.rf_reference_frequency_hz
                                )
                        # Track timestamps from context packets if present
                        t_ctx = _pkt_time_s(pkt.integer_seconds, pkt.fractional_seconds)
                        if t_ctx is not None:
                            if self.first_timestamp_s is None:
                                self.first_timestamp_s = t_ctx
                            self.last_timestamp_s = t_ctx
                    elif header.packet_type in (
                        PacketType.IF_DATA_WITHOUT_STREAM_ID,
                        PacketType.IF_DATA_WITH_STREAM_ID,
                        PacketType.EXTENSION_DATA_WITHOUT_STREAM_ID,
                        PacketType.EXTENSION_DATA_WITH_STREAM_ID,
                    ):
                        pkt = DataPacket.from_bytes(
                            packet_bytes, payload_format=self._last_payload_format
                        )
                        if getattr(pkt, "iq", None) is not None:
                            t_s = _pkt_time_s(pkt.integer_seconds, pkt.fractional_seconds)
                            if t_s is not None:
                                if self.first_timestamp_s is None:
                                    self.first_timestamp_s = t_s
                                self.last_timestamp_s = t_s
                            yield pkt.iq, self.first_sample_rate_hz, t_s
                    else:
                        pass

                    index += 1

    return IQPacketReader(path)


def _stft_waterfall(iq, fs: float | None, fft_size: int, hop: int) -> Tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
    import numpy as np

    x = np.asarray(iq)
    if x.ndim != 1:
        x = x.reshape(-1)
    n = x.size
    if n < fft_size:
        return np.empty((0, 0)), np.array([]), np.array([])

    # Frame the signal
    n_frames = 1 + (n - fft_size) // hop
    idx = np.expand_dims(np.arange(fft_size), 0) + np.expand_dims(np.arange(n_frames) * hop, 1)
    frames = x[idx]

    # Hann window
    win = np.hanning(fft_size).astype(np.float32)
    frames = frames * win

    # FFT and magnitude (in dB)
    spec = np.fft.fftshift(np.fft.fft(frames, n=fft_size, axis=1), axes=1)
    power = np.abs(spec) ** 2
    eps = 1e-12
    db = 10.0 * np.log10(power + eps)

    # Frequency and time axes
    if fs is None or fs <= 0:
        fs = 1.0
    freqs = np.linspace(-fs / 2.0, fs / 2.0, fft_size, endpoint=False)
    times = (np.arange(n_frames) * hop) / fs

    return db, freqs, times


def main(argv: Optional[list[str]] = None) -> int:
    _ensure_src_on_path()

    parser = argparse.ArgumentParser(description="Render a waterfall (spectrogram) from VITA 49 IQ data.")
    parser.add_argument("path", nargs="?", default=r"F:\\VitaFiles\\in_pocsag.v49", help="Path to .v49 file")
    parser.add_argument("--fft", type=int, default=1024, help="FFT size (pixels in frequency)")
    parser.add_argument("--overlap", type=float, default=0.75, help="Frame overlap fraction [0..0.95]")
    parser.add_argument("--max-samples", type=int, default=2_000_000, help="Limit total samples to avoid huge memory use")
    parser.add_argument("--cmap", type=str, default="viridis", help="Matplotlib colormap")
    args = parser.parse_args(argv if argv is not None else None)

    path = args.path
    fft_size = max(64, int(args.fft))
    overlap = min(0.95, max(0.0, float(args.overlap)))
    hop = max(1, int(round(fft_size * (1.0 - overlap))))

    # Accumulate IQ samples (streaming) up to limit
    import numpy as np

    chunks: list[np.ndarray] = []
    total = 0
    fs: Optional[float] = None
    fc: Optional[float] = None  # RF reference from first context
    try:
        reader = read_packets_with_iq(path)
        for iq, sr, _t_s in reader:
            if iq is None or iq.size == 0:
                continue
            if fs is None and sr is not None:
                fs = float(sr)
            iq = np.asarray(iq, dtype=np.complex64)
            if total + iq.size > args.max_samples:
                need = args.max_samples - total
                if need <= 0:
                    break
                chunks.append(iq[:need])
                total += need
                break
            chunks.append(iq)
            total += iq.size
        # Fetch first-context RF reference and sample rate, if available
        if fs is None and reader.first_sample_rate_hz is not None:
            fs = float(reader.first_sample_rate_hz)
        if reader.first_rf_reference_frequency_hz is not None:
            fc = float(reader.first_rf_reference_frequency_hz)
    except FileNotFoundError:
        print(f"File not found: {path}")
        return 1
    except Exception as e:
        print(f"Error reading '{path}': {e}")
        return 1

    if total == 0:
        print("No IQ data decoded. Ensure the file contains CIF0 payload format and data packets.")
        return 2

    x = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]

    # Compute waterfall
    db, freqs, times = _stft_waterfall(x, fs, fft_size, hop)
    if db.size == 0:
        print("Not enough samples for one FFT frame; increase --max-samples or reduce --fft")
        return 2

    # Simpler: do not remap times; keep STFT-relative seconds based on fs

    # Apply RF reference offset from first context, if present
    if fc is not None:
        freqs = freqs + fc

    # Normalize for display (optional: percentile-based)
    lo = float(np.percentile(db, 5))
    hi = float(np.percentile(db, 99))
    db = np.clip(db, lo, hi)

    # Plot
    import matplotlib.pyplot as plt

    # Plot with time on Y (top -> bottom) and frequency on X
    extent = [freqs[0], freqs[-1], times[0] if times.size else 0.0, times[-1] if times.size else 0.0]
    plt.figure(figsize=(10, 6))
    plt.imshow(
        db,  # rows=time, cols=frequency
        origin="upper",  # time increases downward (top -> bottom)
        aspect="auto",
        extent=extent,
        interpolation="nearest",
        cmap=args.cmap,
    )
    plt.colorbar(label="Power (dB)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Time (s)")
    title_fs = f" @ {fs/1e6:.3f} Msps" if fs else ""
    title_fc = f", RF={fc/1e6:.3f} MHz" if fc is not None else ""
    # Add start/end time from first/last packet if available
    time_span = ""
    try:
        from datetime import datetime, timezone
        if reader and (getattr(reader, "first_timestamp_s", None) is not None or getattr(reader, "last_timestamp_s", None) is not None):
            def _fmt(ts: float) -> str:
                return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + "Z"
            if reader.first_timestamp_s is not None and reader.last_timestamp_s is not None:
                time_span = f" | { _fmt(reader.first_timestamp_s) } to { _fmt(reader.last_timestamp_s) }"
            elif reader.first_timestamp_s is not None:
                time_span = f" | start { _fmt(reader.first_timestamp_s) }"
            elif reader.last_timestamp_s is not None:
                time_span = f" | end { _fmt(reader.last_timestamp_s) }"
    except Exception:
        pass
    plt.title(f"{os.path.basename(path)}{title_fs}{title_fc}{time_span}")
    plt.tight_layout()
    plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
