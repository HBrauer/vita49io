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
    """Iterate packets and yield (iq, sample_rate_hz, center_hz) for each DataPacket.

    Keeps track of the last CIF0 payload format, which enables float32/complex64 decoding
    via helper functions. Also extracts sample_rate_hz from CIF0 when present.
    """
    from vita49io.protocol.core import Header
    from vita49io.protocol.enums import PacketType
    from vita49io.protocol.data_packet import DataPacket
    from vita49io.protocol.context_packet import ContextPacket
    from vita49io.protocol.cif0 import PayloadFormat
    from vita49io.io.payload_codec import payload_as_numpy

    last_payload_format: Optional[PayloadFormat] = None
    last_sample_rate_hz: Optional[float] = None
    last_center_hz: Optional[float] = None

    with open(path, "rb") as f:
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
                raise ValueError(f"Invalid packet size (words) at packet {index}: {total_words}")

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
                        last_payload_format = pkt.cif0.payload_format
                    if pkt.cif0.sample_rate_hz is not None:
                        last_sample_rate_hz = pkt.cif0.sample_rate_hz
                    # Try to resolve an absolute center frequency for the capture
                    center = None
                    # Prefer RF reference + offsets if available
                    if pkt.cif0.rf_reference_frequency_hz is not None:
                        center = pkt.cif0.rf_reference_frequency_hz
                        if pkt.cif0.rf_reference_frequency_offset_hz is not None:
                            center += pkt.cif0.rf_reference_frequency_offset_hz
                        if pkt.cif0.if_band_offset_hz is not None:
                            center += pkt.cif0.if_band_offset_hz
                    # Fallback to IF reference if set
                    elif pkt.cif0.if_reference_frequency_hz is not None:
                        center = pkt.cif0.if_reference_frequency_hz
                    if center is not None:
                        last_center_hz = float(center)
            elif header.packet_type in (
                PacketType.IF_DATA_WITHOUT_STREAM_ID,
                PacketType.IF_DATA_WITH_STREAM_ID,
                PacketType.EXTENSION_DATA_WITHOUT_STREAM_ID,
                PacketType.EXTENSION_DATA_WITH_STREAM_ID,
            ):
                if last_payload_format is None:
                    continue
                pkt = DataPacket.from_bytes(packet_bytes)
                payload = pkt.payload
                payload_bytes = payload.tobytes() if isinstance(payload, memoryview) else payload
                iq = payload_as_numpy(payload_bytes, last_payload_format)
                yield iq, last_sample_rate_hz, last_center_hz
            else:
                # Skip unsupported packet types
                pass

            index += 1


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


def _design_fir_lowpass(num_taps: int, cutoff_hz: float, fs: float):
    """Design a real FIR lowpass using a windowed-sinc (Hamming) method.

    - num_taps should be odd for best symmetry.
    - cutoff_hz is the absolute cutoff in Hz (0 < cutoff_hz < fs/2).
    """
    import numpy as np

    num_taps = int(num_taps)
    if num_taps < 3:
        num_taps = 3
    if num_taps % 2 == 0:
        num_taps += 1  # ensure odd
    if fs <= 0:
        fs = 1.0
    fc = float(cutoff_hz)
    # Normalized frequency for sinc (cycles/sample)
    norm = fc / fs
    n = np.arange(num_taps) - (num_taps - 1) / 2
    # Ideal sinc lowpass impulse response
    h = 2 * norm * np.sinc(2 * norm * n)
    # Hamming window
    w = np.hamming(num_taps)
    h *= w
    # Normalize DC gain to 1
    h /= np.sum(h)
    return h.astype(np.float64)


def _decimate_pow2(x, fs: float | None, factor: int, taps_per_stage: int = 63):
    """Decimate complex IQ by a power-of-two factor with FIR anti-aliasing.

    Applies cascaded by-2 stages. Each stage uses a Hamming-windowed sinc lowpass
    with cutoff at ~0.9 of the new Nyquist (0.225 * fs_stage).
    Returns (y, fs_out).
    """
    import numpy as np

    if factor <= 1:
        return np.asarray(x), fs

    # Validate power-of-two
    if factor & (factor - 1) != 0:
        raise ValueError(f"Decimation factor must be a power of two, got {factor}")

    fs_stage = float(fs) if (fs is not None and fs > 0) else 1.0
    y = np.asarray(x, dtype=np.complex64)

    stages = int(round(np.log2(factor)))
    for _ in range(stages):
        # Lowpass cutoff slightly below new Nyquist (Fs/4)
        cutoff = 0.225 * fs_stage
        h = _design_fir_lowpass(taps_per_stage, cutoff, fs_stage)
        # Convolve (same length), then downsample by 2
        yr = np.convolve(y.real, h, mode="same")
        yi = np.convolve(y.imag, h, mode="same")
        y = (yr + 1j * yi)[::2]
        fs_stage = fs_stage / 2.0

    return y.astype(np.complex64), fs_stage


def _freq_shift(x, fs: float | None, delta_hz: float):
    """Frequency-shift complex IQ by delta_hz (x * e^{-j2pi f t})."""
    import numpy as np

    if fs is None or fs <= 0 or abs(delta_hz) < 1e-6:
        return np.asarray(x, dtype=np.complex64)
    
    print(delta_hz)
    x = np.asarray(x, dtype=np.complex64)
    n = np.arange(x.size, dtype=np.float64)
    phase = np.exp(-1j * 2.0 * np.pi * (delta_hz / fs) * n)
    return (x * phase).astype(np.complex64)


def main(argv: Optional[list[str]] = None) -> int:
    _ensure_src_on_path()

    parser = argparse.ArgumentParser(description="Render a waterfall from VITA 49 IQ data with optional decimation.")
    parser.add_argument("path", nargs="?", default=r"F:\\VitaFiles\\in_pocsag.v49", help="Path to .v49 file")
    parser.add_argument("--fft", type=int, default=1024, help="FFT size (pixels in frequency)")
    parser.add_argument("--overlap", type=float, default=0.75, help="Frame overlap fraction [0..0.95]")
    parser.add_argument("--max-samples", type=int, default=2_000_000, help="Limit total samples to avoid huge memory use")
    parser.add_argument("--cmap", type=str, default="viridis", help="Matplotlib colormap")
    parser.add_argument("--decim", type=int, default=1, help="Decimation factor (power of two, e.g. 1,2,4,8)")
    parser.add_argument("--taps", type=int, default=63, help="FIR taps per decimation stage (odd)")
    parser.add_argument("--center-freq", type=float, default=None, help="Absolute RF/IF frequency (Hz) to tune to baseband before decimation")
    args = parser.parse_args(argv if argv is not None else None)

    path = args.path
    fft_size = max(64, int(args.fft))
    overlap = min(0.95, max(0.0, float(args.overlap)))
    hop = max(1, int(round(fft_size * (1.0 - overlap))))
    decim = max(1, int(args.decim))

    # Accumulate samples (streaming) up to limit
    import numpy as np

    chunks: list[np.ndarray] = []
    total = 0
    fs: Optional[float] = None
    capture_center: Optional[float] = None
    try:
        for iq, sr, ctr in read_packets_with_iq(path):
            if iq is None or iq.size == 0:
                continue
            if fs is None and sr is not None:
                fs = float(sr)
            if capture_center is None and ctr is not None:
                capture_center = float(ctr)
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
    except FileNotFoundError:
        print(f"File not found: {path}")
        return 1
    except Exception as e:
        print(f"Error reading '{path}': {e}")
        return 1

    if total == 0:
        print("No sample data decoded. Ensure the file contains CIF0 payload format and data packets.")
        return 2

    x = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]

    # Optional frequency shift: move requested absolute center to DC
    fc_req = float(args.center_freq) if args.center_freq is not None else None
    if fc_req is not None and fs is not None and capture_center is not None:
        delta = fc_req 
        x = _freq_shift(x, fs, delta)

    # Optional decimation with anti-alias filtering
    try:
        x, fs_eff = _decimate_pow2(x, fs, decim, taps_per_stage=int(args.taps))
    except ValueError as ve:
        print(str(ve))
        return 2

    # Compute waterfall
    db, freqs, times = _stft_waterfall(x, fs_eff, fft_size, hop)
    if db.size == 0:
        print("Not enough samples for one FFT frame; increase --max-samples or reduce --fft")
        return 2

    # Normalize for display (optional: percentile-based)
    lo = float(np.percentile(db, 5))
    hi = float(np.percentile(db, 99))
    db = np.clip(db, lo, hi)

    # Plot
    import matplotlib.pyplot as plt

    # Keep baseband axis; include tuned Fc in title
    fc = float(args.center_freq) if args.center_freq is not None else None

    extent = [freqs[0], freqs[-1], times[0] if times.size else 0.0, times[-1] if times.size else 0.0]
    plt.figure(figsize=(10, 6))
    plt.imshow(
        db,
        origin="upper",
        aspect="auto",
        extent=extent,
        interpolation="nearest",
        cmap=args.cmap,
    )
    plt.colorbar(label="Power (dB)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Time (s)")

    title_fs = f" @ {fs_eff/1e6:.3f} Msps" if fs_eff else ""
    title_fc = f" | tuned Fc={fc/1e6:.6f} MHz" if fc is not None else ""
    plt.title(f"Waterfall of {os.path.basename(path)}{title_fs}{title_fc} (decim x{decim})")
    plt.tight_layout()
    plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
