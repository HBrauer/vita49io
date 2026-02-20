"""
Render a frequency-domain waterfall from VITA 49 IQ data packets.

This script decodes IQ samples from data packets and uses the same waterfall
implementation as `validate_ddc_test_matrix.py` (`write_waterfall_svg` from
`vita49io.signal.ddc_testbench`).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from vita49io.io.packet_reader import PacketReader
from vita49io.io.payload_codec import payload_as_numpy
from vita49io.protocol.context_packet import ContextPacket
from vita49io.protocol.data_packet import DataPacket
from vita49io.signal.ddc_testbench import _stft_waterfall, write_waterfall_svg


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render a waterfall from VITA 49 IQ data using the same STFT implementation "
            "as DDC validation."
        ),
    )
    parser.add_argument("input_file", help="Path to input VITA 49 file")
    parser.add_argument(
        "-n",
        "--max-packets",
        type=int,
        default=None,
        help="Maximum number of packets to read (default: all)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of IQ samples to decode (default: all)",
    )
    parser.add_argument(
        "--output-svg",
        default=None,
        help="Optional output SVG path (if omitted, show interactive plot)",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional waterfall title",
    )
    parser.add_argument(
        "--fft-size",
        type=int,
        default=1024,
        help="STFT FFT size",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.75,
        help="STFT overlap in [0, 1)",
    )
    parser.add_argument(
        "--db-span",
        type=float,
        default=90.0,
        help="Displayed dynamic range in dB below peak",
    )
    return parser.parse_args(argv)


def _as_complex_iq(samples: np.ndarray) -> Optional[np.ndarray]:
    arr = np.asarray(samples)
    if arr.size == 0:
        return None
    if np.iscomplexobj(arr):
        return np.asarray(arr.reshape(-1), dtype=np.complex64)

    flat = np.asarray(arr.reshape(-1), dtype=np.float32)
    if flat.size < 2 or (flat.size % 2) != 0:
        return None
    iq = flat[0::2] + 1j * flat[1::2]
    return np.asarray(iq, dtype=np.complex64)


def _collect_iq(
    *,
    path: Path,
    max_packets: Optional[int],
    max_samples: Optional[int],
) -> Tuple[np.ndarray, Optional[float], Optional[float], int, int, int]:
    payload_format = None
    sample_rate_hz: Optional[float] = None
    bandwidth_hz: Optional[float] = None
    chunks: list[np.ndarray] = []

    packets_read = 0
    data_packets_seen = 0
    skipped_without_payload_format = 0
    total_samples = 0

    with path.open("rb") as f:
        reader = PacketReader(f)
        while True:
            if max_packets is not None and packets_read >= int(max_packets):
                break
            if max_samples is not None and total_samples >= int(max_samples):
                break

            pkt = reader.read_packet()
            if pkt is None:
                break
            packets_read += 1

            if isinstance(pkt, ContextPacket):
                cif0 = pkt.cif0
                if cif0 is not None:
                    if cif0.payload_format is not None:
                        payload_format = cif0.payload_format
                    if cif0.sample_rate_hz is not None:
                        sample_rate_hz = float(cif0.sample_rate_hz)
                    if cif0.bandwidth_hz is not None:
                        bandwidth_hz = float(cif0.bandwidth_hz)
                continue

            if not isinstance(pkt, DataPacket):
                continue
            data_packets_seen += 1

            if payload_format is None:
                skipped_without_payload_format += 1
                continue

            payload = pkt.payload.tobytes() if isinstance(pkt.payload, memoryview) else pkt.payload
            try:
                decoded = payload_as_numpy(payload, payload_format)
            except Exception:
                continue

            iq = _as_complex_iq(decoded)
            if iq is None or iq.size == 0:
                continue

            if max_samples is not None:
                remain = int(max_samples) - total_samples
                if remain <= 0:
                    break
                iq = iq[:remain]

            chunks.append(iq)
            total_samples += int(iq.size)

    if chunks:
        iq_all = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
    else:
        iq_all = np.empty(0, dtype=np.complex64)

    return (
        np.asarray(iq_all, dtype=np.complex64),
        sample_rate_hz,
        bandwidth_hz,
        packets_read,
        data_packets_seen,
        skipped_without_payload_format,
    )


def _show_waterfall(
    *,
    iq: np.ndarray,
    sample_rate_hz: float,
    title: str,
    output_bandwidth_hz: Optional[float],
    fft_size: int,
    overlap: float,
    db_span: float,
) -> bool:
    hop = max(1, int(round(float(fft_size) * (1.0 - float(overlap)))))
    db, freqs, times = _stft_waterfall(iq, float(sample_rate_hz), int(fft_size), hop)
    if db.size == 0:
        return False

    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    hi = float(np.max(db))
    lo = float(hi - max(20.0, float(db_span)))
    img = np.clip(db, lo, hi)
    extent = [freqs[0], freqs[-1], times[0] if times.size else 0.0, times[-1] if times.size else 0.0]

    plt.figure(figsize=(10, 5.5))
    plt.imshow(
        img,
        origin="upper",
        aspect="auto",
        extent=extent,
        interpolation="nearest",
        cmap="viridis",
        vmin=lo,
        vmax=hi,
    )
    if output_bandwidth_hz is not None and output_bandwidth_hz > 0:
        half_bw = min(float(output_bandwidth_hz) / 2.0, float(sample_rate_hz) / 2.0)
        if half_bw > 0:
            ax = plt.gca()
            ax.axvspan(freqs[0], -half_bw, color="white", alpha=0.08, lw=0)
            ax.axvspan(half_bw, freqs[-1], color="white", alpha=0.08, lw=0)
            ax.axvline(-half_bw, color="white", linestyle="--", linewidth=1.0, alpha=0.8)
            ax.axvline(half_bw, color="white", linestyle="--", linewidth=1.0, alpha=0.8)
    plt.colorbar(label="Power (dB)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Time (s)")
    plt.title(title)
    plt.tight_layout()
    plt.show()
    plt.close()
    return True


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    input_path = Path(args.input_file).expanduser()
    if not input_path.is_file():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 2

    (
        iq,
        sample_rate_hz,
        bandwidth_hz,
        packets_read,
        data_packets_seen,
        skipped_without_payload_format,
    ) = _collect_iq(
        path=input_path,
        max_packets=args.max_packets,
        max_samples=args.max_samples,
    )

    print(
        f"Packets read: {packets_read}, data packets seen: {data_packets_seen}, "
        f"decoded IQ samples: {int(iq.size)}"
    )
    if skipped_without_payload_format > 0:
        print(
            f"Skipped {skipped_without_payload_format} data packets before payload format context was available."
        )
    if sample_rate_hz is not None:
        print(f"Detected sample_rate_hz: {sample_rate_hz:g}")
    if bandwidth_hz is not None:
        print(f"Detected bandwidth_hz: {bandwidth_hz:g}")

    if sample_rate_hz is None:
        print("Could not determine sample rate from context packets.", file=sys.stderr)
        return 1
    if iq.size == 0:
        print("No IQ data decoded from data packets.", file=sys.stderr)
        return 1

    title = args.title or f"Waterfall: {input_path.name}"
    if args.output_svg:
        output_svg = Path(args.output_svg).expanduser()
        ok = write_waterfall_svg(
            iq=iq,
            sample_rate_hz=float(sample_rate_hz),
            output_svg=output_svg,
            title=title,
            output_bandwidth_hz=bandwidth_hz,
            fft_size=int(args.fft_size),
            overlap=float(args.overlap),
            db_span=float(args.db_span),
        )
    else:
        ok = _show_waterfall(
            iq=iq,
            sample_rate_hz=float(sample_rate_hz),
            title=title,
            output_bandwidth_hz=bandwidth_hz,
            fft_size=int(args.fft_size),
            overlap=float(args.overlap),
            db_span=float(args.db_span),
        )
    if not ok:
        print("Could not render waterfall (matplotlib backend unavailable).", file=sys.stderr)
        return 1

    if args.output_svg:
        print(f"Wrote waterfall: {Path(args.output_svg).expanduser()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
