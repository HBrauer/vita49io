from __future__ import annotations

import argparse
import math
import sys
import textwrap
from pathlib import Path
from typing import Optional

import numpy as np

SAMPLES_PER_PACKET = 1024


class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass


def _ensure_src_on_path() -> None:
    """Allow running this example directly from the repository root."""
    here = Path(__file__).resolve().parent
    src = here.parent / "src"
    src_str = str(src)
    if src.is_dir() and src_str not in sys.path:
        sys.path.insert(0, src_str)


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid float: {value}") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be > 0")
    return parsed


def _non_negative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid float: {value}") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("Value must be >= 0")
    return parsed


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid integer: {value}") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be > 0")
    return parsed


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a synthetic VITA 49 IQ test file with one context packet first, "
            "followed by data packets of exactly 1024 IQ samples each."
        ),
        formatter_class=_HelpFormatter,
        epilog=textwrap.dedent(
            f"""\
            Examples:
              python examples/create_test_v49_file.py test.v49 \\
                --sample-rate-hz 1000000 \\
                --bandwidth-hz 200000 \\
                --rf-reference-frequency-hz 915000000 \\
                --duration-seconds 1.0 \\
                --cosine-frequency-hz 25000 \\
                --cosine-amplitude 0.5 \\
                --noise-amplitude 0.02 \\
                --output-format S16_IQ

              python examples/create_test_v49_file.py test_packets.v49 \\
                --sample-rate-hz 48000 \\
                --bandwidth-hz 20000 \\
                --rf-reference-frequency-hz 100000000 \\
                --num-data-packets 100 \\
                --cosine-frequency-hz 1000 \\
                --output-format F32_IQ

            Notes:
              - Exactly {SAMPLES_PER_PACKET} IQ samples are written to every data packet.
              - If --duration-seconds is used, packet count is ceil(duration * sample_rate / {SAMPLES_PER_PACKET}).
              - The generated tone is always complex (analytic): exp(j*2*pi*f*t).
              - --noise-amplitude follows GNU Radio noise_source_c (GR_GAUSSIAN):
                I/Q std-dev is amplitude/sqrt(2), so E[|n|^2] = amplitude^2.
            """
        ),
    )

    parser.add_argument("output_file", help="Output VITA 49 file path (.v49)")
    parser.add_argument("--sample-rate-hz", required=True, type=_positive_float, help="Sample rate in Hz")
    parser.add_argument("--bandwidth-hz", required=True, type=_positive_float, help="Bandwidth in Hz")
    parser.add_argument(
        "--rf-reference-frequency-hz",
        required=True,
        type=float,
        help="RF reference frequency in Hz",
    )
    parser.add_argument(
        "--output-format",
        required=True,
        choices=("F32_IQ", "S32_IQ", "S24_IQ", "S16_IQ"),
        help="Output payload format",
    )

    length_group = parser.add_mutually_exclusive_group(required=True)
    length_group.add_argument(
        "--duration-seconds",
        type=_positive_float,
        help="Target signal length in seconds",
    )
    length_group.add_argument(
        "--num-data-packets",
        type=_positive_int,
        help="Number of data packets to write",
    )

    parser.add_argument(
        "--cosine-frequency-hz",
        type=float,
        default=1000.0,
        help="Complex tone frequency in Hz",
    )
    parser.add_argument(
        "--cosine-amplitude",
        type=_non_negative_float,
        default=0.5,
        help="Complex tone amplitude (linear full-scale)",
    )
    parser.add_argument(
        "--noise-amplitude",
        type=_non_negative_float,
        default=0.0,
        help="Gaussian complex noise amplitude (GNU Radio noise_source_c equivalent)",
    )
    parser.add_argument(
        "--stream-id",
        type=lambda x: int(x, 0),
        default=0x13572468,
        help="VRT stream ID (decimal or 0x-prefixed hex)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducible noise",
    )
    return parser.parse_args(argv)


def _resolve_packet_count(args: argparse.Namespace, sample_rate_hz: float) -> int:
    if args.num_data_packets is not None:
        return int(args.num_data_packets)

    assert args.duration_seconds is not None
    requested_samples = int(math.ceil(args.duration_seconds * sample_rate_hz))
    return max(1, int(math.ceil(requested_samples / SAMPLES_PER_PACKET)))


def _synthesize_packet_iq(
    *,
    packet_index: int,
    sample_rate_hz: float,
    cosine_frequency_hz: float,
    cosine_amplitude: float,
    noise_amplitude: float,
    rng: np.random.Generator,
) -> np.ndarray:
    start = packet_index * SAMPLES_PER_PACKET
    t = (start + np.arange(SAMPLES_PER_PACKET, dtype=np.float64)) / sample_rate_hz

    phase = 2.0 * np.pi * cosine_frequency_hz * t
    iq = (cosine_amplitude * np.exp(1j * phase)).astype(np.complex64)

    if noise_amplitude > 0.0:
        # Match GNU Radio noise_source_c (GR_GAUSSIAN): per-component sigma = ampl/sqrt(2).
        sigma = np.float32(noise_amplitude / math.sqrt(2.0))
        noise = (
            rng.standard_normal(SAMPLES_PER_PACKET, dtype=np.float32)
            + 1j * rng.standard_normal(SAMPLES_PER_PACKET, dtype=np.float32)
        ) * sigma
        iq = iq + noise.astype(np.complex64)

    return iq.astype(np.complex64, copy=False)


def main(argv: Optional[list[str]] = None) -> int:
    _ensure_src_on_path()

    from vita49io.defaults.default_payload_formats import DefaultPayloadFormats
    from vita49io.io.iq_writer import IQStreamWriter

    args = _parse_args(argv)

    payload_format = getattr(DefaultPayloadFormats, args.output_format)
    sample_rate_hz = float(args.sample_rate_hz)
    num_packets = _resolve_packet_count(args, sample_rate_hz)

    writer = IQStreamWriter(
        stream_id=int(args.stream_id),
        sample_rate_hz=sample_rate_hz,
        payload_format=payload_format,
        bandwidth_hz=float(args.bandwidth_hz),
        rf_reference_frequency_hz=float(args.rf_reference_frequency_hz),
    )

    rng = np.random.default_rng(args.seed)
    out_path = Path(args.output_file).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("wb") as f:
        context_bytes = writer.build_context_packet().to_bytes()
        f.write(context_bytes)

        for packet_index in range(num_packets):
            iq = _synthesize_packet_iq(
                packet_index=packet_index,
                sample_rate_hz=sample_rate_hz,
                cosine_frequency_hz=float(args.cosine_frequency_hz),
                cosine_amplitude=float(args.cosine_amplitude),
                noise_amplitude=float(args.noise_amplitude),
                rng=rng,
            )
            f.write(writer.build_data_packet_bytes(iq))

    total_samples = num_packets * SAMPLES_PER_PACKET
    actual_duration_s = total_samples / sample_rate_hz

    print(f"Wrote {out_path}")
    print(f"Context packets: 1")
    print(f"Data packets:    {num_packets}")
    print(f"Samples/packet:  {SAMPLES_PER_PACKET}")
    print(f"Total samples:   {total_samples}")
    print(f"Actual length:   {actual_duration_s:.9f} s")
    print(f"Output format:   {args.output_format}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
