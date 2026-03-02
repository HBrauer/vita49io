from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from vita49io.signal.ddc_file import (
    DEFAULT_DECIMATOR_CONFIG_PATH,
    convert_v49_ddc,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "DDC a VITA 49 file (resample + re-pack). "
            "No-context mode is supported when required input metadata is passed via CLI."
        ),
        epilog=(
            "Examples:\n"
            "  python -m vita49io.scripts.ddc_v49_file in.v49 out.v49 \\\n"
            "    --output-format S16_IQ \\\n"
            "    --output-sample-rate 2048000\n"
            "\n"
            "  python -m vita49io.scripts.ddc_v49_file in.v49 out_bw.v49 \\\n"
            "    --output-format S16_IQ \\\n"
            "    --bandwidth 10000000\n"
            "\n"
            "  python -m vita49io.scripts.ddc_v49_file in.v49 out_shifted.v49 \\\n"
            "    --output-format F32_IQ \\\n"
            "    --output-sample-rate 1024000 \\\n"
            "    --center-frequency-offset-hz -250000\n"
            "\n"
            "  python -m vita49io.scripts.ddc_v49_file in.v49 out_custom.v49 \\\n"
            "    --output-format S16_IQ \\\n"
            "    --output-sample-rate 1024000 \\\n"
            "    --config examples/ddc_v49_file.toml \\\n"
            "    --chunk-samples 61140 \\\n"
            "    --samples-per-packet 1024\n"
            "\n"
            "  # No-context mode: input stream has no context packets.\n"
            "  # Provide input format, sample rate, and bandwidth explicitly.\n"
            "  python -m vita49io.scripts.ddc_v49_file in_no_ctx.v49 out.v49 \\\n"
            "    --input-format S16_IQ \\\n"
            "    --input-sample-rate 98304000 \\\n"
            "    --input-bandwidth 80000000 \\\n"
            "    --output-format S16_IQ \\\n"
            "    --output-sample-rate 24576000"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("input_file", help="Path to input .v49 file")
    parser.add_argument("output_file", help="Path to output .v49 file")
    parser.add_argument(
        "--output-format",
        required=True,
        help="Output payload format (F32_IQ, S32_IQ, S24_IQ, S16_IQ)",
    )
    parser.add_argument(
        "--input-format",
        default=None,
        help=(
            "Optional input payload format override (F32_IQ, S32_IQ, S24_IQ, S16_IQ). "
            "When set, decoding uses this format starting from the first data packet."
        ),
    )
    parser.add_argument(
        "--input-sample-rate",
        type=int,
        default=None,
        help=(
            "Optional input sample rate in Hz. Required for no-context mode if "
            "sample rate is not available from context."
        ),
    )
    parser.add_argument(
        "--input-bandwidth",
        type=float,
        default=None,
        help=(
            "Optional input bandwidth in Hz. Required for no-context mode if "
            "bandwidth is not available from context."
        ),
    )
    parser.add_argument(
        "--input-rf-reference-frequency-hz",
        type=float,
        default=None,
        help=(
            "Optional input RF reference frequency in Hz. Needed when using "
            "--center-frequency-offset-hz without context RF reference metadata."
        ),
    )
    parser.add_argument(
        "--strict-payload-format",
        "--strict",
        "--stricked",
        dest="strict_payload_format",
        action="store_true",
        help=(
            "Enable strict payload-format validation (exact match including repeat_count "
            "and vector_size). Default mode ignores repeat_count/vector_size."
        ),
    )
    output_selector_group = parser.add_mutually_exclusive_group(required=True)
    output_selector_group.add_argument(
        "--output-sample-rate",
        type=int,
        help="Output sample rate in Hz",
    )
    output_selector_group.add_argument(
        "--bandwidth",
        type=int,
        help="Select output path by configured output bandwidth in Hz",
    )
    tuning_group = parser.add_mutually_exclusive_group()
    tuning_group.add_argument(
        "--center-frequency-hz",
        type=float,
        default=None,
        help=(
            "Target RF center frequency in Hz (absolute). "
            "If input CIF0 rf_reference_frequency_hz is available, this is interpreted "
            "as absolute RF and converted to an offset from the input center."
        ),
    )
    tuning_group.add_argument(
        "--center-frequency-offset-hz",
        type=float,
        default=None,
        help=(
            "Target center frequency offset in Hz relative to input "
            "CIF0 rf_reference_frequency_hz."
        ),
    )
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Path to decimator TOML config. "
            f"Defaults to {DEFAULT_DECIMATOR_CONFIG_PATH}"
        ),
    )
    parser.add_argument(
        "--chunk-samples",
        type=int,
        default=61_140,
        help="Input samples per resampling block",
    )
    parser.add_argument(
        "--samples-per-packet",
        type=int,
        default=1024,
        help="Number of complex samples per output data packet",
    )
    parser.add_argument(
        "--timing",
        action="store_true",
        help="Print elapsed conversion time after completion",
    )
    parser.add_argument(
        "--start-time",
        default=None,
        help=(
            "Optional UTC start time for packet-level time slicing (ISO-8601, "
            "e.g. 2026-01-01T12:34:56Z). Packets overlapping the window are kept."
        ),
    )
    parser.add_argument(
        "--end-time",
        default=None,
        help=(
            "Optional UTC end time for packet-level time slicing (ISO-8601, "
            "e.g. 2026-01-01T12:35:56Z). Packets overlapping the window are kept."
        ),
    )
    return parser.parse_args(argv)


def _parse_iso_utc_time(value: str, *, option_name: str) -> float:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{option_name} must not be empty")
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(
            f"{option_name} must be ISO UTC time like YYYY-MM-DDThh:mm:ssZ"
        ) from exc
    if dt.tzinfo is None:
        raise ValueError(f"{option_name} must include timezone (use trailing 'Z')")
    return float(dt.astimezone(timezone.utc).timestamp())


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    start_time = time.perf_counter() if bool(args.timing) else None

    try:
        summary = convert_v49_ddc(
            input_path=Path(args.input_file),
            output_path=Path(args.output_file),
            output_format_name=args.output_format,
            output_sample_rate_hz=(
                int(args.output_sample_rate)
                if args.output_sample_rate is not None
                else None
            ),
            target_bandwidth_hz=(
                int(args.bandwidth)
                if args.bandwidth is not None
                else None
            ),
            chunk_samples=int(args.chunk_samples),
            samples_per_packet=int(args.samples_per_packet),
            center_frequency_hz=(
                float(args.center_frequency_hz)
                if args.center_frequency_hz is not None
                else None
            ),
            center_frequency_offset_hz=(
                float(args.center_frequency_offset_hz)
                if args.center_frequency_offset_hz is not None
                else None
            ),
            start_time_epoch_s=(
                _parse_iso_utc_time(args.start_time, option_name="--start-time")
                if args.start_time is not None
                else None
            ),
            end_time_epoch_s=(
                _parse_iso_utc_time(args.end_time, option_name="--end-time")
                if args.end_time is not None
                else None
            ),
            input_format_name=args.input_format,
            strict_payload_format=bool(args.strict_payload_format),
            input_sample_rate_hz=(
                int(args.input_sample_rate)
                if args.input_sample_rate is not None
                else None
            ),
            input_bandwidth_hz=(
                float(args.input_bandwidth)
                if args.input_bandwidth is not None
                else None
            ),
            input_rf_reference_frequency_hz=(
                float(args.input_rf_reference_frequency_hz)
                if args.input_rf_reference_frequency_hz is not None
                else None
            ),
            config_path=Path(args.config).expanduser() if args.config else None,
        )
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print(
        "DDC complete. "
        f"In={summary['input_sample_rate_hz']} Hz {summary['input_payload_format']} -> "
        f"Out={summary['output_sample_rate_hz']} Hz {summary['output_payload_format']} "
        f"(BW {summary['output_bandwidth_hz']}), "
        f"Input samples: {summary['input_samples']}, "
        f"Output samples: {summary['output_samples']}, "
        f"Data packets written: {summary['data_packets_written']}"
    )
    if start_time is not None:
        elapsed_s = time.perf_counter() - start_time
        print(f"Timing: elapsed {elapsed_s:.3f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
