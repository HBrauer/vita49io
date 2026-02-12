from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.signal import resample_poly

from vita49io.defaults.default_payload_formats import DefaultPayloadFormats
from vita49io.io.iq_writer import IQStreamWriter
from vita49io.io.packet_reader import PacketReader
from vita49io.io.payload_codec import payload_as_numpy
from vita49io.protocol.context_packet import ContextPacket
from vita49io.protocol.data_packet import DataPacket

DEFAULT_DECIMATOR_CONFIG_PATH = Path(__file__).with_name("ddc_v49_file.toml")


@dataclass(frozen=True)
class DecimatorPath:
    bandwidth_hz: int
    taps: Optional[np.ndarray]


def _ensure_src_on_path() -> None:
    # Allow running the example from the repo root without installation
    here = Path(__file__).resolve().parent
    src = here.parent / "src"
    src_str = str(src)
    if src.is_dir() and src_str not in sys.path:
        sys.path.insert(0, src_str)


def _load_toml_module():
    try:
        import tomllib

        return tomllib
    except ModuleNotFoundError:
        try:
            import tomli

            return tomli
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "TOML parsing requires Python 3.11+ or the 'tomli' package on Python 3.8-3.10"
            ) from exc


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DDC a VITA 49 file (resample + re-pack).")
    parser.add_argument("input_file", help="Path to input .v49 file")
    parser.add_argument("output_file", help="Path to output .v49 file")
    parser.add_argument(
        "--output-format",
        required=True,
        help="Output payload format (F32_IQ, S32_IQ, S24_IQ, S16_IQ)",
    )
    parser.add_argument(
        "--output-sample-rate",
        required=True,
        type=int,
        help="Output sample rate in Hz",
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
    return parser.parse_args(argv)


def _packet_time_s(integer_seconds: Optional[int], fractional_seconds: Optional[int]) -> Optional[float]:
    if integer_seconds is None and fractional_seconds is None:
        return None
    sec = float(integer_seconds or 0)
    frac = float(fractional_seconds or 0)
    return sec + (frac / float(1 << 64))


def _resolve_input_format_name(pf, supported_formats: Dict[str, Any]) -> Optional[str]:
    for name, fmt in supported_formats.items():
        if pf == fmt:
            return name
    return None


def _resample_ratio(in_rate_hz: int, out_rate_hz: int) -> Tuple[int, int]:
    frac = Fraction(int(out_rate_hz), int(in_rate_hz))
    return frac.numerator, frac.denominator


def _load_decimator_paths(config_path: Path) -> Dict[Tuple[int, int], DecimatorPath]:
    config_path = config_path.expanduser()
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Decimator config file not found: {config_path}. "
            "Provide --config or create the default config file."
        )

    toml = _load_toml_module()
    with config_path.open("rb") as f:
        doc = toml.load(f)

    if not isinstance(doc, dict):
        raise ValueError(f"Invalid TOML root in decimator config: {config_path}")

    decimator = doc.get("decimator")
    if not isinstance(decimator, dict):
        raise ValueError(
            f"Missing [decimator] section in decimator config: {config_path}"
        )

    raw_paths = decimator.get("paths")
    if not isinstance(raw_paths, list) or not raw_paths:
        raise ValueError(
            f"Missing or empty [[decimator.paths]] entries in decimator config: {config_path}"
        )

    paths: Dict[Tuple[int, int], DecimatorPath] = {}
    for idx, entry in enumerate(raw_paths, start=1):
        if not isinstance(entry, dict):
            raise ValueError(
                f"Entry #{idx} in [[decimator.paths]] must be a table in {config_path}"
            )

        try:
            in_rate = int(entry["input_sample_rate"])
            out_rate = int(entry["output_sample_rate"])
            bandwidth_hz = int(entry["bandwidth"])
        except KeyError as exc:
            raise ValueError(
                f"Missing required key {exc} in [[decimator.paths]] entry #{idx}"
            ) from exc
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid numeric value in [[decimator.paths]] entry #{idx}"
            ) from exc

        taps_raw = entry.get("taps")
        taps: Optional[np.ndarray]
        if taps_raw is None:
            taps = None
        else:
            if not isinstance(taps_raw, list):
                raise ValueError(
                    f"'taps' must be an array in [[decimator.paths]] entry #{idx}"
                )
            taps = np.asarray(taps_raw, dtype=np.float64)
            if taps.ndim != 1:
                raise ValueError(
                    f"'taps' must be a 1-D array in [[decimator.paths]] entry #{idx}"
                )

        key = (in_rate, out_rate)
        if key in paths:
            raise ValueError(
                f"Duplicate decimator path in config for {in_rate} -> {out_rate}"
            )

        paths[key] = DecimatorPath(
            bandwidth_hz=bandwidth_hz,
            taps=taps,
        )

    return paths


def _get_decimator_path(
    decimator_paths: Dict[Tuple[int, int], DecimatorPath],
    input_sample_rate_hz: int,
    output_sample_rate_hz: int,
) -> DecimatorPath:
    path = decimator_paths.get((input_sample_rate_hz, output_sample_rate_hz))
    if path is not None:
        return path

    configured_output_rates = sorted({out_rate for _, out_rate in decimator_paths})
    if output_sample_rate_hz not in configured_output_rates:
        raise ValueError(
            "Unsupported output sample rate. Configured outputs: "
            + ", ".join(str(x) for x in configured_output_rates)
        )

    supported_input_rates = sorted(
        {
            in_rate
            for (in_rate, out_rate) in decimator_paths
            if out_rate == output_sample_rate_hz
        }
    )
    raise ValueError(
        f"Unsupported input sample rate {input_sample_rate_hz} for output "
        f"{output_sample_rate_hz}. Configured inputs for this output: "
        + ", ".join(str(x) for x in supported_input_rates)
    )


def convert_v49_ddc(
    input_path: Path,
    output_path: Path,
    output_format_name: str,
    output_sample_rate_hz: int,
    chunk_samples: int,
    samples_per_packet: int,
    config_path: Optional[Path] = None,
) -> Dict[str, int]:
    _ensure_src_on_path()

    config_path = (config_path or DEFAULT_DECIMATOR_CONFIG_PATH).expanduser()
    decimator_paths = _load_decimator_paths(config_path)

    supported_formats = {
        "F32_IQ": DefaultPayloadFormats.F32_IQ,
        "S32_IQ": DefaultPayloadFormats.S32_IQ,
        "S24_IQ": DefaultPayloadFormats.S24_IQ,
        "S16_IQ": DefaultPayloadFormats.S16_IQ,
    }

    output_format_name = output_format_name.upper()
    if output_format_name not in supported_formats:
        raise ValueError(
            f"Unsupported output format '{output_format_name}'. "
            f"Supported: {', '.join(sorted(supported_formats))}"
        )
    output_payload_format = supported_formats[output_format_name]

    configured_output_rates = sorted({out_rate for _, out_rate in decimator_paths})
    if output_sample_rate_hz not in configured_output_rates:
        raise ValueError(
            "Unsupported output sample rate. Configured outputs: "
            + ", ".join(str(x) for x in configured_output_rates)
        )

    if chunk_samples <= 0:
        raise ValueError("chunk_samples must be > 0")
    if samples_per_packet <= 0:
        raise ValueError("samples_per_packet must be > 0")

    input_path = input_path.expanduser()
    output_path = output_path.expanduser()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # State from the first context packet
    input_sample_rate_hz: Optional[int] = None
    input_payload_format = None
    input_payload_name: Optional[str] = None
    input_rf_ref_hz: Optional[float] = None
    input_rf_ref_offset_hz: Optional[float] = None
    input_if_ref_hz: Optional[float] = None
    input_if_band_offset_hz: Optional[float] = None
    input_reference_level_dbm: Optional[float] = None
    input_gain_db: Optional[Tuple[float, float]] = None
    input_device_identifier: Optional[Tuple[int, int]] = None
    input_state_event_indicators: Optional[int] = None
    input_context_tsm: bool = False
    input_stream_id: Optional[int] = None
    first_context_time_s: Optional[float] = None

    # Resampling parameters
    decimator_path: Optional[DecimatorPath] = None
    up: Optional[int] = None
    down: Optional[int] = None
    resample_window = None

    # Output stream state
    writer: Optional[IQStreamWriter] = None

    in_chunks: list[np.ndarray] = []
    in_count = 0
    out_buffer = np.empty(0, dtype=np.complex64)

    total_in_samples = 0
    total_out_samples = 0
    data_packets_written = 0
    data_packets_seen = 0
    context_packets_seen = 0
    skipped_packets = 0
    output_bandwidth_hz: Optional[int] = None

    def emit_samples(samples: np.ndarray, out_f) -> None:
        nonlocal out_buffer, data_packets_written, total_out_samples
        if samples.size == 0:
            return
        if out_buffer.size == 0:
            out_buffer = samples
        else:
            out_buffer = np.concatenate([out_buffer, samples])
        while out_buffer.size >= samples_per_packet:
            chunk = out_buffer[:samples_per_packet]
            out_buffer = out_buffer[samples_per_packet:]
            out_f.write(writer.build_data_packet_bytes(chunk))
            data_packets_written += 1
            total_out_samples += samples_per_packet

    with input_path.open("rb") as f_in, output_path.open("wb") as f_out:
        reader = PacketReader(f_in)
        while True:
            pkt = reader.read_packet()
            if pkt is None:
                break

            if isinstance(pkt, ContextPacket):
                context_packets_seen += 1
                if first_context_time_s is None:
                    first_context_time_s = _packet_time_s(pkt.integer_seconds, pkt.fractional_seconds)
                if pkt.cif0 is not None:
                    cif0 = pkt.cif0
                    if input_payload_format is None and cif0.payload_format is not None:
                        input_payload_format = cif0.payload_format
                        input_payload_name = _resolve_input_format_name(
                            input_payload_format,
                            supported_formats,
                        )
                    if input_sample_rate_hz is None and cif0.sample_rate_hz is not None:
                        input_sample_rate_hz = int(round(float(cif0.sample_rate_hz)))
                    if input_rf_ref_hz is None and cif0.rf_reference_frequency_hz is not None:
                        input_rf_ref_hz = float(cif0.rf_reference_frequency_hz)
                    if input_rf_ref_offset_hz is None and cif0.rf_reference_frequency_offset_hz is not None:
                        input_rf_ref_offset_hz = float(cif0.rf_reference_frequency_offset_hz)
                    if input_if_ref_hz is None and cif0.if_reference_frequency_hz is not None:
                        input_if_ref_hz = float(cif0.if_reference_frequency_hz)
                    if input_if_band_offset_hz is None and cif0.if_band_offset_hz is not None:
                        input_if_band_offset_hz = float(cif0.if_band_offset_hz)
                    if input_reference_level_dbm is None and cif0.reference_level_dbm is not None:
                        input_reference_level_dbm = float(cif0.reference_level_dbm)
                    if input_gain_db is None and cif0.gain_db is not None:
                        input_gain_db = cif0.gain_db
                    if input_device_identifier is None and cif0.device_identifier is not None:
                        input_device_identifier = cif0.device_identifier
                    if input_state_event_indicators is None and cif0.state_event_indicators is not None:
                        input_state_event_indicators = cif0.state_event_indicators
                if input_stream_id is None and pkt.stream_id is not None:
                    input_stream_id = pkt.stream_id
                input_context_tsm = bool(pkt.header.indicators_24)
                continue

            if isinstance(pkt, DataPacket):
                data_packets_seen += 1
                if input_payload_format is None or input_sample_rate_hz is None:
                    skipped_packets += 1
                    continue
                if input_payload_name is None:
                    raise ValueError(
                        "Unsupported input payload format. "
                        "Supported: F32_IQ, S32_IQ, S24_IQ, S16_IQ"
                    )

                if up is None or down is None:
                    decimator_path = _get_decimator_path(
                        decimator_paths,
                        input_sample_rate_hz,
                        output_sample_rate_hz,
                    )
                    output_bandwidth_hz = decimator_path.bandwidth_hz
                    up, down = _resample_ratio(input_sample_rate_hz, output_sample_rate_hz)
                    if (up != 1 or down != 1) and decimator_path.taps is None:
                        raise ValueError(
                            f"Decimator path {input_sample_rate_hz} -> {output_sample_rate_hz} "
                            "is missing 'taps' in config"
                        )
                    resample_window = decimator_path.taps

                if writer is None:
                    if input_stream_id is None:
                        input_stream_id = pkt.stream_id
                    if input_stream_id is None:
                        raise ValueError("Input stream_id is missing; cannot write output stream")
                    start_time_s = _packet_time_s(pkt.integer_seconds, pkt.fractional_seconds)
                    if start_time_s is None:
                        start_time_s = first_context_time_s
                    writer = IQStreamWriter(
                        stream_id=input_stream_id,
                        sample_rate_hz=float(output_sample_rate_hz),
                        payload_format=output_payload_format,
                        data_packet_type=pkt.header.packet_type,
                        tsi=pkt.header.tsi,
                        tsf=pkt.header.tsf,
                        class_id=pkt.class_id,
                        requires_vita49_2=bool(pkt.header.indicators_25),
                        frequency_domain=bool(pkt.header.indicators_24),
                        start_time_epoch_s=start_time_s,
                        bandwidth_hz=float(output_bandwidth_hz),
                        rf_reference_frequency_hz=input_rf_ref_hz,
                        rf_reference_frequency_offset_hz=input_rf_ref_offset_hz,
                        if_reference_frequency_hz=input_if_ref_hz,
                        if_band_offset_hz=input_if_band_offset_hz,
                        reference_level_dbm=input_reference_level_dbm,
                        gain_db=input_gain_db,
                        device_identifier=input_device_identifier,
                        state_event_indicators=input_state_event_indicators,
                        context_timestamp_mode_general=input_context_tsm,
                    )
                    f_out.write(writer.build_context_packet().to_bytes())

                payload = pkt.payload
                payload_bytes = payload.tobytes() if isinstance(payload, memoryview) else payload
                iq = payload_as_numpy(payload_bytes, input_payload_format)

                in_chunks.append(iq)
                in_count += int(iq.size)
                total_in_samples += int(iq.size)

                while in_count >= chunk_samples:
                    combined = np.concatenate(in_chunks) if len(in_chunks) > 1 else in_chunks[0]
                    block = combined[:chunk_samples]
                    remainder = combined[chunk_samples:]
                    in_chunks = [remainder] if remainder.size else []
                    in_count = int(remainder.size)

                    if up == 1 and down == 1:
                        resampled = block
                    else:
                        resampled = resample_poly(block, up, down, window=resample_window)
                    emit_samples(np.asarray(resampled, dtype=np.complex64).reshape(-1), f_out)
                continue

            skipped_packets += 1

        # Process remaining samples after loop ends
        if in_count > 0 and input_sample_rate_hz is not None and input_payload_format is not None:
            combined = np.concatenate(in_chunks) if len(in_chunks) > 1 else in_chunks[0]
            if up is None or down is None:
                decimator_path = _get_decimator_path(
                    decimator_paths,
                    input_sample_rate_hz,
                    output_sample_rate_hz,
                )
                output_bandwidth_hz = decimator_path.bandwidth_hz
                up, down = _resample_ratio(input_sample_rate_hz, output_sample_rate_hz)
                if (up != 1 or down != 1) and decimator_path.taps is None:
                    raise ValueError(
                        f"Decimator path {input_sample_rate_hz} -> {output_sample_rate_hz} "
                        "is missing 'taps' in config"
                    )
                resample_window = decimator_path.taps
            if up == 1 and down == 1:
                resampled = combined
            else:
                resampled = resample_poly(combined, up, down, window=resample_window)
            emit_samples(np.asarray(resampled, dtype=np.complex64).reshape(-1), f_out)

        # Pad the last packet with zeros to reach samples_per_packet
        if writer is not None and out_buffer.size > 0:
            pad_len = samples_per_packet - int(out_buffer.size)
            padded = np.concatenate([out_buffer, np.zeros(pad_len, dtype=np.complex64)])
            f_out.write(writer.build_data_packet_bytes(padded))
            data_packets_written += 1
            total_out_samples += samples_per_packet

    return {
        "input_samples": total_in_samples,
        "output_samples": total_out_samples,
        "data_packets_written": data_packets_written,
        "data_packets_seen": data_packets_seen,
        "context_packets_seen": context_packets_seen,
        "skipped_packets": skipped_packets,
        "input_sample_rate_hz": int(input_sample_rate_hz or 0),
        "output_sample_rate_hz": int(output_sample_rate_hz),
        "input_payload_format": input_payload_name or "",
        "output_payload_format": output_format_name,
        "output_bandwidth_hz": int(output_bandwidth_hz or 0),
    }


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)

    try:
        summary = convert_v49_ddc(
            input_path=Path(args.input_file),
            output_path=Path(args.output_file),
            output_format_name=args.output_format,
            output_sample_rate_hz=int(args.output_sample_rate),
            chunk_samples=int(args.chunk_samples),
            samples_per_packet=int(args.samples_per_packet),
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
