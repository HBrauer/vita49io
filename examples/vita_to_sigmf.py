"""Example: Convert a VITA 49 file to SigMF data/meta pair.

This script reads packets from a VITA 49 capture, decodes IQ payloads as complex
float32 samples, and writes them to a SigMF `.sigmf-data` file. A matching
`.sigmf-meta` JSON file is generated with minimal metadata derived from the
latest context information observed in the stream.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from vita49io.protocol.cif0 import PayloadFormat


def _ensure_src_on_path() -> None:
    """Ensure the project src/ directory is importable when running from the repo."""
    here = Path(__file__).resolve().parent
    src = here.parent / "src"
    src_str = str(src)
    if src.is_dir() and src_str not in sys.path:
        sys.path.insert(0, src_str)


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a VITA 49 capture into SigMF data/meta files.",
    )
    parser.add_argument("input_file", help="Path to input VITA 49 binary file")
    parser.add_argument(
        "output_prefix",
        nargs="?",
        help="Base path for SigMF outputs (default: input path without extension)",
    )
    parser.add_argument(
        "-n",
        "--max-packets",
        type=int,
        default=None,
        help="Maximum number of data packets to convert (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Target directory for SigMF outputs (default: directory inferred from output prefix or input file)",
    )
    return parser.parse_args(argv)


def _summarize_payload_format(pf: Optional["PayloadFormat"]) -> Optional[Dict[str, Any]]:
    if pf is None:
        return None
    info: Dict[str, Any] = {
        "packing_method": pf.packing_method.name,
        "sample_type": pf.sample_type.name,
        "item_packing_field_size_bits": int(pf.item_packing_field_size_bits),
        "data_item_size_bits": int(pf.data_item_size_bits),
        "data_item_format_code": int(pf.data_item_format_code),
        "repeat_count": int(pf.repeat_count),
        "vector_size": int(pf.vector_size),
    }
    if pf.data_item_format is not None:
        info["data_item_format"] = pf.data_item_format.name
    return info


def _build_sigmf_metadata(
    *,
    input_path: Path,
    total_samples: int,
    sample_rate: Optional[float],
    center_frequency: Optional[float],
    bandwidth: Optional[float],
    reference_level: Optional[float],
    stream_id: Optional[int],
    payload_format: Optional["PayloadFormat"],
    context_packets: int,
    data_packets: int,
    skipped_packets: int,
) -> Dict[str, Any]:
    global_meta: Dict[str, Any] = {
        "core:version": "1.1.0",
        "core:datatype": "cf32_le",
        "core:num_channels": 1,
        "core:description": f"Converted from {input_path.name} with vita_to_sigmf.py",
        "vita49:data_packets": data_packets,
        "vita49:context_packets": context_packets,
        "vita49:samples_written": total_samples,
        "vita49:packets_skipped": skipped_packets,
    }
    if sample_rate is not None:
        global_meta["core:sample_rate"] = sample_rate
    if stream_id is not None:
        global_meta["vita49:stream_id"] = f"0x{stream_id:08X}"
    if bandwidth is not None:
        global_meta["vita49:bandwidth_hz"] = bandwidth
    if reference_level is not None:
        global_meta["vita49:reference_level_dbm"] = reference_level
    payload_info = _summarize_payload_format(payload_format)
    if payload_info is not None:
        global_meta["vita49:payload_format"] = payload_info

    capture: Dict[str, Any] = {"core:sample_start": 0}
    if center_frequency is not None:
        capture["core:frequency"] = center_frequency

    return {
        "global": global_meta,
        "captures": [capture],
        "annotations": [],
    }


def convert_vita_to_sigmf(
    input_path: Path,
    output_prefix: Path,
    max_packets: Optional[int],
) -> Dict[str, Any]:
    """Stream IQ samples from input_path into SigMF outputs and return a summary."""
    _ensure_src_on_path()
    from vita49io.protocol.context_packet import ContextPacket
    from vita49io.protocol.data_packet import DataPacket
    from vita49io.io.packet_reader import PacketReader
    from vita49io.io.payload_codec import payload_as_numpy

    input_path = input_path.expanduser()
    output_prefix = output_prefix.expanduser()

    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_prefix = output_prefix.resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    data_path = output_prefix.with_suffix(".sigmf-data")
    meta_path = output_prefix.with_suffix(".sigmf-meta")

    total_samples = 0
    data_packets = 0
    context_packets = 0
    skipped_packets = 0
    limited = False

    last_payload_format: Optional["PayloadFormat"] = None
    payload_format_used: Optional["PayloadFormat"] = None
    sample_rate: Optional[float] = None
    center_frequency: Optional[float] = None
    bandwidth: Optional[float] = None
    reference_level: Optional[float] = None
    stream_id: Optional[int] = None

    dtype_le_c8 = np.dtype("<c8")

    with input_path.open("rb") as f, data_path.open("wb") as out_f:
        reader = PacketReader(f)
        packet_index = 0
        while True:
            pkt = reader.read_packet()
            
            if pkt is None:
                break

            if isinstance(pkt, ContextPacket):
                ctx = pkt
                context_packets += 1
                if ctx.stream_id is not None:
                    stream_id = ctx.stream_id
                if ctx.cif0 is not None:
                    cif0 = ctx.cif0
                    if cif0.payload_format is not None:
                        last_payload_format = cif0.payload_format
                    if cif0.sample_rate_hz is not None:
                        sample_rate = float(cif0.sample_rate_hz)
                    if cif0.rf_reference_frequency_hz is not None:
                        center_frequency = float(cif0.rf_reference_frequency_hz)
                    elif cif0.if_reference_frequency_hz is not None:
                        center_frequency = float(cif0.if_reference_frequency_hz)
                    if cif0.bandwidth_hz is not None:
                        bandwidth = float(cif0.bandwidth_hz)
                    if cif0.reference_level_dbm is not None:
                        reference_level = float(cif0.reference_level_dbm)
                packet_index += 1
                continue

            if isinstance(pkt, DataPacket):
                if last_payload_format is None:
                    skipped_packets += 1
                    packet_index += 1
                    continue
                payload = pkt.payload
                payload_bytes = payload.tobytes() if isinstance(payload, memoryview) else payload
                iq = payload_as_numpy(payload_bytes, last_payload_format)
                if iq.size > 0:
                    out_f.write(iq.astype(dtype_le_c8, copy=False).tobytes())
                    total_samples += int(iq.size)
                    payload_format_used = last_payload_format
                data_packets += 1
                packet_index += 1
                if max_packets is not None and data_packets >= max_packets:
                    limited = True
                    break
                continue

            skipped_packets += 1
            packet_index += 1
            continue

    if payload_format_used is None:
        payload_format_used = last_payload_format

    data_bytes = int(total_samples * dtype_le_c8.itemsize)

    meta = _build_sigmf_metadata(
        input_path=input_path,
        total_samples=total_samples,
        sample_rate=sample_rate,
        center_frequency=center_frequency,
        bandwidth=bandwidth,
        reference_level=reference_level,
        stream_id=stream_id,
        payload_format=payload_format_used,
        context_packets=context_packets,
        data_packets=data_packets,
        skipped_packets=skipped_packets,
    )
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    return {
        "data_path": data_path,
        "meta_path": meta_path,
        "total_samples": total_samples,
        "data_packets": data_packets,
        "context_packets": context_packets,
        "data_bytes": data_bytes,
        "limited": limited,
        "skipped_packets": skipped_packets,
    }


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    input_path = Path(args.input_file).expanduser()
    if not input_path.is_file():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 1

    if args.max_packets is not None and args.max_packets <= 0:
        print("--max-packets must be a positive integer", file=sys.stderr)
        return 2

    if args.output_prefix:
        output_prefix = Path(args.output_prefix).expanduser()
    else:
        output_prefix = input_path.with_suffix("")

    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser()
        output_prefix = output_dir / output_prefix.name

    try:
        summary = convert_vita_to_sigmf(input_path, output_prefix, args.max_packets)
    except Exception as exc:
        print(f"Conversion failed: {exc}", file=sys.stderr)
        return 1

    print(
        f"Wrote {summary['total_samples']} complex64 samples ({summary['data_bytes']} bytes) to {summary['data_path']}"
    )
    print(f"Wrote SigMF metadata to {summary['meta_path']}")
    print(
        f"Context packets seen: {summary['context_packets']}, data packets converted: {summary['data_packets']}"
    )
    print(f"Packets skipped: {summary['skipped_packets']}")
    if args.max_packets is not None and summary["limited"]:
        print(f"Stopped after reaching the configured data packet limit ({args.max_packets}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
