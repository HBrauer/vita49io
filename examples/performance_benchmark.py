"""Benchmark VITA 49 encode/decode performance using the example files.

Run from the repo root:
    python examples/performance_benchmark.py

By default all `.v49` and `.vita49` files in `vita_example_files/` are used.
Use `--pattern` or `--limit` to narrow the run, and `--repeat` to average
multiple runs per file.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


MiB = 1024 * 1024


def _ensure_src_on_path() -> None:
    """Allow running the script from the repo root without installation."""
    here = Path(__file__).resolve().parent
    root = here.parent
    src = root / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


@dataclass
class BenchmarkResult:
    path: Path
    packets: int
    size_bytes: int
    decode_s: float
    encode_s: float
    read_s: Optional[float] = None

    @property
    def decode_mib_s(self) -> float:
        return (self.size_bytes / MiB) / self.decode_s if self.decode_s else float("inf")

    @property
    def encode_mib_s(self) -> float:
        return (self.size_bytes / MiB) / self.encode_s if self.encode_s else float("inf")

    @property
    def read_mib_s(self) -> Optional[float]:
        if self.read_s is None or self.read_s == 0:
            return None
        return (self.size_bytes / MiB) / self.read_s


def _iter_packets_from_bytes(buf: bytes) -> Iterable[object]:
    """Parse packets from a VITA 49 buffer."""
    from vita49io.protocol.core import Header
    from vita49io.protocol.enums import PacketType
    from vita49io.protocol.data_packet import DataPacket
    from vita49io.protocol.context_packet import ContextPacket
    offset = 0
    index = 0
    total_len = len(buf)

    while offset < total_len:
        if offset + 4 > total_len:
            raise ValueError(f"Truncated header at packet {index}: expected 4 bytes")

        w0_bytes = buf[offset : offset + 4]
        header = Header.parse(int.from_bytes(w0_bytes, byteorder="big"))
        total_bytes = header.packet_size * 4
        if total_bytes <= 0:
            raise ValueError(f"Invalid packet size at packet {index}: {total_bytes} bytes")
        if offset + total_bytes > total_len:
            raise ValueError(
                f"Truncated packet {index}: expected {total_bytes} bytes, got {total_len - offset}"
            )

        packet_bytes = buf[offset : offset + total_bytes]
        if header.packet_type == PacketType.CONTEXT_PACKET:
            pkt = ContextPacket.from_bytes(packet_bytes)
            yield pkt
        elif header.packet_type in (
            PacketType.IF_DATA_WITHOUT_STREAM_ID,
            PacketType.IF_DATA_WITH_STREAM_ID,
            PacketType.EXTENSION_DATA_WITHOUT_STREAM_ID,
            PacketType.EXTENSION_DATA_WITH_STREAM_ID,
        ):
            yield DataPacket.from_bytes(packet_bytes)
        else:
            raise ValueError(f"Unsupported packet type at index {index}: {header.packet_type}")

        offset += total_bytes
        index += 1


def _benchmark_file(path: Path, repeat: int, measure_read: bool = False) -> BenchmarkResult:
    read_s: Optional[float] = None
    if measure_read:
        read_start = time.perf_counter()
        raw = path.read_bytes()
        read_s = time.perf_counter() - read_start
    else:
        raw = path.read_bytes()

    size_bytes = len(raw)
    decode_acc = 0.0
    encode_acc = 0.0
    packet_count = 0

    for _ in range(repeat):
        decode_start = time.perf_counter()
        packets = list(_iter_packets_from_bytes(raw))
        decode_acc += time.perf_counter() - decode_start
        packet_count = len(packets)

        encode_start = time.perf_counter()
        encoded = b"".join(pkt.to_bytes() for pkt in packets)
        encode_acc += time.perf_counter() - encode_start

    decode_s = decode_acc / repeat
    encode_s = encode_acc / repeat

    return BenchmarkResult(
        path=path,
        packets=packet_count,
        size_bytes=size_bytes,
        decode_s=decode_s,
        encode_s=encode_s,
        read_s=read_s,
    )


def _bar(value: float, max_value: float, width: int) -> str:
    if not max_value or max_value <= 0:
        return "-" * width
    filled = int(round((value / max_value) * width))
    filled = min(width, max(filled, 0))
    return "#" * filled + "-" * (width - filled)


def _render_report(results: List[BenchmarkResult], examples_dir: Path, repeat: int) -> None:
    term_width = shutil.get_terminal_size((120, 40)).columns
    width = max(100, min(140, term_width))
    name_width = min(40, max(len("file"), max(len(r.path.name) for r in results)))

    total_bytes = sum(r.size_bytes for r in results)
    total_packets = sum(r.packets for r in results)
    total_decode_s = sum(r.decode_s for r in results)
    total_encode_s = sum(r.encode_s for r in results)
    total_decode_rate = (total_bytes / MiB) / total_decode_s if total_decode_s else float("inf")
    total_encode_rate = (total_bytes / MiB) / total_encode_s if total_encode_s else float("inf")

    print("=" * width)
    first_with_read = next((r for r in results if r.read_s is not None), None)
    if first_with_read is not None:
        read_rate = first_with_read.read_mib_s
        rate_txt = f" ({read_rate:.1f} MiB/s)" if read_rate is not None else ""
        print(
            f"File read time (first file only, not included in decode/encode): "
            f"{first_with_read.read_s * 1000.0:.1f} ms{rate_txt}".center(width)
        )
        print("=" * width)
    title = "VITA 49 Encode/Decode Benchmark"
    print(title.center(width))
    info = (
        f"files={len(results)} | packets={total_packets} | "
        f"data={total_bytes / MiB:.2f} MiB | repeat={repeat} | dir={examples_dir}"
    )
    print(info.center(width))
    print("=" * width)

    header = (
        f"{'file':<{name_width}} | "
        f"{'packets':>8} | "
        f"{'size (MiB)':>10} | "
        f"{'decode (ms)':>12} | "
        f"{'decode (MiB/s)':>15} | "
        f"{'encode (ms)':>12} | "
        f"{'encode (MiB/s)':>15}"
    )
    divider = "-" * len(header)
    print(header)
    print(divider)

    for res in results:
        decode_ms = res.decode_s * 1000.0
        encode_ms = res.encode_s * 1000.0
        row = (
            f"{res.path.name:<{name_width}} | "
            f"{res.packets:>8} | "
            f"{res.size_bytes / MiB:>10.2f} | "
            f"{decode_ms:>12.1f} | "
            f"{res.decode_mib_s:>15.1f} | "
            f"{encode_ms:>12.1f} | "
            f"{res.encode_mib_s:>15.1f}"
        )
        print(row)

    print(divider)
    total_row = (
        f"{'TOTAL':<{name_width}} | "
        f"{total_packets:>8} | "
        f"{total_bytes / MiB:>10.2f} | "
        f"{total_decode_s * 1000.0:>12.1f} | "
        f"{total_decode_rate:>15.1f} | "
        f"{total_encode_s * 1000.0:>12.1f} | "
        f"{total_encode_rate:>15.1f}"
    )
    print(total_row)

    max_decode = max(r.decode_mib_s for r in results)
    max_encode = max(r.encode_mib_s for r in results)
    bar_width = 18

    print("\nRelative throughput (MiB/s):")
    for res in results:
        decode_bar = _bar(res.decode_mib_s, max_decode, bar_width)
        encode_bar = _bar(res.encode_mib_s, max_encode, bar_width)
        print(
            f"{res.path.name:<{name_width}}  "
            f"dec [{decode_bar}] {res.decode_mib_s:>6.1f}  "
            f"enc [{encode_bar}] {res.encode_mib_s:>6.1f}"
        )

    print("=" * width)


def _collect_files(examples_dir: Path, pattern: Optional[str], limit: Optional[int]) -> List[Path]:
    suffixes = {".v49", ".vita49"}
    files = [
        p
        for p in examples_dir.iterdir()
        if p.is_file() and p.suffix.lower() in suffixes and (pattern is None or pattern in p.name)
    ]
    files.sort()
    if limit is not None:
        files = files[:limit]
    return files


def _parse_args(argv: Optional[list[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure decode/encode performance on vita_example_files.",
    )
    parser.add_argument(
        "--examples-dir",
        type=Path,
        default=None,
        help="Path to the vita_example_files directory (default: repo root / vita_example_files)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=None,
        help="Only include files containing this substring.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of files benchmarked after filtering.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Decode/encode the same file this many times and report the average.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    if args.repeat <= 0:
        raise SystemExit("repeat must be >= 1")

    _ensure_src_on_path()

    repo_root = Path(__file__).resolve().parent.parent
    examples_dir = args.examples_dir or (repo_root / "vita_example_files")
    examples_dir = examples_dir.resolve()

    if not examples_dir.is_dir():
        raise SystemExit(f"Could not find vita_example_files at {examples_dir}")

    files = _collect_files(examples_dir, args.pattern, args.limit)
    if not files:
        raise SystemExit("No matching .v49/.vita49 files found to benchmark.")

    print(f"Benchmarking {len(files)} file(s) from {examples_dir} (repeat={args.repeat})")

    results: List[BenchmarkResult] = []
    for idx, path in enumerate(files, start=1):
        prefix = f"[{idx}/{len(files)}] {path.name}"
        print(f"{prefix} ...", end="\r")
        res = _benchmark_file(path, args.repeat, measure_read=(idx == 1))
        results.append(res)
        read_part = (
            f"read={res.read_s * 1000.0:.1f} ms "
            if res.read_s is not None
            else ""
        )
        print(
            f"{prefix} {read_part}"
            f"decode={res.decode_mib_s:.1f} MiB/s "
            f"encode={res.encode_mib_s:.1f} MiB/s        "
        )

    _render_report(results, examples_dir, args.repeat)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
