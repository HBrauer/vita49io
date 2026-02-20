#!/usr/bin/env python3
"""Live terminal dashboard for a VITA 49 TCP stream."""
from __future__ import annotations

import argparse
import os
import socket
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from vita49io.protocol.context_packet import ContextPacket


def _ensure_src_on_path() -> None:
    """Allow running from the repository root without installing."""
    here = os.path.dirname(__file__)
    src = os.path.normpath(os.path.join(here, "..", "src"))
    if os.path.isdir(src) and src not in sys.path:
        sys.path.insert(0, src)


@dataclass
class DashboardStats:
    rate_window: float
    packet_times: Deque[float] = field(default_factory=lambda: deque(maxlen=16384))
    total_packets: int = 0
    context_packets: int = 0
    data_packets: int = 0
    other_packets: int = 0
    start_time: float = field(default_factory=time.monotonic)
    last_context: Optional["ContextPacket"] = None
    status_message: Optional[str] = None

    def record_packet(self, now: float) -> None:
        self.total_packets += 1
        self.packet_times.append(now)
        self.trim(now)

    def trim(self, now: float) -> None:
        cutoff = max(self.start_time, now - self.rate_window)
        while self.packet_times and self.packet_times[0] < cutoff:
            self.packet_times.popleft()

    def packets_per_second(self, now: float) -> float:
        self.trim(now)
        if not self.packet_times:
            return 0.0
        elapsed = max(now - self.packet_times[0], 1e-9)
        return len(self.packet_times) / elapsed


def _format_context_lines(ctx: "ContextPacket") -> List[str]:
    lines: List[str] = []
    header = ctx.header
    tsi_name = getattr(header.tsi, "name", str(header.tsi))
    tsf_name = getattr(header.tsf, "name", str(header.tsf))

    if ctx.stream_id is None:
        lines.append("Stream ID: (missing)")
    else:
        lines.append(f"Stream ID: 0x{ctx.stream_id:08X} ({ctx.stream_id})")

    lines.append(f"TSI/TSF: {tsi_name} / {tsf_name}")

    if ctx.class_id:
        oui, info_class, pkt_class = ctx.class_id
        lines.append(
            f"Class ID: OUI=0x{oui:06X} info=0x{info_class:04X} pkt=0x{pkt_class:04X}"
        )

    if ctx.integer_seconds is not None:
        lines.append(f"Timestamp (int): {ctx.integer_seconds}")
    if ctx.fractional_seconds is not None:
        lines.append(f"Timestamp (frac): 0x{ctx.fractional_seconds:016X}")

    cif0 = ctx.cif0
    if cif0 is not None:
        if cif0.sample_rate_hz is not None:
            lines.append(f"Sample rate: {cif0.sample_rate_hz:,.3f} Hz")
        if cif0.reference_level_dbm is not None:
            lines.append(f"Reference level: {cif0.reference_level_dbm:.1f} dBm")
        if cif0.payload_format is not None:
            pf = cif0.payload_format
            fmt_name = (
                pf.data_item_format.name
                if pf.data_item_format is not None
                else f"code {pf.data_item_format_code}"
            )
            lines.append(
                "Payload: "
                f"{pf.sample_type.name}, {fmt_name}, "
                f"{pf.data_item_size_bits}-bit items, repeat {pf.repeat_count}"
            )
    return lines


def _format_duration(seconds: float) -> str:
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def render_dashboard(stats: DashboardStats, now: float, peer_display: str) -> None:
    pps = stats.packets_per_second(now)
    uptime = _format_duration(now - stats.start_time)
    status_line = (
        f"Status: {stats.status_message}"
        if stats.status_message
        else (
            "Status: Waiting for packets..."
            if stats.total_packets == 0
            else "Status: Receiving packets"
        )
    )

    sys.stdout.write("\x1b[2J\x1b[H")
    banner_len = max(32, len(peer_display) + 29)
    print(f"VITA 49 TCP Dashboard - {peer_display}")
    print("=" * banner_len)
    print(
        f"Uptime: {uptime}  |  Rate (last {stats.rate_window:.1f}s): "
        f"{pps:8.2f} pkt/s"
    )
    print(
        "Total packets: "
        f"{stats.total_packets:,}  "
        f"(context {stats.context_packets:,} | "
        f"data {stats.data_packets:,} | other {stats.other_packets:,})"
    )
    print(status_line)
    print()
    print("Last context packet:")
    if stats.last_context is None:
        print("  (waiting for context packet)")
    else:
        for line in _format_context_lines(stats.last_context):
            print(f"  {line}")
    sys.stdout.flush()


def consume_stream(
    sock: socket.socket,
    header_cls,
    packet_type_enum,
    context_cls,
    stats: DashboardStats,
    refresh_interval: float,
    peer_display: str,
    buffer_size: int,
) -> None:
    buffer = bytearray()
    last_render = 0.0
    render_dashboard(stats, time.monotonic(), peer_display)

    while True:
        try:
            chunk = sock.recv(buffer_size)
            if not chunk:
                stats.status_message = "Stream closed by peer."
                break
            buffer.extend(chunk)
        except socket.timeout:
            pass
        except OSError as exc:
            stats.status_message = f"Socket error: {exc}"
            break

        # Peel complete packets from the buffer before requesting more bytes.
        while len(buffer) >= 4:
            w0 = int.from_bytes(buffer[:4], byteorder="big")
            try:
                header = header_cls.parse(w0)
            except ValueError as exc:
                stats.status_message = f"Header parse error: {exc}"
                del buffer[:4]
                continue

            total_words = header.packet_size
            if total_words <= 0:
                stats.status_message = f"Invalid packet size: {total_words}"
                del buffer[:4]
                continue

            total_bytes = total_words * 4
            if len(buffer) < total_bytes:
                break

            packet_bytes = bytes(buffer[:total_bytes])
            del buffer[:total_bytes]

            now = time.monotonic()
            stats.record_packet(now)
            stats.status_message = None

            pkt_type = header.packet_type
            if pkt_type == packet_type_enum.CONTEXT_PACKET:
                stats.context_packets += 1
                try:
                    stats.last_context = context_cls.from_bytes(packet_bytes)
                except Exception as exc:
                    stats.status_message = f"Context parse error: {exc}"
            elif pkt_type in (
                packet_type_enum.IF_DATA_WITHOUT_STREAM_ID,
                packet_type_enum.IF_DATA_WITH_STREAM_ID,
                packet_type_enum.EXTENSION_DATA_WITHOUT_STREAM_ID,
                packet_type_enum.EXTENSION_DATA_WITH_STREAM_ID,
            ):
                stats.data_packets += 1
            else:
                stats.other_packets += 1

        now = time.monotonic()
        if now - last_render >= refresh_interval:
            render_dashboard(stats, now, peer_display)
            last_render = now


def _parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Connect to a VITA 49 TCP stream and show a live dashboard."
    )
    parser.add_argument(
        "host",
        nargs="?",
        default="127.0.0.1",
        help="TCP host to connect to (default: 127.0.0.1).",
    )
    parser.add_argument(
        "port",
        nargs="?",
        type=int,
        default=4991,
        help="TCP port to connect to (default: 4991).",
    )
    parser.add_argument(
        "--rate-window",
        type=float,
        default=5.0,
        metavar="SECONDS",
        help="Seconds of recent traffic to use when calculating packets/s (default: 5.0).",
    )
    parser.add_argument(
        "--refresh-interval",
        type=float,
        default=0.25,
        metavar="SECONDS",
        help="Seconds between dashboard refreshes (default: 0.25).",
    )
    parser.add_argument(
        "--connect-timeout",
        type=float,
        default=5.0,
        metavar="SECONDS",
        help="Connection timeout in seconds (default: 5.0).",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=4096,
        metavar="BYTES",
        help="Socket read size in bytes (default: 4096).",
    )
    args = parser.parse_args(argv)

    if not (0 < args.port < 65536):
        parser.error("port must be in the range 1-65535")
    if args.rate_window <= 0:
        parser.error("--rate-window must be greater than 0")
    if args.refresh_interval <= 0:
        parser.error("--refresh-interval must be greater than 0")
    if args.connect_timeout <= 0:
        parser.error("--connect-timeout must be greater than 0")
    if args.buffer_size <= 0:
        parser.error("--buffer-size must be greater than 0")

    return args


def main(argv: Optional[List[str]] = None) -> int:
    _ensure_src_on_path()
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    from vita49io.protocol.core import Header
    from vita49io.protocol.enums import PacketType
    from vita49io.protocol.context_packet import ContextPacket

    peer_display = f"{args.host}:{args.port}"
    stats = DashboardStats(rate_window=args.rate_window)
    refresh_interval = max(args.refresh_interval, 0.05)
    connected = False

    try:
        with socket.create_connection(
            (args.host, args.port), timeout=args.connect_timeout
        ) as sock:
            connected = True
            sock.settimeout(refresh_interval)
            consume_stream(
                sock=sock,
                header_cls=Header,
                packet_type_enum=PacketType,
                context_cls=ContextPacket,
                stats=stats,
                refresh_interval=refresh_interval,
                peer_display=peer_display,
                buffer_size=args.buffer_size,
            )
    except KeyboardInterrupt:
        if connected:
            stats.status_message = "Interrupted by user."
            render_dashboard(stats, time.monotonic(), peer_display)
        return 130
    except OSError as exc:
        print(f"Failed to connect to {peer_display}: {exc}")
        return 1

    if connected:
        render_dashboard(stats, time.monotonic(), peer_display)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())