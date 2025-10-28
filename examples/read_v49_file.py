from __future__ import annotations

import os
import sys
import argparse
from typing import Optional


def _ensure_src_on_path() -> None:
    # Allow running the example from the repo root without installation
    here = os.path.dirname(__file__)
    src = os.path.normpath(os.path.join(here, "..", "src"))
    if os.path.isdir(src) and src not in sys.path:
        sys.path.insert(0, src)


def read_all_packets(path: str):
    """Iterate all VITA 49 packets in a file.

    Uses the header to determine packet size and type, parses context packets
    (including CIF0) and remembers the last context payload format to decode
    subsequent data packets (I/Q extraction when compatible).
    Yields (header, packet) tuples.
    """
    from vita49io.protocol.core import Header
    from vita49io.protocol.enums import PacketType
    from vita49io.protocol.data_packet import DataPacket
    from vita49io.protocol.context_packet import ContextPacket
    from vita49io.protocol.cif0 import PayloadFormat

    last_payload_format: Optional[PayloadFormat] = None

    with open(path, "rb") as f:
        index = 0
        while True:
            w0_bytes = f.read(4)
            if not w0_bytes:
                break  # EOF
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
                # If ContextPacket already parsed CIF0, reuse it to update
                # the last known payload format for subsequent data packets.
                if pkt.cif0 is not None and pkt.cif0.payload_format is not None:
                    last_payload_format = pkt.cif0.payload_format
                yield header, pkt
            elif header.packet_type in (
                PacketType.IF_DATA_WITHOUT_STREAM_ID,
                PacketType.IF_DATA_WITH_STREAM_ID,
                PacketType.EXTENSION_DATA_WITHOUT_STREAM_ID,
                PacketType.EXTENSION_DATA_WITH_STREAM_ID,
            ):
                pkt = DataPacket.from_bytes(packet_bytes, payload_format=last_payload_format)
                yield header, pkt
            else:
                raise ValueError(f"Unsupported packet type at index {index}: {header.packet_type}")

            index += 1


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read and print VITA 49 packets from a file",
    )
    parser.add_argument("input_file", help="Path to input VITA49 binary file")
    parser.add_argument(
        '-n',
        '--max-packets',
        type=int,
        default=None,
        help='Maximum number of packets to read (default: all)',
    )
    parser.add_argument(
        '--context-only',
        action='store_true',
        help='Only print context packets (others are skipped)',
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    _ensure_src_on_path()

    from vita49io.protocol.context_packet import ContextPacket

    args = _parse_args(sys.argv[1:] if argv is None else argv)
    count = 0
    for header, pkt in read_all_packets(args.input_file):
        if args.context_only and not isinstance(pkt, ContextPacket):
            continue

        print(header)
        print(pkt)
        count += 1
        if args.max_packets is not None and count >= args.max_packets:
            break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
