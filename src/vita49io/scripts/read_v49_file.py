from __future__ import annotations

import argparse
import sys
from typing import Optional


def read_all_packets(path: str):
    """Iterate all VITA 49 packets in a file.

    Uses the header to determine packet size and type and parses context packets
    (including CIF0). Sample decoding is handled separately via payload helpers.
    Yields (header, packet) tuples.
    """
    from vita49io.io.packet_reader import PacketReader
    with open(path, "rb") as f:
        reader = PacketReader(f)
        index = 0
        while True:
            pkt = reader.read_packet()
           
            if pkt is None:
                break  # EOF

            yield pkt.header, pkt

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
    from vita49io.protocol.context_packet import ContextPacket

    args = _parse_args(sys.argv[1:] if argv is None else argv)
    count = 0
    for header, pkt in read_all_packets(args.input_file):
        if args.context_only and not isinstance(pkt, ContextPacket):
            continue

        if isinstance(pkt, ContextPacket):
            # Force CIF0 decoding so __repr__ shows parsed content.
            _ = pkt.cif0

        print(header)
        print(pkt)
        count += 1
        if args.max_packets is not None and count >= args.max_packets:
            break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
