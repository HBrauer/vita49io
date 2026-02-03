#!/usr/bin/env python3
import sys
import os
import argparse
from scapy.all import Ether, IP, UDP, Raw, wrpcap

from vita49io.io.packet_reader import PacketReader

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a VITA 49 file to a PCAP capture",
    )
    parser.add_argument("input_file", help="Path to input VITA49 binary file")
    parser.add_argument(
        "-n",
        "--max-packets",
        type=int,
        default=None,
        help="Maximum number of packets to read (default: all)",
    )
    parser.add_argument(
        "--context-only",
        action="store_true",
        help="Only include context packets (others are skipped)",
    )
    return parser.parse_args(argv)


def vita_to_pcap(input_path, max_packets: int | None = None, context_only: bool = False):
    output_path = input_path + ".pcap"
    packets = []
    count = 0

    # Example network wrapper config (adjust if you want different IP/ports)
    src_ip = "192.168.1.100"
    dst_ip = "192.168.1.200"
    sport = 4991
    dport = 4991
    print(f"Reading VITA packets from {input_path}...")
    with open(input_path, "rb") as f:
        reader = PacketReader(f)
        while True:
            try:
                pkt_obj = reader.read_packet()
            except ValueError as exc:
                print(f"Warning: {exc}")
                break

            if pkt_obj is None:
                break

            if context_only and pkt_obj.header.packet_type.name != "CONTEXT_PACKET":
                continue

            header = pkt_obj.header
            word0 = header.pack()
            total_bytes = header.packet_size * 4
            print(f"word0 = 0x{word0:08X}")
            print(f"Read packet with size {header.packet_size} words ({total_bytes} bytes)")

            payload = pkt_obj.to_bytes()
            pkt = (
                Ether() /
                IP(src=src_ip, dst=dst_ip) /
                UDP(sport=sport, dport=dport) /
                Raw(payload)
            )
            packets.append(pkt)
            count += 1
            if max_packets is not None and count >= max_packets:
                break

    if packets:
        wrpcap(output_path, packets)
        print(f"Wrote {len(packets)} packets to {output_path}")
    else:
        print("No valid packets found, no file written")

if __name__ == "__main__":
    args = _parse_args(sys.argv[1:])
    vita_to_pcap(args.input_file, args.max_packets, args.context_only)
