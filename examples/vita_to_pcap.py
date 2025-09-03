#!/usr/bin/env python3
import sys
import os
import argparse
from struct import unpack
from scapy.all import Ether, IP, UDP, Raw, wrpcap


def vita_to_pcap(input_path: str, max_packets: int | None = None) -> None:
    """Convert a raw VITA49 capture file to a PCAP with UDP-encapsulated frames.

    - input_path: path to the .v49-like file containing concatenated VITA49 packets.
    - max_packets: if provided, stop after emitting at most this many UDP packets.
      Default is None (process all packets in the file).
    """
    output_path = input_path + ".pcap"
    packets: list = []

    # Example network wrapper config (adjust if you want different IP/ports)
    src_ip = "192.168.1.100"
    dst_ip = "192.168.1.200"
    sport = 5000
    dport = 5000

    emitted = 0
    with open(input_path, "rb") as f:
        while True:
            if max_packets is not None and emitted >= max_packets:
                break

            hdr = f.read(4)
            if not hdr:
                break
            if len(hdr) < 4:
                print("Warning: file ends with incomplete header")
                break

            (word0,) = unpack(">I", hdr)  # VITA is big-endian
            pkt_words = word0 & 0xFFFF  # lower 16 bits = packet size (in 32-bit words)
            total_bytes = pkt_words * 4

            if pkt_words < 4 or total_bytes < 4:
                print("Warning: invalid packet size, skipping...")
                continue

            rest = f.read(total_bytes - 4)
            if len(rest) != total_bytes - 4:
                print("Warning: file ends with incomplete packet")
                break

            payload = hdr + rest
            pkt = Ether() / IP(src=src_ip, dst=dst_ip) / UDP(sport=sport, dport=dport) / Raw(payload)
            packets.append(pkt)
            emitted += 1

    if packets:
        wrpcap(output_path, packets)
        print(f"Wrote {len(packets)} packets to {output_path}")
    else:
        print("No valid packets found, no file written")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Wrap VITA49 packets from a file into UDP and write to PCAP")
    p.add_argument("input_file", help="Path to input VITA49 binary file")
    p.add_argument(
        "-n",
        "--max-packets",
        type=int,
        default=None,
        help="Maximum number of packets to process (default: all)",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    vita_to_pcap(args.input_file, max_packets=args.max_packets)
