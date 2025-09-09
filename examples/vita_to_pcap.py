#!/usr/bin/env python3
import sys
import os
from struct import unpack
from scapy.all import Ether, IP, UDP, Raw, wrpcap

def vita_to_pcap(input_path):
    output_path = input_path + ".pcap"
    packets = []

    # Example network wrapper config (adjust if you want different IP/ports)
    src_ip = "192.168.1.100"
    dst_ip = "192.168.1.200"
    sport = 4991
    dport = 4991

    with open(input_path, "rb") as f:
        while True:
            hdr = f.read(4)
            if not hdr:
                break
            if len(hdr) < 4:
                print("Warning: file ends with incomplete header")
                break

            word0, = unpack(">I", hdr)   # VITA is big-endian
            pkt_words = word0 & 0xFFFF   # lower 16 bits = packet size (in 32-bit words)
            total_bytes = pkt_words * 4

            if pkt_words < 4 or total_bytes < 4:
                print("Warning: invalid packet size, skipping...")
                continue

            rest = f.read(total_bytes - 4)
            if len(rest) != total_bytes - 4:
                print("Warning: file ends with incomplete packet")
                break

            payload = hdr + rest
            pkt = (
                Ether() /
                IP(src=src_ip, dst=dst_ip) /
                UDP(sport=sport, dport=dport) /
                Raw(payload)
            )
            packets.append(pkt)

    if packets:
        wrpcap(output_path, packets)
        print(f"Wrote {len(packets)} packets to {output_path}")
    else:
        print("No valid packets found, no file written")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {os.path.basename(sys.argv[0])} <input_file>")
        sys.exit(1)
    vita_to_pcap(sys.argv[1])
