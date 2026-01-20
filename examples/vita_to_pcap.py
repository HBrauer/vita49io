#!/usr/bin/env python3
import sys
import os
from scapy.all import Ether, IP, UDP, Raw, wrpcap

from vita49io.io.packet_reader import PacketReader

def vita_to_pcap(input_path):
    output_path = input_path + ".pcap"
    packets = []

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
