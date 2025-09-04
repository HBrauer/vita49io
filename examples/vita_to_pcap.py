#!/usr/bin/env python3
import sys
import os
import argparse
from struct import unpack
from dataclasses import replace
from scapy.all import Ether, IP, UDP, Raw
from scapy.utils import PcapWriter
from typing import Optional, List


def _ensure_src_on_path() -> None:
    here = os.path.dirname(__file__)
    src = os.path.normpath(os.path.join(here, "..", "src"))
    if os.path.isdir(src) and src not in sys.path:
        sys.path.insert(0, src)


def vita_to_pcap(
    input_path: str,
    max_packets: int | None = None,
    src_ip: str = "192.168.1.100",
    dst_ip: str = "192.168.1.200",
    sport: int = 4991,
    dport: int = 4991,
    max_vrt_payload: Optional[int] = None,
) -> None:
    """Convert a raw VITA49 capture file to a PCAP with UDP-encapsulated frames.

    - input_path: path to the .v49-like file containing concatenated VITA49 packets.
    - max_packets: if provided, stop after emitting at most this many UDP packets.
      Default is None (process all packets in the file).
    - src_ip/dst_ip/sport/dport: network wrapper fields. Default UDP port 4991 to
      match Wireshark's VITA 49 dissector defaults.
    - max_vrt_payload: if set, splits oversized VITA IF Data packets into multiple
      VITA packets so each UDP payload stays under this size (and word-aligned).
      If None, keeps original sizes unless they exceed IPv4/UDP maximum.
    """
    output_path = input_path + ".pcap"

    # Max UDP payload for IPv4 without options: 65535 - 20 (IP) - 8 (UDP)
    MAX_UDP_PAYLOAD = 65507

    emitted = 0
    written = 0
    # Lazy import of VITA helpers (local src path if running from repo)
    _ensure_src_on_path()
    from vita49.core import Header
    from vita49.enums import PacketType
    from vita49.data_packet import DataPacket
    from vita49.context_packet import ContextPacket

    with open(input_path, "rb") as f, PcapWriter(output_path, append=False, sync=True) as pcap:
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

            # Evaluate whether we need to segment this VITA packet at the VRT layer
            need_segment = total_bytes > MAX_UDP_PAYLOAD

            rest = f.read(total_bytes - 4)
            if len(rest) != total_bytes - 4:
                print("Warning: file ends with incomplete packet")
                break

            payload = hdr + rest

            # Optionally segment IF Data packets so they fit into UDP comfortably
            if need_segment or (max_vrt_payload is not None and total_bytes > max_vrt_payload):
                h = Header.parse(word0)
                if h.packet_type in (
                    PacketType.IF_DATA_WITHOUT_STREAM_ID,
                    PacketType.IF_DATA_WITH_STREAM_ID,
                    PacketType.EXTENSION_DATA_WITHOUT_STREAM_ID,
                    PacketType.EXTENSION_DATA_WITH_STREAM_ID,
                ):
                    try:
                        dp = DataPacket.parse(payload)
                    except Exception as e:
                        print(f"Warning: failed to parse oversized IF Data packet at index {emitted}: {e}. Skipping.")
                        emitted += 1
                        continue

                    # Choose per-chunk payload limit (bytes), word-aligned
                    # If user didn't specify, default to a conservative 1400 to avoid IP fragmentation
                    limit = max_vrt_payload if max_vrt_payload is not None else 1400
                    limit &= ~0x3  # ensure 4-byte alignment
                    if limit <= 0:
                        limit = 1400

                    start_cnt = h.packet_count & 0xF
                    trailer_for_last = dp.trailer

                    data = dp.payload
                    offs = 0
                    seg_index = 0
                    while offs < len(data):
                        chunk = data[offs : offs + limit]
                        # Ensure 4-byte alignment so we don't pad mid-stream
                        clen = len(chunk) & ~0x3
                        if clen == 0:
                            # If near the end with <4 bytes remaining, take the rest here
                            chunk = data[offs:]
                            clen = len(chunk)
                        else:
                            chunk = chunk[:clen]

                        is_last = (offs + clen) >= len(data)
                        new_trailer = trailer_for_last if is_last else None
                        new_pc = (start_cnt + seg_index) & 0xF

                        dp_seg = DataPacket(
                            packet_type=dp.packet_type,
                            stream_id=dp.stream_id,
                            class_id=dp.class_id,
                            tsi=dp.tsi,
                            tsf=dp.tsf,
                            integer_seconds=dp.integer_seconds,
                            fractional_seconds=dp.fractional_seconds,
                            payload=chunk,
                            trailer=new_trailer,
                            packet_count=new_pc,
                            iq=None,
                        )
                        seg_bytes = dp_seg.pack()
                        if len(seg_bytes) > MAX_UDP_PAYLOAD:
                            print(
                                f"Warning: segment still too large ({len(seg_bytes)} bytes). Consider smaller --max-vrt-payload. Skipping this segment."
                            )
                        else:
                            pkt = (
                                Ether()
                                / IP(src=src_ip, dst=dst_ip)
                                / UDP(sport=sport, dport=dport)
                                / Raw(seg_bytes)
                            )
                            pcap.write(pkt)
                            written += 1
                        seg_index += 1
                        offs += clen

                    emitted += 1
                    continue
                else:
                    # Not a data packet (e.g., Context). We cannot safely segment without
                    # deeper knowledge, so skip if it violates UDP.
                    if total_bytes > MAX_UDP_PAYLOAD:
                        print(
                            f"Warning: Context/other VITA packet {emitted} size {total_bytes} exceeds UDP max; skipping."
                        )
                        emitted += 1
                        continue

            # Emit as-is
            pkt = Ether() / IP(src=src_ip, dst=dst_ip) / UDP(sport=sport, dport=dport) / Raw(payload)
            pcap.write(pkt)
            emitted += 1
            written += 1

    if written:
        print(f"Wrote {written} packets to {output_path}")
    else:
        print("No valid packets written; output file not created or empty")


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
    p.add_argument("--src-ip", default="192.168.1.100", help="Source IPv4 address (default: 192.168.1.100)")
    p.add_argument("--dst-ip", default="192.168.1.200", help="Destination IPv4 address (default: 192.168.1.200)")
    p.add_argument("--sport", type=int, default=4991, help="Source UDP port (default: 4991)")
    p.add_argument("--dport", type=int, default=4991, help="Destination UDP port (default: 4991)")
    p.add_argument(
        "--max-vrt-payload",
        type=int,
        default=None,
        help=(
            "If set, split IF Data into multiple VITA packets with at most this many payload bytes per packet (word-aligned)."
        ),
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    vita_to_pcap(
        args.input_file,
        max_packets=args.max_packets,
        src_ip=args.src_ip,
        dst_ip=args.dst_ip,
        sport=args.sport,
        dport=args.dport,
        max_vrt_payload=args.max_vrt_payload,
    )
