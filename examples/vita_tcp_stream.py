#!/usr/bin/env python3
"""Stream a VITA 49 capture file to TCP clients."""
from __future__ import annotations

import argparse
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Optional

from vita49io.io.packet_reader import PacketReader
from vita49io.protocol.cif0 import PayloadFormat, SampleType
from vita49io.protocol.context_packet import ContextPacket
from vita49io.protocol.core import Header
from vita49io.protocol.data_packet import DataPacket
from vita49io.protocol.enums import PacketType

DATA_PACKET_TYPES = {
    PacketType.IF_DATA_WITHOUT_STREAM_ID,
    PacketType.IF_DATA_WITH_STREAM_ID,
    PacketType.EXTENSION_DATA_WITHOUT_STREAM_ID,
    PacketType.EXTENSION_DATA_WITH_STREAM_ID,
}


@dataclass
class StreamTiming:
    """Track pacing metadata for a single VITA stream."""

    sample_rate_hz: Optional[float] = None
    payload_format: Optional[PayloadFormat] = None
    samples_override: Optional[int] = None
    cumulative_seconds: float = 0.0
    start_monotonic: Optional[float] = None
    warned_missing_timing: bool = False

    def reset_schedule(self) -> None:
        """Reset the pacing schedule while keeping configuration hints."""

        self.cumulative_seconds = 0.0
        self.start_monotonic = None

    def apply_sample_rate(self, value: Optional[float]) -> None:
        """Update the cached sample rate and reset pacing if it changed."""

        if value is None or value <= 0:
            return
        if self.sample_rate_hz != float(value):
            self.sample_rate_hz = float(value)
            self.reset_schedule()
            self.warned_missing_timing = False

    def apply_payload_format(self, payload_format: Optional[PayloadFormat]) -> None:
        """Update the payload format used to derive samples per packet."""

        if payload_format is None:
            return
        if self.payload_format != payload_format:
            self.payload_format = payload_format
            self.reset_schedule()
            self.warned_missing_timing = False


def iter_vita_packets(path: Path) -> Iterator[bytes]:
    """Yield raw VITA 49 packets from *path* in the order they appear."""
    index = 0
    with path.open('rb') as f:
        reader = PacketReader(f)
        while True:
            pkt = reader.read_packet()
            
            if pkt is None:
                break
            yield pkt.to_bytes()
            index += 1


def _stream_label(stream_id: Optional[int]) -> str:
    return f'0x{stream_id:08X}' if stream_id is not None else 'default'


def _components_per_sample(pf: PayloadFormat) -> Optional[int]:
    if pf.sample_type == SampleType.REAL:
        base = 1
    elif pf.sample_type in (SampleType.COMPLEX_CARTESIAN, SampleType.COMPLEX_POLAR):
        base = 2
    else:
        return None

    repeat = pf.repeat_count if pf.repeat_count else 1
    if repeat <= 0:
        repeat = 1
    vector = pf.vector_size if pf.vector_size else 1
    if vector <= 0:
        vector = 1
    return base * repeat * vector


def _estimate_samples(state: StreamTiming, payload_len: int) -> Optional[int]:
    if state.samples_override is not None:
        return state.samples_override

    pf = state.payload_format
    if pf is None:
        return None
    components = _components_per_sample(pf)
    if components is None or components <= 0:
        return None
    ipf_bits = pf.item_packing_field_size_bits
    if ipf_bits <= 0 or ipf_bits % 8 != 0:
        return None
    total_bits = payload_len * 8
    denom = ipf_bits * components
    if denom <= 0 or total_bits % denom != 0:
        return None
    return total_bits // denom


def _extract_payload_format(ctx: ContextPacket) -> Optional[PayloadFormat]:
    cif0 = ctx.cif0
    if cif0 is None:
        return None
    if cif0.payload_format is not None:
        return cif0.payload_format
    return None


def _compute_wait_time(
    state: StreamTiming, payload_len: int, stream_id: Optional[int]
) -> Optional[float]:
    stream_label = _stream_label(stream_id)
    if state.sample_rate_hz is None:
        if not state.warned_missing_timing:
            print(
                f'No sample rate known for stream {stream_label}; streaming at maximum speed.'
            )
            state.warned_missing_timing = True
        return None

    samples = _estimate_samples(state, payload_len)
    if samples is None or samples <= 0:
        if not state.warned_missing_timing:
            print(
                f'Unable to infer samples per packet for stream {stream_label}; '
                'streaming at maximum speed.'
            )
            state.warned_missing_timing = True
        return None

    now = time.monotonic()
    if state.start_monotonic is None:
        state.start_monotonic = now
        target = now
    else:
        target = state.start_monotonic + state.cumulative_seconds

    wait = target - now
    state.cumulative_seconds += samples / state.sample_rate_hz
    if wait <= 0:
        return 0.0
    return wait


def stream_vita_file(
    conn: socket.socket,
    path: Path,
    repeat: bool,
    sample_rate_hint: Optional[float],
    samples_per_packet_hint: Optional[int],
    enable_pacing: bool,
) -> int:
    """Send the contents of *path* to *conn*, pacing output when requested."""

    total_packets = 0
    states: Dict[Optional[int], StreamTiming] = {}

    def get_state(stream_id: Optional[int]) -> StreamTiming:
        state = states.get(stream_id)
        if state is None:
            state = StreamTiming(
                sample_rate_hz=(
                    sample_rate_hint if sample_rate_hint and sample_rate_hint > 0 else None
                ),
                samples_override=(
                    samples_per_packet_hint
                    if samples_per_packet_hint and samples_per_packet_hint > 0
                    else None
                ),
            )
            states[stream_id] = state
        return state

    while True:
        sent_this_pass = 0
        for state in states.values():
            state.reset_schedule()

        for packet in iter_vita_packets(path):
            header_word = int.from_bytes(packet[:4], byteorder='big')
            try:
                header = Header.parse(header_word)
            except ValueError:
                header = None

            if enable_pacing and header is not None and header.packet_type is PacketType.CONTEXT_PACKET:
                try:
                    ctx = ContextPacket.from_bytes(packet)
                except ValueError as exc:
                    print(f'Failed to parse context packet for pacing: {exc}')
                else:
                    state = get_state(ctx.stream_id)
                    if ctx.cif0 is not None:
                        state.apply_sample_rate(ctx.cif0.sample_rate_hz)
                    state.apply_payload_format(_extract_payload_format(ctx))
            elif enable_pacing and header is not None and header.packet_type in DATA_PACKET_TYPES:
                try:
                    data_pkt = DataPacket.from_bytes(packet)
                except ValueError as exc:
                    print(f'Failed to parse data packet for pacing: {exc}')
                else:
                    state = get_state(data_pkt.stream_id)
                    wait_time = _compute_wait_time(state, len(data_pkt.payload), data_pkt.stream_id)
                    if wait_time and wait_time > 0:
                        time.sleep(wait_time)

            conn.sendall(packet)
            sent_this_pass += 1

        total_packets += sent_this_pass
        if not repeat or sent_this_pass == 0:
            break

    return total_packets


def serve(
    path: Path,
    bind: str,
    port: int,
    repeat: bool,
    sample_rate_hint: Optional[float],
    samples_per_packet_hint: Optional[int],
    enable_pacing: bool,
) -> None:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((bind, port))
            server.listen(1)
            server.settimeout(1.0)
            loop_note = ' (looping)' if repeat else ''
            pace_note = ' with pacing' if enable_pacing else ' (fast mode)'
            print(f'Listening on {bind}:{port} - streaming {path}{loop_note}{pace_note}')
            print(f'Connect with e.g. "nc {bind} {port}" to receive raw VITA 49 packets.')
            print('Press Ctrl+C to stop the server.')
            while True:
                try:
                    conn, addr = server.accept()
                except socket.timeout:
                    continue
                except KeyboardInterrupt:
                    print()
                    print('Interrupted while waiting for a connection; shutting down.')
                    break
                print(f'Client connected: {addr[0]}:{addr[1]}')
                with conn:
                    try:
                        total_packets = stream_vita_file(
                            conn,
                            path,
                            repeat,
                            sample_rate_hint,
                            samples_per_packet_hint,
                            enable_pacing,
                        )
                        if total_packets:
                            print(f'Streamed {total_packets} packets to {addr[0]}:{addr[1]}')
                        else:
                            print(f'No packets found in {path}; nothing was sent.')
                    except KeyboardInterrupt:
                        print()
                        print('Interrupted during streaming; shutting down.')
                        return
                    except (BrokenPipeError, ConnectionResetError):
                        print(f'Client disconnected early: {addr[0]}:{addr[1]}')
                    except ValueError as exc:
                        print(f'Error reading {path}: {exc}')
                        return
                print('Connection closed. Waiting for the next client...')
    finally:
        print('Server stopped.')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description='Replay a VITA 49 capture file over TCP.'
    )
    parser.add_argument('path', help='Path to the VITA 49 file (e.g. .v49) to stream.')
    parser.add_argument('--bind', default='127.0.0.1', help='Address to bind the TCP server to (default: 127.0.0.1).')
    parser.add_argument('--port', type=int, default=4991, help='TCP port to listen on (default: 4991).')
    parser.add_argument('--loop', dest='repeat', action='store_true', help='Loop the file once it reaches the end.')
    parser.add_argument(
        '--sample-rate',
        type=float,
        default=None,
        help='Optional sample-rate hint in Hz used when the capture lacks context metadata.',
    )
    parser.add_argument(
        '--samples-per-packet',
        type=int,
        default=None,
        help='Override samples per data packet when context metadata is unavailable.',
    )
    parser.add_argument(
        '--fast',
        action='store_true',
        help='Disable pacing and stream data as fast as possible.',
    )
    args = parser.parse_args(argv)

    if args.sample_rate is not None and args.sample_rate <= 0:
        parser.error('--sample-rate must be positive.')
    if args.samples_per_packet is not None and args.samples_per_packet <= 0:
        parser.error('--samples-per-packet must be positive.')

    file_path = Path(args.path).expanduser()
    if not file_path.is_file():
        parser.error(f'File not found: {file_path}')
    try:
        serve(
            file_path,
            args.bind,
            args.port,
            args.repeat,
            args.sample_rate,
            args.samples_per_packet,
            enable_pacing=not args.fast,
        )
    except KeyboardInterrupt:
        print()
        print('Interrupted; shutting down.')
        return 130
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
