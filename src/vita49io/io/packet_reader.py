"""Read VITA 49 packets from any stream supporting ``read()``."""

from __future__ import annotations

from typing import Protocol

from ..protocol.core import Header
from ..protocol.context_packet import ContextPacket
from ..protocol.data_packet import DataPacket
from ..protocol.enums import PacketType


class Readable(Protocol):
    """Protocol for stream-like objects that provide read()."""

    def read(self, n: int = ...) -> bytes: ...


class PacketReader:
    """Stateless reader that pulls a single packet from a stream."""

    def __init__(self, stream: Readable) -> None:
        self._stream = stream

    def read_packet(self) -> ContextPacket | DataPacket | None:
        """Read and parse the next packet from the stream."""
        w0_buf = _read_exact(self._stream, 4)
        if not w0_buf:
            return None

        if len(w0_buf) != 4:
            raise ValueError(f"Truncated header: expected 4 bytes, got {len(w0_buf)}")

        w0 = int.from_bytes(w0_buf, byteorder="big")
        header = Header.parse(w0)
        total_words = header.packet_size
        if total_words <= 0:
            raise ValueError(f"Invalid packet size (words): {total_words}")

        total_bytes = total_words * 4
        packet_bytes = bytearray(total_bytes)
        packet_bytes[:4] = w0_buf
        _read_exact_into(self._stream, memoryview(packet_bytes)[4:])

        if header.packet_type == PacketType.CONTEXT_PACKET:
            return ContextPacket.from_bytes(memoryview(packet_bytes))
        if header.packet_type in (
            PacketType.IF_DATA_WITHOUT_STREAM_ID,
            PacketType.IF_DATA_WITH_STREAM_ID,
            PacketType.EXTENSION_DATA_WITHOUT_STREAM_ID,
            PacketType.EXTENSION_DATA_WITH_STREAM_ID,
        ):
            return DataPacket.from_bytes(memoryview(packet_bytes))

        raise ValueError(f"Unsupported packet type: {header.packet_type}")


def _read_exact(stream: Readable, n: int) -> bytes:
    """Read exactly n bytes or raise if the stream is truncated."""
    if n == 0:
        return b""

    buf = bytearray(n)
    view = memoryview(buf)
    _read_exact_into(stream, view)
    return bytes(buf)


def _read_exact_into(stream: Readable, view: memoryview) -> None:
    """Fill the view completely or raise if the stream is truncated."""
    if not view:
        return

    remaining = len(view)
    offset = 0
    readinto = getattr(stream, "readinto", None)
    if callable(readinto):
        while remaining > 0:
            nread = readinto(view[offset:])
            if not nread:
                break
            remaining -= nread
            offset += nread
    else:
        while remaining > 0:
            chunk = stream.read(remaining)
            if not chunk:
                break
            view[offset : offset + len(chunk)] = chunk
            remaining -= len(chunk)
            offset += len(chunk)

    if remaining != 0:
        raise ValueError(
            f"Truncated packet: expected {len(view)} bytes, got {len(view) - remaining}"
        )
