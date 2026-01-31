"""Read VITA 49 packets from any stream supporting ``read()``."""

from __future__ import annotations

from typing import Protocol

from ..protocol.core import Header
from ..protocol.context_packet import ContextPacket
from ..protocol.data_packet import DataPacket
from ..protocol.enums import PacketType

_DATA_PACKET_TYPES = (
    PacketType.IF_DATA_WITHOUT_STREAM_ID,
    PacketType.IF_DATA_WITH_STREAM_ID,
    PacketType.EXTENSION_DATA_WITHOUT_STREAM_ID,
    PacketType.EXTENSION_DATA_WITH_STREAM_ID,
)


class Readable(Protocol):
    """Protocol for stream-like objects that provide read()."""

    def read(self, n: int = ...) -> bytes: ...


class PacketReader:
    """Stateless reader that pulls a single packet from a stream."""

    def __init__(self, stream: Readable) -> None:
        self._stream = stream

    def read_packet(self) -> ContextPacket | DataPacket | None:
        """Read and parse the next packet from the stream."""
        w0_buf = _read_exact(self._stream, 4, allow_eof=True)
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
        if header.packet_type in _DATA_PACKET_TYPES:
            return DataPacket.from_bytes(memoryview(packet_bytes))

        raise ValueError(f"Unsupported packet type: {header.packet_type}")

    def skip_packets(self, n: int) -> int:
        """Skip the next n packets, returning the number actually skipped."""
        if n <= 0:
            return 0

        skipped = 0
        while skipped < n:
            header, w0_buf, total_bytes = _read_packet_header(self._stream)
            if header is None:
                break
            _skip_exact(self._stream, total_bytes - len(w0_buf))
            skipped += 1
        return skipped

    def skip_data_packets(self, n: int) -> int:
        """Skip packets until n data packets have been skipped."""
        if n <= 0:
            return 0

        skipped = 0
        while skipped < n:
            header, w0_buf, total_bytes = _read_packet_header(self._stream)
            if header is None:
                break
            _skip_exact(self._stream, total_bytes - len(w0_buf))
            if header.packet_type in _DATA_PACKET_TYPES:
                skipped += 1
        return skipped

    def skip_context_packets(self, n: int) -> int:
        """Skip packets until n context packets have been skipped."""
        if n <= 0:
            return 0

        skipped = 0
        while skipped < n:
            header, w0_buf, total_bytes = _read_packet_header(self._stream)
            if header is None:
                break
            _skip_exact(self._stream, total_bytes - len(w0_buf))
            if header.packet_type == PacketType.CONTEXT_PACKET:
                skipped += 1
        return skipped

    def skip_until_next_context_packet(self) -> ContextPacket | None:
        """Skip packets until the next context packet, returning it if found."""
        while True:
            header, w0_buf, total_bytes = _read_packet_header(self._stream)
            if header is None:
                return None
            if header.packet_type == PacketType.CONTEXT_PACKET:
                packet_bytes = bytearray(total_bytes)
                packet_bytes[:4] = w0_buf
                _read_exact_into(self._stream, memoryview(packet_bytes)[4:])
                return ContextPacket.from_bytes(memoryview(packet_bytes))
            _skip_exact(self._stream, total_bytes - len(w0_buf))

    def skip_until_next_data_packet(self) -> DataPacket | None:
        """Skip packets until the next data packet, returning it if found."""
        while True:
            header, w0_buf, total_bytes = _read_packet_header(self._stream)
            if header is None:
                return None
            if header.packet_type in _DATA_PACKET_TYPES:
                packet_bytes = bytearray(total_bytes)
                packet_bytes[:4] = w0_buf
                _read_exact_into(self._stream, memoryview(packet_bytes)[4:])
                return DataPacket.from_bytes(memoryview(packet_bytes))
            _skip_exact(self._stream, total_bytes - len(w0_buf))


def _read_exact(stream: Readable, n: int, *, allow_eof: bool = False) -> bytes:
    """Read exactly n bytes or raise if the stream is truncated.

    If allow_eof is True, return b"" when EOF is encountered before any bytes
    are read. Partial reads still raise.
    """
    if n == 0:
        return b""

    buf = bytearray(n)
    view = memoryview(buf)
    nread = _read_exact_into(stream, view, allow_eof=allow_eof)
    if allow_eof and nread == 0:
        return b""
    return bytes(buf)


def _read_packet_header(stream: Readable) -> tuple[Header | None, bytes, int]:
    """Read the next packet header, returning (header, w0_buf, total_bytes)."""
    w0_buf = _read_exact(stream, 4, allow_eof=True)
    if not w0_buf:
        return None, b"", 0
    if len(w0_buf) != 4:
        raise ValueError(f"Truncated header: expected 4 bytes, got {len(w0_buf)}")

    w0 = int.from_bytes(w0_buf, byteorder="big")
    header = Header.parse(w0)
    total_words = header.packet_size
    if total_words <= 0:
        raise ValueError(f"Invalid packet size (words): {total_words}")
    return header, w0_buf, total_words * 4


def _skip_exact(stream: Readable, n: int) -> None:
    """Read and discard exactly n bytes."""
    if n <= 0:
        return
    buf = bytearray(min(65536, n))
    view = memoryview(buf)
    remaining = n
    while remaining > 0:
        chunk = min(remaining, len(buf))
        _read_exact_into(stream, view[:chunk])
        remaining -= chunk


def _read_exact_into(stream: Readable, view: memoryview, *, allow_eof: bool = False) -> int:
    """Fill the view completely or raise if the stream is truncated.

    Returns the number of bytes read when allow_eof is True.
    """
    if not view:
        return 0

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
        if allow_eof and offset == 0:
            return 0
        raise ValueError(
            f"Truncated packet: expected {len(view)} bytes, got {len(view) - remaining}"
        )
    return len(view)
