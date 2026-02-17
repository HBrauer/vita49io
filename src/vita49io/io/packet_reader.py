"""Read VITA 49 packets from any stream supporting ``read()``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..protocol.core import Header, WORD
from ..protocol.context_packet import ContextPacket
from ..protocol.data_packet import DataPacket
from ..protocol.enums import PacketType
from ..protocol.vrt_types import ClassID

_DATA_PACKET_TYPES = (
    PacketType.IF_DATA_WITHOUT_STREAM_ID,
    PacketType.IF_DATA_WITH_STREAM_ID,
    PacketType.EXTENSION_DATA_WITHOUT_STREAM_ID,
    PacketType.EXTENSION_DATA_WITH_STREAM_ID,
)
_DATA_PACKET_TYPE_VALUES = tuple(int(x) for x in _DATA_PACKET_TYPES)
_CONTEXT_PACKET_TYPE_VALUE = int(PacketType.CONTEXT_PACKET)
_DATA_PACKET_TYPE_WITH_STREAM_VALUES = (
    int(PacketType.IF_DATA_WITH_STREAM_ID),
    int(PacketType.EXTENSION_DATA_WITH_STREAM_ID),
)

_HDR_C_MASK = 0x08000000
_HDR_PSI_MASK = 0x07000000
_HDR_TSI_MASK = 0x00C00000
_HDR_TSF_MASK = 0x00300000
_HDR_PKT_CNT_MASK = 0x000F0000


@dataclass(slots=True)
class RawDataPacket:
    """Lightweight data-packet view for high-throughput streaming reads."""

    packet_type: int
    packet_count: int
    tsi: int
    tsf: int
    indicators_26: bool
    indicators_25: bool
    indicators_24: bool
    stream_id: int | None
    class_id: ClassID | None
    integer_seconds: int | None
    fractional_seconds: int | None
    payload: memoryview
    trailer: int | None


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

    def read_packet_fast(self) -> ContextPacket | RawDataPacket | None:
        """Read the next packet using a lightweight data-packet fast path.

        Returns ContextPacket for context packets and RawDataPacket for data packets.
        """
        w0_buf = _read_exact(self._stream, 4, allow_eof=True)
        if not w0_buf:
            return None
        if len(w0_buf) != 4:
            raise ValueError(f"Truncated header: expected 4 bytes, got {len(w0_buf)}")

        w0 = WORD.unpack_from(w0_buf, 0)[0]
        packet_type = (w0 >> 28) & 0xF
        total_words = w0 & 0xFFFF
        if total_words <= 0:
            raise ValueError(f"Invalid packet size (words): {total_words}")

        total_bytes = total_words * 4
        packet_bytes = bytearray(total_bytes)
        packet_bytes[:4] = w0_buf
        _read_exact_into(self._stream, memoryview(packet_bytes)[4:])
        mv = memoryview(packet_bytes)

        if packet_type == _CONTEXT_PACKET_TYPE_VALUE:
            return ContextPacket.from_bytes(mv)
        if packet_type not in _DATA_PACKET_TYPE_VALUES:
            raise ValueError(f"Unsupported packet type: {packet_type}")

        return _parse_raw_data_packet(mv, w0, packet_type)

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


def _parse_raw_data_packet(mv: memoryview, w0: int, packet_type: int) -> RawDataPacket:
    total_len = len(mv)
    idx = 4

    class_id_present = bool(w0 & _HDR_C_MASK)
    psi_bits = (w0 & _HDR_PSI_MASK) >> 24
    indicators_26 = bool(psi_bits & 0b100)
    indicators_25 = bool(psi_bits & 0b010)
    indicators_24 = bool(psi_bits & 0b001)
    tsi = (w0 & _HDR_TSI_MASK) >> 22
    tsf = (w0 & _HDR_TSF_MASK) >> 20
    packet_count = (w0 & _HDR_PKT_CNT_MASK) >> 16

    stream_id: int | None = None
    class_id: ClassID | None = None
    integer_seconds: int | None = None
    fractional_seconds: int | None = None

    if packet_type in _DATA_PACKET_TYPE_WITH_STREAM_VALUES:
        if idx + 4 > total_len:
            raise ValueError("Truncated packet: missing stream_id")
        stream_id = WORD.unpack_from(mv, idx)[0]
        idx += 4

    if class_id_present:
        if idx + 8 > total_len:
            raise ValueError("Truncated packet: missing class_id")
        w_a = WORD.unpack_from(mv, idx)[0]
        w_b = WORD.unpack_from(mv, idx + 4)[0]
        idx += 8
        oui = (w_a >> 8) & 0xFFFFFF
        information_class = (w_b >> 16) & 0xFFFF
        packet_class = w_b & 0xFFFF
        class_id = (oui, information_class, packet_class)

    if tsi != 0:
        if idx + 4 > total_len:
            raise ValueError("Truncated packet: missing integer_seconds")
        integer_seconds = WORD.unpack_from(mv, idx)[0]
        idx += 4

    if tsf != 0:
        if idx + 8 > total_len:
            raise ValueError("Truncated packet: missing fractional_seconds")
        msw = WORD.unpack_from(mv, idx)[0]
        lsw = WORD.unpack_from(mv, idx + 4)[0]
        fractional_seconds = ((msw & 0xFFFFFFFF) << 32) | (lsw & 0xFFFFFFFF)
        idx += 8

    payload_end = total_len - (4 if indicators_26 else 0)
    if payload_end < idx:
        raise ValueError("Truncated packet: invalid payload bounds")
    payload = mv[idx:payload_end]

    trailer = WORD.unpack_from(mv, payload_end)[0] if indicators_26 else None

    return RawDataPacket(
        packet_type=int(packet_type),
        packet_count=int(packet_count),
        tsi=int(tsi),
        tsf=int(tsf),
        indicators_26=indicators_26,
        indicators_25=indicators_25,
        indicators_24=indicators_24,
        stream_id=stream_id,
        class_id=class_id,
        integer_seconds=integer_seconds,
        fractional_seconds=fractional_seconds,
        payload=payload,
        trailer=trailer,
    )


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
