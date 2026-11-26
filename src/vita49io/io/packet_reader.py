"""Read VITA 49 packets from any stream supporting ``read()``.

Streams may contain bare VRT packets (VITA 49.0 / 49.2) or VRT packets wrapped
in VITA 49.1 VRL (VITA Radio Link Layer) frames. VRL framing is detected per
packet boundary and stripped transparently: the frame header (``VRLP`` frame
alignment word + frame count/size word) and the frame trailer (CRC-32 or the
``VEND`` no-CRC marker) are consumed and discarded, and the contained VRT
packets are returned exactly as if they had been stored without framing. The
CRC is not validated.
"""

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

# VITA 49.1 VRL frame constants.
VRL_FRAME_ALIGNMENT_WORD = 0x56524C50  # "VRLP"
VRL_NO_CRC_TRAILER = 0x56454E44  # "VEND"
_VRL_FAW_BYTES = b"VRLP"
_VRL_FRAME_SIZE_MASK = 0x000FFFFF
_VRL_FRAME_OVERHEAD_WORDS = 3  # 2 header words + 1 trailer word

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
        # Words left in the current VITA 49.1 VRL frame (packets + trailer).
        # 0 means "not inside a VRL frame".
        self._vrl_words_left = 0
        self.vrl_frame_count: int | None = None
        """Frame count of the most recent VRL frame, or None if none seen."""

    def _read_header_word(self) -> tuple[bytes, int, int]:
        """Read the next VRT header word, stripping any VITA 49.1 VRL framing.

        Returns (w0_buf, total_words). At EOF on a packet/frame boundary,
        returns (b"", 0).
        """
        stream = self._stream
        while True:
            if self._vrl_words_left == 1:
                # Only the frame trailer (CRC-32 or "VEND") remains; drop it.
                _read_exact(stream, 4)
                self._vrl_words_left = 0

            w0_buf = _read_exact(stream, 4, allow_eof=True)
            if not w0_buf:
                if self._vrl_words_left:
                    raise ValueError("Truncated VRL frame: EOF inside frame")
                return b"", 0

            if self._vrl_words_left == 0 and w0_buf == _VRL_FAW_BYTES:
                frame_word = WORD.unpack(_read_exact(stream, 4))[0]
                frame_words = frame_word & _VRL_FRAME_SIZE_MASK
                if frame_words < _VRL_FRAME_OVERHEAD_WORDS:
                    raise ValueError(f"Invalid VRL frame size (words): {frame_words}")
                self.vrl_frame_count = (frame_word >> 20) & 0xFFF
                # Remaining = contained packets + trailer word.
                self._vrl_words_left = frame_words - 2
                continue

            w0 = WORD.unpack(w0_buf)[0]
            total_words = w0 & 0xFFFF
            if total_words <= 0:
                raise ValueError(f"Invalid packet size (words): {total_words}")
            if self._vrl_words_left:
                if total_words > self._vrl_words_left - 1:
                    raise ValueError(
                        "VRT packet exceeds enclosing VRL frame: "
                        f"{total_words} words, {self._vrl_words_left - 1} left"
                    )
                self._vrl_words_left -= total_words
            return w0_buf, total_words

    def _read_packet_header(self) -> tuple[Header | None, bytes, int]:
        """Read the next packet header, returning (header, w0_buf, total_bytes)."""
        w0_buf, total_words = self._read_header_word()
        if not w0_buf:
            return None, b"", 0
        header = Header.parse(int.from_bytes(w0_buf, byteorder="big"))
        return header, w0_buf, total_words * 4

    def read_packet(self) -> ContextPacket | DataPacket | None:
        """Read and parse the next packet from the stream."""
        header, w0_buf, total_bytes = self._read_packet_header()
        if header is None:
            return None

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
        w0_buf, total_words = self._read_header_word()
        if not w0_buf:
            return None

        w0 = WORD.unpack_from(w0_buf, 0)[0]
        packet_type = (w0 >> 28) & 0xF

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
            header, w0_buf, total_bytes = self._read_packet_header()
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
            header, w0_buf, total_bytes = self._read_packet_header()
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
            header, w0_buf, total_bytes = self._read_packet_header()
            if header is None:
                break
            _skip_exact(self._stream, total_bytes - len(w0_buf))
            if header.packet_type == PacketType.CONTEXT_PACKET:
                skipped += 1
        return skipped

    def skip_until_next_context_packet(self) -> ContextPacket | None:
        """Skip packets until the next context packet, returning it if found."""
        while True:
            header, w0_buf, total_bytes = self._read_packet_header()
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
            header, w0_buf, total_bytes = self._read_packet_header()
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
