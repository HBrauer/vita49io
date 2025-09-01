from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Tuple, List


class PacketType(IntEnum):
    IF_DATA_WITHOUT_STREAM_ID = 0x0
    IF_DATA_WITH_STREAM_ID = 0x1
    EXTENSION_DATA_WITHOUT_STREAM_ID = 0x2
    EXTENSION_DATA_WITH_STREAM_ID = 0x3
    CONTEXT_PACKET = 0x4
    # 0x5–0xF reserved


class TSI(IntEnum):
    NONE = 0
    UTC = 1
    GPS = 2
    OTHER = 3


class TSF(IntEnum):
    NONE = 0
    SAMPLE_COUNT = 1  # Or Real-time fractional seconds depending on type
    FRACTIONAL = 2
    FREE_RUNNING = 3


# Header bit masks (based on common VITA 49.0 usage)
_HDR_PACKET_TYPE_MASK = 0xF0000000
_HDR_C_MASK = 0x08000000  # Class ID present
_HDR_T_MASK = 0x04000000  # Trailer present
_HDR_R_MASK = 0x02000000  # Reserved (must be 0)
_HDR_S_MASK = 0x01000000  # Stream ID present
_HDR_TSI_MASK = 0x00C00000  # 2 bits
_HDR_TSF_MASK = 0x00300000  # 2 bits
_HDR_PKT_CNT_MASK = 0x000F0000  # 4 bits
_HDR_PKT_SIZE_MASK = 0x0000FFFF  # 16 bits (32-bit words)


def _u32(v: int) -> int:
    return v & 0xFFFFFFFF


def _pack_u32_le(v: int) -> bytes:
    return _u32(v).to_bytes(4, byteorder="big")


def _unpack_u32_be(b: bytes) -> int:
    return int.from_bytes(b, byteorder="big")


ClassID = Tuple[int, int, int]  # (OUI 24-bit, Information Class 16-bit, Packet Class 16-bit)


# ----- Internal helpers to consolidate duplicate logic -----
@dataclass
class _Common:
    packet_type: PacketType
    stream_id: Optional[int]
    class_id: Optional[ClassID]
    tsi: TSI
    tsf: TSF
    integer_seconds: Optional[int]
    fractional_seconds: Optional[int]
    trailer: Optional[int]
    packet_count: int


def _pack_common_prefix(c: _Common) -> List[int]:
    words: List[int] = []
    w0 = 0
    w0 |= (int(c.packet_type) & 0xF) << 28
    if c.class_id is not None:
        w0 |= _HDR_C_MASK
    if c.trailer is not None:
        w0 |= _HDR_T_MASK
    if c.stream_id is not None:
        w0 |= _HDR_S_MASK
    w0 |= (int(c.tsi) & 0x3) << 22
    w0 |= (int(c.tsf) & 0x3) << 20
    w0 |= (c.packet_count & 0xF) << 16
    words.append(w0)

    if c.stream_id is not None:
        words.append(_u32(c.stream_id))

    if c.class_id is not None:
        oui, ic, pc = c.class_id
        words.append(_u32((oui & 0xFFFFFF) << 8))
        words.append(_u32(((ic & 0xFFFF) << 16) | (pc & 0xFFFF)))

    if c.tsi != TSI.NONE:
        if c.integer_seconds is None:
            raise ValueError("TSI set but integer_seconds is None")
        words.append(_u32(c.integer_seconds))

    if c.tsf != TSF.NONE:
        if c.fractional_seconds is None:
            raise ValueError("TSF set but fractional_seconds is None")
        words.append(_u32(c.fractional_seconds))

    return words


def _finalize_words_to_bytes(words: List[int]) -> bytes:
    words[0] = (words[0] & ~_HDR_PKT_SIZE_MASK) | (len(words) & _HDR_PKT_SIZE_MASK)
    out = bytearray()
    for w in words:
        out += _pack_u32_le(w)
    return bytes(out)


def _payload_bytes_to_words(payload: bytes) -> List[int]:
    data = payload or b""
    if len(data) % 4 != 0:
        data += b"\x00" * (4 - (len(data) % 4))
    return [_unpack_u32_be(data[i : i + 4]) for i in range(0, len(data), 4)]


def _payload_words_to_bytes(words: List[int]) -> bytes:
    b = bytearray()
    for w in words:
        b += _pack_u32_le(w)
    return bytes(b)


def _parse_common_from_words(words: List[int]) -> tuple[_Common, int, int]:
    if not words:
        raise ValueError("Empty words for VRT packet")
    w0 = words[0]
    pkt_type = PacketType((w0 & _HDR_PACKET_TYPE_MASK) >> 28)
    c_present = bool(w0 & _HDR_C_MASK)
    t_present = bool(w0 & _HDR_T_MASK)
    s_present = bool(w0 & _HDR_S_MASK)
    tsi = TSI((w0 & _HDR_TSI_MASK) >> 22)
    tsf = TSF((w0 & _HDR_TSF_MASK) >> 20)
    pkt_cnt = (w0 & _HDR_PKT_CNT_MASK) >> 16
    pkt_size_words = w0 & _HDR_PKT_SIZE_MASK

    if pkt_size_words != len(words):
        raise ValueError("Packet size mismatch")

    idx = 1
    stream_id: Optional[int] = None
    class_id: Optional[ClassID] = None
    integer_seconds: Optional[int] = None
    fractional_seconds: Optional[int] = None

    if s_present:
        if idx >= len(words):
            raise ValueError("Truncated after header: missing Stream ID")
        stream_id = words[idx]
        idx += 1

    if c_present:
        if idx + 1 >= len(words):
            raise ValueError("Truncated: missing Class ID words")
        w_a = words[idx]
        w_b = words[idx + 1]
        idx += 2
        oui = (w_a >> 8) & 0xFFFFFF
        information_class = (w_b >> 16) & 0xFFFF
        packet_class = w_b & 0xFFFF
        class_id = (oui, information_class, packet_class)

    if tsi != TSI.NONE:
        if idx >= len(words):
            raise ValueError("Truncated: missing integer seconds")
        integer_seconds = words[idx]
        idx += 1

    if tsf != TSF.NONE:
        if idx >= len(words):
            raise ValueError("Truncated: missing fractional seconds")
        fractional_seconds = words[idx]
        idx += 1

    trailer: Optional[int] = None
    end_idx = len(words)
    if t_present:
        trailer = words[-1]
        end_idx -= 1

    common = _Common(
        packet_type=pkt_type,
        stream_id=stream_id,
        class_id=class_id,
        tsi=tsi,
        tsf=tsf,
        integer_seconds=integer_seconds,
        fractional_seconds=fractional_seconds,
        trailer=trailer,
        packet_count=pkt_cnt,
    )
    return common, idx, end_idx


@dataclass
class DataPacket:
    packet_type: PacketType
    stream_id: Optional[int] = None
    class_id: Optional[ClassID] = None
    tsi: TSI = TSI.NONE
    tsf: TSF = TSF.NONE
    integer_seconds: Optional[int] = None
    fractional_seconds: Optional[int] = None
    payload: bytes = b""
    trailer: Optional[int] = None
    packet_count: int = 0

    def pack(self) -> bytes:
        if self.packet_type not in (
            PacketType.IF_DATA_WITHOUT_STREAM_ID,
            PacketType.IF_DATA_WITH_STREAM_ID,
            PacketType.EXTENSION_DATA_WITHOUT_STREAM_ID,
            PacketType.EXTENSION_DATA_WITH_STREAM_ID,
        ):
            raise ValueError(
                "DataPacket must be IF/EXT data (with/without Stream ID)"
            )

        # Enforce consistency between packet type and Stream ID presence
        if self.packet_type in (
            PacketType.IF_DATA_WITH_STREAM_ID,
            PacketType.EXTENSION_DATA_WITH_STREAM_ID,
        ) and self.stream_id is None:
            raise ValueError("Packet type requires a Stream ID, but none provided")
        if self.packet_type in (
            PacketType.IF_DATA_WITHOUT_STREAM_ID,
            PacketType.EXTENSION_DATA_WITHOUT_STREAM_ID,
        ) and self.stream_id is not None:
            raise ValueError("Packet type forbids a Stream ID, but one was provided")

        common = _Common(
            packet_type=self.packet_type,
            stream_id=self.stream_id,
            class_id=self.class_id,
            tsi=self.tsi,
            tsf=self.tsf,
            integer_seconds=self.integer_seconds,
            fractional_seconds=self.fractional_seconds,
            trailer=self.trailer,
            packet_count=self.packet_count,
        )
        words = _pack_common_prefix(common)
        words.extend(_payload_bytes_to_words(self.payload))
        if self.trailer is not None:
            words.append(_u32(self.trailer))
        return _finalize_words_to_bytes(words)

    @staticmethod
    def parse(data: bytes) -> "DataPacket":
        if len(data) < 4 or len(data) % 4 != 0:
            raise ValueError("Invalid VRT packet length")
        words = [_unpack_u32_be(data[i : i + 4]) for i in range(0, len(data), 4)]
        common, idx, end_idx = _parse_common_from_words(words)
        if common.packet_type not in (
            PacketType.IF_DATA_WITHOUT_STREAM_ID,
            PacketType.IF_DATA_WITH_STREAM_ID,
            PacketType.EXTENSION_DATA_WITHOUT_STREAM_ID,
            PacketType.EXTENSION_DATA_WITH_STREAM_ID,
        ):
            raise ValueError("Not a Data packet type")
        payload = _payload_words_to_bytes(words[idx:end_idx])
        return DataPacket(
            packet_type=common.packet_type,
            stream_id=common.stream_id,
            class_id=common.class_id,
            tsi=common.tsi,
            tsf=common.tsf,
            integer_seconds=common.integer_seconds,
            fractional_seconds=common.fractional_seconds,
            payload=payload,
            trailer=common.trailer,
            packet_count=common.packet_count,
        )


@dataclass
class ContextPacket:
    packet_type: PacketType
    stream_id: Optional[int] = None
    class_id: Optional[ClassID] = None
    tsi: TSI = TSI.NONE
    tsf: TSF = TSF.NONE
    integer_seconds: Optional[int] = None
    fractional_seconds: Optional[int] = None
    payload: bytes = b""
    trailer: Optional[int] = None
    packet_count: int = 0

    def pack(self) -> bytes:
        if self.packet_type is not PacketType.CONTEXT_PACKET:
            raise ValueError("ContextPacket must have CONTEXT_PACKET packet_type")

        common = _Common(
            packet_type=self.packet_type,
            stream_id=self.stream_id,
            class_id=self.class_id,
            tsi=self.tsi,
            tsf=self.tsf,
            integer_seconds=self.integer_seconds,
            fractional_seconds=self.fractional_seconds,
            trailer=self.trailer,
            packet_count=self.packet_count,
        )
        words = _pack_common_prefix(common)
        words.extend(_payload_bytes_to_words(self.payload))
        if self.trailer is not None:
            words.append(_u32(self.trailer))
        return _finalize_words_to_bytes(words)

    @staticmethod
    def parse(data: bytes) -> "ContextPacket":
        if len(data) < 4 or len(data) % 4 != 0:
            raise ValueError("Invalid VRT packet length")
        words = [_unpack_u32_be(data[i : i + 4]) for i in range(0, len(data), 4)]
        common, idx, end_idx = _parse_common_from_words(words)
        if common.packet_type is not PacketType.CONTEXT_PACKET:
            raise ValueError("Not a Context packet type")
        payload = _payload_words_to_bytes(words[idx:end_idx])
        return ContextPacket(
            packet_type=common.packet_type,
            stream_id=common.stream_id,
            class_id=common.class_id,
            tsi=common.tsi,
            tsf=common.tsf,
            integer_seconds=common.integer_seconds,
            fractional_seconds=common.fractional_seconds,
            payload=payload,
            trailer=common.trailer,
            packet_count=common.packet_count,
        )
