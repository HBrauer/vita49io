from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Tuple, List


class PacketType(IntEnum):
    IF_DATA = 0
    IF_CONTEXT = 1
    EXT_DATA = 2
    EXT_CONTEXT = 3
    COMMAND = 4
    # 5-14 reserved, 15 = context assoc list (per some impls).


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
        if self.packet_type not in (PacketType.IF_DATA, PacketType.EXT_DATA):
            raise ValueError("DataPacket must have IF_DATA or EXT_DATA packet_type")

        words: List[int] = []
        w0 = 0
        w0 |= (int(self.packet_type) & 0xF) << 28
        if self.class_id is not None:
            w0 |= _HDR_C_MASK
        if self.trailer is not None:
            w0 |= _HDR_T_MASK
        if self.stream_id is not None:
            w0 |= _HDR_S_MASK
        w0 |= (int(self.tsi) & 0x3) << 22
        w0 |= (int(self.tsf) & 0x3) << 20
        w0 |= (self.packet_count & 0xF) << 16
        words.append(w0)

        if self.stream_id is not None:
            words.append(_u32(self.stream_id))

        if self.class_id is not None:
            oui, ic, pc = self.class_id
            words.append(_u32((oui & 0xFFFFFF) << 8))
            words.append(_u32(((ic & 0xFFFF) << 16) | (pc & 0xFFFF)))

        if self.tsi != TSI.NONE:
            if self.integer_seconds is None:
                raise ValueError("TSI set but integer_seconds is None")
            words.append(_u32(self.integer_seconds))

        if self.tsf != TSF.NONE:
            if self.fractional_seconds is None:
                raise ValueError("TSF set but fractional_seconds is None")
            words.append(_u32(self.fractional_seconds))

        payload = self.payload or b""
        if len(payload) % 4 != 0:
            payload += b"\x00" * (4 - (len(payload) % 4))
        for i in range(0, len(payload), 4):
            words.append(_unpack_u32_be(payload[i : i + 4]))

        if self.trailer is not None:
            words.append(_u32(self.trailer))

        words[0] = (words[0] & ~_HDR_PKT_SIZE_MASK) | (len(words) & _HDR_PKT_SIZE_MASK)

        out = bytearray()
        for w in words:
            out += _pack_u32_le(w)
        return bytes(out)

    @staticmethod
    def parse(data: bytes) -> "DataPacket":
        if len(data) < 4 or len(data) % 4 != 0:
            raise ValueError("Invalid VRT packet length")
        words = [_unpack_u32_be(data[i : i + 4]) for i in range(0, len(data), 4)]
        w0 = words[0]
        pkt_type = PacketType((w0 & _HDR_PACKET_TYPE_MASK) >> 28)
        if pkt_type not in (PacketType.IF_DATA, PacketType.EXT_DATA):
            raise ValueError("Not a Data packet type")

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
            stream_id = words[idx]; idx += 1

        if c_present:
            w_a = words[idx]; w_b = words[idx + 1]; idx += 2
            oui = (w_a >> 8) & 0xFFFFFF
            information_class = (w_b >> 16) & 0xFFFF
            packet_class = w_b & 0xFFFF
            class_id = (oui, information_class, packet_class)

        if tsi != TSI.NONE:
            integer_seconds = words[idx]; idx += 1
        if tsf != TSF.NONE:
            fractional_seconds = words[idx]; idx += 1

        trailer: Optional[int] = None
        end_idx = len(words)
        if t_present:
            trailer = words[-1]
            end_idx -= 1

        payload_words = words[idx:end_idx]
        payload_bytes = bytearray()
        for w in payload_words:
            payload_bytes += _pack_u32_le(w)

        return DataPacket(
            packet_type=pkt_type,
            stream_id=stream_id,
            class_id=class_id,
            tsi=tsi,
            tsf=tsf,
            integer_seconds=integer_seconds,
            fractional_seconds=fractional_seconds,
            payload=bytes(payload_bytes),
            trailer=trailer,
            packet_count=pkt_cnt,
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
        if self.packet_type not in (PacketType.IF_CONTEXT, PacketType.EXT_CONTEXT):
            raise ValueError("ContextPacket must have IF_CONTEXT or EXT_CONTEXT packet_type")

        words: List[int] = []
        w0 = 0
        w0 |= (int(self.packet_type) & 0xF) << 28
        if self.class_id is not None:
            w0 |= _HDR_C_MASK
        if self.trailer is not None:
            w0 |= _HDR_T_MASK
        if self.stream_id is not None:
            w0 |= _HDR_S_MASK
        w0 |= (int(self.tsi) & 0x3) << 22
        w0 |= (int(self.tsf) & 0x3) << 20
        w0 |= (self.packet_count & 0xF) << 16
        words.append(w0)

        if self.stream_id is not None:
            words.append(_u32(self.stream_id))

        if self.class_id is not None:
            oui, ic, pc = self.class_id
            words.append(_u32((oui & 0xFFFFFF) << 8))
            words.append(_u32(((ic & 0xFFFF) << 16) | (pc & 0xFFFF)))

        if self.tsi != TSI.NONE:
            if self.integer_seconds is None:
                raise ValueError("TSI set but integer_seconds is None")
            words.append(_u32(self.integer_seconds))

        if self.tsf != TSF.NONE:
            if self.fractional_seconds is None:
                raise ValueError("TSF set but fractional_seconds is None")
            words.append(_u32(self.fractional_seconds))

        payload = self.payload or b""
        if len(payload) % 4 != 0:
            payload += b"\x00" * (4 - (len(payload) % 4))
        for i in range(0, len(payload), 4):
            words.append(_unpack_u32_be(payload[i : i + 4]))

        if self.trailer is not None:
            words.append(_u32(self.trailer))

        words[0] = (words[0] & ~_HDR_PKT_SIZE_MASK) | (len(words) & _HDR_PKT_SIZE_MASK)

        out = bytearray()
        for w in words:
            out += _pack_u32_le(w)
        return bytes(out)

    @staticmethod
    def parse(data: bytes) -> "ContextPacket":
        if len(data) < 4 or len(data) % 4 != 0:
            raise ValueError("Invalid VRT packet length")
        words = [_unpack_u32_be(data[i : i + 4]) for i in range(0, len(data), 4)]
        w0 = words[0]
        pkt_type = PacketType((w0 & _HDR_PACKET_TYPE_MASK) >> 28)
        if pkt_type not in (PacketType.IF_CONTEXT, PacketType.EXT_CONTEXT):
            raise ValueError("Not a Context packet type")

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
            stream_id = words[idx]; idx += 1

        if c_present:
            w_a = words[idx]; w_b = words[idx + 1]; idx += 2
            oui = (w_a >> 8) & 0xFFFFFF
            information_class = (w_b >> 16) & 0xFFFF
            packet_class = w_b & 0xFFFF
            class_id = (oui, information_class, packet_class)

        if tsi != TSI.NONE:
            integer_seconds = words[idx]; idx += 1
        if tsf != TSF.NONE:
            fractional_seconds = words[idx]; idx += 1

        trailer: Optional[int] = None
        end_idx = len(words)
        if t_present:
            trailer = words[-1]
            end_idx -= 1

        payload_words = words[idx:end_idx]
        payload_bytes = bytearray()
        for w in payload_words:
            payload_bytes += _pack_u32_le(w)

        return ContextPacket(
            packet_type=pkt_type,
            stream_id=stream_id,
            class_id=class_id,
            tsi=tsi,
            tsf=tsf,
            integer_seconds=integer_seconds,
            fractional_seconds=fractional_seconds,
            payload=bytes(payload_bytes),
            trailer=trailer,
            packet_count=pkt_cnt,
        )
