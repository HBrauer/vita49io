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


# ----- CIF0 helpers -----

def _to_s64_fixed20(v: float) -> Tuple[int, int]:
    scale = 1 << 20
    i = int(round(v * scale)) & ((1 << 64) - 1)
    hi = (i >> 32) & 0xFFFFFFFF
    lo = i & 0xFFFFFFFF
    return hi, lo


def _from_s64_fixed20(hi: int, lo: int) -> float:
    i = ((hi & 0xFFFFFFFF) << 32) | (lo & 0xFFFFFFFF)
    if i & (1 << 63):
        i -= 1 << 64
    return i / float(1 << 20)


def _to_u64_fixed20(v: float) -> Tuple[int, int]:
    if v < 0:
        raise ValueError("Unsigned fixed-point cannot be negative")
    scale = 1 << 20
    i = int(round(v * scale)) & ((1 << 64) - 1)
    hi = (i >> 32) & 0xFFFFFFFF
    lo = i & 0xFFFFFFFF
    return hi, lo


def _from_u64_fixed20(hi: int, lo: int) -> float:
    i = ((hi & 0xFFFFFFFF) << 32) | (lo & 0xFFFFFFFF)
    return i / float(1 << 20)


def _to_s16_fixed7(v: float) -> int:
    scale = 1 << 7
    i = int(round(v * scale))
    if i < -32768:
        i = -32768
    if i > 32767:
        i = 32767
    return i & 0xFFFF


def _from_s16_fixed7(w: int) -> float:
    v = w & 0xFFFF
    if v & 0x8000:
        v -= 0x10000
    return v / float(1 << 7)


@dataclass
class CIF0Fields:
    # Bit 31
    context_field_change_indicator: Optional[int] = None  # 32-bit mask/value
    # Bit 30
    reference_point_identifier: Optional[int] = None  # 32-bit SID
    # Bits 29..25 (2 words each, s64 fp20)
    bandwidth_hz: Optional[float] = None
    if_reference_frequency_hz: Optional[float] = None
    rf_reference_frequency_hz: Optional[float] = None
    rf_reference_frequency_offset_hz: Optional[float] = None
    if_band_offset_hz: Optional[float] = None
    # Bit 24 (1 word, s16 fp7 in low 16 bits)
    reference_level_dbm: Optional[float] = None
    # Bit 23 (1 word, two s16 fp7)
    gain_db: Optional[Tuple[float, float]] = None  # (stage1, stage2)
    # Bit 22 (1 word u32)
    over_range_count: Optional[int] = None
    # Bit 21 (2 words u64 fp20)
    sample_rate_hz: Optional[float] = None
    # Bit 20 (2 words s64, LSB=1 fs)
    timestamp_adjustment_fs: Optional[int] = None
    # Bit 19 (1 word u32 seconds)
    timestamp_calibration_time_s: Optional[int] = None
    # Bit 18 (1 word s32 degrees C)
    temperature_c: Optional[int] = None
    # Bit 17 (2 words: OUI (24-bit) + device code (32-bit))
    device_identifier: Optional[Tuple[int, int]] = None  # (OUI24, device_code32)
    # Bit 16 (1 word mask)
    state_event_indicators: Optional[int] = None
    # Bit 15 (2 words raw)
    data_packet_payload_format: Optional[Tuple[int, int]] = None

    def _presence_mask(self) -> int:
        m = 0
        if self.context_field_change_indicator is not None:
            m |= 1 << 31
        if self.reference_point_identifier is not None:
            m |= 1 << 30
        if self.bandwidth_hz is not None:
            m |= 1 << 29
        if self.if_reference_frequency_hz is not None:
            m |= 1 << 28
        if self.rf_reference_frequency_hz is not None:
            m |= 1 << 27
        if self.rf_reference_frequency_offset_hz is not None:
            m |= 1 << 26
        if self.if_band_offset_hz is not None:
            m |= 1 << 25
        if self.reference_level_dbm is not None:
            m |= 1 << 24
        if self.gain_db is not None:
            m |= 1 << 23
        if self.over_range_count is not None:
            m |= 1 << 22
        if self.sample_rate_hz is not None:
            m |= 1 << 21
        if self.timestamp_adjustment_fs is not None:
            m |= 1 << 20
        if self.timestamp_calibration_time_s is not None:
            m |= 1 << 19
        if self.temperature_c is not None:
            m |= 1 << 18
        if self.device_identifier is not None:
            m |= 1 << 17
        if self.state_event_indicators is not None:
            m |= 1 << 16
        if self.data_packet_payload_format is not None:
            m |= 1 << 15
        return m

    def pack(self) -> bytes:
        words: List[int] = []
        words.append(self._presence_mask() & 0xFFFFFFFF)

        # Emit fields in descending bit order, 31 -> 15
        if self.context_field_change_indicator is not None:
            words.append(_u32(self.context_field_change_indicator))
        if self.reference_point_identifier is not None:
            words.append(_u32(self.reference_point_identifier))
        if self.bandwidth_hz is not None:
            hi, lo = _to_s64_fixed20(self.bandwidth_hz)
            words.extend([hi, lo])
        if self.if_reference_frequency_hz is not None:
            hi, lo = _to_s64_fixed20(self.if_reference_frequency_hz)
            words.extend([hi, lo])
        if self.rf_reference_frequency_hz is not None:
            hi, lo = _to_s64_fixed20(self.rf_reference_frequency_hz)
            words.extend([hi, lo])
        if self.rf_reference_frequency_offset_hz is not None:
            hi, lo = _to_s64_fixed20(self.rf_reference_frequency_offset_hz)
            words.extend([hi, lo])
        if self.if_band_offset_hz is not None:
            hi, lo = _to_s64_fixed20(self.if_band_offset_hz)
            words.extend([hi, lo])
        if self.reference_level_dbm is not None:
            w = _to_s16_fixed7(self.reference_level_dbm)
            words.append(w)
        if self.gain_db is not None:
            a, b = self.gain_db
            hi16 = _to_s16_fixed7(a)
            lo16 = _to_s16_fixed7(b)
            words.append(((hi16 & 0xFFFF) << 16) | (lo16 & 0xFFFF))
        if self.over_range_count is not None:
            words.append(_u32(self.over_range_count))
        if self.sample_rate_hz is not None:
            hi, lo = _to_u64_fixed20(self.sample_rate_hz)
            words.extend([hi, lo])
        if self.timestamp_adjustment_fs is not None:
            i = self.timestamp_adjustment_fs & ((1 << 64) - 1)
            words.extend([(i >> 32) & 0xFFFFFFFF, i & 0xFFFFFFFF])
        if self.timestamp_calibration_time_s is not None:
            words.append(_u32(self.timestamp_calibration_time_s))
        if self.temperature_c is not None:
            # Store as signed 32-bit integer degrees Celsius
            words.append(_u32(self.temperature_c & 0xFFFFFFFF))
        if self.device_identifier is not None:
            oui, dev = self.device_identifier
            words.extend([_u32(oui & 0xFFFFFF), _u32(dev)])
        if self.state_event_indicators is not None:
            words.append(_u32(self.state_event_indicators))
        if self.data_packet_payload_format is not None:
            w0, w1 = self.data_packet_payload_format
            words.extend([_u32(w0), _u32(w1)])

        return _payload_words_to_bytes(words)

    @staticmethod
    def parse(payload: bytes) -> Tuple["CIF0Fields", int]:
        """Parse CIF0 payload. Returns (cif0, bytes_consumed).

        Expects fields to be encoded in descending bit order (31 -> 15).
        """
        words = _payload_bytes_to_words(payload)
        if not words:
            raise ValueError("Empty payload for CIF0")
        mask = words[0]
        idx = 1

        def need(n: int):
            nonlocal idx
            if idx + n > len(words):
                raise ValueError("Truncated CIF0 payload")

        f = CIF0Fields()

        if mask & (1 << 31):
            need(1)
            f.context_field_change_indicator = words[idx]
            idx += 1
        if mask & (1 << 30):
            need(1)
            f.reference_point_identifier = words[idx]
            idx += 1
        if mask & (1 << 29):
            need(2)
            f.bandwidth_hz = _from_s64_fixed20(words[idx], words[idx + 1])
            idx += 2
        if mask & (1 << 28):
            need(2)
            f.if_reference_frequency_hz = _from_s64_fixed20(words[idx], words[idx + 1])
            idx += 2
        if mask & (1 << 27):
            need(2)
            f.rf_reference_frequency_hz = _from_s64_fixed20(words[idx], words[idx + 1])
            idx += 2
        if mask & (1 << 26):
            need(2)
            f.rf_reference_frequency_offset_hz = _from_s64_fixed20(words[idx], words[idx + 1])
            idx += 2
        if mask & (1 << 25):
            need(2)
            f.if_band_offset_hz = _from_s64_fixed20(words[idx], words[idx + 1])
            idx += 2
        if mask & (1 << 24):
            need(1)
            f.reference_level_dbm = _from_s16_fixed7(words[idx] & 0xFFFF)
            idx += 1
        if mask & (1 << 23):
            need(1)
            w = words[idx]
            a = _from_s16_fixed7((w >> 16) & 0xFFFF)
            b = _from_s16_fixed7(w & 0xFFFF)
            f.gain_db = (a, b)
            idx += 1
        if mask & (1 << 22):
            need(1)
            f.over_range_count = words[idx] & 0xFFFFFFFF
            idx += 1
        if mask & (1 << 21):
            need(2)
            f.sample_rate_hz = _from_u64_fixed20(words[idx], words[idx + 1])
            idx += 2
        if mask & (1 << 20):
            need(2)
            hi, lo = words[idx], words[idx + 1]
            i = ((hi & 0xFFFFFFFF) << 32) | (lo & 0xFFFFFFFF)
            if i & (1 << 63):
                i -= 1 << 64
            f.timestamp_adjustment_fs = i
            idx += 2
        if mask & (1 << 19):
            need(1)
            f.timestamp_calibration_time_s = words[idx] & 0xFFFFFFFF
            idx += 1
        if mask & (1 << 18):
            need(1)
            w = words[idx]
            if w & 0x80000000:
                w = (w - 0x100000000)  # s32
            f.temperature_c = w
            idx += 1
        if mask & (1 << 17):
            need(2)
            oui = words[idx] & 0xFFFFFF
            dev = words[idx + 1] & 0xFFFFFFFF
            f.device_identifier = (oui, dev)
            idx += 2
        if mask & (1 << 16):
            need(1)
            f.state_event_indicators = words[idx] & 0xFFFFFFFF
            idx += 1
        if mask & (1 << 15):
            need(2)
            f.data_packet_payload_format = (words[idx] & 0xFFFFFFFF, words[idx + 1] & 0xFFFFFFFF)
            idx += 2

        # Return bytes consumed (words*4)
        return f, idx * 4


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
    # Optional structured CIF0. If provided when packing, it takes precedence
    # over the raw `payload` field and will be encoded as the payload.
    cif0: Optional[CIF0Fields] = None
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
        payload_bytes = self.cif0.pack() if self.cif0 is not None else self.payload
        words.extend(_payload_bytes_to_words(payload_bytes))
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
            cif0=None,
            trailer=common.trailer,
            packet_count=common.packet_count,
        )
