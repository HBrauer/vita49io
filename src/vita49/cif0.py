from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import IntEnum

from .core import _payload_bytes_to_words, _payload_words_to_bytes, _u32


# ---------------------------------------------
# Data Packet Payload Format (CIF0 bit 15)
# ---------------------------------------------


class PackingMethod(IntEnum):
    PROCESSING_EFFICIENT = 0
    LINK_EFFICIENT = 1


class SampleType(IntEnum):
    REAL = 0
    COMPLEX_CARTESIAN = 1  # I/Q
    COMPLEX_POLAR = 2


class DataItemFormat(IntEnum):
    # 00000: Signed fixed-point (two's-complement), normalized [-1, 1-2^-(N-1)]
    SIGNED_FIXED_POINT = 0b00000
    IEEE754_SINGLE = 0b01110  # 32-bit float
    # 10000: Unsigned fixed-point, normalized [0, 1-2^-N]
    UNSIGNED_FIXED_POINT = 0b10000

@dataclass
class PayloadFormat:
    # Word 1 fields (bit positions per spec)
    packing_method: PackingMethod  # 0=processing-efficient, 1=link-efficient
    sample_type: SampleType  # 0=real, 1=complex cartesian, 2=complex polar
    # Data item format: both raw 5-bit code and optional enum
    data_item_format_code: int  # 5-bit code (0..31)
    sample_component_repeat: bool  # if True, components repeat (I,I,... then Q,Q,...)
    event_tag_size_bits: int  # 0..7
    channel_tag_size_bits: int  # 0..15
    data_item_fraction_size_bits: int  # 0..15
    item_packing_field_size_bits: int  # decoded size in bits (1..64)
    data_item_size_bits: int  # decoded size in bits (1..64)

    # Word 2 fields
    repeat_count: int  # decoded (1..65536)
    vector_size: int  # decoded (1..65536)
    data_item_format: Optional["DataItemFormat"] = None

    @staticmethod
    def parse(w0: int, w1: int) -> "PayloadFormat":
        w0 &= 0xFFFFFFFF
        w1 &= 0xFFFFFFFF

        packing_method = PackingMethod((w0 >> 31) & 0x1)
        sample_type = SampleType((w0 >> 29) & 0x3)
        data_item_format_code = (w0 >> 24) & 0x1F
        # Try to map to enum; keep None if unknown
        try:
            data_item_format = DataItemFormat(data_item_format_code)
        except ValueError:
            raise ValueError(f"Unsuppported DataItemFormat code: {data_item_format_code}")
        sample_component_repeat = bool((w0 >> 23) & 0x1)
        event_tag_size_bits = (w0 >> 20) & 0x7
        channel_tag_size_bits = (w0 >> 16) & 0xF
        data_item_fraction_size_bits = (w0 >> 12) & 0xF
        item_packing_field_size_bits = ((w0 >> 6) & 0x3F) + 1  # stored as value-1
        data_item_size_bits = (w0 & 0x3F) + 1  # stored as value-1

        repeat_count = ((w1 >> 16) & 0xFFFF) 
        vector_size = (w1 & 0xFFFF) 

        return PayloadFormat(
            packing_method=packing_method,
            sample_type=sample_type,
            data_item_format_code=data_item_format_code,
            data_item_format=data_item_format,
            sample_component_repeat=sample_component_repeat,
            event_tag_size_bits=event_tag_size_bits,
            channel_tag_size_bits=channel_tag_size_bits,
            data_item_fraction_size_bits=data_item_fraction_size_bits,
            item_packing_field_size_bits=item_packing_field_size_bits,
            data_item_size_bits=data_item_size_bits,
            repeat_count=repeat_count,
            vector_size=vector_size,
        )

    def pack_words(self) -> Tuple[int, int]:
        # Encode back to two 32-bit words
        w0 = 0
        w0 |= (int(self.packing_method) & 0x1) << 31
        w0 |= (int(self.sample_type) & 0x3) << 29
        code = self.data_item_format_code
        if self.data_item_format is not None:
            code = int(self.data_item_format) & 0x1F
        w0 |= (code & 0x1F) << 24
        w0 |= (1 if self.sample_component_repeat else 0) << 23
        w0 |= (self.event_tag_size_bits & 0x7) << 20
        w0 |= (self.channel_tag_size_bits & 0xF) << 16
        w0 |= (self.data_item_fraction_size_bits & 0xF) << 12
        w0 |= ((max(1, self.item_packing_field_size_bits) - 1) & 0x3F) << 6
        w0 |= (max(1, self.data_item_size_bits) - 1) & 0x3F

        w1 = 0
        w1 |= ((max(1, self.repeat_count)) & 0xFFFF) << 16
        w1 |= (max(1, self.vector_size) ) & 0xFFFF

        return _u32(w0), _u32(w1)


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
    context_field_change_indicator: bool = False  
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
    # Bit 19 (1 word u32)
    timestamp_calibration_time_s: Optional[int] = None
    # Bit 18 (1 word s32 Celsius)
    temperature_c: Optional[int] = None
    # Bit 17 (2 words)
    device_identifier: Optional[Tuple[int, int]] = None  # (OUI 24-bit, device 32-bit)
    # Bit 16 (1 word u32)
    state_event_indicators: Optional[int] = None
    # Bit 15 (2 words payload format)
    data_packet_payload_format: Optional[Tuple[int, int]] = None
    # Decoded helper (not part of on-wire format). If provided when packing
    # and raw tuple is None, it will be used to generate the two words.
    payload_format: Optional[PayloadFormat] = None
    # Bit 1 (no additional words): CIF 1 Enable (flag)
    cif1_enable: Optional[bool] = None

    def _presence_mask(self) -> int:
        m = 0
        if self.context_field_change_indicator:
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
        if self.cif1_enable:
            m |= 1 << 1
        return m

    def pack(self) -> bytes:
        words: List[int] = []
        words.append(self._presence_mask() & 0xFFFFFFFF)

        # Emit fields in descending bit order, 31 -> 15
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
        if self.data_packet_payload_format is not None or self.payload_format is not None:
            if self.data_packet_payload_format is not None:
                w0, w1 = self.data_packet_payload_format
            else:
                w0, w1 = self.payload_format.pack_words()  # type: ignore[union-attr]
            words.extend([_u32(w0), _u32(w1)])
        # Bit 1 (CIF 1 Enable) adds no additional words

        return _payload_words_to_bytes(words)

    @staticmethod
    def parse_from_mask(mask: int, field_words: List[int]) -> Tuple["CIF0Fields", int]:
        """Parse CIF0 fields given a known presence mask and subsequent words.

        Returns (cif0, words_consumed_for_fields). Does not include the mask word.
        """

        # Reject unsupported lower bits explicitly requested by caller
        unsupported_bits = [14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 0]
        bad = [b for b in unsupported_bits if (mask >> b) & 1]
        if bad:
            raise ValueError(
                f"Unsupported CIF0 fields present (presence bits set): {bad} {bin(mask)}"
            )

        idx = 0

        def need(n: int):
            nonlocal idx
            if idx + n > len(field_words):
                raise ValueError("Truncated CIF0 payload")

        f = CIF0Fields()
        print(f"mask: {bin(mask)}")


        f.context_field_change_indicator = bool(mask & (1 << 31))
        if mask & (1 << 30):
            need(1)
            f.reference_point_identifier = field_words[idx]
            print(f'Reference point: {hex(f.reference_point_identifier)}')
            idx += 1
        if mask & (1 << 29):
            need(2)
            f.bandwidth_hz = _from_s64_fixed20(field_words[idx], field_words[idx + 1])
            idx += 2
        if mask & (1 << 28):
            need(2)
            f.if_reference_frequency_hz = _from_s64_fixed20(field_words[idx], field_words[idx + 1])
            idx += 2
        if mask & (1 << 27):
            need(2)
            f.rf_reference_frequency_hz = _from_s64_fixed20(field_words[idx], field_words[idx + 1])
            idx += 2
        if mask & (1 << 26):
            need(2)
            f.rf_reference_frequency_offset_hz = _from_s64_fixed20(field_words[idx], field_words[idx + 1])
            idx += 2
        if mask & (1 << 25):
            need(2)
            f.if_band_offset_hz = _from_s64_fixed20(field_words[idx], field_words[idx + 1])
            idx += 2
        if mask & (1 << 24):
            need(1)
            f.reference_level_dbm = _from_s16_fixed7(field_words[idx] & 0xFFFF)
            idx += 1
        if mask & (1 << 23):
            need(1)
            w = field_words[idx]
            a = _from_s16_fixed7((w >> 16) & 0xFFFF)
            b = _from_s16_fixed7(w & 0xFFFF)
            f.gain_db = (a, b)
            idx += 1
        if mask & (1 << 22):
            need(1)
            f.over_range_count = field_words[idx] & 0xFFFFFFFF
            idx += 1
        if mask & (1 << 21):
            need(2)
            f.sample_rate_hz = _from_u64_fixed20(field_words[idx], field_words[idx + 1])
            idx += 2
        if mask & (1 << 20):
            need(2)
            hi, lo = field_words[idx], field_words[idx + 1]
            i = ((hi & 0xFFFFFFFF) << 32) | (lo & 0xFFFFFFFF)
            if i & (1 << 63):
                i -= 1 << 64
            f.timestamp_adjustment_fs = i
            idx += 2
        if mask & (1 << 19):
            need(1)
            f.timestamp_calibration_time_s = field_words[idx] & 0xFFFFFFFF
            idx += 1
        if mask & (1 << 18):
            need(1)
            w = field_words[idx]
            if w & 0x80000000:
                w = (w - 0x100000000)  # s32
            f.temperature_c = w
            idx += 1
        if mask & (1 << 17):
            need(2)
            oui = field_words[idx] & 0xFFFFFF
            dev = field_words[idx + 1] & 0xFFFFFFFF
            f.device_identifier = (oui, dev)
            idx += 2
        if mask & (1 << 16):
            need(1)
            f.state_event_indicators = field_words[idx] & 0xFFFFFFFF
            idx += 1
        if mask & (1 << 15):
            need(2)
            w0 = field_words[idx] & 0xFFFFFFFF
            w1 = field_words[idx + 1] & 0xFFFFFFFF
            f.data_packet_payload_format = (w0, w1)
            # Also provide a decoded, user-friendly view
            try:
                f.payload_format = PayloadFormat.parse(w0, w1)
            except Exception:
                f.payload_format = None
            idx += 2
        # Bit 1: CIF 1 Enable (flag only)
        if mask & (1 << 1):
            f.cif1_enable = True

        return f, idx

    def parse(payload: bytes) -> Tuple["CIF0Fields", int]:
        """Parse CIF0 payload. Returns (cif0, bytes_consumed).

        Expects fields to be encoded in descending bit order (31 -> 15).
        """
        words = _payload_bytes_to_words(payload)
        if not words:
            raise ValueError("Empty payload for CIF0")
        mask = words[0]
        f, used_field_words = CIF0Fields.parse_from_mask(mask, words[1:])
        # Return bytes consumed including the mask word
        return f, (1 + used_field_words) * 4




__all__ = [
    "CIF0Fields",
    "PayloadFormat",
    "PackingMethod",
    "SampleType",
    "DataItemFormat",
]
