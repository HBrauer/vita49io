from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from .core import _payload_bytes_to_words, _payload_words_to_bytes, _u32


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


__all__ = ["CIF0Fields"]

