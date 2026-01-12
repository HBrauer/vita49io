"""Implement VITA 49 CIF3 helpers for temporal/environmental fields (Section 9.7, 9.9)."""

from __future__ import annotations

from dataclasses import dataclass
import struct
from enum import IntFlag
from typing import List, Tuple

from .utils import _u32


def _encode_fractional_time_fs(value: int) -> Tuple[int, int]:
    i = value & ((1 << 64) - 1)
    return _u32(i >> 32), _u32(i)


def _decode_fractional_time_fs(hi: int, lo: int) -> int:
    i = ((hi & 0xFFFFFFFF) << 32) | (lo & 0xFFFFFFFF)
    if i & (1 << 63):
        i -= 1 << 64
    return i


class CIF3Flags(IntFlag):
    """Bit positions for CIF3 presence mask (temporal/environmental).

    Attributes:
        NONE (int): No flags set.
        TIMESTAMP_DETAILS (int): Timestamp details present.
        TIMESTAMP_SKEW (int): Timestamp skew present.
        RISE_TIME (int): Rise time present.
        FALL_TIME (int): Fall time present.
        OFFSET_TIME (int): Offset time present.
        PULSE_WIDTH (int): Pulse width present.
        PERIOD (int): Period present.
        DURATION (int): Duration present.
        DWELL (int): Dwell present.
        JITTER (int): Jitter present.
        AGE (int): Age present.
        SHELF_LIFE (int): Shelf life present.
    """

    NONE = 0
    TIMESTAMP_DETAILS = 1 << 31
    TIMESTAMP_SKEW = 1 << 30
    RISE_TIME = 1 << 27
    FALL_TIME = 1 << 26
    OFFSET_TIME = 1 << 25
    PULSE_WIDTH = 1 << 24
    PERIOD = 1 << 23
    DURATION = 1 << 22
    DWELL = 1 << 21
    JITTER = 1 << 20
    AGE = 1 << 17
    SHELF_LIFE = 1 << 16


@dataclass
class CIF3Fields:
    """Represent the CIF3 temporal/environmental fields.

    Attributes:
        timestamp_details (Tuple[int, int] | None): Timestamp details words.
        timestamp_skew_fs (int | None): Timestamp skew in femtoseconds.
        rise_time_fs (int | None): Rise time in femtoseconds.
        fall_time_fs (int | None): Fall time in femtoseconds.
        offset_time_fs (int | None): Offset time in femtoseconds.
        pulse_width_fs (int | None): Pulse width in femtoseconds.
        period_fs (int | None): Period in femtoseconds.
        duration_fs (int | None): Duration in femtoseconds.
        dwell_fs (int | None): Dwell in femtoseconds.
        jitter_fs (int | None): Jitter in femtoseconds.
        age_word (int | None): Age word (raw 32-bit, TSI/TSF dependent).
        shelf_life_word (int | None): Shelf life word (raw 32-bit, TSI/TSF dependent).
    """

    timestamp_details: Tuple[int, int] | None = None
    timestamp_skew_fs: int | None = None
    rise_time_fs: int | None = None
    fall_time_fs: int | None = None
    offset_time_fs: int | None = None
    pulse_width_fs: int | None = None
    period_fs: int | None = None
    duration_fs: int | None = None
    dwell_fs: int | None = None
    jitter_fs: int | None = None
    age_word: int | None = None  # stored as raw 32-bit (TSI/TSF dependent)
    shelf_life_word: int | None = None  # stored as raw 32-bit (TSI/TSF dependent)

    SUPPORTED_MASK = (
        CIF3Flags.TIMESTAMP_DETAILS
        | CIF3Flags.TIMESTAMP_SKEW
        | CIF3Flags.RISE_TIME
        | CIF3Flags.FALL_TIME
        | CIF3Flags.OFFSET_TIME
        | CIF3Flags.PULSE_WIDTH
        | CIF3Flags.PERIOD
        | CIF3Flags.DURATION
        | CIF3Flags.DWELL
        | CIF3Flags.JITTER
        | CIF3Flags.AGE
        | CIF3Flags.SHELF_LIFE
    )

    def _presence_mask(self) -> int:
        m = CIF3Flags.NONE
        if self.timestamp_details is not None:
            m |= CIF3Flags.TIMESTAMP_DETAILS
        if self.timestamp_skew_fs is not None:
            m |= CIF3Flags.TIMESTAMP_SKEW
        if self.rise_time_fs is not None:
            m |= CIF3Flags.RISE_TIME
        if self.fall_time_fs is not None:
            m |= CIF3Flags.FALL_TIME
        if self.offset_time_fs is not None:
            m |= CIF3Flags.OFFSET_TIME
        if self.pulse_width_fs is not None:
            m |= CIF3Flags.PULSE_WIDTH
        if self.period_fs is not None:
            m |= CIF3Flags.PERIOD
        if self.duration_fs is not None:
            m |= CIF3Flags.DURATION
        if self.dwell_fs is not None:
            m |= CIF3Flags.DWELL
        if self.jitter_fs is not None:
            m |= CIF3Flags.JITTER
        if self.age_word is not None:
            m |= CIF3Flags.AGE
        if self.shelf_life_word is not None:
            m |= CIF3Flags.SHELF_LIFE
        return int(m)

    def pack(self) -> bytes:
        words: List[int] = []
        if self.timestamp_details is not None:
            words.extend([_u32(self.timestamp_details[0]), _u32(self.timestamp_details[1])])
        if self.timestamp_skew_fs is not None:
            hi, lo = _encode_fractional_time_fs(self.timestamp_skew_fs)
            words.extend([hi, lo])
        if self.rise_time_fs is not None:
            hi, lo = _encode_fractional_time_fs(self.rise_time_fs)
            words.extend([hi, lo])
        if self.fall_time_fs is not None:
            hi, lo = _encode_fractional_time_fs(self.fall_time_fs)
            words.extend([hi, lo])
        if self.offset_time_fs is not None:
            hi, lo = _encode_fractional_time_fs(self.offset_time_fs)
            words.extend([hi, lo])
        if self.pulse_width_fs is not None:
            hi, lo = _encode_fractional_time_fs(self.pulse_width_fs)
            words.extend([hi, lo])
        if self.period_fs is not None:
            hi, lo = _encode_fractional_time_fs(self.period_fs)
            words.extend([hi, lo])
        if self.duration_fs is not None:
            hi, lo = _encode_fractional_time_fs(self.duration_fs)
            words.extend([hi, lo])
        if self.dwell_fs is not None:
            hi, lo = _encode_fractional_time_fs(self.dwell_fs)
            words.extend([hi, lo])
        if self.jitter_fs is not None:
            hi, lo = _encode_fractional_time_fs(self.jitter_fs)
            words.extend([hi, lo])
        if self.age_word is not None:
            words.append(_u32(self.age_word))
        if self.shelf_life_word is not None:
            words.append(_u32(self.shelf_life_word))

        out = bytearray(len(words) * 4)
        for i, w in enumerate(words):
            struct.pack_into(">I", out, i * 4, _u32(w))
        return bytes(out)

    @staticmethod
    def parse_from_mask(mask: int, field_words: memoryview | bytes | bytearray) -> Tuple["CIF3Fields", int]:
        mv = field_words if isinstance(field_words, memoryview) else memoryview(field_words)
        if mv.format != "B":
            mv = mv.cast("B")
        if len(mv) % 4 != 0:
            raise ValueError("CIF3 payload must be a whole number of 32-bit words")

        flags = CIF3Flags(mask)
        unsupported = int(flags & ~CIF3Fields.SUPPORTED_MASK)
        if unsupported:
            raise ValueError(f"Unsupported CIF3 bits set: 0x{unsupported:08X}")

        idx_bytes = 0

        def take_words(n: int) -> List[int]:
            nonlocal idx_bytes
            end = idx_bytes + 4 * n
            if end > len(mv):
                raise ValueError("Truncated CIF3 payload")
            vals = list(struct.unpack_from(f">{n}I", mv, idx_bytes))
            idx_bytes = end
            return vals

        def take_time() -> int:
            hi, lo = take_words(2)
            return _decode_fractional_time_fs(hi, lo)

        ts_details = None
        ts_skew = None
        rise = fall = offset = None
        pulse_w = period = duration = None
        dwell = jitter = None
        age = shelf = None

        if flags & CIF3Flags.TIMESTAMP_DETAILS:
            vals = take_words(2)
            ts_details = (vals[0], vals[1])
        if flags & CIF3Flags.TIMESTAMP_SKEW:
            ts_skew = take_time()
        if flags & CIF3Flags.RISE_TIME:
            rise = take_time()
        if flags & CIF3Flags.FALL_TIME:
            fall = take_time()
        if flags & CIF3Flags.OFFSET_TIME:
            offset = take_time()
        if flags & CIF3Flags.PULSE_WIDTH:
            pulse_w = take_time()
        if flags & CIF3Flags.PERIOD:
            period = take_time()
        if flags & CIF3Flags.DURATION:
            duration = take_time()
        if flags & CIF3Flags.DWELL:
            dwell = take_time()
        if flags & CIF3Flags.JITTER:
            jitter = take_time()
        if flags & CIF3Flags.AGE:
            age = take_words(1)[0]
        if flags & CIF3Flags.SHELF_LIFE:
            shelf = take_words(1)[0]

        fields = CIF3Fields(
            timestamp_details=ts_details,
            timestamp_skew_fs=ts_skew,
            rise_time_fs=rise,
            fall_time_fs=fall,
            offset_time_fs=offset,
            pulse_width_fs=pulse_w,
            period_fs=period,
            duration_fs=duration,
            dwell_fs=dwell,
            jitter_fs=jitter,
            age_word=age,
            shelf_life_word=shelf,
        )

        return fields, idx_bytes // 4


__all__ = ["CIF3Flags", "CIF3Fields"]
