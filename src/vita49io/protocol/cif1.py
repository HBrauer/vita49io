"""Implement VITA 49 CIF1 helpers for spectral context fields (Spectrum field).

Only the Spectrum field (Section 9.6.1) is supported for now. The field is
present when CIF1 bit 10 is set and uses a fixed 13-word layout as defined in
the standard.
"""

from __future__ import annotations

from dataclasses import dataclass
import struct
from enum import IntEnum, IntFlag
from typing import List, Sequence, Tuple, Union

from .utils import _from_s64_fixed20, _to_s64_fixed20, _u32


class SpectrumType(IntEnum):
    """Enumerate values for the Spectrum Type bit-field (Table 9.6.1.1.1-1)."""

    DEFAULT = 0
    LOG_POWER_DB = 1
    CARTESIAN = 2
    POLAR = 3
    MAGNITUDE = 4


class AveragingType(IntFlag):
    """Bit-mapped averaging type (Table 9.6.1.1.2-1)."""

    NONE = 0
    LINEAR = 1
    PEAK_HOLD = 2
    MIN_HOLD = 4
    EXPONENTIAL = 8
    MEDIAN = 16
    SMOOTHING = 32


class WindowTimeDeltaInterpretation(IntEnum):
    """Interpretation of the Window Time-Delta field (Table 9.6.1.1.3-1)."""

    NOT_CONTROLLED = 0
    PERCENT = 1
    SAMPLES = 2
    TIME_NS = 3


class CIF1Flags(IntFlag):
    """Bit positions for CIF1 presence mask."""

    NONE = 0
    SPECTRUM = 1 << 10


def _decode_signed_32(raw: int) -> int:
    """Decode a 32-bit two's-complement value to Python int."""
    v = raw & 0xFFFFFFFF
    if v & 0x80000000:
        v -= 0x100000000
    return v


def _encode_signed_32(value: int) -> int:
    """Encode a Python int into 32-bit two's-complement."""
    return _u32(value & 0xFFFFFFFF)


def _decode_percent_overlap(raw: int) -> float:
    """Percent overlap uses signed 20.12 fixed-point."""
    return _decode_signed_32(raw) / float(1 << 12)


def _encode_percent_overlap(percent: float) -> int:
    scaled = int(round(percent * (1 << 12)))
    return _encode_signed_32(scaled)


def _pack_window_time_delta(
    value: Union[int, float], interpretation: WindowTimeDeltaInterpretation
) -> int:
    if interpretation is WindowTimeDeltaInterpretation.PERCENT:
        return _encode_percent_overlap(float(value))
    if interpretation is WindowTimeDeltaInterpretation.SAMPLES:
        return _u32(int(value))
    if interpretation is WindowTimeDeltaInterpretation.TIME_NS:
        return _u32(int(value))
    # NOT_CONTROLLED or reserved fall back to zero
    return 0


def _parse_window_time_delta(
    raw: int, interpretation: WindowTimeDeltaInterpretation
) -> Union[int, float]:
    if interpretation is WindowTimeDeltaInterpretation.PERCENT:
        return _decode_percent_overlap(raw)
    if interpretation is WindowTimeDeltaInterpretation.SAMPLES:
        return raw & 0xFFFFFFFF
    if interpretation is WindowTimeDeltaInterpretation.TIME_NS:
        return raw & 0xFFFFFFFF
    return raw & 0xFFFFFFFF


@dataclass
class SpectrumField:
    """Represent the CIF1 Spectrum field (fixed 13-word structure)."""

    spectrum_type: Union[SpectrumType, int] # Union because of possible user defined values
    averaging_type: Union[AveragingType, int] # Union because of possible user defined values
    window_time_delta_interpretation: WindowTimeDeltaInterpretation
    window_type: int
    num_transform_points: int
    num_window_points: int
    resolution_hz: float
    span_hz: float
    number_of_averages: int
    weighting_factor: float
    f1_index: int
    f2_index: int
    window_time_delta: Union[int, float]

    NUM_WORDS = 13

    def pack_words(self) -> List[int]:
        """Encode the Spectrum field into a list of 32-bit words."""
        w0 = 0
        w0 |= int(self.window_time_delta_interpretation) << 16
        w0 |= (int(self.averaging_type) & 0xFF) << 8
        w0 |= int(self.spectrum_type) & 0xFF

        res_hi, res_lo = _to_s64_fixed20(self.resolution_hz)
        span_hi, span_lo = _to_s64_fixed20(self.span_hz)
        weight_word = struct.unpack(">I", struct.pack(">f", float(self.weighting_factor)))[0]

        words = [
            _u32(w0),
            _u32(self.window_type),
            _u32(self.num_transform_points),
            _u32(self.num_window_points),
            _u32(res_hi),
            _u32(res_lo),
            _u32(span_hi),
            _u32(span_lo),
            _u32(self.number_of_averages),
            _u32(weight_word),
            _encode_signed_32(self.f1_index),
            _encode_signed_32(self.f2_index),
            _pack_window_time_delta(self.window_time_delta, self.window_time_delta_interpretation),
        ]
        return words

    @staticmethod
    def parse_words(words: Sequence[int]) -> "SpectrumField":
        """Decode a SpectrumField from 13 consecutive words."""
        if len(words) < SpectrumField.NUM_WORDS:
            raise ValueError("Truncated Spectrum field")
        w0 = words[0]
        spectrum_type_val = w0 & 0xFF
        try:
            spectrum_type: Union[SpectrumType, int] = SpectrumType(spectrum_type_val)
        except ValueError:
            spectrum_type = spectrum_type_val

        averaging_val = (w0 >> 8) & 0xFF
        averaging_type: Union[AveragingType, int]
        try:
            averaging_type = AveragingType(averaging_val)
        except ValueError:
            averaging_type = averaging_val

        interp_val = (w0 >> 16) & 0xF
        interpretation = WindowTimeDeltaInterpretation(interp_val)

        window_type = words[1] & 0xFFFFFFFF
        npoints = words[2] & 0xFFFFFFFF
        wnpoints = words[3] & 0xFFFFFFFF
        resolution_hz = _from_s64_fixed20(words[4], words[5])
        span_hz = _from_s64_fixed20(words[6], words[7])
        num_avgs = words[8] & 0xFFFFFFFF
        weight_word = words[9] & 0xFFFFFFFF
        weighting_factor = struct.unpack(">f", struct.pack(">I", weight_word))[0]
        f1 = _decode_signed_32(words[10])
        f2 = _decode_signed_32(words[11])
        window_delta = _parse_window_time_delta(words[12], interpretation)

        return SpectrumField(
            spectrum_type=spectrum_type,
            averaging_type=averaging_type,
            window_time_delta_interpretation=interpretation,
            window_type=window_type,
            num_transform_points=npoints,
            num_window_points=wnpoints,
            resolution_hz=resolution_hz,
            span_hz=span_hz,
            number_of_averages=num_avgs,
            weighting_factor=weighting_factor,
            f1_index=f1,
            f2_index=f2,
            window_time_delta=window_delta,
        )


@dataclass
class CIF1Fields:
    """Represent the CIF1 fields currently supported (Spectrum only)."""

    spectrum: SpectrumField | None = None

    SUPPORTED_MASK = int(CIF1Flags.SPECTRUM)

    def _presence_mask(self) -> int:
        m = CIF1Flags.NONE
        if self.spectrum is not None:
            m |= CIF1Flags.SPECTRUM
        return int(m)

    def pack(self) -> bytes:
        """Serialize CIF1 fields (without the mask word)."""
        words: List[int] = []
        if self.spectrum is not None:
            words.extend(self.spectrum.pack_words())
        out = bytearray(len(words) * 4)
        for i, w in enumerate(words):
            struct.pack_into(">I", out, i * 4, _u32(w))
        return bytes(out)

    @staticmethod
    def parse_from_mask(mask: int, field_words: memoryview | bytes | bytearray) -> Tuple["CIF1Fields", int]:
        """Parse CIF1 fields based on a mask and return (fields, words_consumed)."""
        mv = field_words if isinstance(field_words, memoryview) else memoryview(field_words)
        if mv.format != "B":
            mv = mv.cast("B")
        if len(mv) % 4 != 0:
            raise ValueError("CIF1 payload must be a whole number of 32-bit words")

        flags = CIF1Flags(mask)

        unsupported = int(flags & ~CIF1Flags.SPECTRUM)
        if unsupported:
            raise ValueError(f"Unsupported CIF1 bits set: 0x{unsupported:08X}")

        idx = 0  # byte index
        spectrum: SpectrumField | None = None
        if flags & CIF1Flags.SPECTRUM:
            needed = SpectrumField.NUM_WORDS * 4
            if idx + needed > len(mv):
                raise ValueError("Truncated Spectrum field")
            seg = struct.unpack_from(f">{SpectrumField.NUM_WORDS}I", mv, idx)
            spectrum = SpectrumField.parse_words(seg)
            idx += needed

        return CIF1Fields(spectrum=spectrum), idx // 4


__all__ = [
    "AveragingType",
    "CIF1Fields",
    "SpectrumField",
    "SpectrumType",
    "WindowTimeDeltaInterpretation",
    "CIF1Flags",
]
