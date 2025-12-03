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

from .utils import _from_s16_fixed7, _from_s64_fixed20, _to_s16_fixed7, _to_s64_fixed20, _u32


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
    PHASE = 1 << 31
    EB_NO_AND_BER = 1 << 20
    THRESHOLD = 1 << 19
    COMPRESSION_POINT = 1 << 18
    INTERCEPT_POINTS = 1 << 17
    SNR_AND_NOISE_FIGURE = 1 << 16
    AUX_FREQUENCY = 1 << 15
    AUX_GAIN = 1 << 14
    AUX_BANDWIDTH = 1 << 13
    ARRAY_OF_CIF = 1 << 11
    SPECTRUM = 1 << 10
    SECTOR_STEP_SCAN = 1 << 9
    ATTRIBUTES = 1 << 7
    DISCRETE_IO_32 = 1 << 6
    DISCRETE_IO_64 = 1 << 5
    HEALTH_STATUS = 1 << 4
    V49_SPEC_COMPLIANCE = 1 << 3
    BUILD_INFO = 1 << 2
    BUFFER_SIZE = 1 << 1


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


def _encode_fractional_time_fs(value: int) -> Tuple[int, int]:
    """Encode a signed 64-bit fractional time value (LSB = 1 fs)."""
    i = value & ((1 << 64) - 1)
    return _u32(i >> 32), _u32(i)


def _decode_fractional_time_fs(hi: int, lo: int) -> int:
    """Decode a signed 64-bit fractional time value (LSB = 1 fs)."""
    i = ((hi & 0xFFFFFFFF) << 32) | (lo & 0xFFFFFFFF)
    if i & (1 << 63):
        i -= 1 << 64
    return i


def _decode_u64(hi: int, lo: int) -> int:
    return ((hi & 0xFFFFFFFF) << 32) | (lo & 0xFFFFFFFF)


def _encode_u64(value: int) -> Tuple[int, int]:
    i = value & ((1 << 64) - 1)
    return _u32(i >> 32), _u32(i)


@dataclass
class BuildInformation:
    """Encode/decode the build information word (Section 9.10.4)."""

    year: int  # actual calendar year (e.g., 2025)
    day: int  # day of year (1..366)
    revision: int
    user_defined: int

    def pack_word(self) -> int:
        year_field = self.year - 2000 if self.year >= 2000 else self.year
        word = ((year_field & 0x7F) << 25) | ((self.day & 0x1FF) << 16)
        word |= (self.revision & 0x3F) << 10
        word |= self.user_defined & 0x3FF
        return _u32(word)

    @staticmethod
    def parse(word: int) -> "BuildInformation":
        year_field = (word >> 25) & 0x7F
        day = (word >> 16) & 0x1FF
        revision = (word >> 10) & 0x3F
        user_defined = word & 0x3FF
        return BuildInformation(year=year_field + 2000, day=day, revision=revision, user_defined=user_defined)


@dataclass
class BufferSizeField:
    """Represent the Buffer Size field (two-word structure)."""

    buffer_size_bytes: int
    level: int
    status: int

    def pack_words(self) -> List[int]:
        word0 = _u32(self.buffer_size_bytes)
        word1 = ((self.level & 0xFF) << 8) | (self.status & 0xFF)
        return [word0, _u32(word1)]

    @staticmethod
    def parse(words: Sequence[int]) -> "BufferSizeField":
        if len(words) < 2:
            raise ValueError("Truncated Buffer Size field")
        buffer_size = words[0] & 0xFFFFFFFF
        level = (words[1] >> 8) & 0xFF
        status = words[1] & 0xFF
        return BufferSizeField(buffer_size_bytes=buffer_size, level=level, status=status)


@dataclass
class SectorStepRecord:
    """Represent a single Sector/Step-Scan record."""

    sector_number: int
    f1_start_frequency_hz: float
    f2_stop_frequency_hz: float | None = None
    resolution_bandwidth_hz: float | None = None
    tune_step_size_hz: float | None = None
    number_of_points: int | None = None
    default_gain_db: Tuple[float, float] | None = None
    threshold_db: Tuple[float, float] | None = None
    dwell_time_fs: int | None = None
    start_time_fs: int | None = None
    time3_fs: int | None = None
    time4_fs: int | None = None


def _sector_indicator_for_record(rec: SectorStepRecord) -> int:
    indicator = 0
    indicator |= 1 << 31  # Sector Number is required
    indicator |= 1 << 30  # F1 Start Frequency is required
    if rec.f2_stop_frequency_hz is not None:
        indicator |= 1 << 29
    if rec.resolution_bandwidth_hz is not None:
        indicator |= 1 << 28
    if rec.tune_step_size_hz is not None:
        indicator |= 1 << 27
    if rec.number_of_points is not None:
        indicator |= 1 << 26
    if rec.default_gain_db is not None:
        indicator |= 1 << 25
    if rec.threshold_db is not None:
        indicator |= 1 << 24
    if rec.dwell_time_fs is not None:
        indicator |= 1 << 23
    if rec.start_time_fs is not None:
        indicator |= 1 << 22
    if rec.time3_fs is not None:
        indicator |= 1 << 21
    if rec.time4_fs is not None:
        indicator |= 1 << 20
    return indicator


_SECTOR_FIELD_WORD_LENGTHS = {
    31: 1,  # sector number
    30: 2,  # F1 start
    29: 2,  # F2 stop
    28: 2,  # resolution BW
    27: 2,  # tune step size
    26: 1,  # number of points
    25: 1,  # default gain
    24: 1,  # threshold
    23: 2,  # dwell time (fractional time)
    22: 2,  # start time (fractional time)
    21: 2,  # time3 (fractional time)
    20: 2,  # time4 (fractional time)
}


def _sector_record_word_count(indicator: int) -> int:
    count = 0
    for bit, words in _SECTOR_FIELD_WORD_LENGTHS.items():
        if indicator & (1 << bit):
            count += words
    return count


def _pack_gain_tuple(values: Tuple[float, float]) -> int:
    hi16 = _to_s16_fixed7(values[0])
    lo16 = _to_s16_fixed7(values[1])
    return ((hi16 & 0xFFFF) << 16) | (lo16 & 0xFFFF)


def _parse_gain_tuple(word: int) -> Tuple[float, float]:
    hi = _from_s16_fixed7((word >> 16) & 0xFFFF)
    lo = _from_s16_fixed7(word & 0xFFFF)
    return hi, lo


@dataclass
class SectorStepScanField:
    """Represent the Sector/Step-Scan field (array-of-records structure)."""

    records: List[SectorStepRecord]

    def _indicator(self) -> int:
        if not self.records:
            raise ValueError("SectorStepScanField requires at least one record")
        base = _sector_indicator_for_record(self.records[0])
        for rec in self.records[1:]:
            if _sector_indicator_for_record(rec) != base:
                raise ValueError("All sector/step records must use the same subfield selection")
        # Required bits must always be present
        if not (base & (1 << 31)) or not (base & (1 << 30)):
            raise ValueError("Sector number and F1 start frequency are required")
        return base

    def pack_words(self) -> List[int]:
        indicator = self._indicator()
        per_record = _sector_record_word_count(indicator)
        header_size = 3
        num_records = len(self.records)
        size_of_array = header_size + num_records * per_record
        header_word = ((header_size & 0xFF) << 24) | ((per_record & 0xFFF) << 12) | (num_records & 0xFFF)
        words: List[int] = [_u32(size_of_array), _u32(header_word), _u32(indicator)]

        for rec in self.records:
            words.extend(self._pack_record(rec, indicator))
        return words

    def _pack_record(self, rec: SectorStepRecord, indicator: int) -> List[int]:
        words: List[int] = []
        if indicator & (1 << 31):
            words.append(_u32(rec.sector_number))
        if indicator & (1 << 30):
            hi, lo = _to_s64_fixed20(rec.f1_start_frequency_hz)
            words.extend([_u32(hi), _u32(lo)])
        if indicator & (1 << 29):
            if rec.f2_stop_frequency_hz is None:
                raise ValueError("F2 stop frequency required by indicator")
            hi, lo = _to_s64_fixed20(rec.f2_stop_frequency_hz)
            words.extend([_u32(hi), _u32(lo)])
        if indicator & (1 << 28):
            if rec.resolution_bandwidth_hz is None:
                raise ValueError("Resolution bandwidth required by indicator")
            hi, lo = _to_s64_fixed20(rec.resolution_bandwidth_hz)
            words.extend([_u32(hi), _u32(lo)])
        if indicator & (1 << 27):
            if rec.tune_step_size_hz is None:
                raise ValueError("Tune step size required by indicator")
            hi, lo = _to_s64_fixed20(rec.tune_step_size_hz)
            words.extend([_u32(hi), _u32(lo)])
        if indicator & (1 << 26):
            if rec.number_of_points is None:
                raise ValueError("Number of points required by indicator")
            words.append(_u32(rec.number_of_points))
        if indicator & (1 << 25):
            if rec.default_gain_db is None:
                raise ValueError("Default gain required by indicator")
            words.append(_pack_gain_tuple(rec.default_gain_db))
        if indicator & (1 << 24):
            if rec.threshold_db is None:
                raise ValueError("Threshold required by indicator")
            words.append(_pack_gain_tuple(rec.threshold_db))
        if indicator & (1 << 23):
            if rec.dwell_time_fs is None:
                raise ValueError("Dwell time required by indicator")
            hi, lo = _encode_fractional_time_fs(rec.dwell_time_fs)
            words.extend([hi, lo])
        if indicator & (1 << 22):
            if rec.start_time_fs is None:
                raise ValueError("Start time required by indicator")
            hi, lo = _encode_fractional_time_fs(rec.start_time_fs)
            words.extend([hi, lo])
        if indicator & (1 << 21):
            if rec.time3_fs is None:
                raise ValueError("Time3 required by indicator")
            hi, lo = _encode_fractional_time_fs(rec.time3_fs)
            words.extend([hi, lo])
        if indicator & (1 << 20):
            if rec.time4_fs is None:
                raise ValueError("Time4 required by indicator")
            hi, lo = _encode_fractional_time_fs(rec.time4_fs)
            words.extend([hi, lo])
        return [_u32(w) for w in words]

    @staticmethod
    def parse(words: Sequence[int]) -> Tuple["SectorStepScanField", int]:
        if len(words) < 3:
            raise ValueError("Truncated Sector/Step-Scan field header")
        size_of_array = words[0]
        header_word = words[1]
        header_size = (header_word >> 24) & 0xFF
        num_words_per_record = (header_word >> 12) & 0xFFF
        num_records = header_word & 0xFFF
        indicator = words[2]

        if header_size < 3:
            raise ValueError("Invalid Sector/Step-Scan header size")
        expected_record_words = _sector_record_word_count(indicator)
        if expected_record_words != num_words_per_record:
            raise ValueError("Sector/Step-Scan header word count mismatch")

        total_words_needed = header_size + num_records * num_words_per_record
        if len(words) < total_words_needed or size_of_array != total_words_needed:
            raise ValueError("Truncated Sector/Step-Scan field")

        idx = header_size
        records: List[SectorStepRecord] = []
        for _ in range(num_records):
            rec_words = words[idx : idx + num_words_per_record]
            if len(rec_words) < num_words_per_record:
                raise ValueError("Truncated Sector/Step-Scan record")
            rec = _parse_sector_record(rec_words, indicator)
            records.append(rec)
            idx += num_words_per_record

        return SectorStepScanField(records=records), total_words_needed


def _parse_sector_record(rec_words: Sequence[int], indicator: int) -> SectorStepRecord:
    idx = 0

    def take(n: int) -> Tuple[int, ...]:
        nonlocal idx
        if idx + n > len(rec_words):
            raise ValueError("Truncated Sector/Step-Scan record content")
        vals = tuple(rec_words[idx : idx + n])
        idx += n
        return vals

    sector_number = f1_freq = None
    f2_freq = res_bw = tune_step = None
    num_points = None
    default_gain = threshold = None
    dwell_fs = start_fs = time3_fs = time4_fs = None

    if indicator & (1 << 31):
        sector_number = take(1)[0]
    if indicator & (1 << 30):
        hi, lo = take(2)
        f1_freq = _from_s64_fixed20(hi, lo)
    if indicator & (1 << 29):
        hi, lo = take(2)
        f2_freq = _from_s64_fixed20(hi, lo)
    if indicator & (1 << 28):
        hi, lo = take(2)
        res_bw = _from_s64_fixed20(hi, lo)
    if indicator & (1 << 27):
        hi, lo = take(2)
        tune_step = _from_s64_fixed20(hi, lo)
    if indicator & (1 << 26):
        num_points = take(1)[0] & 0xFFFFFFFF
    if indicator & (1 << 25):
        default_gain = _parse_gain_tuple(take(1)[0])
    if indicator & (1 << 24):
        threshold = _parse_gain_tuple(take(1)[0])
    if indicator & (1 << 23):
        hi, lo = take(2)
        dwell_fs = _decode_fractional_time_fs(hi, lo)
    if indicator & (1 << 22):
        hi, lo = take(2)
        start_fs = _decode_fractional_time_fs(hi, lo)
    if indicator & (1 << 21):
        hi, lo = take(2)
        time3_fs = _decode_fractional_time_fs(hi, lo)
    if indicator & (1 << 20):
        hi, lo = take(2)
        time4_fs = _decode_fractional_time_fs(hi, lo)

    if sector_number is None or f1_freq is None:
        raise ValueError("Sector/Step-Scan record missing required fields")

    return SectorStepRecord(
        sector_number=sector_number,
        f1_start_frequency_hz=f1_freq,
        f2_stop_frequency_hz=f2_freq,
        resolution_bandwidth_hz=res_bw,
        tune_step_size_hz=tune_step,
        number_of_points=num_points,
        default_gain_db=default_gain,
        threshold_db=threshold,
        dwell_time_fs=dwell_fs,
        start_time_fs=start_fs,
        time3_fs=time3_fs,
        time4_fs=time4_fs,
    )


@dataclass
class ArrayOfCifFields:
    """Opaque support for the Array of CIF Fields capability."""

    cif0_mask: int
    cif1_mask: int
    cif2_mask: int
    cif3_mask: int
    cif7_mask: int
    records: List[List[int]]
    header_size: int = 8

    def pack_words(self) -> List[int]:
        per_record_words = self._per_record_words()
        num_records = len(self.records)
        size_of_array = self.header_size + per_record_words * num_records
        header_word = ((self.header_size & 0xFF) << 24) | ((per_record_words & 0xFFF) << 12) | (num_records & 0xFFF)
        words: List[int] = [
            _u32(size_of_array),
            _u32(header_word),
            0,  # Bitmapped Control/Context indicator (always zero per spec)
            _u32(self.cif0_mask),
            _u32(self.cif1_mask),
            _u32(self.cif2_mask),
            _u32(self.cif3_mask),
            _u32(self.cif7_mask),
        ]
        for rec in self.records:
            if len(rec) != per_record_words:
                raise ValueError("Array-of-CIF record length mismatch")
            words.extend(_u32(w) for w in rec)
        return words

    def _per_record_words(self) -> int:
        if not self.records:
            return 0
        return len(self.records[0])

    @staticmethod
    def parse(words: Sequence[int]) -> Tuple["ArrayOfCifFields", int]:
        if len(words) < 8:
            raise ValueError("Truncated Array-of-CIF header")
        size_of_array = words[0]
        header_word = words[1]
        header_size = (header_word >> 24) & 0xFF
        per_record_words = (header_word >> 12) & 0xFFF
        num_records = header_word & 0xFFF
        if header_size < 7:
            raise ValueError("Invalid Array-of-CIF header size")
        total_needed = header_size + num_records * per_record_words
        if size_of_array != total_needed or len(words) < total_needed:
            raise ValueError("Truncated Array-of-CIF field")
        # Skip the 3rd word (bitmapped control/context indicator)
        cif0_mask = words[3]
        cif1_mask = words[4]
        cif2_mask = words[5]
        cif3_mask = words[6]
        cif7_mask = words[7]
        idx = header_size
        records: List[List[int]] = []
        for _ in range(num_records):
            rec = list(words[idx : idx + per_record_words])
            if len(rec) < per_record_words:
                raise ValueError("Truncated Array-of-CIF record")
            records.append(rec)
            idx += per_record_words
        return (
            ArrayOfCifFields(
                cif0_mask=cif0_mask,
                cif1_mask=cif1_mask,
                cif2_mask=cif2_mask,
                cif3_mask=cif3_mask,
                cif7_mask=cif7_mask,
                records=records,
                header_size=header_size,
            ),
            total_needed,
        )


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
    """Represent the CIF1 fields."""

    # Bit 31
    phase_radians: float | None = None
    # Bit 20
    eb_no_and_ber_db: Tuple[float, float] | None = None  # (Eb/No, BER), both in dB
    # Bit 19
    threshold_db: Tuple[float, float] | None = None
    # Bit 18
    compression_point_dbm: float | None = None
    # Bit 17
    intercept_points_dbm: Tuple[float, float] | None = None  # (2nd, 3rd order)
    # Bit 16
    snr_and_noise_figure_db: Tuple[float, float] | None = None  # (SNR, Noise Figure)
    # Bit 15
    aux_frequency_hz: float | None = None
    # Bit 14
    aux_gain_db: Tuple[float, float] | None = None
    # Bit 13
    aux_bandwidth_hz: float | None = None
    # Bit 11
    array_of_cif_fields: ArrayOfCifFields | None = None
    # Bit 10
    spectrum: SpectrumField | None = None
    # Bit 9
    sector_step_scan: SectorStepScanField | None = None
    # Bit 7
    attributes: int | None = None
    # Bits 6/5
    discrete_io_32: int | None = None
    discrete_io_64: int | None = None
    # Bit 4
    health_status: int | None = None
    # Bit 3
    v49_spec_compliance: int | None = None
    # Bit 2
    build_info: BuildInformation | None = None
    # Bit 1
    buffer_size: BufferSizeField | None = None

    SUPPORTED_MASK = int(
        CIF1Flags.PHASE
        | CIF1Flags.EB_NO_AND_BER
        | CIF1Flags.THRESHOLD
        | CIF1Flags.COMPRESSION_POINT
        | CIF1Flags.INTERCEPT_POINTS
        | CIF1Flags.SNR_AND_NOISE_FIGURE
        | CIF1Flags.AUX_FREQUENCY
        | CIF1Flags.AUX_GAIN
        | CIF1Flags.AUX_BANDWIDTH
        | CIF1Flags.ARRAY_OF_CIF
        | CIF1Flags.SPECTRUM
        | CIF1Flags.SECTOR_STEP_SCAN
        | CIF1Flags.ATTRIBUTES
        | CIF1Flags.DISCRETE_IO_32
        | CIF1Flags.DISCRETE_IO_64
        | CIF1Flags.HEALTH_STATUS
        | CIF1Flags.V49_SPEC_COMPLIANCE
        | CIF1Flags.BUILD_INFO
        | CIF1Flags.BUFFER_SIZE
    )

    def _presence_mask(self) -> int:
        m = CIF1Flags.NONE
        if self.phase_radians is not None:
            m |= CIF1Flags.PHASE
        if self.eb_no_and_ber_db is not None:
            m |= CIF1Flags.EB_NO_AND_BER
        if self.threshold_db is not None:
            m |= CIF1Flags.THRESHOLD
        if self.compression_point_dbm is not None:
            m |= CIF1Flags.COMPRESSION_POINT
        if self.intercept_points_dbm is not None:
            m |= CIF1Flags.INTERCEPT_POINTS
        if self.snr_and_noise_figure_db is not None:
            m |= CIF1Flags.SNR_AND_NOISE_FIGURE
        if self.aux_frequency_hz is not None:
            m |= CIF1Flags.AUX_FREQUENCY
        if self.aux_gain_db is not None:
            m |= CIF1Flags.AUX_GAIN
        if self.aux_bandwidth_hz is not None:
            m |= CIF1Flags.AUX_BANDWIDTH
        if self.array_of_cif_fields is not None:
            m |= CIF1Flags.ARRAY_OF_CIF
        if self.spectrum is not None:
            m |= CIF1Flags.SPECTRUM
        if self.sector_step_scan is not None:
            m |= CIF1Flags.SECTOR_STEP_SCAN
        if self.attributes is not None:
            m |= CIF1Flags.ATTRIBUTES
        if self.discrete_io_32 is not None:
            m |= CIF1Flags.DISCRETE_IO_32
        if self.discrete_io_64 is not None:
            m |= CIF1Flags.DISCRETE_IO_64
        if self.health_status is not None:
            m |= CIF1Flags.HEALTH_STATUS
        if self.v49_spec_compliance is not None:
            m |= CIF1Flags.V49_SPEC_COMPLIANCE
        if self.build_info is not None:
            m |= CIF1Flags.BUILD_INFO
        if self.buffer_size is not None:
            m |= CIF1Flags.BUFFER_SIZE
        return int(m)

    def pack(self) -> bytes:
        """Serialize CIF1 fields (without the mask word)."""
        words: List[int] = []
        if self.phase_radians is not None:
            words.append(_u32(_to_s16_fixed7(self.phase_radians)))
        if self.eb_no_and_ber_db is not None:
            words.append(_pack_gain_tuple(self.eb_no_and_ber_db))
        if self.threshold_db is not None:
            words.append(_pack_gain_tuple(self.threshold_db))
        if self.compression_point_dbm is not None:
            words.append(_u32(_to_s16_fixed7(self.compression_point_dbm)))
        if self.intercept_points_dbm is not None:
            words.append(_pack_gain_tuple(self.intercept_points_dbm))
        if self.snr_and_noise_figure_db is not None:
            words.append(_pack_gain_tuple(self.snr_and_noise_figure_db))
        if self.aux_frequency_hz is not None:
            hi, lo = _to_s64_fixed20(self.aux_frequency_hz)
            words.extend([_u32(hi), _u32(lo)])
        if self.aux_gain_db is not None:
            words.append(_pack_gain_tuple(self.aux_gain_db))
        if self.aux_bandwidth_hz is not None:
            hi, lo = _to_s64_fixed20(self.aux_bandwidth_hz)
            words.extend([_u32(hi), _u32(lo)])
        if self.array_of_cif_fields is not None:
            words.extend(self.array_of_cif_fields.pack_words())
        if self.spectrum is not None:
            words.extend(self.spectrum.pack_words())
        if self.sector_step_scan is not None:
            words.extend(self.sector_step_scan.pack_words())
        if self.attributes is not None:
            words.append(_u32(self.attributes))
        if self.discrete_io_32 is not None:
            words.append(_u32(self.discrete_io_32))
        if self.discrete_io_64 is not None:
            hi, lo = _encode_u64(self.discrete_io_64)
            words.extend([hi, lo])
        if self.health_status is not None:
            words.append(_u32(self.health_status & 0xFFFF))
        if self.v49_spec_compliance is not None:
            words.append(_u32(self.v49_spec_compliance))
        if self.build_info is not None:
            words.append(self.build_info.pack_word())
        if self.buffer_size is not None:
            words.extend(self.buffer_size.pack_words())
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

        unsupported = int(flags & ~CIF1Fields.SUPPORTED_MASK)
        if unsupported:
            raise ValueError(f"Unsupported CIF1 bits set: 0x{unsupported:08X}")

        idx = 0  # byte index
        phase: float | None = None
        eb_no_ber: Tuple[float, float] | None = None
        threshold: Tuple[float, float] | None = None
        compression: float | None = None
        intercept: Tuple[float, float] | None = None
        snr_nf: Tuple[float, float] | None = None
        aux_freq: float | None = None
        aux_gain: Tuple[float, float] | None = None
        aux_bw: float | None = None
        array_of_cif: ArrayOfCifFields | None = None
        spectrum: SpectrumField | None = None
        sector_field: SectorStepScanField | None = None
        attributes: int | None = None
        dio32: int | None = None
        dio64: int | None = None
        health: int | None = None
        spec_ver: int | None = None
        build: BuildInformation | None = None
        buffer_size: BufferSizeField | None = None

        def take_word() -> int:
            nonlocal idx
            if idx + 4 > len(mv):
                raise ValueError("Truncated CIF1 field data")
            w = struct.unpack_from(">I", mv, idx)[0]
            idx += 4
            return w

        if flags & CIF1Flags.PHASE:
            w = take_word()
            phase = _from_s16_fixed7(w & 0xFFFF)
        if flags & CIF1Flags.EB_NO_AND_BER:
            w = take_word()
            eb_no_ber = _parse_gain_tuple(w)
        if flags & CIF1Flags.THRESHOLD:
            w = take_word()
            threshold = _parse_gain_tuple(w)
        if flags & CIF1Flags.COMPRESSION_POINT:
            w = take_word()
            compression = _from_s16_fixed7(w & 0xFFFF)
        if flags & CIF1Flags.INTERCEPT_POINTS:
            w = take_word()
            intercept = _parse_gain_tuple(w)
        if flags & CIF1Flags.SNR_AND_NOISE_FIGURE:
            w = take_word()
            snr_nf = _parse_gain_tuple(w)
        if flags & CIF1Flags.AUX_FREQUENCY:
            if idx + 8 > len(mv):
                raise ValueError("Truncated Aux Frequency field")
            hi, lo = struct.unpack_from(">II", mv, idx)
            aux_freq = _from_s64_fixed20(hi, lo)
            idx += 8
        if flags & CIF1Flags.AUX_GAIN:
            w = take_word()
            aux_gain = _parse_gain_tuple(w)
        if flags & CIF1Flags.AUX_BANDWIDTH:
            if idx + 8 > len(mv):
                raise ValueError("Truncated Aux Bandwidth field")
            hi, lo = struct.unpack_from(">II", mv, idx)
            aux_bw = _from_s64_fixed20(hi, lo)
            idx += 8
        if flags & CIF1Flags.ARRAY_OF_CIF:
            remaining_words = [struct.unpack_from(">I", mv, offset)[0] for offset in range(idx, len(mv), 4)]
            array_of_cif, consumed = ArrayOfCifFields.parse(remaining_words)
            idx += consumed * 4
        if flags & CIF1Flags.SPECTRUM:
            needed = SpectrumField.NUM_WORDS * 4
            if idx + needed > len(mv):
                raise ValueError("Truncated Spectrum field")
            seg = struct.unpack_from(f">{SpectrumField.NUM_WORDS}I", mv, idx)
            spectrum = SpectrumField.parse_words(seg)
            idx += needed
        if flags & CIF1Flags.SECTOR_STEP_SCAN:
            remaining_words = [struct.unpack_from(">I", mv, offset)[0] for offset in range(idx, len(mv), 4)]
            sector_field, consumed = SectorStepScanField.parse(remaining_words)
            idx += consumed * 4
        if flags & CIF1Flags.ATTRIBUTES:
            attributes = take_word()
        if flags & CIF1Flags.DISCRETE_IO_32:
            dio32 = take_word()
        if flags & CIF1Flags.DISCRETE_IO_64:
            if idx + 8 > len(mv):
                raise ValueError("Truncated Discrete IO 64 field")
            hi, lo = struct.unpack_from(">II", mv, idx)
            dio64 = _decode_u64(hi, lo)
            idx += 8
        if flags & CIF1Flags.HEALTH_STATUS:
            health = take_word() & 0xFFFF
        if flags & CIF1Flags.V49_SPEC_COMPLIANCE:
            spec_ver = take_word()
        if flags & CIF1Flags.BUILD_INFO:
            build = BuildInformation.parse(take_word())
        if flags & CIF1Flags.BUFFER_SIZE:
            remaining_words = [struct.unpack_from(">I", mv, offset)[0] for offset in range(idx, len(mv), 4)]
            buffer_size = BufferSizeField.parse(remaining_words)
            idx += 8

        fields = CIF1Fields(
            phase_radians=phase,
            eb_no_and_ber_db=eb_no_ber,
            threshold_db=threshold,
            compression_point_dbm=compression,
            intercept_points_dbm=intercept,
            snr_and_noise_figure_db=snr_nf,
            aux_frequency_hz=aux_freq,
            aux_gain_db=aux_gain,
            aux_bandwidth_hz=aux_bw,
            array_of_cif_fields=array_of_cif,
            spectrum=spectrum,
            sector_step_scan=sector_field,
            attributes=attributes,
            discrete_io_32=dio32,
            discrete_io_64=dio64,
            health_status=health,
            v49_spec_compliance=spec_ver,
            build_info=build,
            buffer_size=buffer_size,
        )

        return fields, idx // 4


__all__ = [
    "AveragingType",
    "CIF1Fields",
    "SpectrumField",
    "SpectrumType",
    "WindowTimeDeltaInterpretation",
    "CIF1Flags",
    "SectorStepRecord",
    "SectorStepScanField",
    "ArrayOfCifFields",
    "BufferSizeField",
    "BuildInformation",
]
