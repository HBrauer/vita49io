"""Implement VITA 49 Context Indicator Field 0 (CIF0) helpers and enumerations.

Args:
    None.

Returns:
    None.

Raises:
    None.

Side Effects:
    None.

Examples:
    >>> from vita49io.protocol.cif0 import PayloadFormat
    >>> PayloadFormat.parse(0x00000000, 0x00010001).repeat_count
    1
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import IntEnum

from .core import _payload_bytes_to_words, _payload_words_to_bytes, _u32


# ---------------------------------------------
# Data Packet Payload Format (CIF0 bit 15)
# ---------------------------------------------


class PackingMethod(IntEnum):
    """Enumerate CIF0 packing method options.

    Examples:
        >>> from vita49io.protocol.cif0 import PackingMethod
        >>> PackingMethod.PROCESSING_EFFICIENT.value
        0
    """
    PROCESSING_EFFICIENT = 0
    LINK_EFFICIENT = 1


class SampleType(IntEnum):
    """Represent CIF0 sample type enumerations.

    Examples:
        >>> from vita49io.protocol.cif0 import SampleType
        >>> SampleType.COMPLEX_CARTESIAN.name
        'COMPLEX_CARTESIAN'
    """
    REAL = 0
    COMPLEX_CARTESIAN = 1  # I/Q
    COMPLEX_POLAR = 2


class DataItemFormat(IntEnum):
    """Enumerate Payload data item format codes.

    Examples:
        >>> from vita49io.protocol.cif0 import DataItemFormat
        >>> DataItemFormat.IEEE754_SINGLE.value
        14
    """
    
    SIGNED_FIXED_POINT = 0b00000 # Signed fixed-point (two's-complement), normalized [-1, 1-2^-(N-1)]
    UNSIGNED_FIXED_POINT = 0b10000 # Unsigned fixed-point, normalized [0, 1-2^-N]
    IEEE754_SINGLE = 0b01110  # Standard 32 float.       
    


@dataclass
class PayloadFormat:
    """Describe the payload format two-word structure.

    Args:
        packing_method (PackingMethod): Packing method bit (processing vs link efficient).
        sample_type (SampleType): Sample type bit field describing complex vs real layout.
        data_item_format_code (int): Raw five-bit data item format code.
        sample_component_repeat (bool): Indicates repeated component ordering.
        event_tag_size_bits (int): Width in bits for event tags.
        channel_tag_size_bits (int): Width in bits for channel tags.
        data_item_fraction_size_bits (int): Fractional bits count for fixed-point formats.
        item_packing_field_size_bits (int): Total bits per packing field.
        data_item_size_bits (int): Bits per data item.
        repeat_count (int): Repeat count encoded in word two.
        vector_size (int): Vector size encoded in word two.
        data_item_format (Optional[DataItemFormat]): Optional friendly enum for the format code.

    Examples:
        >>> from vita49io.protocol.cif0 import PayloadFormat, PackingMethod, SampleType, DataItemFormat
        >>> PayloadFormat(
        ... packing_method=PackingMethod.PROCESSING_EFFICIENT,
        ... sample_type=SampleType.COMPLEX_CARTESIAN,
        ... data_item_format_code=int(DataItemFormat.IEEE754_SINGLE),
        ... sample_component_repeat=False,
        ... event_tag_size_bits=0,
        ... channel_tag_size_bits=0,
        ... data_item_fraction_size_bits=0,
        ... item_packing_field_size_bits=32,
        ... data_item_size_bits=32,
        ... repeat_count=1,
        ... vector_size=1,
        ... data_item_format=DataItemFormat.IEEE754_SINGLE
        ... ).repeat_count
        1
    """
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
        """Decode a PayloadFormat from two 32-bit words.

        Args:
            w0 (int): First 32-bit word containing packing metadata.
            w1 (int): Second 32-bit word containing repeat and vector sizes.

        Returns:
            PayloadFormat: Parsed structure capturing the payload layout.

        Raises:
            ValueError: If the data item format code is unsupported.

        Examples:
            >>> from vita49io.protocol.cif0 import PayloadFormat
            >>> PayloadFormat.parse(0x00000000, 0x00000000).vector_size
            1
        """
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

        repeat_count = ((w1 >> 16) & 0xFFFF) + 1  # stored as value-1
        vector_size = (w1 & 0xFFFF) + 1  # stored as value-1

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
        """Encode the payload format back into two 32-bit words.

        Returns:
            Tuple[int, int]: Pair of words corresponding to the CIF0 payload format fields.

        Examples:
            >>> from vita49io.protocol.cif0 import PayloadFormat, PackingMethod, SampleType, DataItemFormat
            >>> PayloadFormat(
            ... packing_method=PackingMethod.PROCESSING_EFFICIENT,
            ... sample_type=SampleType.COMPLEX_CARTESIAN,
            ... data_item_format_code=int(DataItemFormat.IEEE754_SINGLE),
            ... sample_component_repeat=False,
            ... event_tag_size_bits=0,
            ... channel_tag_size_bits=0,
            ... data_item_fraction_size_bits=0,
            ... item_packing_field_size_bits=32,
            ... data_item_size_bits=32,
            ... repeat_count=1,
            ... vector_size=1,
            ... data_item_format=DataItemFormat.IEEE754_SINGLE
            ... ).pack_words()[0] >> 31
            0
        """
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
        w1 |= ((max(1, self.repeat_count) - 1) & 0xFFFF) << 16 
        w1 |= ((max(1, self.vector_size) - 1) & 0xFFFF) 

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
    """Represent the structured fields contained in CIF0 payloads.

    Args:
        context_field_change_indicator (bool): Indicates if fields have changed.
        reference_point_identifier (Optional[int]): Optional stream identifier reference.
        bandwidth_hz (Optional[float]): Receiver bandwidth in Hz.
        if_reference_frequency_hz (Optional[float]): IF reference frequency in Hz.
        rf_reference_frequency_hz (Optional[float]): RF reference frequency in Hz.
        rf_reference_frequency_offset_hz (Optional[float]): RF frequency offset in Hz.
        if_band_offset_hz (Optional[float]): IF band offset in Hz.
        reference_level_dbm (Optional[float]): Reference level in dBm.
        gain_db (Optional[Tuple[float, float]]): Optional (gain1, gain2) tuple in dB.
        over_range_count (Optional[int]): Overrange counter value.
        sample_rate_hz (Optional[float]): Sample rate in Hz.
        timestamp_adjustment_fs (Optional[int]): Timestamp adjustment in femtoseconds.
        timestamp_calibration_time_s (Optional[int]): Timestamp calibration integer seconds.
        temperature_c (Optional[int]): Temperature in Celsius (signed).
        device_identifier (Optional[Tuple[int, int]]): Device identifier (OUI, device).
        state_event_indicators (Optional[int]): State and event indicator bits.
        data_packet_payload_format (Optional[Tuple[int, int]]): Raw payload format words.
        payload_format (Optional[PayloadFormat]): Parsed payload format helper.
        raw_low_bits (int): Unparsed lower mask bits reserved for future use.

    Examples:
        >>> from vita49io.protocol.cif0 import CIF0Fields
        >>> CIF0Fields().sample_rate_hz is None
        True
    """
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
    # Raw presence bits 14..0 preserved as-is (no additional words handled)
    raw_low_bits: int = 0

    def _presence_mask(self) -> int:
        m = 0
        # Include raw low bits (14..0) as-is
        m |= (self.raw_low_bits & 0x7FFF)
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
        return m

    def pack(self) -> bytes:
        """Serialize the CIF0 fields into bytes beginning with the mask word.

        Returns:
            bytes: Serialized CIF0 payload including mask and encoded fields.

        Examples:
            >>> from vita49io.protocol.cif0 import CIF0Fields
            >>> CIF0Fields().pack()[:4]
            b"\x00\x00\x00\x00"
        """
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

        return _payload_words_to_bytes(words)

    @staticmethod
    def parse_from_mask(mask: int, field_words: List[int]) -> Tuple["CIF0Fields", int]:
        """Interpret CIF0 fields based on a mask word and payload words.

        Args:
            mask (int): CIF0 mask word indicating which fields are present.
            field_words (List[int]): Subsequent 32-bit words holding encoded field data.

        Returns:
            Tuple[CIF0Fields, int]: Parsed fields and the number of words consumed.

        Raises:
            ValueError: If the mask requires more field words than provided.

        Examples:
            >>> from vita49io.protocol.cif0 import CIF0Fields
            >>> CIF0Fields.parse_from_mask(0, [])[1]
            0
        """

        # Reject unsupported lower bits explicitly requested by caller
        # Preserve raw lower bits (14..0) without attempting to parse them
            

        idx = 0

        def need(n: int) -> None:
            nonlocal idx
            if idx + n > len(field_words):
                raise ValueError("Truncated CIF0 payload")

        f = CIF0Fields()

        f.context_field_change_indicator = bool(mask & (1 << 31))
        # Save raw low bits 14..0
        f.raw_low_bits = mask & 0x7FFF
        if mask & (1 << 30):
            need(1)
            f.reference_point_identifier = field_words[idx]
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
        return f, idx

    @staticmethod
    def parse(payload: bytes) -> Tuple["CIF0Fields", int]:
        """Parse a CIF0 payload from bytes and report the number of bytes consumed.

        Args:
            payload (bytes): Raw CIF0 payload beginning with the mask word.

        Returns:
            Tuple[CIF0Fields, int]: Parsed fields and the number of bytes consumed.

        Raises:
            ValueError: If the payload is empty or malformed.

        Side Effects:
            None.

        Examples:
            >>> from vita49io.protocol.cif0 import CIF0Fields
            >>> CIF0Fields.parse(b"\x00\x00\x00\x00")[1]
            4
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
