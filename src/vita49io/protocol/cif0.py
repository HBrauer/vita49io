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

from dataclasses import dataclass, field
import struct
from typing import List, Sequence, Tuple, Union
from enum import IntEnum, IntFlag

from .cif1 import CIF1Fields
from .cif2 import CIF2Fields
from .cif3 import CIF3Fields
from .cif2 import CIF2Fields
from .enums import TSI, TSF
from .utils import (
    _decode_fixed_point,
    _encode_fixed_point,
    _from_s16_fixed7,
    _from_s64_fixed20,
    _from_u64_fixed20,
    _payload_bytes_to_words,
    _payload_words_to_bytes,
    _to_s16_fixed7,
    _to_s64_fixed20,
    _to_u64_fixed20,
    _u32,
)


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
    

class CIF0Flags(IntFlag):
    """Bit positions for CIF0 presence mask."""

    NONE = 0
    CONTEXT_FIELD_CHANGE_INDICATOR = 1 << 31
    REFERENCE_POINT_IDENTIFIER = 1 << 30
    BANDWIDTH = 1 << 29
    IF_REFERENCE_FREQUENCY = 1 << 28
    RF_REFERENCE_FREQUENCY = 1 << 27
    RF_REFERENCE_FREQUENCY_OFFSET = 1 << 26
    IF_BAND_OFFSET = 1 << 25
    REFERENCE_LEVEL_DBM = 1 << 24
    GAIN_DB = 1 << 23
    OVER_RANGE_COUNT = 1 << 22
    SAMPLE_RATE = 1 << 21
    TIMESTAMP_ADJUSTMENT = 1 << 20
    TIMESTAMP_CALIBRATION = 1 << 19
    TEMPERATURE = 1 << 18
    DEVICE_IDENTIFIER = 1 << 17
    STATE_EVENT_INDICATORS = 1 << 16
    DATA_PACKET_PAYLOAD_FORMAT = 1 << 15
    FORMATTED_GPS_GEOLOCATION = 1 << 14
    FORMATTED_INS_GEOLOCATION = 1 << 13
    ECEF_EPHEMERIS = 1 << 12
    RELATIVE_EPHEMERIS = 1 << 11
    EPHEMERIS_REFERENCE_IDENTIFIER = 1 << 10
    GPS_ASCII = 1 << 9
    CONTEXT_ASSOCIATION_LISTS = 1 << 8
    FIELD_ATTRIBUTES_ENABLE = 1 << 7
    CIF3_ENABLE = 1 << 3
    CIF2_ENABLE = 1 << 2
    CIF1_ENABLE = 1 << 1
    RESERVED_0 = 1 << 0


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
    data_item_format: "DataItemFormat | None" = None

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


ANGLE_SCALE = 1 << 22  # Geolocation Angle Format radix (bit 22)
ALT_SCALE = 1 << 5  # Altitude radix (bit 5)
SPEED_SCALE = 1 << 16  # Speed over ground radix (bit 16)
POS_SCALE = 1 << 5  # ECEF position radix (bit 5)
VEL_SCALE = 1 << 16  # ECEF velocity radix (bit 16)
ATT_SCALE = 1 << 22  # ECEF attitude radix (bit 22)


@dataclass
class FormattedGeolocation:
    """Represent the formatted GPS/INS geolocation field (CIF0 bits 14 and 13)."""

    tsi: TSI
    tsf: TSF
    manufacturer_oui: int
    integer_seconds: int
    fractional_seconds: int
    latitude_deg: float
    longitude_deg: float
    altitude_m: float
    speed_over_ground_m_s: float
    heading_angle_deg: float
    track_angle_deg: float
    magnetic_variation_deg: float

    NUM_WORDS = 11

    def pack_words(self) -> List[int]:
        # Word 0: bits [27:26]=TSI, [25:24]=TSF, [23:0]=OUI
        w0 = (int(self.tsi) & 0x3) << 26
        w0 |= (int(self.tsf) & 0x3) << 24
        w0 |= self.manufacturer_oui & 0xFFFFFF
        frac_hi = (self.fractional_seconds >> 32) & 0xFFFFFFFF
        frac_lo = self.fractional_seconds & 0xFFFFFFFF
        return [
            _u32(w0),
            _u32(self.integer_seconds),
            _u32(frac_hi),
            _u32(frac_lo),
            _encode_fixed_point(self.latitude_deg, ANGLE_SCALE),
            _encode_fixed_point(self.longitude_deg, ANGLE_SCALE),
            _encode_fixed_point(self.altitude_m, ALT_SCALE),
            _encode_fixed_point(self.speed_over_ground_m_s, SPEED_SCALE),
            _encode_fixed_point(self.heading_angle_deg, ANGLE_SCALE),
            _encode_fixed_point(self.track_angle_deg, ANGLE_SCALE),
            _encode_fixed_point(self.magnetic_variation_deg, ANGLE_SCALE),
        ]

    @staticmethod
    def parse(words: Sequence[int]) -> "FormattedGeolocation":
        if len(words) < FormattedGeolocation.NUM_WORDS:
            raise ValueError("Truncated FormattedGeolocation field")
        w0 = words[0]
        tsi = TSI((w0 >> 26) & 0x3)
        tsf = TSF((w0 >> 24) & 0x3)
        manufacturer_oui = w0 & 0xFFFFFF
        integer_seconds = words[1]
        frac_hi = words[2]
        frac_lo = words[3]
        fractional_seconds = ((frac_hi << 32) | frac_lo) & ((1 << 64) - 1)
        return FormattedGeolocation(
            tsi=tsi,
            tsf=tsf,
            manufacturer_oui=manufacturer_oui,
            integer_seconds=integer_seconds,
            fractional_seconds=fractional_seconds,
            latitude_deg=_decode_fixed_point(words[4], ANGLE_SCALE),
            longitude_deg=_decode_fixed_point(words[5], ANGLE_SCALE),
            altitude_m=_decode_fixed_point(words[6], ALT_SCALE),
            speed_over_ground_m_s=_decode_fixed_point(words[7], SPEED_SCALE),
            heading_angle_deg=_decode_fixed_point(words[8], ANGLE_SCALE),
            track_angle_deg=_decode_fixed_point(words[9], ANGLE_SCALE),
            magnetic_variation_deg=_decode_fixed_point(words[10], ANGLE_SCALE),
        )


@dataclass
class Ephemeris:
    """Represent either the ECEF Ephemeris or Relative Ephemeris (CIF0 bits 12/11)."""

    tsi: TSI
    tsf: TSF
    manufacturer_oui: int
    integer_seconds: int
    fractional_seconds: int
    position_x_m: float
    position_y_m: float
    position_z_m: float
    attitude_alpha_deg: float
    attitude_beta_deg: float
    attitude_phi_deg: float
    velocity_dx_m_s: float
    velocity_dy_m_s: float
    velocity_dz_m_s: float

    NUM_WORDS = 13

    def pack_words(self) -> List[int]:
        # Word 0: bits [27:26]=TSI, [25:24]=TSF, [23:0]=OUI
        w0 = (int(self.tsi) & 0x3) << 26
        w0 |= (int(self.tsf) & 0x3) << 24
        w0 |= self.manufacturer_oui & 0xFFFFFF
        frac_hi = (self.fractional_seconds >> 32) & 0xFFFFFFFF
        frac_lo = self.fractional_seconds & 0xFFFFFFFF
        return [
            _u32(w0),
            _u32(self.integer_seconds),
            _u32(frac_hi),
            _u32(frac_lo),
            _encode_fixed_point(self.position_x_m, POS_SCALE),
            _encode_fixed_point(self.position_y_m, POS_SCALE),
            _encode_fixed_point(self.position_z_m, POS_SCALE),
            _encode_fixed_point(self.attitude_alpha_deg, ATT_SCALE),
            _encode_fixed_point(self.attitude_beta_deg, ATT_SCALE),
            _encode_fixed_point(self.attitude_phi_deg, ATT_SCALE),
            _encode_fixed_point(self.velocity_dx_m_s, VEL_SCALE),
            _encode_fixed_point(self.velocity_dy_m_s, VEL_SCALE),
            _encode_fixed_point(self.velocity_dz_m_s, VEL_SCALE),
        ]

    @staticmethod
    def parse(words: Sequence[int]) -> "Ephemeris":
        if len(words) < Ephemeris.NUM_WORDS:
            raise ValueError("Truncated Ephemeris field")
        w0 = words[0]
        tsi = TSI((w0 >> 26) & 0x3)
        tsf = TSF((w0 >> 24) & 0x3)
        manufacturer_oui = w0 & 0xFFFFFF
        integer_seconds = words[1]
        frac_hi = words[2]
        frac_lo = words[3]
        fractional_seconds = ((frac_hi << 32) | frac_lo) & ((1 << 64) - 1)
        return Ephemeris(
            tsi=tsi,
            tsf=tsf,
            manufacturer_oui=manufacturer_oui,
            integer_seconds=integer_seconds,
            fractional_seconds=fractional_seconds,
            position_x_m=_decode_fixed_point(words[4], POS_SCALE),
            position_y_m=_decode_fixed_point(words[5], POS_SCALE),
            position_z_m=_decode_fixed_point(words[6], POS_SCALE),
            attitude_alpha_deg=_decode_fixed_point(words[7], ATT_SCALE),
            attitude_beta_deg=_decode_fixed_point(words[8], ATT_SCALE),
            attitude_phi_deg=_decode_fixed_point(words[9], ATT_SCALE),
            velocity_dx_m_s=_decode_fixed_point(words[10], VEL_SCALE),
            velocity_dy_m_s=_decode_fixed_point(words[11], VEL_SCALE),
            velocity_dz_m_s=_decode_fixed_point(words[12], VEL_SCALE),
        )


@dataclass
class GPSASCIIField:
    """Represent the GPS ASCII field (CIF0 bit 9)."""

    manufacturer_oui: int
    sentences: str

    def pack_words(self) -> List[int]:
        payload_bytes = (self.sentences or "").encode("ascii", errors="strict")
        payload = payload_bytes
        pad_len = (4 - (len(payload) % 4)) % 4
        payload += b"\x00" * pad_len
        payload_words = _payload_bytes_to_words(payload)
        num_words = len(payload_words)
        header_0 = _u32(self.manufacturer_oui)
        header_1 = _u32(num_words)
        return [header_0, header_1, *payload_words]

    @staticmethod
    def parse(words: List[int]) -> Tuple["GPSASCIIField", int]:
        if len(words) < 2:
            raise ValueError("Truncated GPS ASCII header")
        manufacturer_oui = words[0]
        num_words = words[1]
        total_needed = 2 + num_words
        if len(words) < total_needed:
            raise ValueError("Truncated GPS ASCII payload")
        payload_words = words[2:total_needed]
        payload_bytes = _payload_words_to_bytes(payload_words)
        # Strip trailing null padding
        payload_bytes = payload_bytes.rstrip(b"\x00")
        sentences = payload_bytes.decode("ascii", errors="strict")
        return GPSASCIIField(manufacturer_oui=manufacturer_oui, sentences=sentences), total_needed


@dataclass
class ContextAssociationLists:
    """Represent the Context Association Lists section (CIF0 bit 8)."""

    source_list: List[int]
    system_list: List[int]
    vector_component_list: List[int]
    async_channel_list: List[int]
    async_channel_tags: List[int] | None = None

    def pack_words(self) -> List[int]:
        src_size = len(self.source_list)
        sys_size = len(self.system_list)
        vec_size = len(self.vector_component_list)
        async_size = len(self.async_channel_list)
        tag_list = self.async_channel_tags
        if src_size > 511 or sys_size > 511:
            raise ValueError("Source/System list sizes must be <= 511")
        if vec_size > 0xFFFF:
            raise ValueError("Vector-component list size must fit in 16 bits")
        if async_size > 0x7FFF:
            raise ValueError("Async-channel list size must be <= 32767")
        if tag_list is not None and len(tag_list) != async_size:
            raise ValueError("Async-channel tag list length must match async-channel list size")
        word0 = ((src_size & 0x1FF) << 23) | ((sys_size & 0x1FF) << 14)
        word1 = ((vec_size & 0xFFFF) << 16) | ((1 if tag_list else 0) << 15) | (async_size & 0x7FFF)
        words: List[int] = [_u32(word0), _u32(word1)]
        words.extend(_u32(x) for x in self.source_list)
        words.extend(_u32(x) for x in self.system_list)
        words.extend(_u32(x) for x in self.vector_component_list)
        words.extend(_u32(x) for x in self.async_channel_list)
        if tag_list is not None:
            words.extend(_u32(x) for x in tag_list)
        return words

    @staticmethod
    def parse(words: List[int]) -> Tuple["ContextAssociationLists", int]:
        if len(words) < 2:
            raise ValueError("Truncated Context Association Lists header")
        word0, word1 = words[0], words[1]
        src_size = (word0 >> 23) & 0x1FF
        sys_size = (word0 >> 14) & 0x1FF
        vec_size = (word1 >> 16) & 0xFFFF
        async_has_tags = bool((word1 >> 15) & 0x1)
        async_size = word1 & 0x7FFF
        idx = 2

        def take(n: int) -> List[int]:
            nonlocal idx
            end = idx + n
            if end > len(words):
                raise ValueError("Truncated Context Association Lists payload")
            segment = words[idx:end]
            idx = end
            return segment

        source_list = take(src_size)
        system_list = take(sys_size)
        vector_component_list = take(vec_size)
        async_channel_list = take(async_size)
        async_tags: List[int] | None = None
        if async_has_tags:
            async_tags = take(async_size)

        cal = ContextAssociationLists(
            source_list=[_u32(x) for x in source_list],
            system_list=[_u32(x) for x in system_list],
            vector_component_list=[_u32(x) for x in vector_component_list],
            async_channel_list=[_u32(x) for x in async_channel_list],
            async_channel_tags=[_u32(x) for x in async_tags] if async_tags is not None else None,
        )
        return cal, idx


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

    Examples:
        >>> from vita49io.protocol.cif0 import CIF0Fields
        >>> CIF0Fields().sample_rate_hz is None
        True
    """
    # Bit 31
    context_field_change_indicator: bool = False  
    # Bit 30
    reference_point_identifier: int | None = None  # 32-bit SID
    # Bits 29..25 (2 words each, s64 fp20)
    bandwidth_hz: float | None = None
    if_reference_frequency_hz: float | None = None
    rf_reference_frequency_hz: float | None = None
    rf_reference_frequency_offset_hz: float | None = None
    if_band_offset_hz: float | None = None
    # Bit 24 (1 word, s16 fp7 in low 16 bits)
    reference_level_dbm: float | None = None
    # Bit 23 (1 word, two s16 fp7)
    gain_db: Tuple[float, float] | None = None  # (stage1, stage2)
    # Bit 22 (1 word u32)
    over_range_count: int | None = None
    # Bit 21 (2 words u64 fp20)
    sample_rate_hz: float | None = None
    # Bit 20 (2 words s64, LSB=1 fs)
    timestamp_adjustment_fs: int | None = None
    # Bit 19 (1 word u32)
    timestamp_calibration_time_s: int | None = None
    # Bit 18 (1 word s32 Celsius)
    temperature_c: int | None = None
    # Bit 17 (2 words)
    device_identifier: Tuple[int, int] | None = None  # (OUI 24-bit, device 32-bit)
    # Bit 16 (1 word u32)
    state_event_indicators: int | None = None
    # Bit 15 (2 words payload format)
    data_packet_payload_format: Tuple[int, int] | None = None
    # Decoded helper (not part of on-wire format). If provided when packing
    # and raw tuple is None, it will be used to generate the two words.
    payload_format: PayloadFormat | None = None

    # Bit 14 (multi-word formatted GPS geolocation)
    formatted_gps_geolocation: "FormattedGeolocation | None" = None
    # Bit 13 (multi-word formatted INS geolocation)
    formatted_ins_geolocation: "FormattedGeolocation | None" = None
    # Bit 12 (multi-word ECEF ephemeris)
    ecef_ephemeris: "Ephemeris | None" = None
    # Bit 11 (multi-word Relative ephemeris)
    relative_ephemeris: "Ephemeris | None" = None
    # Bit 10 (single-word ephemeris reference identifier)
    ephemeris_reference_identifier: int | None = None
    # Bit 9 (variable-length GPS ASCII field)
    gps_ascii: "GPSASCIIField | None" = None
    # Bit 8 (variable-length Context Association Lists section)
    context_association_lists: "ContextAssociationLists | None" = None
    # Bit 1 (CIF1 mask + payload)
    cif1: CIF1Fields | None = None
    # Bit 2 (CIF2 mask + payload)
    cif2: CIF2Fields | None = None
    # Bit 3 (CIF3 mask + payload)
    cif3: CIF3Fields | None = None
    # Internal caches to preserve original payload words on roundtrip
    _raw_rf_reference_frequency_words: Tuple[int, int] | None = field(default=None, repr=False, init=False)
    _parsed_rf_reference_frequency_hz: float | None = field(default=None, repr=False, init=False)
    _raw_reference_level_word: int | None = field(default=None, repr=False, init=False)
    _parsed_reference_level_dbm: float | None = field(default=None, repr=False, init=False)
    _raw_device_identifier_words: Tuple[int, int] | None = field(default=None, repr=False, init=False)
    _parsed_device_identifier: Tuple[int, int] | None = field(default=None, repr=False, init=False)

    def _presence_mask(self) -> int:
        mask = CIF0Flags.NONE
        if self.context_field_change_indicator:
            mask |= CIF0Flags.CONTEXT_FIELD_CHANGE_INDICATOR
        if self.reference_point_identifier is not None:
            mask |= CIF0Flags.REFERENCE_POINT_IDENTIFIER
        if self.bandwidth_hz is not None:
            mask |= CIF0Flags.BANDWIDTH
        if self.if_reference_frequency_hz is not None:
            mask |= CIF0Flags.IF_REFERENCE_FREQUENCY
        if self.rf_reference_frequency_hz is not None:
            mask |= CIF0Flags.RF_REFERENCE_FREQUENCY
        if self.rf_reference_frequency_offset_hz is not None:
            mask |= CIF0Flags.RF_REFERENCE_FREQUENCY_OFFSET
        if self.if_band_offset_hz is not None:
            mask |= CIF0Flags.IF_BAND_OFFSET
        if self.reference_level_dbm is not None:
            mask |= CIF0Flags.REFERENCE_LEVEL_DBM
        if self.gain_db is not None:
            mask |= CIF0Flags.GAIN_DB
        if self.over_range_count is not None:
            mask |= CIF0Flags.OVER_RANGE_COUNT
        if self.sample_rate_hz is not None:
            mask |= CIF0Flags.SAMPLE_RATE
        if self.timestamp_adjustment_fs is not None:
            mask |= CIF0Flags.TIMESTAMP_ADJUSTMENT
        if self.timestamp_calibration_time_s is not None:
            mask |= CIF0Flags.TIMESTAMP_CALIBRATION
        if self.temperature_c is not None:
            mask |= CIF0Flags.TEMPERATURE
        if self.device_identifier is not None:
            mask |= CIF0Flags.DEVICE_IDENTIFIER
        if self.state_event_indicators is not None:
            mask |= CIF0Flags.STATE_EVENT_INDICATORS
        if self.data_packet_payload_format is not None:
            mask |= CIF0Flags.DATA_PACKET_PAYLOAD_FORMAT
        if self.formatted_gps_geolocation is not None:
            mask |= CIF0Flags.FORMATTED_GPS_GEOLOCATION
        if self.formatted_ins_geolocation is not None:
            mask |= CIF0Flags.FORMATTED_INS_GEOLOCATION
        if self.ecef_ephemeris is not None:
            mask |= CIF0Flags.ECEF_EPHEMERIS
        if self.relative_ephemeris is not None:
            mask |= CIF0Flags.RELATIVE_EPHEMERIS
        if self.ephemeris_reference_identifier is not None:
            mask |= CIF0Flags.EPHEMERIS_REFERENCE_IDENTIFIER
        if self.gps_ascii is not None:
            mask |= CIF0Flags.GPS_ASCII
        if self.context_association_lists is not None:
            mask |= CIF0Flags.CONTEXT_ASSOCIATION_LISTS
        if self.cif1 is not None:
            mask |= CIF0Flags.CIF1_ENABLE
        if self.cif2 is not None:
            mask |= CIF0Flags.CIF2_ENABLE
        if self.cif3 is not None:
            mask |= CIF0Flags.CIF3_ENABLE
        return int(mask)

    def pack(self) -> bytes:
        """Serialize the CIF0 fields into bytes beginning with the mask word."""
        words: List[int] = []
        words.append(self._presence_mask())

        if self.reference_point_identifier is not None:
            words.append(_u32(self.reference_point_identifier))
        if self.bandwidth_hz is not None:
            hi, lo = _to_s64_fixed20(self.bandwidth_hz)
            words.extend([hi, lo])
        if self.if_reference_frequency_hz is not None:
            hi, lo = _to_s64_fixed20(self.if_reference_frequency_hz)
            words.extend([hi, lo])
        if self.rf_reference_frequency_hz is not None:
            if (
                self._raw_rf_reference_frequency_words is not None
                and self._parsed_rf_reference_frequency_hz == self.rf_reference_frequency_hz
            ):
                hi, lo = self._raw_rf_reference_frequency_words
            else:
                hi, lo = _to_s64_fixed20(self.rf_reference_frequency_hz)
            words.extend([hi, lo])
        if self.rf_reference_frequency_offset_hz is not None:
            hi, lo = _to_s64_fixed20(self.rf_reference_frequency_offset_hz)
            words.extend([hi, lo])
        if self.if_band_offset_hz is not None:
            hi, lo = _to_s64_fixed20(self.if_band_offset_hz)
            words.extend([hi, lo])
        if self.reference_level_dbm is not None:
            if (
                self._raw_reference_level_word is not None
                and self._parsed_reference_level_dbm == self.reference_level_dbm
            ):
                w = self._raw_reference_level_word
            else:
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
            words.append(_u32(self.temperature_c))
        if self.device_identifier is not None:
            if self._raw_device_identifier_words is not None and self.device_identifier == self._parsed_device_identifier:
                oui_word, dev_word = self._raw_device_identifier_words
                words.extend([oui_word, dev_word])
            else:
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
        if self.formatted_gps_geolocation is not None:
            words.extend(self.formatted_gps_geolocation.pack_words())
        if self.formatted_ins_geolocation is not None:
            words.extend(self.formatted_ins_geolocation.pack_words())
        if self.ecef_ephemeris is not None:
            words.extend(self.ecef_ephemeris.pack_words())
        if self.relative_ephemeris is not None:
            words.extend(self.relative_ephemeris.pack_words())
        if self.ephemeris_reference_identifier is not None:
            words.append(_u32(self.ephemeris_reference_identifier))
        if self.gps_ascii is not None:
            words.extend(self.gps_ascii.pack_words())
        if self.context_association_lists is not None:
            words.extend(self.context_association_lists.pack_words())
        if self.cif1 is not None:
            words.append(_u32(self.cif1._presence_mask()))
            words.extend(_payload_bytes_to_words(self.cif1.pack()))
        if self.cif2 is not None:
            words.append(_u32(self.cif2._presence_mask()))
            words.extend(_payload_bytes_to_words(self.cif2.pack()))
        if self.cif3 is not None:
            words.append(_u32(self.cif3._presence_mask()))
            words.extend(_payload_bytes_to_words(self.cif3.pack()))

        out = bytearray(len(words) * 4)
        for i, w in enumerate(words):
            struct.pack_into(">I", out, i * 4, _u32(w))
        return bytes(out)

    @staticmethod
    def parse_from_mask(
        mask: int,
        field_words: memoryview | bytes | bytearray,
    ) -> Tuple["CIF0Fields", int]:
        """Interpret CIF0 fields based on a mask word and payload words.

        Args:
            mask (int): CIF0 mask word indicating which fields are present.
            field_words (memoryview | bytes | bytearray): Subsequent 32-bit words
                holding encoded field data. Prefer passing a ``memoryview`` to avoid
                copies.

        Returns:
            Tuple[CIF0Fields, int]: Parsed fields and the number of words consumed.

        Raises:
            ValueError: If the mask requires more field words than provided.

        Examples:
            >>> from vita49io.protocol.cif0 import CIF0Fields
            >>> CIF0Fields.parse_from_mask(0, memoryview(b""))[1]
            0
        """
        mv = field_words if isinstance(field_words, memoryview) else memoryview(field_words)
        if mv.format != "B":
            mv = mv.cast("B")
        if len(mv) % 4 != 0:
            raise ValueError("CIF0 payload must be a whole number of 32-bit words")

        idx = 0  # byte index into mv

        f = CIF0Fields()

        flags = CIF0Flags(mask)

        f.context_field_change_indicator = bool(flags & CIF0Flags.CONTEXT_FIELD_CHANGE_INDICATOR)
        if flags & CIF0Flags.REFERENCE_POINT_IDENTIFIER:
            f.reference_point_identifier = struct.unpack_from(">I", mv, idx)[0]
            idx += 4
        if flags & CIF0Flags.BANDWIDTH:
            hi, lo = struct.unpack_from(">II", mv, idx)
            f.bandwidth_hz = _from_s64_fixed20(hi, lo)
            idx += 8
        if flags & CIF0Flags.IF_REFERENCE_FREQUENCY:
            hi, lo = struct.unpack_from(">II", mv, idx)
            f.if_reference_frequency_hz = _from_s64_fixed20(hi, lo)
            idx += 8
        if flags & CIF0Flags.RF_REFERENCE_FREQUENCY:
            hi, lo = struct.unpack_from(">II", mv, idx)
            f.rf_reference_frequency_hz = _from_s64_fixed20(hi, lo)
            f._raw_rf_reference_frequency_words = (_u32(hi), _u32(lo))
            f._parsed_rf_reference_frequency_hz = f.rf_reference_frequency_hz
            idx += 8
        if flags & CIF0Flags.RF_REFERENCE_FREQUENCY_OFFSET:
            hi, lo = struct.unpack_from(">II", mv, idx)
            f.rf_reference_frequency_offset_hz = _from_s64_fixed20(hi, lo)
            idx += 8
        if flags & CIF0Flags.IF_BAND_OFFSET:
            hi, lo = struct.unpack_from(">II", mv, idx)
            f.if_band_offset_hz = _from_s64_fixed20(hi, lo)
            idx += 8
        if flags & CIF0Flags.REFERENCE_LEVEL_DBM:
            w = struct.unpack_from(">I", mv, idx)[0]
            f.reference_level_dbm = _from_s16_fixed7(w & 0xFFFF)
            f._raw_reference_level_word = _u32(w)
            f._parsed_reference_level_dbm = f.reference_level_dbm
            idx += 4
        if flags & CIF0Flags.GAIN_DB:
            w = struct.unpack_from(">I", mv, idx)[0]
            a = _from_s16_fixed7((w >> 16) & 0xFFFF)
            b = _from_s16_fixed7(w & 0xFFFF)
            f.gain_db = (a, b)
            idx += 4
        if flags & CIF0Flags.OVER_RANGE_COUNT:
            f.over_range_count = struct.unpack_from(">I", mv, idx)[0]
            idx += 4
        if flags & CIF0Flags.SAMPLE_RATE:
            hi, lo = struct.unpack_from(">II", mv, idx)
            f.sample_rate_hz = _from_u64_fixed20(hi, lo)
            idx += 8
        if flags & CIF0Flags.TIMESTAMP_ADJUSTMENT:
            hi, lo = struct.unpack_from(">II", mv, idx)
            i = (hi << 32) | lo
            if i & (1 << 63):
                i -= 1 << 64
            f.timestamp_adjustment_fs = i
            idx += 8
        if flags & CIF0Flags.TIMESTAMP_CALIBRATION:
            f.timestamp_calibration_time_s = struct.unpack_from(">I", mv, idx)[0]
            idx += 4
        if flags & CIF0Flags.TEMPERATURE:
            w = struct.unpack_from(">I", mv, idx)[0]
            if w & 0x80000000:
                w = (w - 0x100000000)  # s32
            f.temperature_c = w
            idx += 4
        if flags & CIF0Flags.DEVICE_IDENTIFIER:
            oui_word, dev = struct.unpack_from(">II", mv, idx)
            oui = oui_word & 0xFFFFFF
            f.device_identifier = (oui, dev)
            f._raw_device_identifier_words = (_u32(oui_word), _u32(dev))
            f._parsed_device_identifier = f.device_identifier
            idx += 8
        if flags & CIF0Flags.STATE_EVENT_INDICATORS:
            f.state_event_indicators = struct.unpack_from(">I", mv, idx)[0]
            idx += 4
        if flags & CIF0Flags.DATA_PACKET_PAYLOAD_FORMAT:
            w0, w1 = struct.unpack_from(">II", mv, idx)
            f.data_packet_payload_format = (w0, w1)
            # Also provide a decoded, user-friendly view
            try:
                f.payload_format = PayloadFormat.parse(w0, w1)
            except Exception:
                f.payload_format = None
            idx += 8
        if flags & CIF0Flags.FORMATTED_GPS_GEOLOCATION:
            segment = struct.unpack_from(f">{FormattedGeolocation.NUM_WORDS}I", mv, idx)
            f.formatted_gps_geolocation = FormattedGeolocation.parse(segment)
            idx += FormattedGeolocation.NUM_WORDS * 4
        if flags & CIF0Flags.FORMATTED_INS_GEOLOCATION:
            segment = struct.unpack_from(f">{FormattedGeolocation.NUM_WORDS}I", mv, idx)
            f.formatted_ins_geolocation = FormattedGeolocation.parse(segment)
            idx += FormattedGeolocation.NUM_WORDS * 4
        if flags & CIF0Flags.ECEF_EPHEMERIS:
            segment = struct.unpack_from(f">{Ephemeris.NUM_WORDS}I", mv, idx)
            f.ecef_ephemeris = Ephemeris.parse(segment)
            idx += Ephemeris.NUM_WORDS * 4
        if flags & CIF0Flags.RELATIVE_EPHEMERIS:
            segment = struct.unpack_from(f">{Ephemeris.NUM_WORDS}I", mv, idx)
            f.relative_ephemeris = Ephemeris.parse(segment)
            idx += Ephemeris.NUM_WORDS * 4
        if flags & CIF0Flags.EPHEMERIS_REFERENCE_IDENTIFIER:
            f.ephemeris_reference_identifier = struct.unpack_from(">I", mv, idx)[0]
            idx += 4
        if flags & CIF0Flags.GPS_ASCII:
            # Need two header words to learn payload length
            remaining_words = [struct.unpack_from(">I", mv, offset)[0] for offset in range(idx, len(mv), 4)]
            gps_ascii, consumed = GPSASCIIField.parse(remaining_words)
            f.gps_ascii = gps_ascii
            idx += consumed * 4
        if flags & CIF0Flags.CONTEXT_ASSOCIATION_LISTS:
            remaining_words = [struct.unpack_from(">I", mv, offset)[0] for offset in range(idx, len(mv), 4)]
            cal, consumed = ContextAssociationLists.parse(remaining_words)
            f.context_association_lists = cal
            idx += consumed * 4
        if flags & CIF0Flags.CIF1_ENABLE:
            if idx + 4 > len(mv):
                raise ValueError("Truncated CIF1 mask")
            cif1_mask = struct.unpack_from(">I", mv, idx)[0]
            idx += 4
            cif1_fields, cif1_used_words = CIF1Fields.parse_from_mask(cif1_mask, mv[idx:])
            f.cif1 = cif1_fields
            idx += cif1_used_words * 4
        if flags & CIF0Flags.CIF2_ENABLE:
            if idx + 4 > len(mv):
                raise ValueError("Truncated CIF2 mask")
            cif2_mask = struct.unpack_from(">I", mv, idx)[0]
            idx += 4
            cif2_fields, cif2_used_words = CIF2Fields.parse_from_mask(cif2_mask, mv[idx:])
            f.cif2 = cif2_fields
            idx += cif2_used_words * 4
        if flags & CIF0Flags.CIF3_ENABLE:
            if idx + 4 > len(mv):
                raise ValueError("Truncated CIF3 mask")
            cif3_mask = struct.unpack_from(">I", mv, idx)[0]
            idx += 4
            cif3_fields, cif3_used_words = CIF3Fields.parse_from_mask(cif3_mask, mv[idx:])
            f.cif3 = cif3_fields
            idx += cif3_used_words * 4
        return f, idx // 4

    @staticmethod
    def parse(payload: Union[bytes, memoryview]) -> Tuple["CIF0Fields", int]:
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
        mv = memoryview(payload)
        if len(mv) < 4:
            raise ValueError("Empty payload for CIF0")
        mask = struct.unpack_from(">I", mv, 0)[0]
        f, used_field_words = CIF0Fields.parse_from_mask(mask, mv[4:])
        return f, (1 + used_field_words) * 4


__all__ = [
    "CIF0Fields",
    "PayloadFormat",
    "PackingMethod",
    "SampleType",
    "DataItemFormat",
    "FormattedGeolocation",
    "Ephemeris",
    "GPSASCIIField",
    "ContextAssociationLists",
    "CIF0Flags",
    "_encode_fixed_point",
    "_decode_fixed_point",
]
