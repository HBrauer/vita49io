"""Common PayloadFormat presets and helpers.

Naming pattern for presets:
  <FMT><BITS>[_Q7]_<SAMPLE>
  FMT: F = IEEE754_SINGLE, S = signed fixed-point, U = unsigned fixed-point
  BITS: data_item_size_bits
  SAMPLE: REAL, IQ (complex Cartesian), POLAR (complex polar)
"""

from __future__ import annotations

from vita49io.protocol.cif0 import DataItemFormat, PackingMethod, PayloadFormat, SampleType


def build_payload_format(
    data_item_format: DataItemFormat,
    data_item_size_bits: int,
    item_packing_field_size_bits: int | None = None,
    *,
    packing_method: PackingMethod = PackingMethod.PROCESSING_EFFICIENT,
    sample_type: SampleType = SampleType.COMPLEX_CARTESIAN,
    sample_component_repeat: bool = False,
    event_tag_size_bits: int = 0,
    channel_tag_size_bits: int = 0,
    data_item_fraction_size_bits: int = 0,
    repeat_count: int = 1,
    vector_size: int = 1,
) -> PayloadFormat:
    """Build a PayloadFormat with common defaults for CIF0 payloads."""
    if item_packing_field_size_bits is None:
        item_packing_field_size_bits = 32 if data_item_size_bits == 24 else data_item_size_bits
    return PayloadFormat(
        packing_method=packing_method,
        sample_type=sample_type,
        data_item_format_code=int(data_item_format),
        sample_component_repeat=sample_component_repeat,
        event_tag_size_bits=event_tag_size_bits,
        channel_tag_size_bits=channel_tag_size_bits,
        data_item_fraction_size_bits=data_item_fraction_size_bits,
        item_packing_field_size_bits=item_packing_field_size_bits,
        data_item_size_bits=data_item_size_bits,
        repeat_count=repeat_count,
        vector_size=vector_size,
        data_item_format=data_item_format,
    )

class DefaultPayloadFormats:
    """Processing-efficient presets using the naming pattern above.

    Link-efficient presets can be added later using the same suffixes.
    """

    # Real
    F32_REAL = build_payload_format(DataItemFormat.IEEE754_SINGLE, 32, sample_type=SampleType.REAL)
    S32_REAL = build_payload_format(DataItemFormat.SIGNED_FIXED_POINT, 32, sample_type=SampleType.REAL)
    S24_REAL = build_payload_format(DataItemFormat.SIGNED_FIXED_POINT, 24, sample_type=SampleType.REAL)
    S16_REAL = build_payload_format(DataItemFormat.SIGNED_FIXED_POINT, 16, sample_type=SampleType.REAL)
    S8_REAL = build_payload_format(DataItemFormat.SIGNED_FIXED_POINT, 8, sample_type=SampleType.REAL)
    S32_Q7_REAL = build_payload_format(
        DataItemFormat.SIGNED_FIXED_POINT, 32, sample_type=SampleType.REAL, data_item_fraction_size_bits=7
    )
    S24_Q7_REAL = build_payload_format(
        DataItemFormat.SIGNED_FIXED_POINT, 24, sample_type=SampleType.REAL, data_item_fraction_size_bits=7
    )
    S16_Q7_REAL = build_payload_format(
        DataItemFormat.SIGNED_FIXED_POINT, 16, sample_type=SampleType.REAL, data_item_fraction_size_bits=7
    )
    S8_Q7_REAL = build_payload_format(
        DataItemFormat.SIGNED_FIXED_POINT, 8, sample_type=SampleType.REAL, data_item_fraction_size_bits=7
    )
    U32_REAL = build_payload_format(DataItemFormat.UNSIGNED_FIXED_POINT, 32, sample_type=SampleType.REAL)
    U24_REAL = build_payload_format(DataItemFormat.UNSIGNED_FIXED_POINT, 24, sample_type=SampleType.REAL)
    U16_REAL = build_payload_format(DataItemFormat.UNSIGNED_FIXED_POINT, 16, sample_type=SampleType.REAL)
    U8_REAL = build_payload_format(DataItemFormat.UNSIGNED_FIXED_POINT, 8, sample_type=SampleType.REAL)

    # Complex Cartesian
    F32_IQ = build_payload_format(DataItemFormat.IEEE754_SINGLE, 32, sample_type=SampleType.COMPLEX_CARTESIAN)
    S32_IQ = build_payload_format(DataItemFormat.SIGNED_FIXED_POINT, 32, sample_type=SampleType.COMPLEX_CARTESIAN)
    S24_IQ = build_payload_format(DataItemFormat.SIGNED_FIXED_POINT, 24, sample_type=SampleType.COMPLEX_CARTESIAN)
    S16_IQ = build_payload_format(DataItemFormat.SIGNED_FIXED_POINT, 16, sample_type=SampleType.COMPLEX_CARTESIAN)
    S8_IQ = build_payload_format(DataItemFormat.SIGNED_FIXED_POINT, 8, sample_type=SampleType.COMPLEX_CARTESIAN)
    S32_Q7_IQ = build_payload_format(
        DataItemFormat.SIGNED_FIXED_POINT, 32, sample_type=SampleType.COMPLEX_CARTESIAN, data_item_fraction_size_bits=7
    )
    S24_Q7_IQ = build_payload_format(
        DataItemFormat.SIGNED_FIXED_POINT, 24, sample_type=SampleType.COMPLEX_CARTESIAN, data_item_fraction_size_bits=7
    )
    S16_Q7_IQ = build_payload_format(
        DataItemFormat.SIGNED_FIXED_POINT, 16, sample_type=SampleType.COMPLEX_CARTESIAN, data_item_fraction_size_bits=7
    )
    S8_Q7_IQ = build_payload_format(
        DataItemFormat.SIGNED_FIXED_POINT, 8, sample_type=SampleType.COMPLEX_CARTESIAN, data_item_fraction_size_bits=7
    )
    U32_IQ = build_payload_format(DataItemFormat.UNSIGNED_FIXED_POINT, 32, sample_type=SampleType.COMPLEX_CARTESIAN)
    U24_IQ = build_payload_format(DataItemFormat.UNSIGNED_FIXED_POINT, 24, sample_type=SampleType.COMPLEX_CARTESIAN)
    U16_IQ = build_payload_format(DataItemFormat.UNSIGNED_FIXED_POINT, 16, sample_type=SampleType.COMPLEX_CARTESIAN)
    U8_IQ = build_payload_format(DataItemFormat.UNSIGNED_FIXED_POINT, 8, sample_type=SampleType.COMPLEX_CARTESIAN)

    # Complex Polar
    F32_POLAR = build_payload_format(DataItemFormat.IEEE754_SINGLE, 32, sample_type=SampleType.COMPLEX_POLAR)
    S32_POLAR = build_payload_format(DataItemFormat.SIGNED_FIXED_POINT, 32, sample_type=SampleType.COMPLEX_POLAR)
    S24_POLAR = build_payload_format(DataItemFormat.SIGNED_FIXED_POINT, 24, sample_type=SampleType.COMPLEX_POLAR)
    S16_POLAR = build_payload_format(DataItemFormat.SIGNED_FIXED_POINT, 16, sample_type=SampleType.COMPLEX_POLAR)
    S8_POLAR = build_payload_format(DataItemFormat.SIGNED_FIXED_POINT, 8, sample_type=SampleType.COMPLEX_POLAR)
    S32_Q7_POLAR = build_payload_format(
        DataItemFormat.SIGNED_FIXED_POINT, 32, sample_type=SampleType.COMPLEX_POLAR, data_item_fraction_size_bits=7
    )
    S24_Q7_POLAR = build_payload_format(
        DataItemFormat.SIGNED_FIXED_POINT, 24, sample_type=SampleType.COMPLEX_POLAR, data_item_fraction_size_bits=7
    )
    S16_Q7_POLAR = build_payload_format(
        DataItemFormat.SIGNED_FIXED_POINT, 16, sample_type=SampleType.COMPLEX_POLAR, data_item_fraction_size_bits=7
    )
    S8_Q7_POLAR = build_payload_format(
        DataItemFormat.SIGNED_FIXED_POINT, 8, sample_type=SampleType.COMPLEX_POLAR, data_item_fraction_size_bits=7
    )
    U32_POLAR = build_payload_format(DataItemFormat.UNSIGNED_FIXED_POINT, 32, sample_type=SampleType.COMPLEX_POLAR)
    U24_POLAR = build_payload_format(DataItemFormat.UNSIGNED_FIXED_POINT, 24, sample_type=SampleType.COMPLEX_POLAR)
    U16_POLAR = build_payload_format(DataItemFormat.UNSIGNED_FIXED_POINT, 16, sample_type=SampleType.COMPLEX_POLAR)
    U8_POLAR = build_payload_format(DataItemFormat.UNSIGNED_FIXED_POINT, 8, sample_type=SampleType.COMPLEX_POLAR)


__all__ = ["DefaultPayloadFormats", "build_payload_format"]
