import pytest

from vita49io import (
    CIF0Fields,
    CIF1Fields,
    ContextPacket,
    PacketType,
    SpectrumField,
    SpectrumType,
    AveragingType,
    WindowTimeDeltaInterpretation,
)
from vita49io.protocol.cif1 import CIF1Flags
from vita49io.protocol.utils import _payload_bytes_to_words


def test_spectrum_field_pack_parse_roundtrip():
    spectrum = SpectrumField(
        spectrum_type=SpectrumType.CARTESIAN,
        averaging_type=AveragingType.LINEAR | AveragingType.PEAK_HOLD,
        window_time_delta_interpretation=WindowTimeDeltaInterpretation.PERCENT,
        window_type=10,
        num_transform_points=4096,
        num_window_points=4096,
        resolution_hz=50.0,
        span_hz=200_000.0,
        number_of_averages=4,
        weighting_factor=0.5,
        f1_index=-1024,
        f2_index=1023,
        window_time_delta=12.5,
    )
    cif1 = CIF1Fields(spectrum=spectrum)
    mask = cif1._presence_mask()
    assert mask == 1 << 10

    words = _payload_bytes_to_words(cif1.pack())
    parsed, used = CIF1Fields.parse_from_mask(mask, cif1.pack())
    assert used == SpectrumField.NUM_WORDS
    parsed_spec = parsed.spectrum
    assert parsed_spec is not None
    assert parsed_spec.spectrum_type == SpectrumType.CARTESIAN
    assert parsed_spec.averaging_type & AveragingType.LINEAR
    assert parsed_spec.num_transform_points == 4096
    assert parsed_spec.num_window_points == 4096
    assert parsed_spec.f1_index == -1024
    assert parsed_spec.f2_index == 1023
    assert pytest.approx(parsed_spec.resolution_hz, rel=1e-6) == 50.0
    assert pytest.approx(parsed_spec.span_hz, rel=1e-6) == 200_000.0
    assert pytest.approx(parsed_spec.weighting_factor, rel=1e-6) == 0.5
    assert pytest.approx(parsed_spec.window_time_delta, rel=1e-6) == 12.5


def test_context_packet_with_cif1_spectrum():
    spectrum = SpectrumField(
        spectrum_type=SpectrumType.MAGNITUDE,
        averaging_type=AveragingType.LINEAR,
        window_time_delta_interpretation=WindowTimeDeltaInterpretation.SAMPLES,
        window_type=0,
        num_transform_points=2048,
        num_window_points=2048,
        resolution_hz=25.0,
        span_hz=100_000.0,
        number_of_averages=2,
        weighting_factor=0.25,
        f1_index=0,
        f2_index=2047,
        window_time_delta=256,
    )
    cif1 = CIF1Fields(spectrum=spectrum)

    ctx = ContextPacket(
        packet_type=PacketType.CONTEXT_PACKET,
        stream_id=0x13579BDF,
        cif0=CIF0Fields(),
        cif1=cif1,
        packet_count=3,
    )
    raw = ctx.to_bytes()
    parsed = ContextPacket.from_bytes(raw)

    assert parsed.cif_extra_masks == [(1, 1 << 10)]
    parsed_cif1 = parsed.cif1
    assert parsed_cif1 is not None
    parsed_spec = parsed_cif1.spectrum
    assert parsed_spec is not None
    assert parsed_spec.num_transform_points == 2048
    assert parsed_spec.number_of_averages == 2
    assert parsed.raw_cif_fields is None


@pytest.mark.parametrize(
    "interpretation,value,expect",
    [
        (WindowTimeDeltaInterpretation.PERCENT, 37.5, 37.5),
        (WindowTimeDeltaInterpretation.SAMPLES, 512, 512),
        (WindowTimeDeltaInterpretation.TIME_NS, 123_456, 123_456),
        (WindowTimeDeltaInterpretation.NOT_CONTROLLED, 0xDEADBEEF, 0xDEADBEEF & 0xFFFFFFFF),
    ],
)
def test_spectrum_window_time_delta_variants(interpretation, value, expect):
    spectrum = SpectrumField(
        spectrum_type=SpectrumType.DEFAULT,
        averaging_type=AveragingType.LINEAR,
        window_time_delta_interpretation=interpretation,
        window_type=1,
        num_transform_points=1024,
        num_window_points=1024,
        resolution_hz=1.0,
        span_hz=10.0,
        number_of_averages=1,
        weighting_factor=1.0,
        f1_index=0,
        f2_index=10,
        window_time_delta=value,
    )
    cif1 = CIF1Fields(spectrum=spectrum)
    mask = cif1._presence_mask()
    parsed, used = CIF1Fields.parse_from_mask(mask, cif1.pack())
    assert used == SpectrumField.NUM_WORDS
    parsed_spec = parsed.spectrum
    assert parsed_spec is not None
    if interpretation is WindowTimeDeltaInterpretation.PERCENT:
        assert pytest.approx(parsed_spec.window_time_delta, rel=1e-6) == expect
    else:
        assert parsed_spec.window_time_delta == expect


def test_spectrum_unknown_type_and_averaging_bits_roundtrip():
    custom_type_val = 99
    custom_avg_bits = 0xFF
    spectrum = SpectrumField(
        spectrum_type=custom_type_val,
        averaging_type=custom_avg_bits,
        window_time_delta_interpretation=WindowTimeDeltaInterpretation.SAMPLES,
        window_type=7,
        num_transform_points=128,
        num_window_points=128,
        resolution_hz=2.5,
        span_hz=320.0,
        number_of_averages=3,
        weighting_factor=0.75,
        f1_index=-5,
        f2_index=5,
        window_time_delta=42,
    )
    cif1 = CIF1Fields(spectrum=spectrum)
    parsed, _ = CIF1Fields.parse_from_mask(cif1._presence_mask(), cif1.pack())
    parsed_spec = parsed.spectrum
    assert parsed_spec is not None
    assert parsed_spec.spectrum_type == custom_type_val
    assert int(parsed_spec.averaging_type) & custom_avg_bits == custom_avg_bits
    assert parsed_spec.window_type == 7
    assert parsed_spec.number_of_averages == 3


def test_cif1_parse_rejects_unsupported_bits():
    with pytest.raises(ValueError):
        CIF1Fields.parse_from_mask(int(CIF1Flags.SPECTRUM) | (1 << 9), b"")
