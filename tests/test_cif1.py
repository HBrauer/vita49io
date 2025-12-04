import json
import os
import struct
import subprocess
from pathlib import Path
from typing import List, Optional

import pytest

from vita49io import (
    CIF0Fields,
    CIF1Fields,
    BufferSizeField,
    BuildInformation,
    ContextPacket,
    PacketType,
    SectorStepRecord,
    SectorStepScanField,
    SpectrumField,
    SpectrumType,
    AveragingType,
    WindowTimeDeltaInterpretation,
)
from vita49io.protocol.cif0 import CIF0Flags
from vita49io.protocol.cif1 import ArrayOfCifFields, CIF1Flags
from vita49io.protocol.utils import _payload_bytes_to_words


def _load_tshark_path() -> Optional[Path]:
    # Prefer environment variable to allow CI overrides
    env_val = os.environ.get("TSHARK_PATH")
    if env_val:
        return Path(env_val)
    env_file = Path(__file__).resolve().parents[1] / ".env"
    if not env_file.exists():
        return None
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        if key.strip() == "TSHARK_PATH":
            return Path(val.strip().strip('"').strip("'"))
    return None


_TSHARK_PATH = _load_tshark_path()
_TSHARK_MISSING = _TSHARK_PATH is None or not _TSHARK_PATH.exists()


def _ip_checksum(data: bytes) -> int:
    if len(data) % 2:
        data += b"\x00"
    s = sum(int.from_bytes(data[i:i + 2], "big") for i in range(0, len(data), 2))
    while s >> 16:
        s = (s & 0xFFFF) + (s >> 16)
    return (~s) & 0xFFFF


def _build_udp_frame(payload: bytes, *, sport: int = 4991, dport: int = 4991) -> bytes:
    # Minimal Ethernet + IPv4 + UDP wrapper for a VRT payload.
    dst = b"\x00\x11\x22\x33\x44\x55"
    src = b"\x66\x77\x88\x99\xaa\xbb"
    eth_type = 0x0800

    ver_ihl = (4 << 4) | 5
    tos = 0
    total_len = 20 + 8 + len(payload)
    identification = 0
    flags_fragment = 0
    ttl = 64
    proto = 17
    checksum = 0
    src_ip = b"\x0a\x00\x00\x01"
    dst_ip = b"\x0a\x00\x00\x02"
    ip_header = struct.pack(
        "!BBHHHBBH4s4s",
        ver_ihl,
        tos,
        total_len,
        identification,
        flags_fragment,
        ttl,
        proto,
        checksum,
        src_ip,
        dst_ip,
    )
    checksum = _ip_checksum(ip_header)
    ip_header = struct.pack(
        "!BBHHHBBH4s4s",
        ver_ihl,
        tos,
        total_len,
        identification,
        flags_fragment,
        ttl,
        proto,
        checksum,
        src_ip,
        dst_ip,
    )

    udp_len = 8 + len(payload)
    udp_header = struct.pack("!HHHH", sport, dport, udp_len, 0)

    return dst + src + struct.pack("!H", eth_type) + ip_header + udp_header + payload


def _write_pcap(path: Path, frames: List[bytes]) -> None:
    # Classic pcap (LE) with Ethernet link type.
    global_hdr = struct.pack("<IHHIIII", 0xA1B2C3D4, 2, 4, 0, 0, 0xFFFF, 1)
    with path.open("wb") as f:
        f.write(global_hdr)
        for frame in frames:
            ts_sec = 0
            ts_usec = 0
            incl = len(frame)
            orig = len(frame)
            f.write(struct.pack("<IIII", ts_sec, ts_usec, incl, orig))
            f.write(frame)


def _run_tshark_json(pcap_path: Path) -> List[dict]:
    assert _TSHARK_PATH is not None
    cmd = [
        str(_TSHARK_PATH),
        "-r",
        str(pcap_path),
        "-T",
        "json",
        "-d",
        "udp.port==4991,vrt",
    ]
    output = subprocess.check_output(cmd)
    return json.loads(output)


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
    ctx = ContextPacket(
        packet_type=PacketType.CONTEXT_PACKET,
        stream_id=0x13579BDF,
        cif0=CIF0Fields(cif1=CIF1Fields(spectrum=spectrum)),
        packet_count=3,
    )
    raw = ctx.to_bytes()
    parsed = ContextPacket.from_bytes(raw)

    parsed_cif1 = parsed.cif0.cif1
    assert parsed_cif1 is not None
    parsed_spec = parsed_cif1.spectrum
    assert parsed_spec is not None
    assert parsed_spec.num_transform_points == 2048
    assert parsed_spec.number_of_averages == 2
    assert parsed.cif0.cif1 is not None
    assert parsed.cif0._presence_mask() & int(CIF0Flags.CIF1_ENABLE)


@pytest.mark.skipif(_TSHARK_MISSING, reason="Set TSHARK_PATH in .env to enable tshark integration tests")
def test_cif1_external_decoder_masks_and_lengths(tmp_path: Path):
    spec_a = SpectrumField(
        spectrum_type=SpectrumType.MAGNITUDE,
        averaging_type=AveragingType.LINEAR,
        window_time_delta_interpretation=WindowTimeDeltaInterpretation.SAMPLES,
        window_type=1,
        num_transform_points=256,
        num_window_points=256,
        resolution_hz=10.0,
        span_hz=1000.0,
        number_of_averages=5,
        weighting_factor=0.25,
        f1_index=0,
        f2_index=10,
        window_time_delta=128,
    )
    spec_b = SpectrumField(
        spectrum_type=SpectrumType.CARTESIAN,
        averaging_type=AveragingType.PEAK_HOLD | AveragingType.MIN_HOLD,
        window_time_delta_interpretation=WindowTimeDeltaInterpretation.TIME_NS,
        window_type=7,
        num_transform_points=1024,
        num_window_points=1024,
        resolution_hz=5.0,
        span_hz=2500.0,
        number_of_averages=3,
        weighting_factor=0.75,
        f1_index=-12,
        f2_index=12,
        window_time_delta=250,
    )

    ctx_packets = [
        ContextPacket(packet_type=PacketType.CONTEXT_PACKET, stream_id=0xABCDEF01, cif0=CIF0Fields()),
        ContextPacket(
            packet_type=PacketType.CONTEXT_PACKET,
            stream_id=0xABCDEF02,
            cif0=CIF0Fields(cif1=CIF1Fields(spectrum=spec_a)),
        ),
        ContextPacket(
            packet_type=PacketType.CONTEXT_PACKET,
            stream_id=0xABCDEF03,
            cif0=CIF0Fields(cif1=CIF1Fields(spectrum=spec_b)),
        ),
    ]

    frames = [_build_udp_frame(pkt.to_bytes()) for pkt in ctx_packets]
    pcap_path = tmp_path / "cif1_external.pcap"
    _write_pcap(pcap_path, frames)
    print(pcap_path)

    packets = _run_tshark_json(pcap_path)
    assert len(packets) == len(ctx_packets)

    for ctx, decoded in zip(ctx_packets, packets):
        vrt_layer = decoded["_source"]["layers"]["vrt"]
        assert int(vrt_layer["vrt.sid"], 16) == ctx.stream_id
        assert int(vrt_layer["vrt.hdr_tree"]["vrt.len"]) == len(ctx.to_bytes()) // 4

        expected_cif0_mask = ctx.cif0._presence_mask()
        assert int(vrt_layer["vrt.cif0"], 16) == expected_cif0_mask

        if ctx.cif0.cif1 is None:
            assert "vrt.cif1" not in vrt_layer
        else:
            assert int(vrt_layer["vrt.cif1"], 16) == ctx.cif0.cif1._presence_mask()


@pytest.mark.parametrize(
    "interpretation,value,expect",
    [
        (WindowTimeDeltaInterpretation.PERCENT, 37.5, 37.5),
        (WindowTimeDeltaInterpretation.SAMPLES, 512, 512),
        (WindowTimeDeltaInterpretation.TIME_NS, 123_456, 123_456),
        (WindowTimeDeltaInterpretation.NOT_CONTROLLED, 0xDEADBEEF, 0),
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
    for mask in (int(CIF1Flags.SPECTRUM) | (1 << 8), int(CIF1Flags.ARRAY_OF_CIF)):
        with pytest.raises(ValueError):
            CIF1Fields.parse_from_mask(mask, b"")


def test_cif1_array_of_cif_fields_is_rejected():
    stub = ArrayOfCifFields(
        cif0_mask=0,
        cif1_mask=0,
        cif2_mask=0,
        cif3_mask=0,
        cif7_mask=0,
        records=[],
    )
    cif1 = CIF1Fields(array_of_cif_fields=stub)
    with pytest.raises(ValueError):
        cif1._presence_mask()
    with pytest.raises(ValueError):
        cif1.pack()


def test_cif1_additional_fields_roundtrip():
    build = BuildInformation(year=2025, day=123, revision=7, user_defined=0x1FF)
    buffer = BufferSizeField(buffer_size_bytes=4096, level=0x80, status=0x02)
    cif1 = CIF1Fields(
        phase_radians=1.5,
        eb_no_and_ber_db=(12.5, -45.0),
        threshold_db=(5.0, 10.0),
        compression_point_dbm=1.0,
        intercept_points_dbm=(30.0, 25.0),
        snr_and_noise_figure_db=(45.25, 3.75),
        aux_frequency_hz=1_000_000.25,
        aux_gain_db=(2.0, -1.5),
        aux_bandwidth_hz=200_000.0,
        attributes=0xA5A5A5A5,
        discrete_io_32=0xDEADBEEF,
        discrete_io_64=0x123456789ABCDEF0,
        health_status=0x1234,
        v49_spec_compliance=4,
        build_info=build,
        buffer_size=buffer,
    )

    mask = cif1._presence_mask()
    raw = cif1.pack()
    parsed, used = CIF1Fields.parse_from_mask(mask, raw)
    assert used == len(_payload_bytes_to_words(raw))
    assert parsed.phase_radians is not None
    assert pytest.approx(parsed.phase_radians, rel=1e-6) == 1.5
    assert parsed.eb_no_and_ber_db == pytest.approx((12.5, -45.0))
    assert parsed.threshold_db == pytest.approx((5.0, 10.0))
    assert parsed.compression_point_dbm == pytest.approx(1.0)
    assert parsed.intercept_points_dbm == pytest.approx((30.0, 25.0))
    assert parsed.snr_and_noise_figure_db == pytest.approx((45.25, 3.75))
    assert parsed.aux_frequency_hz == pytest.approx(1_000_000.25)
    assert parsed.aux_gain_db == pytest.approx((2.0, -1.5))
    assert parsed.aux_bandwidth_hz == pytest.approx(200_000.0)
    assert parsed.attributes == 0xA5A5A5A5
    assert parsed.discrete_io_32 == 0xDEADBEEF
    assert parsed.discrete_io_64 == 0x123456789ABCDEF0
    assert parsed.health_status == 0x1234
    assert parsed.v49_spec_compliance == 4
    assert parsed.build_info == build
    assert parsed.buffer_size == buffer


def test_sector_step_scan_roundtrip():
    rec1 = SectorStepRecord(
        sector_number=1,
        f1_start_frequency_hz=1000.0,
        f2_stop_frequency_hz=2000.0,
        resolution_bandwidth_hz=10.0,
        tune_step_size_hz=50.0,
        number_of_points=10,
        default_gain_db=(1.0, 2.0),
        threshold_db=(3.0, 4.0),
        dwell_time_fs=1_000_000,
        start_time_fs=2_000_000,
        time3_fs=3_000_000,
        time4_fs=4_000_000,
    )
    rec2 = SectorStepRecord(
        sector_number=2,
        f1_start_frequency_hz=3000.0,
        f2_stop_frequency_hz=4000.0,
        resolution_bandwidth_hz=20.0,
        tune_step_size_hz=75.0,
        number_of_points=12,
        default_gain_db=(1.5, 2.5),
        threshold_db=(5.0, 6.0),
        dwell_time_fs=5_000_000,
        start_time_fs=6_000_000,
        time3_fs=7_000_000,
        time4_fs=8_000_000,
    )
    sector_field = SectorStepScanField(records=[rec1, rec2])
    cif1 = CIF1Fields(sector_step_scan=sector_field)
    mask = cif1._presence_mask()
    raw = cif1.pack()

    parsed, used_words = CIF1Fields.parse_from_mask(mask, raw)
    assert used_words == len(_payload_bytes_to_words(raw))
    parsed_sector = parsed.sector_step_scan
    assert parsed_sector is not None
    assert len(parsed_sector.records) == 2
    assert parsed_sector.records[0].sector_number == 1
    assert pytest.approx(parsed_sector.records[0].f1_start_frequency_hz) == 1000.0
    assert pytest.approx(parsed_sector.records[1].tune_step_size_hz) == 75.0
    assert parsed_sector.records[1].number_of_points == 12
