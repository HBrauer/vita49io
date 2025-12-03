import pytest

from vita49io import CIF0Fields, CIF2Fields
from vita49io.protocol.cif0 import CIF0Flags
from vita49io.protocol.cif2 import CIF2Flags
from vita49io.protocol.utils import _payload_bytes_to_words


def test_cif2_pack_parse_roundtrip():
    cif2 = CIF2Fields(
        bind=0x1,
        cited_sid=0xABCDEF00,
        sibling_sid=0x12345678,
        parent_sid=0x11112222,
        child_sid=0x33334444,
        cited_message_id=0x55,
        controllee_id=0x66,
        controllee_uuid=uuid.UUID("00000001-0002-0003-0004-000000000000"),
        controller_id=0x77,
        controller_uuid=uuid.UUID("00000005-0006-0007-0008-000000000000"),
        information_source_id=0x99,
        track_id=0xAA,
        country_code=0xBB,
        operator_id=0xCC,
        platform_class=0xDD,
        platform_instance=0xEE,
        platform_display_type=0xFF,
        ems_device_class=0x100,
        ems_device_type=0x101,
        ems_device_instance=0x102,
        modulation_class=0x103,
        modulation_type=0x104,
        function_id=0x105,
        mode_id=0x106,
        event_id=0x107,
        function_priority_id=0x108,
        comms_priority_id=0x109,
        rf_footprint_range=0x10A,
        rf_footprint=0x10B,
    )
    mask = cif2._presence_mask()
    raw = cif2.pack()
    parsed, used = CIF2Fields.parse_from_mask(mask, raw)
    assert used == len(_payload_bytes_to_words(raw))
    assert parsed.controller_uuid == uuid.UUID("00000005-0006-0007-0008-000000000000")
    assert parsed.cited_sid == 0xABCDEF00
    assert parsed.rf_footprint == 0x10B


def test_cif0_with_cif2_enable_roundtrip():
    cif2 = CIF2Fields(track_id=0x1234, modulation_type=0x10, rf_footprint=0x2222)
    cif0 = CIF0Fields(cif2=cif2)
    mask = cif0._presence_mask()
    assert mask & int(CIF0Flags.CIF2_ENABLE)
    raw = cif0.pack()
    # mask word is first; skip it in parse
    parsed, used_words = CIF0Fields.parse_from_mask(mask, raw[4:])
    assert parsed.cif2 is not None
    assert parsed.cif2.track_id == 0x1234
    assert parsed.cif2.modulation_type == 0x10
    assert parsed.cif2.rf_footprint == 0x2222


def test_cif2_unsupported_bits_rejected():
    with pytest.raises(ValueError):
        CIF2Fields.parse_from_mask(int(CIF2Flags.BIND) | (1 << 2), b"")
import uuid
