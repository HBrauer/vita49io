from vita49io import CIF0Fields, CIF3Fields
from vita49io.protocol.cif0 import CIF0Flags
from vita49io.protocol.cif3 import CIF3Flags
from vita49io.protocol.utils import _payload_bytes_to_words


def test_cif3_pack_parse_roundtrip():
    cif3 = CIF3Fields(
        timestamp_details=(0xAAAA, 0xBBBB),
        timestamp_skew_fs=-123456789,
        rise_time_fs=111,
        fall_time_fs=222,
        offset_time_fs=333,
        pulse_width_fs=444,
        period_fs=555,
        duration_fs=666,
        dwell_fs=777,
        jitter_fs=888,
        age_word=0xCAFEBABE,
        shelf_life_word=0xDEADBEEF,
    )
    mask = cif3._presence_mask()
    raw = cif3.pack()
    parsed, used = CIF3Fields.parse_from_mask(mask, raw)
    assert used == len(_payload_bytes_to_words(raw))
    assert parsed.timestamp_details == (0xAAAA, 0xBBBB)
    assert parsed.timestamp_skew_fs == -123456789
    assert parsed.jitter_fs == 888
    assert parsed.age_word == 0xCAFEBABE
    assert parsed.shelf_life_word == 0xDEADBEEF


def test_cif0_with_cif3_enable_roundtrip():
    cif3 = CIF3Fields(duration_fs=9999)
    cif0 = CIF0Fields(cif3=cif3)
    mask = cif0._presence_mask()
    assert mask & int(CIF0Flags.CIF3_ENABLE)
    raw = cif0.pack()
    parsed, used_words = CIF0Fields.parse_from_mask(mask, raw[4:])
    assert parsed.cif3 is not None
    assert parsed.cif3.duration_fs == 9999


def test_cif3_unsupported_bits_rejected():
    try:
        CIF3Fields.parse_from_mask(int(CIF3Flags.DURATION) | (1 << 15), b"")
    except ValueError:
        pass
    else:
        assert False, "Unsupported bits should raise"
