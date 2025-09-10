import unittest

from vita49io import ContextPacket, PacketType, TSI, TSF, CIF0Fields


class TestCIF0(unittest.TestCase):
    def test_cif0_pack_parse_roundtrip_subset(self):
        # Build a context packet using CIF0 for a few fields
        cif0 = CIF0Fields(
            reference_point_identifier=0x12345678,
            sample_rate_hz=10_000_000.0,
            bandwidth_hz=8_000_000.0,
            reference_level_dbm=-3.5,
            gain_db=(10.0, -2.0),
            over_range_count=5,
            data_packet_payload_format=(0xA5A5A5A5, 0x5A5A5A5A),
        )

        p = ContextPacket(
            packet_type=PacketType.CONTEXT_PACKET,
            stream_id=0xABCDEF01,
            class_id=None,
            tsi=TSI.UTC,
            tsf=TSF.FRACTIONAL,
            integer_seconds=1700000000,
            fractional_seconds=0,
            cif0=cif0,
            packet_count=1,
        )

        b = p.to_bytes()
        q = ContextPacket.from_bytes(b)

        # Validate parsed CIF0 fields directly
        self.assertIsNotNone(q.cif0)
        parsed = q.cif0
        self.assertEqual(parsed.reference_point_identifier, 0x12345678)
        self.assertAlmostEqual(parsed.sample_rate_hz, 10_000_000.0, places=3)
        self.assertAlmostEqual(parsed.bandwidth_hz, 8_000_000.0, places=3)
        self.assertAlmostEqual(parsed.reference_level_dbm, -3.5, places=2)
        self.assertAlmostEqual(parsed.gain_db[0], 10.0, places=2)
        self.assertAlmostEqual(parsed.gain_db[1], -2.0, places=2)
        self.assertEqual(parsed.over_range_count, 5)
        self.assertEqual(parsed.data_packet_payload_format, (0xA5A5A5A5, 0x5A5A5A5A))


if __name__ == "__main__":
    unittest.main()
