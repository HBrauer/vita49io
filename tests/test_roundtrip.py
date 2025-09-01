import unittest

from vita49 import DataPacket, ContextPacket, PacketType, TSI, TSF


class TestVRT(unittest.TestCase):
    def test_minimal_if_data_roundtrip(self):
        p = DataPacket(
            packet_type=PacketType.IF_DATA,
            stream_id=0x12345678,
            tsi=TSI.UTC,
            tsf=TSF.FRACTIONAL,
            integer_seconds=1700000000,
            fractional_seconds=0x01020304,
            payload=b"\xDE\xAD\xBE\xEF\x00\x11\x22\x33",
            packet_count=7,
        )
        b = p.pack()
        q = DataPacket.parse(b)

        self.assertEqual(q.packet_type, PacketType.IF_DATA)
        self.assertEqual(q.stream_id, 0x12345678)
        self.assertEqual(q.tsi, TSI.UTC)
        self.assertEqual(q.tsf, TSF.FRACTIONAL)
        self.assertEqual(q.integer_seconds, 1700000000)
        self.assertEqual(q.fractional_seconds, 0x01020304)
        self.assertEqual(q.payload, b"\xDE\xAD\xBE\xEF\x00\x11\x22\x33")
        self.assertEqual(q.packet_count, 7)

    def test_with_class_id_and_trailer(self):
        p = ContextPacket(
            packet_type=PacketType.IF_CONTEXT,
            stream_id=None,
            class_id=(0x00AABB, 0x1122, 0x3344),
            tsi=TSI.NONE,
            tsf=TSF.NONE,
            payload=b"\x01\x02\x03\x04",
            trailer=0xCAFEBABE,
        )
        b = p.pack()
        q = ContextPacket.parse(b)
        self.assertEqual(q.packet_type, PacketType.IF_CONTEXT)
        self.assertIsNone(q.stream_id)
        self.assertEqual(q.class_id, (0x00AABB, 0x1122, 0x3344))
        self.assertEqual(q.tsi, TSI.NONE)
        self.assertEqual(q.tsf, TSF.NONE)
        self.assertEqual(q.payload, b"\x01\x02\x03\x04")
        self.assertEqual(q.trailer, 0xCAFEBABE)


if __name__ == "__main__":
    unittest.main()
