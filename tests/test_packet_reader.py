import io
import unittest

from vita49io import CIF0Fields, ContextPacket, DataPacket, PacketType, TSI, TSF
from vita49io.io.packet_reader import PacketReader


def _data_packet(stream_id: int, payload: bytes = b"\x00") -> bytes:
    pkt = DataPacket(
        packet_type=PacketType.IF_DATA_WITH_STREAM_ID,
        stream_id=stream_id,
        tsi=TSI.NONE,
        tsf=TSF.NONE,
        payload=payload,
        packet_count=0,
    )
    return pkt.to_bytes()


def _context_packet(stream_id: int) -> bytes:
    pkt = ContextPacket(
        packet_type=PacketType.CONTEXT_PACKET,
        stream_id=stream_id,
        tsi=TSI.NONE,
        tsf=TSF.NONE,
        cif0=CIF0Fields(),
    )
    return pkt.to_bytes()


class TestPacketReader(unittest.TestCase):
    def test_skip_packets(self) -> None:
        stream = io.BytesIO(
            _data_packet(1)
            + _context_packet(2)
            + _data_packet(3, payload=b"\xAA")
        )
        reader = PacketReader(stream)
        skipped = reader.skip_packets(2)
        self.assertEqual(skipped, 2)
        pkt = reader.read_packet()
        self.assertIsNotNone(pkt)
        self.assertEqual(pkt.stream_id, 3)

    def test_skip_data_packets(self) -> None:
        stream = io.BytesIO(
            _data_packet(1)
            + _context_packet(10)
            + _data_packet(2, payload=b"\xBB")
            + _context_packet(20)
        )
        reader = PacketReader(stream)
        skipped = reader.skip_data_packets(2)
        self.assertEqual(skipped, 2)
        pkt = reader.read_packet()
        self.assertIsInstance(pkt, ContextPacket)
        self.assertEqual(pkt.stream_id, 20)

    def test_skip_context_packets(self) -> None:
        stream = io.BytesIO(
            _context_packet(1)
            + _data_packet(10)
            + _context_packet(2)
            + _data_packet(20)
        )
        reader = PacketReader(stream)
        skipped = reader.skip_context_packets(2)
        self.assertEqual(skipped, 2)
        pkt = reader.read_packet()
        self.assertIsInstance(pkt, DataPacket)
        self.assertEqual(pkt.stream_id, 20)

    def test_skip_until_next_context_packet(self) -> None:
        stream = io.BytesIO(
            _data_packet(1)
            + _data_packet(2)
            + _context_packet(42)
            + _data_packet(3)
        )
        reader = PacketReader(stream)
        pkt = reader.skip_until_next_context_packet()
        self.assertIsInstance(pkt, ContextPacket)
        self.assertEqual(pkt.stream_id, 42)
        next_pkt = reader.read_packet()
        self.assertIsInstance(next_pkt, DataPacket)
        self.assertEqual(next_pkt.stream_id, 3)

    def test_skip_until_next_data_packet(self) -> None:
        stream = io.BytesIO(
            _context_packet(7)
            + _context_packet(8)
            + _data_packet(9)
            + _context_packet(10)
        )
        reader = PacketReader(stream)
        pkt = reader.skip_until_next_data_packet()
        self.assertIsInstance(pkt, DataPacket)
        self.assertEqual(pkt.stream_id, 9)
        next_pkt = reader.read_packet()
        self.assertIsInstance(next_pkt, ContextPacket)
        self.assertEqual(next_pkt.stream_id, 10)
