import io
import struct
import unittest
import zlib

from vita49io import CIF0Fields, ContextPacket, DataPacket, PacketType, TSI, TSF
from vita49io.io.packet_reader import PacketReader, RawDataPacket


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


def _vrl_frame(packets: bytes, frame_count: int = 0, crc: bool = False) -> bytes:
    """Wrap VRT packets in a VITA 49.1 VRL frame."""
    assert len(packets) % 4 == 0
    frame_words = len(packets) // 4 + 3
    body = b"VRLP" + struct.pack(">I", ((frame_count & 0xFFF) << 20) | frame_words) + packets
    trailer = struct.pack(">I", zlib.crc32(body)) if crc else b"VEND"
    return body + trailer


class TestPacketReader(unittest.TestCase):
    def test_read_packet_fast_data_packet(self) -> None:
        stream = io.BytesIO(_data_packet(0x12345678, payload=b"\x11\x22\x33\x44"))
        reader = PacketReader(stream)
        pkt = reader.read_packet_fast()
        self.assertIsInstance(pkt, RawDataPacket)
        assert isinstance(pkt, RawDataPacket)
        self.assertEqual(pkt.packet_type, int(PacketType.IF_DATA_WITH_STREAM_ID))
        self.assertEqual(pkt.stream_id, 0x12345678)
        self.assertEqual(bytes(pkt.payload), b"\x11\x22\x33\x44")

    def test_read_packet_fast_context_packet(self) -> None:
        stream = io.BytesIO(_context_packet(0xCAFEBABE))
        reader = PacketReader(stream)
        pkt = reader.read_packet_fast()
        self.assertIsInstance(pkt, ContextPacket)
        assert isinstance(pkt, ContextPacket)
        self.assertEqual(pkt.stream_id, 0xCAFEBABE)

    def test_read_packet_fast_context_then_data(self) -> None:
        stream = io.BytesIO(
            _context_packet(7) + _data_packet(7, payload=b"\xAA\xBB\xCC\xDD")
        )
        reader = PacketReader(stream)

        first = reader.read_packet_fast()
        second = reader.read_packet_fast()

        self.assertIsInstance(first, ContextPacket)
        self.assertIsInstance(second, RawDataPacket)
        assert isinstance(second, RawDataPacket)
        self.assertEqual(second.stream_id, 7)
        self.assertEqual(bytes(second.payload), b"\xAA\xBB\xCC\xDD")

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


class TestPacketReaderVRL(unittest.TestCase):
    """VITA 49.1 VRL frames are stripped transparently."""

    def _read_all(self, data: bytes, fast: bool = False) -> list:
        reader = PacketReader(io.BytesIO(data))
        out = []
        while True:
            pkt = reader.read_packet_fast() if fast else reader.read_packet()
            if pkt is None:
                return out
            out.append(pkt)

    def test_single_packet_per_frame(self) -> None:
        raw = _context_packet(1) + _data_packet(1, payload=b"\x01\x02\x03\x04")
        framed = _vrl_frame(_context_packet(1), 0) + _vrl_frame(
            _data_packet(1, payload=b"\x01\x02\x03\x04"), 1, crc=True
        )
        plain = self._read_all(raw)
        vrl = self._read_all(framed)
        self.assertEqual([p.to_bytes() for p in vrl], [p.to_bytes() for p in plain])

    def test_multiple_packets_per_frame_fast(self) -> None:
        framed = _vrl_frame(
            _context_packet(7) + _data_packet(7, payload=b"\xAA\xBB\xCC\xDD") + _data_packet(8)
        )
        pkts = self._read_all(framed, fast=True)
        self.assertEqual(len(pkts), 3)
        self.assertIsInstance(pkts[0], ContextPacket)
        self.assertIsInstance(pkts[1], RawDataPacket)
        self.assertEqual(bytes(pkts[1].payload), b"\xAA\xBB\xCC\xDD")
        self.assertEqual(pkts[2].stream_id, 8)

    def test_frame_count_exposed(self) -> None:
        reader = PacketReader(io.BytesIO(_vrl_frame(_data_packet(1), frame_count=0xABC)))
        self.assertIsNone(reader.vrl_frame_count)
        reader.read_packet()
        self.assertEqual(reader.vrl_frame_count, 0xABC)

    def test_empty_frame_and_mixed_unframed(self) -> None:
        data = _vrl_frame(b"") + _data_packet(1) + _vrl_frame(_data_packet(2))
        pkts = self._read_all(data)
        self.assertEqual([p.stream_id for p in pkts], [1, 2])

    def test_skip_methods_across_frames(self) -> None:
        data = (
            _vrl_frame(_data_packet(1) + _data_packet(2))
            + _vrl_frame(_context_packet(3) + _data_packet(4))
            + _vrl_frame(_context_packet(5))
        )
        reader = PacketReader(io.BytesIO(data))
        self.assertEqual(reader.skip_data_packets(2), 2)
        ctx = reader.skip_until_next_context_packet()
        self.assertEqual(ctx.stream_id, 3)
        self.assertEqual(reader.skip_until_next_data_packet().stream_id, 4)
        self.assertEqual(reader.skip_packets(5), 1)
        self.assertIsNone(reader.read_packet())

    def test_packet_larger_than_frame_raises(self) -> None:
        pkt = _data_packet(1, payload=b"\x00" * 8)
        frame = bytearray(_vrl_frame(pkt))
        frame[4:8] = struct.pack(">I", 4)  # claim frame holds only 1 word
        with self.assertRaises(ValueError):
            PacketReader(io.BytesIO(bytes(frame))).read_packet()

    def test_truncated_frame_raises(self) -> None:
        frame = _vrl_frame(_data_packet(1))[:-4]  # drop trailer
        reader = PacketReader(io.BytesIO(frame))
        reader.read_packet()
        with self.assertRaises(ValueError):
            reader.read_packet()
