import unittest

from vita49io import DataPacket, ContextPacket, PacketType, TSI, TSF, CIF0Fields


class TestVRT(unittest.TestCase):
    def test_minimal_if_data_roundtrip(self):
        p = DataPacket(
            packet_type=PacketType.IF_DATA_WITH_STREAM_ID,
            stream_id=0x12345678,
            tsi=TSI.UTC,
            tsf=TSF.FRACTIONAL,
            integer_seconds=1700000000,
            fractional_seconds=0x01020304,
            payload=b"\xDE\xAD\xBE\xEF\x00\x11\x22\x33",
            packet_count=7,
        )
        b = p.to_bytes()
        q = DataPacket.from_bytes(b)

        self.assertEqual(q.packet_type, PacketType.IF_DATA_WITH_STREAM_ID)
        self.assertEqual(q.stream_id, 0x12345678)
        self.assertEqual(q.tsi, TSI.UTC)
        self.assertEqual(q.tsf, TSF.FRACTIONAL)
        self.assertEqual(q.integer_seconds, 1700000000)
        self.assertEqual(q.fractional_seconds, 0x01020304)
        self.assertEqual(q.payload, b"\xDE\xAD\xBE\xEF\x00\x11\x22\x33")
        self.assertEqual(q.packet_count, 7)

    def test_with_class_id_and_no_trailer(self):
        # Raw CIF fields to carry alongside CIF0 (1 word)
        extra_bytes = b"\x01\x02\x03\x04"
        extra_word = int.from_bytes(extra_bytes, byteorder="big")
        p = ContextPacket(
            packet_type=PacketType.CONTEXT_PACKET,
            stream_id=0x01020304,
            class_id=(0x00AABB, 0x1122, 0x3344),
            tsi=TSI.NONE,
            tsf=TSF.NONE,
            cif0=CIF0Fields(),
            raw_cif_fields=[extra_word],
        )
        b = p.to_bytes()
        q = ContextPacket.from_bytes(b)
        self.assertEqual(q.packet_type, PacketType.CONTEXT_PACKET)
        self.assertEqual(q.stream_id, 0x01020304)
        self.assertEqual(q.class_id, (0x00AABB, 0x1122, 0x3344))
        self.assertEqual(q.tsi, TSI.NONE)
        self.assertEqual(q.tsf, TSF.NONE)
        # Reconstruct bytes from raw_cif_fields and compare
        self.assertIsNotNone(q.raw_cif_fields)
        got_bytes = b"".join(int(w).to_bytes(4, byteorder="big") for w in q.raw_cif_fields)
        self.assertEqual(got_bytes, extra_bytes)
        # Context packets have no trailer; bit has no meaning
        # Ensure no trailer attribute/field is present
        self.assertFalse(hasattr(q, "trailer"))

    def test_data_header_roundtrip_all_types_tsi_tsf(self):
        # Exercise all data packet types with and without timestamps
        from vita49io import PacketType, TSI, TSF, DataPacket

        pkt_types = [
            PacketType.IF_DATA_WITHOUT_STREAM_ID,
            PacketType.IF_DATA_WITH_STREAM_ID,
            PacketType.EXTENSION_DATA_WITHOUT_STREAM_ID,
            PacketType.EXTENSION_DATA_WITH_STREAM_ID,
        ]

        tsi_values = [TSI.NONE, TSI.UTC, TSI.GPS, TSI.OTHER]
        tsf_values = [TSF.NONE, TSF.SAMPLE_COUNT, TSF.FRACTIONAL, TSF.FREE_RUNNING]

        for pt in pkt_types:
            for tsi in tsi_values:
                for tsf in tsf_values:
                    stream_id = None
                    if pt in (
                        PacketType.IF_DATA_WITH_STREAM_ID,
                        PacketType.EXTENSION_DATA_WITH_STREAM_ID,
                    ):
                        stream_id = 0x11112222

                    integer_seconds = 1700000123 if tsi != TSI.NONE else None
                    fractional_seconds = 0x0000000100000002 if tsf != TSF.NONE else None

                    p = DataPacket(
                        packet_type=pt,
                        stream_id=stream_id,
                        class_id=(0x00A0B1, 0x1234, 0xABCD),
                        tsi=tsi,
                        tsf=tsf,
                        integer_seconds=integer_seconds,
                        fractional_seconds=fractional_seconds,
                        payload=b"\xDE\xAD\xBE\xEF\x00\x11\x22\x33",
                        trailer=0xFEEDC0DE,
                        packet_count=9,
                    )
                    b = p.to_bytes()
                    q = DataPacket.from_bytes(b)

                    self.assertEqual(q.packet_type, pt)
                    self.assertEqual(q.stream_id, stream_id)
                    self.assertEqual(q.class_id, (0x00A0B1, 0x1234, 0xABCD))
                    self.assertEqual(q.tsi, tsi)
                    self.assertEqual(q.tsf, tsf)
                    if tsi == TSI.NONE:
                        self.assertIsNone(q.integer_seconds)
                    else:
                        self.assertEqual(q.integer_seconds, integer_seconds)
                    if tsf == TSF.NONE:
                        self.assertIsNone(q.fractional_seconds)
                    else:
                        self.assertEqual(q.fractional_seconds, fractional_seconds)
                    self.assertEqual(q.payload, b"\xDE\xAD\xBE\xEF\x00\x11\x22\x33")
                    self.assertEqual(q.trailer, 0xFEEDC0DE)
                    self.assertEqual(q.packet_count, 9)

    def test_iq_roundtrip_all_supported_payload_formats(self):
        # Validate encode/decode of IQ using every supported PayloadFormat combination
        import numpy as np
        from vita49io.protocol.cif0 import (
            PayloadFormat,
            PackingMethod,
            SampleType,
            DataItemFormat,
        )

        def make_pf(fmt: DataItemFormat, ipf_bits: int, di_bits: int) -> PayloadFormat:
            return PayloadFormat(
                packing_method=PackingMethod.PROCESSING_EFFICIENT,
                sample_type=SampleType.COMPLEX_CARTESIAN,
                data_item_format_code=int(fmt),
                sample_component_repeat=False,
                event_tag_size_bits=0,
                channel_tag_size_bits=0,
                data_item_fraction_size_bits=0,
                item_packing_field_size_bits=ipf_bits,
                data_item_size_bits=di_bits,
                repeat_count=1,
                vector_size=0,
                data_item_format=fmt,
            )

        # Test vectors
        iq_signed = np.array(
            [0.0 + 0.0j, 0.5 - 0.5j, -0.75 + 0.2j, 0.9999 - 0.9999j], dtype=np.complex64
        )
        iq_unsigned = np.array(
            [0.0 + 0.0j, 0.5 + 0.25j, 0.9 + 0.1j, 1.0 + 0.0j], dtype=np.complex64
        )

        combos = []
        # Signed fixed-point
        for ipf, di in [(16, 16), (32, 16), (32, 24), (32, 32)]:
            combos.append((DataItemFormat.SIGNED_FIXED_POINT, ipf, di, iq_signed))
        # Unsigned fixed-point
        for ipf, di in [(16, 16), (32, 16), (32, 24), (32, 32)]:
            combos.append((DataItemFormat.UNSIGNED_FIXED_POINT, ipf, di, iq_unsigned))
        # IEEE754 single
        combos.append((DataItemFormat.IEEE754_SINGLE, 32, 32, iq_signed))

        for fmt, ipf, di, base_iq in combos:
            pf = make_pf(fmt, ipf, di)
            p = DataPacket(
                packet_type=PacketType.IF_DATA_WITH_STREAM_ID,
                stream_id=0x22223333,
                tsi=TSI.NONE,
                tsf=TSF.NONE,
                iq=base_iq,
                packet_count=3,
            )
            b = p.to_bytes(payload_format=pf)
            q = DataPacket.from_bytes(b, payload_format=pf)

            self.assertEqual(q.packet_type, PacketType.IF_DATA_WITH_STREAM_ID)
            self.assertEqual(q.stream_id, 0x22223333)
            self.assertIsNotNone(q.iq)
            self.assertEqual(q.packet_count, 3)

            got = q.iq  # type: ignore[assignment]
            # Tolerance based on quantization step
            if fmt == DataItemFormat.IEEE754_SINGLE:
                atol = 1e-6
            else:
                if fmt == DataItemFormat.SIGNED_FIXED_POINT:
                    step = 1.0 / (1 << (di - 1))
                else:
                    step = 1.0 / (1 << di)
                # Account for float32 rounding noise when di >= 24
                atol = max(step * 2.0, 2e-7)
                # Empirically, u32/32 can accumulate slightly larger float32 error
                if fmt == DataItemFormat.UNSIGNED_FIXED_POINT and di == 32:
                    atol = max(atol, 1e-6)
            self.assertTrue(np.allclose(got, base_iq, atol=atol), msg=f"Mismatch for fmt={fmt.name}, ipf={ipf}, di={di}")

    def test_data_packet_reference_level_scaling(self):
        import numpy as np
        from vita49io.protocol.cif0 import (
            PayloadFormat,
            PackingMethod,
            SampleType,
            DataItemFormat,
        )

        ref_dbm = -3.0
        power_w = 10 ** ((ref_dbm - 30.0) / 10.0)
        vpk = float(np.sqrt(2.0 * 50.0 * power_w))

        pf = PayloadFormat(
            packing_method=PackingMethod.PROCESSING_EFFICIENT,
            sample_type=SampleType.COMPLEX_CARTESIAN,
            data_item_format_code=int(DataItemFormat.SIGNED_FIXED_POINT),
            sample_component_repeat=False,
            event_tag_size_bits=0,
            channel_tag_size_bits=0,
            data_item_fraction_size_bits=0,
            item_packing_field_size_bits=16,
            data_item_size_bits=16,
            repeat_count=1,
            vector_size=0,
            data_item_format=DataItemFormat.SIGNED_FIXED_POINT,
        )

        iq_volts = np.array([0.0 + 0.0j, 0.5 * vpk - 0.25 * vpk * 1j], dtype=np.complex64)
        packet = DataPacket(
            packet_type=PacketType.IF_DATA_WITH_STREAM_ID,
            stream_id=0x55556666,
            tsi=TSI.NONE,
            tsf=TSF.NONE,
            iq=iq_volts,
        )
        payload = packet.to_bytes(payload_format=pf, reference_level_dbm=ref_dbm)
        decoded = DataPacket.from_bytes(payload, payload_format=pf)

        self.assertIsNotNone(decoded.iq)
        expected = iq_volts / vpk
        step = 1.0 / (1 << 15)
        self.assertTrue(
            np.allclose(decoded.iq, expected, atol=step * 2.5),
            msg="Reference-level scaling did not normalize IQ as expected",
        )

    def test_iq_stream_writer_reference_level_bytes(self):
        import numpy as np
        from vita49io.io import IQStreamWriter

        ref_dbm = -6.0
        writer = IQStreamWriter(
            stream_id=3,
            sample_rate_hz=1e6,
            normalize_iq_to_reference_level=True,
            reference_level_dbm=ref_dbm,
        )

        power_w = 10 ** ((ref_dbm - 30.0) / 10.0)
        vpk = float(np.sqrt(2.0 * 50.0 * power_w))
        iq_volts = np.array([0.0 + 0.0j, 0.5 * vpk + 0.0j, 0.0 + 0.25 * vpk * 1j], dtype=np.complex64)
        payload = writer.build_data_packet_bytes(iq_volts)
        packet = DataPacket.from_bytes(payload, payload_format=writer.payload_format)

        self.assertIsNotNone(packet.iq)
        expected = np.array([0.0 + 0.0j, 0.5 + 0.0j, 0.0 + 0.25j], dtype=np.complex64)
        self.assertTrue(np.allclose(packet.iq, expected, atol=1e-6))

    def test_iq_stream_writer_reference_level_requires_value(self):
        import numpy as np
        from vita49io.io import IQStreamWriter

        writer = IQStreamWriter(
            stream_id=4,
            sample_rate_hz=1e6,
            normalize_iq_to_reference_level=True,
        )

        with self.assertRaises(ValueError):
            writer.build_data_packet_bytes(np.zeros(4, dtype=np.complex64))


if __name__ == "__main__":
    unittest.main()
