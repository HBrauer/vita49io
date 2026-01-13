import tempfile
import unittest
from pathlib import Path

import numpy as np

from vita49io.protocol.cif0 import (
    CIF0Fields,
    DataItemFormat,
    PackingMethod,
    PayloadFormat,
    SampleType,
)
from vita49io.protocol.context_packet import ContextPacket
from vita49io.protocol.core import Header
from vita49io.protocol.data_packet import DataPacket
from vita49io.protocol.enums import PacketType, TSI, TSF
from vita49io.io.payload_codec import payload_as_numpy, payload_from_numpy


class TestProcessingEfficientFile(unittest.TestCase):
    def test_signed_fixed_point_payload_round_trip(self) -> None:
        """Write a VITA 49 file with a processing-efficient payload format and verify decoded IQ."""
        payload_format = PayloadFormat(
            packing_method=PackingMethod.PROCESSING_EFFICIENT,
            sample_type=SampleType.COMPLEX_CARTESIAN,
            sample_component_repeat=False,
            event_tag_size_bits=0,
            channel_tag_size_bits=0,
            data_item_fraction_size_bits=0,
            item_packing_field_size_bits=16,
            data_item_size_bits=16,
            repeat_count=1,
            vector_size=1,
            data_item_format=DataItemFormat.SIGNED_FIXED_POINT,
        )

        stream_id = 0x91A2B3C4

        context_packet = ContextPacket(
            packet_type=PacketType.CONTEXT_PACKET,
            stream_id=stream_id,
            tsi=TSI.NONE,
            tsf=TSF.NONE,
            cif0=CIF0Fields(
                payload_format=payload_format,
            ),
        )

        scale = float(1 << 15)
        max_raw = (1 << 15) - 1  # 32767
        min_raw = -(1 << 15)  # -32768
        half_raw = max_raw // 2  # 16383

        def _normalize(value: int) -> float:
            return value / scale

        # Include max, min, and half-scale values across I/Q components
        expected_iq = np.array(
            [
                complex(_normalize(max_raw), _normalize(min_raw)),
                complex(_normalize(min_raw), _normalize(half_raw)),
                complex(_normalize(half_raw), _normalize(max_raw)),
            ],
            dtype=np.complex64,
        )

        data_packet = DataPacket(
            packet_type=PacketType.IF_DATA_WITH_STREAM_ID,
            stream_id=stream_id,
            tsi=TSI.NONE,
            tsf=TSF.NONE,
            payload=payload_from_numpy(expected_iq, payload_format),
            packet_count=1,
        )

        context_bytes = context_packet.to_bytes()
        data_bytes = data_packet.to_bytes()

  
        

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "processing_efficient.v49"
            with path.open("wb") as f:
                f.write(context_bytes)
                f.write(data_bytes)

            last_payload_format = None
            decoded_iq = None

            with path.open("rb") as f:
                while True:
                    header_bytes = f.read(4)
                    if not header_bytes:
                        break

                    header_word = int.from_bytes(header_bytes, byteorder="big")
                    header = Header.parse(header_word)
                    payload_length_bytes = (header.packet_size - 1) * 4
                    payload_bytes = f.read(payload_length_bytes)
                    self.assertEqual(
                        len(payload_bytes),
                        payload_length_bytes,
                        "Truncated packet payload while reading test file",
                    )
                    packet_bytes = header_bytes + payload_bytes

                    if header.packet_type == PacketType.CONTEXT_PACKET:
                        ctx = ContextPacket.from_bytes(packet_bytes)
                        self.assertIsNotNone(ctx.cif0)
                        self.assertIsNotNone(ctx.cif0.payload_format)
                        last_payload_format = ctx.cif0.payload_format
                    else:
                        self.assertIsNotNone(
                            last_payload_format,
                            "Data packet encountered before payload format was defined",
                        )
                        data = DataPacket.from_bytes(packet_bytes)
                        payload = data.payload
                        payload_bytes = payload.tobytes() if isinstance(payload, memoryview) else payload
                        decoded_iq = payload_as_numpy(payload_bytes, last_payload_format)

            self.assertIsNotNone(decoded_iq, "No data packet decoded from test file")
            iq_array = decoded_iq
            assert iq_array is not None  # Narrow type for mypy

            self.assertEqual(iq_array.dtype, np.complex64)
            quantization_step = 1.0 / (1 << 15)
            print(iq_array)
            self.assertTrue(
                np.allclose(iq_array, expected_iq, atol=quantization_step),
                msg=f"Decoded IQ values {iq_array} did not match expected {expected_iq}",
            )

            # Ensure round-tripped integers match the original full-scale values
            decoded_components = np.column_stack((iq_array.real, iq_array.imag))
            decoded_fixed = np.rint(decoded_components * scale).astype(np.int32)
            expected_fixed = np.array(
                [
                    [max_raw, min_raw],
                    [min_raw, half_raw],
                    [half_raw, max_raw],
                ],
                dtype=np.int32,
            )
            np.testing.assert_array_equal(decoded_fixed, expected_fixed)


if __name__ == "__main__":
    unittest.main()
