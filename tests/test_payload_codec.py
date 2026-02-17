import unittest

import numpy as np

from vita49io.io.payload_codec import (
    build_payload_decoder,
    payload_as_numpy,
    payload_as_numpy_view,
    payload_from_numpy,
)
from vita49io.protocol.cif0 import (
    DataItemFormat,
    PackingMethod,
    PayloadFormat,
    SampleType,
)


def _make_pf(
    *,
    sample_type: SampleType,
    data_item_format: DataItemFormat,
    item_packing_field_size_bits: int,
    data_item_size_bits: int,
) -> PayloadFormat:
    return PayloadFormat(
        packing_method=PackingMethod.PROCESSING_EFFICIENT,
        sample_type=sample_type,
        data_item_format=data_item_format,
        sample_component_repeat=False,
        event_tag_size_bits=0,
        channel_tag_size_bits=0,
        data_item_fraction_size_bits=0,
        item_packing_field_size_bits=int(item_packing_field_size_bits),
        data_item_size_bits=int(data_item_size_bits),
        repeat_count=1,
        vector_size=0,
    )


class TestPayloadCodec(unittest.TestCase):
    def test_build_payload_decoder_reuse_and_memoryview(self) -> None:
        pf = _make_pf(
            sample_type=SampleType.COMPLEX_CARTESIAN,
            data_item_format=DataItemFormat.SIGNED_FIXED_POINT,
            item_packing_field_size_bits=16,
            data_item_size_bits=16,
        )
        samples = np.array([0.25 - 0.125j, -0.5 + 0.75j], dtype=np.complex64)
        payload = payload_from_numpy(samples, pf)

        decoder1 = build_payload_decoder(pf)
        decoder2 = build_payload_decoder(pf)
        self.assertIs(decoder1, decoder2)

        decoded_fast = decoder1(memoryview(payload))
        decoded_ref = payload_as_numpy(payload, pf)
        self.assertEqual(decoded_fast.dtype, np.dtype(np.complex64))
        np.testing.assert_allclose(decoded_fast, decoded_ref, atol=1e-6)

    def test_real_s32_decodes_to_float32(self) -> None:
        pf = _make_pf(
            sample_type=SampleType.REAL,
            data_item_format=DataItemFormat.SIGNED_FIXED_POINT,
            item_packing_field_size_bits=32,
            data_item_size_bits=32,
        )
        samples = np.array([0.0, 0.5, -0.5], dtype=np.float32)
        payload = payload_from_numpy(samples, pf)
        decoded = payload_as_numpy(payload, pf)
        self.assertEqual(decoded.dtype, np.dtype(np.float32))
        np.testing.assert_allclose(decoded, samples, atol=1e-6)

    def test_complex_float32_roundtrip(self) -> None:
        pf = _make_pf(
            sample_type=SampleType.COMPLEX_CARTESIAN,
            data_item_format=DataItemFormat.IEEE754_SINGLE,
            item_packing_field_size_bits=32,
            data_item_size_bits=32,
        )
        samples = np.array([1.0 + 0.5j, -0.25 - 0.75j], dtype=np.complex64)
        payload = payload_from_numpy(samples, pf)
        decoded = payload_as_numpy(payload, pf)
        np.testing.assert_allclose(decoded, samples, atol=1e-6)

        # Check exact on-wire bytes for IEEE754 single (big-endian)
        expected = np.array([1.0, 0.5, -0.25, -0.75], dtype=">f4").tobytes()
        self.assertEqual(payload, expected)

    def test_real_float32_roundtrip(self) -> None:
        pf = _make_pf(
            sample_type=SampleType.REAL,
            data_item_format=DataItemFormat.IEEE754_SINGLE,
            item_packing_field_size_bits=32,
            data_item_size_bits=32,
        )
        samples = np.array([1.0, -0.5, 0.25], dtype=np.float32)
        payload = payload_from_numpy(samples, pf)
        decoded = payload_as_numpy(payload, pf)
        np.testing.assert_allclose(decoded, samples, atol=1e-6)

        expected = np.array([1.0, -0.5, 0.25], dtype=">f4").tobytes()
        self.assertEqual(payload, expected)

    def test_real_fixed_point_view(self) -> None:
        pf = _make_pf(
            sample_type=SampleType.REAL,
            data_item_format=DataItemFormat.SIGNED_FIXED_POINT,
            item_packing_field_size_bits=16,
            data_item_size_bits=16,
        )
        samples = np.array([0.0, 0.5, -0.5], dtype=np.float32)
        payload = payload_from_numpy(samples, pf)
        view = payload_as_numpy_view(payload, pf)
        self.assertEqual(view.dtype, np.dtype(">i2"))
        self.assertEqual(view.shape, (3,))

    def test_complex_signed_fixed_point_roundtrip(self) -> None:
        pf = _make_pf(
            sample_type=SampleType.COMPLEX_CARTESIAN,
            data_item_format=DataItemFormat.SIGNED_FIXED_POINT,
            item_packing_field_size_bits=16,
            data_item_size_bits=16,
        )
        samples = np.array([0.0 + 0.0j, 0.5 - 0.25j, -1.0 + 0.75j], dtype=np.complex64)
        payload = payload_from_numpy(samples, pf)
        decoded = payload_as_numpy(payload, pf)
        quantization = 1.0 / (1 << 15)
        np.testing.assert_allclose(decoded, samples, atol=quantization)

        expected_i16 = np.array([0, 0, 16384, -8192, -32768, 24576], dtype=">i2").tobytes()
        self.assertEqual(payload, expected_i16)

    def test_complex_unsigned_fixed_point_roundtrip(self) -> None:
        pf = _make_pf(
            sample_type=SampleType.COMPLEX_CARTESIAN,
            data_item_format=DataItemFormat.UNSIGNED_FIXED_POINT,
            item_packing_field_size_bits=16,
            data_item_size_bits=16,
        )
        samples = np.array([0.0 + 0.0j, 0.5 + 0.25j, 1.0 + 0.75j], dtype=np.complex64)
        payload = payload_from_numpy(samples, pf)
        decoded = payload_as_numpy(payload, pf)
        quantization = 1.0 / (1 << 16)
        np.testing.assert_allclose(decoded, samples, atol=quantization)

        expected_u16 = np.array([0, 0, 32768, 16384, 65535, 49152], dtype=">u2").tobytes()
        self.assertEqual(payload, expected_u16)

    def test_real_signed_fixed_point_24_in_32_encoding(self) -> None:
        pf = _make_pf(
            sample_type=SampleType.REAL,
            data_item_format=DataItemFormat.SIGNED_FIXED_POINT,
            item_packing_field_size_bits=32,
            data_item_size_bits=24,
        )
        samples = np.array([0.5, -0.5], dtype=np.float32)
        payload = payload_from_numpy(samples, pf)
        decoded = payload_as_numpy(payload, pf)
        quantization = 1.0 / (1 << 23)
        np.testing.assert_allclose(decoded, samples, atol=quantization)

        expected_u32 = np.array([0x00400000, 0x00C00000], dtype=">u4").tobytes()
        self.assertEqual(payload, expected_u32)

    def test_complex_view_for_float32(self) -> None:
        pf = _make_pf(
            sample_type=SampleType.COMPLEX_CARTESIAN,
            data_item_format=DataItemFormat.IEEE754_SINGLE,
            item_packing_field_size_bits=32,
            data_item_size_bits=32,
        )
        samples = np.array([0.75 + 0.5j, -0.5 + 0.25j], dtype=np.complex64)
        payload = payload_from_numpy(samples, pf)
        view = payload_as_numpy_view(payload, pf)
        self.assertEqual(view.dtype, np.dtype(">c8"))
        np.testing.assert_allclose(view.astype(np.complex64), samples, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
