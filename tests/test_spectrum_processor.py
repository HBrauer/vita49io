import io
import unittest

import numpy as np

from vita49io.io import IQStreamWriter, SpectrumStreamProcessor
from vita49io.protocol.context_packet import ContextPacket
from vita49io.protocol.data_packet import DataPacket
from vita49io.protocol.cif0 import SampleType
from vita49io.protocol.cif1 import SpectrumType


class TestSpectrumStreamProcessor(unittest.TestCase):
    def test_iq_to_spectrum_packets(self) -> None:
        stream_id = 0x1234ABCD
        sample_rate_hz = 1_000_000.0
        bandwidth_hz = 200_000.0

        writer = IQStreamWriter(
            stream_id=stream_id,
            sample_rate_hz=sample_rate_hz,
            bandwidth_hz=bandwidth_hz,
        )

        iq1 = (np.arange(64) + 1j * np.arange(64)).astype(np.complex64)
        iq2 = (np.arange(64, 128) + 1j * np.arange(64, 128)).astype(np.complex64)

        data1 = writer.build_data_packet(iq1)
        ctx = writer.build_context_packet()
        data2 = writer.build_data_packet(iq2)

        # Intentionally place data before context to exercise the scan/restart logic.
        stream = io.BytesIO(data1.to_bytes() + ctx.to_bytes() + data2.to_bytes())

        processor = SpectrumStreamProcessor(
            stream=stream,
            fft_size=16,
            hop_size=8,
            window_type="hann",
            averaging_mode="mean",
            averaging_param=2,
            output_fps=sample_rate_hz / 8,
            band_mode="inband",
        )

        out_packets = list(processor.read_packets())
        self.assertTrue(out_packets, "No output packets produced")

        ctx_packets = [p for p in out_packets if isinstance(p, ContextPacket)]
        data_packets = [p for p in out_packets if isinstance(p, DataPacket)]

        self.assertTrue(ctx_packets, "No context packet emitted")
        self.assertTrue(data_packets, "No data packets emitted")

        ctx_out = ctx_packets[0]
        self.assertIsNotNone(ctx_out.cif0)
        assert ctx_out.cif0 is not None
        self.assertEqual(float(ctx_out.cif0.sample_rate_hz), sample_rate_hz)
        self.assertEqual(float(ctx_out.cif0.bandwidth_hz), bandwidth_hz)
        self.assertIsNotNone(ctx_out.cif0.payload_format)
        self.assertEqual(ctx_out.cif0.payload_format.sample_type, SampleType.REAL)
        self.assertIsNotNone(ctx_out.cif0.cif1)
        assert ctx_out.cif0.cif1 is not None
        self.assertIsNotNone(ctx_out.cif0.cif1.spectrum)
        assert ctx_out.cif0.cif1.spectrum is not None
        self.assertEqual(ctx_out.cif0.cif1.spectrum.spectrum_type, SpectrumType.LOG_POWER_DB)
        self.assertEqual(ctx_out.cif0.cif1.spectrum.window_type, 2)
        self.assertEqual(ctx_out.cif0.cif1.spectrum.span_hz, bandwidth_hz)

        freqs = np.fft.fftshift(np.fft.fftfreq(16, d=1.0 / sample_rate_hz))
        half_bw = bandwidth_hz / 2.0
        expected_bins = int(np.count_nonzero((freqs >= -half_bw) & (freqs <= half_bw)))

        for pkt in data_packets:
            self.assertTrue(pkt.header.indicators_24, "Output data packet missing S-bit")
            payload = pkt.payload
            payload_bytes = payload.tobytes() if isinstance(payload, memoryview) else payload
            vals = np.frombuffer(payload_bytes, dtype=">f4")
            self.assertEqual(vals.size, expected_bins)


if __name__ == "__main__":
    unittest.main()
