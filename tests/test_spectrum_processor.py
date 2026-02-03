import os
import sys

# Ensure the repo's src/ layout is importable even when an older version of the
# package is installed in the environment.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC = os.path.join(_ROOT, "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import io
import unittest

import numpy as np

from vita49io.io import IQStreamWriter, SpectrumStreamProcessor
from vita49io.protocol.context_packet import ContextPacket
from vita49io.protocol.data_packet import DataPacket
from vita49io.protocol.cif0 import SampleType
from vita49io.protocol.cif1 import AveragingType, SpectrumType
from vita49io.signal.spectrum import SpectrumProcessor


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

    def test_peak_hold_window(self) -> None:
        stream_id = 0x0A0B0C0D
        sample_rate_hz = 1_000_000.0
        bandwidth_hz = 500_000.0

        writer = IQStreamWriter(
            stream_id=stream_id,
            sample_rate_hz=sample_rate_hz,
            bandwidth_hz=bandwidth_hz,
        )

        iq = (np.arange(256) + 1j * np.arange(256)).astype(np.complex64)
        data = writer.build_data_packet(iq)
        ctx = writer.build_context_packet()

        stream = io.BytesIO(ctx.to_bytes() + data.to_bytes())

        processor = SpectrumStreamProcessor(
            stream=stream,
            fft_size=64,
            hop_size=32,
            window_type="kaiser",
            window_param=6.0,
            averaging_mode="peak_hold",
            averaging_param=0,
            output_fps=sample_rate_hz / 32,
            band_mode="full",
        )

        out_packets = list(processor.read_packets())
        ctx_packets = [p for p in out_packets if isinstance(p, ContextPacket)]
        data_packets = [p for p in out_packets if isinstance(p, DataPacket)]

        self.assertTrue(ctx_packets, "No context packet emitted")
        self.assertTrue(data_packets, "No data packets emitted")

        ctx_out = ctx_packets[0]
        assert ctx_out.cif0 is not None
        assert ctx_out.cif0.cif1 is not None
        assert ctx_out.cif0.cif1.spectrum is not None
        self.assertEqual(ctx_out.cif0.cif1.spectrum.averaging_type, AveragingType.PEAK_HOLD)
        self.assertEqual(ctx_out.cif0.cif1.spectrum.span_hz, sample_rate_hz)

        payload = data_packets[0].payload
        payload_bytes = payload.tobytes() if isinstance(payload, memoryview) else payload
        vals = np.frombuffer(payload_bytes, dtype=">f4")
        self.assertEqual(vals.size, 64)

    def test_frame_mean_packets(self) -> None:
        stream_id = 0xDEADBEEF
        sample_rate_hz = 1_000_000.0
        bandwidth_hz = 1_000_000.0

        fft_size = 1024
        hop_size = 1024
        output_fps = 20.0
        expected_nominal = int(round(sample_rate_hz / (hop_size * output_fps)))
        self.assertEqual(expected_nominal, 49)

        # Use 16-bit fixed point so the packet stays within the 16-bit packet_size (words) field.
        from vita49io.protocol.cif0 import DataItemFormat

        writer = IQStreamWriter(
            stream_id=stream_id,
            sample_rate_hz=sample_rate_hz,
            bandwidth_hz=bandwidth_hz,
            data_item_format=DataItemFormat.SIGNED_FIXED_POINT,
            item_packing_field_size_bits=16,
            data_item_size_bits=16,
        )

        num_ffts = expected_nominal + 5
        n = fft_size + num_ffts * hop_size
        rng_i = np.random.default_rng(0)
        rng_q = np.random.default_rng(1)
        iq = (0.1 * (rng_i.standard_normal(n) + 1j * rng_q.standard_normal(n))).astype(np.complex64)

        data = writer.build_data_packet(iq)
        ctx = writer.build_context_packet()
        stream = io.BytesIO(ctx.to_bytes() + data.to_bytes())

        processor = SpectrumStreamProcessor(
            stream=stream,
            fft_size=fft_size,
            hop_size=hop_size,
            window_type="rect",
            averaging_mode="frame_mean",
            averaging_param=0,
            output_fps=output_fps,
            band_mode="full",
        )

        out_packets = list(processor.read_packets())
        ctx_packets = [p for p in out_packets if isinstance(p, ContextPacket)]
        data_packets = [p for p in out_packets if isinstance(p, DataPacket)]

        self.assertTrue(ctx_packets, "No context packet emitted")
        self.assertTrue(data_packets, "No data packets emitted")

        ctx_out = ctx_packets[0]
        assert ctx_out.cif0 is not None
        assert ctx_out.cif0.cif1 is not None
        assert ctx_out.cif0.cif1.spectrum is not None
        self.assertEqual(ctx_out.cif0.cif1.spectrum.averaging_type, AveragingType.LINEAR)
        self.assertEqual(ctx_out.cif0.cif1.spectrum.number_of_averages, expected_nominal)

        payload = data_packets[0].payload
        payload_bytes = payload.tobytes() if isinstance(payload, memoryview) else payload
        vals = np.frombuffer(payload_bytes, dtype=">f4")
        self.assertEqual(vals.size, fft_size)

    def test_frame_mean_counts(self) -> None:
        sample_rate_hz = 1_000_000.0
        fft_size = 1024
        hop_size = 1024
        output_fps = 20.0

        expected_nominal = int(round(sample_rate_hz / (hop_size * output_fps)))
        self.assertEqual(expected_nominal, 49)

        proc = SpectrumProcessor(
            sample_rate_hz=sample_rate_hz,
            bandwidth_hz=sample_rate_hz,
            band_mode="full",
            fft_size=fft_size,
            hop_size=hop_size,
            window_type="rect",
            window_param=None,
            dc_block=False,
            averaging_mode="frame_mean",
            averaging_param=0,
            output_fps=output_fps,
            output_bins=None,
        )

        num_ffts = expected_nominal + 5
        rng = np.random.default_rng(0)
        iq = (rng.standard_normal(fft_size + num_ffts * hop_size) + 1j * rng.standard_normal(fft_size + num_ffts * hop_size)).astype(
            np.complex64
        )
        frames = proc.push(iq)
        self.assertTrue(frames, "No frames emitted")
        n = frames[0].meta["num_ffts_averaged"]
        self.assertLessEqual(abs(int(n) - expected_nominal), 1)


if __name__ == "__main__":
    unittest.main()
