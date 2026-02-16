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

    def test_default_hop_size_matches_fft_size(self) -> None:
        sample_rate_hz = 640.0
        fft_size = 64
        output_fps = 10.0
        n_samples = 8 * fft_size

        rng = np.random.default_rng(123)
        iq = (rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)).astype(np.complex64)

        proc_default = SpectrumProcessor(
            sample_rate_hz=sample_rate_hz,
            bandwidth_hz=sample_rate_hz,
            band_mode="full",
            fft_size=fft_size,
            window_type="hann",
            averaging_mode="none",
            averaging_param=0,
            output_fps=output_fps,
            output_bins=None,
            power_scale="raw",
            processing_mode="continuous",
        )
        proc_explicit = SpectrumProcessor(
            sample_rate_hz=sample_rate_hz,
            bandwidth_hz=sample_rate_hz,
            band_mode="full",
            fft_size=fft_size,
            hop_size=fft_size,
            window_type="hann",
            averaging_mode="none",
            averaging_param=0,
            output_fps=output_fps,
            output_bins=None,
            power_scale="raw",
            processing_mode="continuous",
        )

        frames_a = proc_default.push(iq)
        frames_b = proc_explicit.push(iq)
        self.assertEqual(len(frames_a), len(frames_b))
        self.assertTrue(frames_a)
        for fa, fb in zip(frames_a, frames_b):
            self.assertAlmostEqual(fa.timestamp, fb.timestamp, places=12)
            np.testing.assert_allclose(fa.spectrum_db, fb.spectrum_db, rtol=0.0, atol=0.0)

    def test_stream_default_hop_size_matches_fft_size(self) -> None:
        sample_rate_hz = 640.0
        fft_size = 64
        output_fps = 10.0
        stream_id = 0x10203040

        writer = IQStreamWriter(
            stream_id=stream_id,
            sample_rate_hz=sample_rate_hz,
            bandwidth_hz=sample_rate_hz,
        )

        rng = np.random.default_rng(55)
        n_samples = 8 * fft_size
        iq = (rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)).astype(np.complex64)
        blob = writer.build_context_packet().to_bytes() + writer.build_data_packet(iq).to_bytes()

        proc_default = SpectrumStreamProcessor(
            stream=io.BytesIO(blob),
            fft_size=fft_size,
            window_type="hann",
            averaging_mode="none",
            averaging_param=0,
            output_fps=output_fps,
            processing_mode="continuous",
            band_mode="full",
            power_scale="raw",
        )
        proc_explicit = SpectrumStreamProcessor(
            stream=io.BytesIO(blob),
            fft_size=fft_size,
            hop_size=fft_size,
            window_type="hann",
            averaging_mode="none",
            averaging_param=0,
            output_fps=output_fps,
            processing_mode="continuous",
            band_mode="full",
            power_scale="raw",
        )

        out_a = [p for p in proc_default.read_packets() if isinstance(p, DataPacket)]
        out_b = [p for p in proc_explicit.read_packets() if isinstance(p, DataPacket)]
        self.assertEqual(len(out_a), len(out_b))
        self.assertTrue(out_a)

        for pa, pb in zip(out_a, out_b):
            ba = pa.payload.tobytes() if isinstance(pa.payload, memoryview) else pa.payload
            bb = pb.payload.tobytes() if isinstance(pb.payload, memoryview) else pb.payload
            self.assertEqual(ba, bb)

    def test_stream_snapshot_mode_hop_size_independent(self) -> None:
        sample_rate_hz = 640.0
        fft_size = 64
        output_fps = 10.0
        stream_id = 0x2468ACE0

        writer = IQStreamWriter(
            stream_id=stream_id,
            sample_rate_hz=sample_rate_hz,
            bandwidth_hz=sample_rate_hz,
        )

        rng = np.random.default_rng(5)
        n_samples = 8 * fft_size
        iq = (rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)).astype(np.complex64)
        blob = writer.build_context_packet().to_bytes() + writer.build_data_packet(iq).to_bytes()

        proc_a = SpectrumStreamProcessor(
            stream=io.BytesIO(blob),
            fft_size=fft_size,
            hop_size=1,
            window_type="hann",
            averaging_mode="none",
            averaging_param=0,
            output_fps=output_fps,
            processing_mode="snapshot",
            band_mode="full",
            power_scale="raw",
        )
        proc_b = SpectrumStreamProcessor(
            stream=io.BytesIO(blob),
            fft_size=fft_size,
            hop_size=4096,
            window_type="hann",
            averaging_mode="none",
            averaging_param=0,
            output_fps=output_fps,
            processing_mode="snapshot",
            band_mode="full",
            power_scale="raw",
        )

        out_a = [p for p in proc_a.read_packets() if isinstance(p, DataPacket)]
        out_b = [p for p in proc_b.read_packets() if isinstance(p, DataPacket)]
        self.assertEqual(len(out_a), len(out_b))
        self.assertTrue(out_a)

        for pa, pb in zip(out_a, out_b):
            ba = pa.payload.tobytes() if isinstance(pa.payload, memoryview) else pa.payload
            bb = pb.payload.tobytes() if isinstance(pb.payload, memoryview) else pb.payload
            self.assertEqual(ba, bb)

    def test_dbfs_scaling_tone(self) -> None:
        # With dbfs scaling enabled, a full-scale bin-centered complex tone should
        # land near 0 dBFS independent of fft_size. Raw scaling grows with fft_size.
        sample_rate_hz = 1024.0
        fft_size = 1024
        hop_size = 1024
        k = 17

        n = np.arange(fft_size, dtype=np.float32)
        tone = np.exp(2j * np.pi * k * n / float(fft_size)).astype(np.complex64)

        proc_dbfs = SpectrumProcessor(
            sample_rate_hz=sample_rate_hz,
            bandwidth_hz=sample_rate_hz,
            band_mode="full",
            fft_size=fft_size,
            hop_size=hop_size,
            window_type="rect",
            averaging_mode="none",
            averaging_param=0,
            output_fps=1.0,
            output_bins=None,
            power_scale="dbfs",
        )
        frames_dbfs = proc_dbfs.push(tone)
        self.assertTrue(frames_dbfs, "No frames emitted for dbfs scaling")
        peak_dbfs = float(np.max(frames_dbfs[0].spectrum_db))
        self.assertLessEqual(abs(peak_dbfs - 0.0), 0.25)

        proc_raw = SpectrumProcessor(
            sample_rate_hz=sample_rate_hz,
            bandwidth_hz=sample_rate_hz,
            band_mode="full",
            fft_size=fft_size,
            hop_size=hop_size,
            window_type="rect",
            averaging_mode="none",
            averaging_param=0,
            output_fps=1.0,
            output_bins=None,
            power_scale="raw",
        )
        frames_raw = proc_raw.push(tone)
        self.assertTrue(frames_raw, "No frames emitted for raw scaling")
        peak_raw = float(np.max(frames_raw[0].spectrum_db))
        self.assertGreater(peak_raw, 40.0)

    def test_snapshot_mode_fixed_snapshot_windows(self) -> None:
        sample_rate_hz = 1024.0
        fft_size = 128
        output_fps = 8.0
        n_frames = 6
        n_samples = n_frames * fft_size

        rng = np.random.default_rng(7)
        iq = (rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)).astype(np.complex64)

        proc = SpectrumProcessor(
            sample_rate_hz=sample_rate_hz,
            bandwidth_hz=sample_rate_hz,
            band_mode="full",
            fft_size=fft_size,
            hop_size=17,  # ignored in snapshot mode
            window_type="rect",
            averaging_mode="none",
            averaging_param=0,
            output_fps=output_fps,
            output_bins=None,
            power_scale="raw",
            processing_mode="snapshot",
        )

        frames = proc.push(iq)
        self.assertEqual(len(frames), n_frames)

        for idx, frame in enumerate(frames):
            seg = iq[idx * fft_size : (idx + 1) * fft_size]
            expected_power = np.abs(np.fft.fft(seg, n=fft_size)) ** 2
            expected_db = (10.0 * np.log10(np.fft.fftshift(expected_power) + 1e-20)).astype(np.float32)

            np.testing.assert_allclose(frame.spectrum_db, expected_db, rtol=1e-5, atol=1e-4)
            self.assertAlmostEqual(frame.timestamp, float(idx + 1) / output_fps, places=9)
            self.assertEqual(frame.meta["num_ffts_averaged"], 1)
            self.assertEqual(frame.meta["processing_mode"], "snapshot")

    def test_snapshot_mode_hop_size_independent(self) -> None:
        sample_rate_hz = 640.0
        fft_size = 64
        output_fps = 10.0
        n_samples = 8 * fft_size

        rng = np.random.default_rng(42)
        iq = (rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)).astype(np.complex64)

        proc_small_hop = SpectrumProcessor(
            sample_rate_hz=sample_rate_hz,
            bandwidth_hz=sample_rate_hz,
            band_mode="full",
            fft_size=fft_size,
            hop_size=1,
            window_type="hann",
            averaging_mode="none",
            averaging_param=0,
            output_fps=output_fps,
            output_bins=None,
            power_scale="raw",
            processing_mode="snapshot",
        )
        proc_large_hop = SpectrumProcessor(
            sample_rate_hz=sample_rate_hz,
            bandwidth_hz=sample_rate_hz,
            band_mode="full",
            fft_size=fft_size,
            hop_size=4096,
            window_type="hann",
            averaging_mode="none",
            averaging_param=0,
            output_fps=output_fps,
            output_bins=None,
            power_scale="raw",
            processing_mode="snapshot",
        )

        frames_a = proc_small_hop.push(iq)
        frames_b = proc_large_hop.push(iq)

        self.assertEqual(len(frames_a), len(frames_b))
        for fa, fb in zip(frames_a, frames_b):
            self.assertAlmostEqual(fa.timestamp, fb.timestamp, places=12)
            np.testing.assert_allclose(fa.spectrum_db, fb.spectrum_db, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()
