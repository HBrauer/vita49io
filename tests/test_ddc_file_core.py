import tempfile
import unittest
from pathlib import Path

import numpy as np

from vita49io.defaults.default_payload_formats import DefaultPayloadFormats
from vita49io.io.iq_writer import IQStreamWriter
from vita49io.signal.ddc_file import convert_v49_ddc
from vita49io.signal.ddc_testbench import read_v49_iq


def _write_input_v49(path: Path, *, fs_in: float, iq: np.ndarray) -> None:
    writer = IQStreamWriter(
        stream_id=0x12345678,
        sample_rate_hz=fs_in,
        payload_format=DefaultPayloadFormats.S16_IQ,
        bandwidth_hz=80_000_000.0,
        rf_reference_frequency_hz=915_000_000.0,
    )

    with path.open("wb") as f:
        f.write(writer.build_context_packet().to_bytes())
        for start in range(0, iq.size, 1024):
            block = iq[start : start + 1024]
            if block.size < 1024:
                pad = np.zeros(1024 - block.size, dtype=np.complex64)
                block = np.concatenate([block, pad])
            f.write(writer.build_data_packet_bytes(block))


class TestDDCFileCore(unittest.TestCase):
    def test_convert_v49_ddc_core_api_direct(self) -> None:
        fs_in = 98_304_000.0
        fs_out = 24_576_000
        tone_hz = 1_000_000.0
        shift_hz = 1_000_000.0

        n_samples = 98_304  # 1 ms at 98.304 Msps
        t = np.arange(n_samples, dtype=np.float64) / fs_in
        iq = (0.6 * np.exp(1j * 2.0 * np.pi * tone_hz * t)).astype(np.complex64)

        config_path = Path(__file__).resolve().parents[1] / "examples" / "ddc_v49_file.toml"

        with tempfile.TemporaryDirectory() as td:
            in_path = Path(td) / "in.v49"
            out_path = Path(td) / "out.v49"
            _write_input_v49(in_path, fs_in=fs_in, iq=iq)

            summary = convert_v49_ddc(
                input_path=in_path,
                output_path=out_path,
                output_format_name="S16_IQ",
                output_sample_rate_hz=fs_out,
                chunk_samples=24_576,
                samples_per_packet=1024,
                config_path=config_path,
                center_frequency_offset_hz=shift_hz,
            )

            self.assertEqual(summary["output_sample_rate_hz"], fs_out)
            self.assertGreater(summary["data_packets_written"], 0)

            out_iq, out_fs, _ = read_v49_iq(out_path, max_samples=200_000)
            self.assertEqual(int(round(out_fs)), fs_out)
            self.assertGreater(out_iq.size, 1024)

            nfft = min(16384, out_iq.size)
            seg = out_iq[:nfft]
            spec = np.fft.fftshift(np.fft.fft(seg * np.hanning(nfft)))
            freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / out_fs))
            peak_hz = float(freqs[int(np.argmax(np.abs(spec)))])

            # Tone was shifted by its own frequency, so output should be near DC.
            self.assertLess(abs(peak_hz), 5_000.0)

    def test_non_integer_decimation_path_rejected(self) -> None:
        fs_in = 98_304_000.0
        n_samples = 98_304
        t = np.arange(n_samples, dtype=np.float64) / fs_in
        iq = (0.5 * np.exp(1j * 2.0 * np.pi * 10_000.0 * t)).astype(np.complex64)

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            in_path = Path(td) / "in.v49"
            out_path = Path(td) / "out.v49"
            config_path = td_path / "non_integer.toml"
            _write_input_v49(in_path, fs_in=fs_in, iq=iq)
            config_path.write_text(
                "\n".join(
                    [
                        "[[decimator.paths]]",
                        "input_sample_rate = 98304000",
                        "output_sample_rate = 12345000",
                        "bandwidth = 5000000",
                        "taps = [1.0]",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Non-integer decimation"):
                convert_v49_ddc(
                    input_path=in_path,
                    output_path=out_path,
                    output_format_name="S16_IQ",
                    output_sample_rate_hz=12_345_000,
                    chunk_samples=24_576,
                    samples_per_packet=1024,
                    config_path=config_path,
                    center_frequency_offset_hz=0.0,
                )


if __name__ == "__main__":
    unittest.main()
