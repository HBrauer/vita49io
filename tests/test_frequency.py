import unittest

import numpy as np

from vita49io.io.frequency import StreamingFrequencyShifter


class TestStreamingFrequencyShifter(unittest.TestCase):
    def test_zero_offset_passthrough(self) -> None:
        x = np.array([1 + 2j, -0.5 + 0.25j, 0.1 - 0.3j], dtype=np.complex64)
        shifter = StreamingFrequencyShifter(sample_rate_hz=1_000_000.0, frequency_offset_hz=0.0)
        y = shifter.process(x)
        np.testing.assert_allclose(y, x, atol=0.0)

    def test_chunked_processing_matches_reference(self) -> None:
        rng = np.random.default_rng(1234)
        x = (
            rng.standard_normal(4096).astype(np.float32)
            + 1j * rng.standard_normal(4096).astype(np.float32)
        ).astype(np.complex64)

        fs = 98_304_000.0
        foff = 24_000.0
        shifter = StreamingFrequencyShifter(sample_rate_hz=fs, frequency_offset_hz=foff)

        out_chunks = []
        cursor = 0
        for n in (511, 611, 1024, 37, 901, 1012):
            block = x[cursor : cursor + n]
            if block.size == 0:
                break
            out_chunks.append(shifter.process(block))
            cursor += n
        if cursor < x.size:
            out_chunks.append(shifter.process(x[cursor:]))
        y = np.concatenate(out_chunks) if out_chunks else np.empty(0, dtype=np.complex64)

        phase = -2.0 * np.pi * foff * np.arange(x.size, dtype=np.float64) / fs
        ref = x * np.exp(1j * phase).astype(np.complex64)
        np.testing.assert_allclose(y, ref, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
