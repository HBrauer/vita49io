import tempfile
import unittest
from pathlib import Path

import numpy as np

from vita49io.signal.ddc_testbench import (
    OutputMetrics,
    Thresholds,
    check_metrics,
    fold_frequency_hz,
    load_scenario,
    read_v49_iq,
    synthesize_composite_iq,
    write_v49_from_iq,
)


class TestDDCTestbench(unittest.TestCase):
    def test_fold_frequency_hz(self) -> None:
        fs = 16_000.0
        self.assertAlmostEqual(fold_frequency_hz(17_000.0, fs), 1_000.0)
        self.assertAlmostEqual(fold_frequency_hz(-17_000.0, fs), -1_000.0)

    def test_scenario_synthesis_and_v49_roundtrip(self) -> None:
        scenario_path = Path(__file__).resolve().parents[1] / "scripts" / "ddc_test_scenario.toml"
        scenario = load_scenario(scenario_path)

        iq, meta = synthesize_composite_iq(scenario)
        self.assertEqual(iq.dtype, np.complex64)
        self.assertGreater(iq.size, 0)
        self.assertGreater(meta["fsk_n_symbols"], 0)
        self.assertLessEqual(float(np.max(np.abs(iq))), scenario.max_abs + 1e-5)

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "input.v49"
            write_v49_from_iq(out, scenario=scenario, iq=iq)
            decoded, fs, _ = read_v49_iq(out, max_samples=200_000)

        self.assertGreater(decoded.size, 0)
        self.assertEqual(int(round(fs)), int(round(scenario.sample_rate_hz)))

    def test_unwanted_spur_threshold_check(self) -> None:
        metrics = OutputMetrics(
            tone_frequency_hz=0.0,
            tone_frequency_error_hz=0.0,
            am_correlation=1.0,
            fsk_ber=0.0,
            worst_blocker_db_relative_to_tone=-60.0,
            worst_unwanted_db_relative_to_tone=-15.0,
        )
        thresholds = Thresholds(
            tone_freq_tolerance_hz=200.0,
            am_min_correlation=0.4,
            fsk_max_ber=0.3,
            max_blocker_db_relative_to_tone=-20.0,
            max_unwanted_db_relative_to_tone=-30.0,
        )

        passed, failures = check_metrics(metrics, thresholds)
        self.assertFalse(passed)
        self.assertTrue(any("unwanted in-band spur" in f for f in failures))


if __name__ == "__main__":
    unittest.main()
