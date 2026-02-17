from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


def _ensure_src_on_path() -> None:
    here = Path(__file__).resolve().parent
    src = here.parent / "src"
    src_str = str(src)
    if src.is_dir() and src_str not in sys.path:
        sys.path.insert(0, src_str)


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a deterministic DDC validation V49 file containing tone + AM + FSK + blockers + noise."
        )
    )
    parser.add_argument(
        "output_file",
        help="Output V49 file path",
    )
    parser.add_argument(
        "--scenario",
        default=str(Path(__file__).with_name("ddc_test_scenario.toml")),
        help="Path to DDC test scenario TOML",
    )
    parser.add_argument(
        "--meta-json",
        default=None,
        help="Optional output path for synthesis metadata JSON",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    _ensure_src_on_path()

    from vita49io.signal.ddc_testbench import (
        load_scenario,
        synthesize_composite_iq,
        write_v49_from_iq,
    )

    args = _parse_args(argv)

    scenario_path = Path(args.scenario).expanduser()
    output_path = Path(args.output_file).expanduser()

    scenario = load_scenario(scenario_path)
    iq, synth_meta = synthesize_composite_iq(scenario)
    write_meta = {
        "scenario_path": str(scenario_path),
        "sample_rate_hz": scenario.sample_rate_hz,
        "bandwidth_hz": scenario.bandwidth_hz,
        "rf_reference_frequency_hz": scenario.rf_reference_frequency_hz,
        "duration_seconds": scenario.duration_seconds,
        "output_format": scenario.output_format,
        "center_frequency_offset_hz": scenario.center_frequency_offset_hz,
        "tone_frequency_hz": scenario.tone.frequency_hz,
        "am_carrier_frequency_hz": scenario.am.carrier_frequency_hz,
        "fsk_mark_frequency_hz": scenario.fsk.mark_frequency_hz,
        "fsk_space_frequency_hz": scenario.fsk.space_frequency_hz,
    }
    write_meta.update({
        "generated_samples": int(synth_meta["n_samples"]),
        "generated_duration_seconds": float(synth_meta["duration_seconds"]),
        "fsk_n_symbols": int(synth_meta["fsk_n_symbols"]),
        "scale_applied": float(synth_meta["scale_applied"]),
    })

    write_stats = write_v49_from_iq(output_path, scenario=scenario, iq=iq)

    meta_path: Path
    if args.meta_json is not None:
        meta_path = Path(args.meta_json).expanduser()
    else:
        meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump({**write_meta, **write_stats}, f, indent=2, sort_keys=True)

    print(f"Wrote test file: {output_path}")
    print(f"Wrote metadata:  {meta_path}")
    print(f"Samples: {write_meta['generated_samples']}")
    print(f"Data packets: {write_stats['data_packets']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
