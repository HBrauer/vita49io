from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


def _ensure_src_on_path() -> None:
    here = Path(__file__).resolve().parent
    src = here.parent / "src"
    src_str = str(src)
    if src.is_dir() and src_str not in sys.path:
        sys.path.insert(0, src_str)


def _parse_rates_arg(rates_raw: Optional[str]) -> Optional[set[int]]:
    if rates_raw is None:
        return None
    parts = [p.strip() for p in rates_raw.split(",") if p.strip()]
    if not parts:
        return None
    return {int(float(p)) for p in parts}


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run DDC validation matrix over all output rates configured for the scenario input rate."
        )
    )
    parser.add_argument(
        "--scenario",
        default=str(Path(__file__).with_name("ddc_test_scenario.toml")),
        help="Path to DDC test scenario TOML",
    )
    parser.add_argument(
        "--ddc-config",
        default=str(Path(__file__).with_name("ddc_v49_file.toml")),
        help="Path to decimator config TOML",
    )
    parser.add_argument(
        "--input-file",
        default=None,
        help=(
            "Optional existing input V49 file. If omitted, a deterministic input is generated from scenario."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path("artifacts") / "ddc_validation"),
        help="Directory for generated outputs/reports/plots",
    )
    parser.add_argument(
        "--chunk-samples",
        type=int,
        default=491_520,
        help="chunk_samples for DDC processing",
    )
    parser.add_argument(
        "--samples-per-packet",
        type=int,
        default=1024,
        help="samples_per_packet for DDC output",
    )
    parser.add_argument(
        "--rates",
        default=None,
        help="Optional comma-separated output sample rates to validate",
    )
    parser.add_argument(
        "--max-analysis-samples",
        type=int,
        default=1_000_000,
        help="Max output IQ samples to decode/analyze per output file",
    )
    parser.add_argument(
        "--no-waterfall",
        action="store_true",
        help="Skip waterfall SVG generation",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing output-dir before running",
    )
    return parser.parse_args(argv)


def _rates_for_input(
    decimator_paths: Dict[Tuple[int, int], object],
    input_sample_rate_hz: int,
) -> List[int]:
    rates = sorted(
        {
            out_rate
            for (in_rate, out_rate) in decimator_paths
            if int(in_rate) == int(input_sample_rate_hz)
        }
    )
    return [int(r) for r in rates]


def main(argv: Optional[list[str]] = None) -> int:
    _ensure_src_on_path()

    from vita49io.signal import convert_v49_ddc, load_decimator_paths
    from vita49io.signal.ddc_testbench import (
        OutputMetrics,
        ValidationResult,
        check_metrics,
        evaluate_output_metrics,
        load_scenario,
        read_v49_iq,
        synthesize_composite_iq,
        write_v49_from_iq,
        write_waterfall_svg,
    )

    args = _parse_args(argv)

    scenario_path = Path(args.scenario).expanduser()
    ddc_config_path = Path(args.ddc_config).expanduser()
    output_dir = Path(args.output_dir).expanduser()

    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs_dir = output_dir / "outputs"
    plots_dir = output_dir / "waterfalls"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_waterfall:
        plots_dir.mkdir(parents=True, exist_ok=True)

    scenario = load_scenario(scenario_path)
    _, synth_meta = synthesize_composite_iq(scenario)

    if args.input_file is None:
        input_file = output_dir / "ddc_input.v49"
        input_iq, synth_meta = synthesize_composite_iq(scenario)
        write_v49_from_iq(input_file, scenario=scenario, iq=input_iq)
    else:
        input_file = Path(args.input_file).expanduser()
        if not input_file.is_file():
            print(f"Input file not found: {input_file}", file=sys.stderr)
            return 2

    decimator_paths = load_decimator_paths(ddc_config_path)
    rates = _rates_for_input(decimator_paths, int(round(scenario.sample_rate_hz)))
    requested_rates = _parse_rates_arg(args.rates)
    if requested_rates is not None:
        rates = [r for r in rates if r in requested_rates]

    if not rates:
        print(
            f"No output rates configured for input {int(round(scenario.sample_rate_hz))} in {ddc_config_path}",
            file=sys.stderr,
        )
        return 3

    print(f"Scenario: {scenario_path}")
    print(f"Input:    {input_file}")
    print(f"Config:   {ddc_config_path}")
    print(f"Rates:    {', '.join(str(r) for r in rates)}")

    results: List[ValidationResult] = []
    any_failed = False

    for rate in rates:
        out_file = outputs_dir / f"ddc_{rate}.v49"
        out_bw = float(
            decimator_paths[(int(round(scenario.sample_rate_hz)), int(rate))].bandwidth_hz
        )
        summary = None
        try:
            summary = convert_v49_ddc(
                input_path=input_file,
                output_path=out_file,
                output_format_name=scenario.output_format,
                output_sample_rate_hz=int(rate),
                chunk_samples=int(args.chunk_samples),
                samples_per_packet=int(args.samples_per_packet),
                config_path=ddc_config_path,
                center_frequency_offset_hz=float(scenario.center_frequency_offset_hz),
            )

            out_iq, out_fs, _ = read_v49_iq(
                out_file, max_samples=int(args.max_analysis_samples)
            )
            metrics = evaluate_output_metrics(
                iq=out_iq,
                sample_rate_hz=float(out_fs),
                output_bandwidth_hz=out_bw,
                scenario=scenario,
                synthesis_meta=synth_meta,
            )
            passed, failures = check_metrics(metrics, scenario.thresholds)
        except Exception as exc:
            metrics = OutputMetrics(
                tone_frequency_hz=float("nan"),
                tone_frequency_error_hz=float("nan"),
                am_correlation=float("nan"),
                fsk_ber=float("nan"),
                worst_blocker_db_relative_to_tone=float("nan"),
                worst_unwanted_db_relative_to_tone=float("nan"),
            )
            passed = False
            failures = (str(exc),)

        any_failed = any_failed or (not passed)

        svg_path: Optional[Path] = None
        if not args.no_waterfall and passed:
            candidate = plots_dir / f"waterfall_{rate}.svg"
            ok = write_waterfall_svg(
                iq=out_iq,
                sample_rate_hz=float(out_fs),
                output_svg=candidate,
                title=(
                    f"DDC {int(round(scenario.sample_rate_hz))} -> {rate} Hz | "
                    f"tone_err={metrics.tone_frequency_error_hz:.1f} Hz"
                ),
            )
            if ok:
                svg_path = candidate

        result = ValidationResult(
            output_sample_rate_hz=int(rate),
            output_bandwidth_hz=float(out_bw),
            output_file=out_file,
            waterfall_svg=svg_path,
            metrics=metrics,
            passed=bool(passed),
            failures=tuple(failures),
        )
        results.append(result)

        status = "PASS" if passed else "FAIL"
        packets_written = (
            summary["data_packets_written"] if summary is not None else "n/a"
        )
        print(
            f"[{status}] {rate:>8d} Hz | tone_err={metrics.tone_frequency_error_hz:8.2f} Hz "
            f"| am_corr={metrics.am_correlation:6.3f} "
            f"| fsk_ber={metrics.fsk_ber:6.3f} "
            f"| blocker={metrics.worst_blocker_db_relative_to_tone:7.2f} dBc "
            f"| unwanted={metrics.worst_unwanted_db_relative_to_tone:7.2f} dBc "
            f"| packets={packets_written}"
        )
        if failures:
            for failure in failures:
                print(f"  - {failure}")

    report = {
        "scenario_path": str(scenario_path),
        "ddc_config_path": str(ddc_config_path),
        "input_file": str(input_file),
        "output_dir": str(output_dir),
        "rates": rates,
        "overall_passed": not any_failed,
        "results": [
            {
                **{
                    k: v
                    for k, v in asdict(res).items()
                    if k not in {"output_file", "waterfall_svg"}
                },
                "output_file": str(res.output_file),
                "waterfall_svg": str(res.waterfall_svg) if res.waterfall_svg else None,
            }
            for res in results
        ],
    }
    report_path = output_dir / "validation_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    print(f"Report: {report_path}")
    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
