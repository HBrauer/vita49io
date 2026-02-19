from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


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
            "Run DDC validation matrix over configured decimator rates."
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
        "--input-rates",
        default=None,
        help=(
            "Optional comma-separated input sample rates to validate. "
            "Defaults to scenario.sample_rate_hz."
        ),
    )
    parser.add_argument(
        "--all-input-rates",
        action="store_true",
        help="Validate all input sample rates found in decimator config",
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


def _input_rates_from_paths(decimator_paths: Dict[Tuple[int, int], object]) -> List[int]:
    rates = sorted({int(in_rate) for (in_rate, _) in decimator_paths})
    return [int(r) for r in rates]


def main(argv: Optional[list[str]] = None) -> int:
    _ensure_src_on_path()

    from vita49io.signal import convert_v49_ddc, load_decimator_paths
    from vita49io.signal.ddc_testbench import (
        AMSpec,
        BlockerSpec,
        FSKSpec,
        OutputMetrics,
        ToneSpec,
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

    inputs_dir = output_dir / "inputs"
    outputs_dir = output_dir / "outputs"
    plots_dir = output_dir / "waterfalls"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_waterfall:
        plots_dir.mkdir(parents=True, exist_ok=True)

    base_scenario = load_scenario(scenario_path)
    decimator_paths = load_decimator_paths(ddc_config_path)
    config_input_rates = _input_rates_from_paths(decimator_paths)
    requested_output_rates = _parse_rates_arg(args.rates)
    requested_input_rates = _parse_rates_arg(args.input_rates)

    if args.all_input_rates:
        input_rates = list(config_input_rates)
    elif requested_input_rates is not None:
        input_rates = [r for r in config_input_rates if r in requested_input_rates]
    else:
        input_rates = [int(round(base_scenario.sample_rate_hz))]

    if not input_rates:
        print(
            "No input rates selected. "
            f"Configured inputs in {ddc_config_path}: {', '.join(str(r) for r in config_input_rates)}",
            file=sys.stderr,
        )
        return 3

    external_input_file: Optional[Path] = None
    external_input_rate: Optional[int] = None
    if args.input_file is not None:
        external_input_file = Path(args.input_file).expanduser()
        if not external_input_file.is_file():
            print(f"Input file not found: {external_input_file}", file=sys.stderr)
            return 2
        _, fs_external, _ = read_v49_iq(external_input_file, max_samples=0)
        external_input_rate = int(round(fs_external))
        if external_input_rate not in config_input_rates:
            print(
                f"Input file sample rate {external_input_rate} is not configured in {ddc_config_path}",
                file=sys.stderr,
            )
            return 3
        if requested_input_rates is not None and external_input_rate not in requested_input_rates:
            print(
                f"Input file sample rate {external_input_rate} is not in requested --input-rates",
                file=sys.stderr,
            )
            return 3
        if args.all_input_rates or len(input_rates) > 1:
            print(
                "--input-file can only be used with a single effective input rate. "
                "Use generated inputs for multi-rate validation.",
                file=sys.stderr,
            )
            return 3
        input_rates = [external_input_rate]

    rates_by_input = {
        int(in_rate): _rates_for_input(decimator_paths, int(in_rate))
        for in_rate in input_rates
    }
    if requested_output_rates is not None:
        rates_by_input = {
            in_rate: [r for r in rates if r in requested_output_rates]
            for in_rate, rates in rates_by_input.items()
        }

    if not any(rates_by_input.values()):
        print(
            "No output rates selected for the chosen input rates. "
            f"Config: {ddc_config_path}",
            file=sys.stderr,
        )
        return 3

    print(f"Scenario: {scenario_path}")
    print(f"Config:   {ddc_config_path}")
    if external_input_file is not None:
        print(
            f"Input:    {external_input_file} "
            f"(sample_rate={external_input_rate})"
        )
    else:
        print(f"Input rates: {', '.join(str(r) for r in input_rates)}")
    if requested_output_rates is not None:
        print(f"Requested output rates: {', '.join(str(r) for r in sorted(requested_output_rates))}")

    results: List[Dict[str, Any]] = []
    input_files: Dict[int, Dict[str, str]] = {}
    any_failed = False
    result_modes: List[str] = []

    for input_rate in input_rates:
        rates = rates_by_input.get(int(input_rate), [])
        if not rates:
            print(f"[SKIP] input {input_rate}: no matching output rates")
            any_failed = True
            continue

        input_bw_hz = float(min(float(base_scenario.bandwidth_hz), float(input_rate)))
        max_output_bw_hz = max(
            float(decimator_paths[(int(input_rate), int(r))].bandwidth_hz) for r in rates
        )
        input_half_hz = float(input_bw_hz / 2.0)
        protected_half_hz = float(max_output_bw_hz / 2.0)
        available_stopband_hz = max(0.0, input_half_hz - protected_half_hz)
        max_allowed_offset_hz = min(
            max(
                0.0,
                (input_bw_hz - float(decimator_paths[(int(input_rate), int(r))].bandwidth_hz))
                / 2.0,
            )
            for r in rates
        )
        requested_offset_hz = float(base_scenario.center_frequency_offset_hz)
        effective_offset_hz = max(
            -float(max_allowed_offset_hz),
            min(float(max_allowed_offset_hz), requested_offset_hz),
        )
        offset_shift_hz = float(effective_offset_hz - requested_offset_hz)

        remapped_blockers: List[BlockerSpec] = []
        if base_scenario.blockers:
            n_blockers = len(base_scenario.blockers)
            for idx, blocker in enumerate(base_scenario.blockers):
                rel_base_hz = float(blocker.frequency_hz - requested_offset_hz)
                sign = 1.0 if rel_base_hz >= 0.0 else -1.0
                if available_stopband_hz > 0.0:
                    frac = 0.5 if n_blockers == 1 else (0.25 + 0.5 * idx / (n_blockers - 1))
                    magnitude_hz = protected_half_hz + frac * available_stopband_hz
                else:
                    magnitude_hz = max(0.0, protected_half_hz * 0.95)
                remapped_blockers.append(
                    BlockerSpec(
                        frequency_hz=float(effective_offset_hz + sign * magnitude_hz),
                        amplitude=float(blocker.amplitude),
                    )
                )

        scenario = replace(
            base_scenario,
            sample_rate_hz=float(input_rate),
            bandwidth_hz=float(input_bw_hz),
            center_frequency_offset_hz=float(effective_offset_hz),
            tone=ToneSpec(
                frequency_hz=float(base_scenario.tone.frequency_hz + offset_shift_hz),
                amplitude=float(base_scenario.tone.amplitude),
            ),
            am=AMSpec(
                carrier_frequency_hz=float(
                    base_scenario.am.carrier_frequency_hz + offset_shift_hz
                ),
                amplitude=float(base_scenario.am.amplitude),
                modulation_frequency_hz=float(base_scenario.am.modulation_frequency_hz),
                modulation_index=float(base_scenario.am.modulation_index),
            ),
            fsk=FSKSpec(
                mark_frequency_hz=float(
                    base_scenario.fsk.mark_frequency_hz + offset_shift_hz
                ),
                space_frequency_hz=float(
                    base_scenario.fsk.space_frequency_hz + offset_shift_hz
                ),
                symbol_rate_hz=float(base_scenario.fsk.symbol_rate_hz),
                amplitude=float(base_scenario.fsk.amplitude),
                seed=int(base_scenario.fsk.seed),
            ),
            blockers=tuple(remapped_blockers),
        )

        if abs(offset_shift_hz) > 1e-12:
            print(
                f"Input {input_rate} Hz: adjusted center offset "
                f"{requested_offset_hz:.3f} -> {effective_offset_hz:.3f} Hz "
                f"for selected output bandwidths"
            )

        if external_input_file is not None:
            mode_scenarios = {"composite": scenario}
        else:
            spur_thresholds = replace(
                scenario.thresholds,
                am_min_correlation=0.0,
                fsk_max_ber=1.0,
            )
            modulation_thresholds = replace(
                scenario.thresholds,
                max_blocker_db_relative_to_tone=0.0,
                max_unwanted_db_relative_to_tone=10.0,
            )
            mode_scenarios = {
                "spur": replace(
                    scenario,
                    am=AMSpec(
                        carrier_frequency_hz=float(scenario.am.carrier_frequency_hz),
                        amplitude=0.0,
                        modulation_frequency_hz=float(scenario.am.modulation_frequency_hz),
                        modulation_index=0.0,
                    ),
                    fsk=FSKSpec(
                        mark_frequency_hz=float(scenario.fsk.mark_frequency_hz),
                        space_frequency_hz=float(scenario.fsk.space_frequency_hz),
                        symbol_rate_hz=float(scenario.fsk.symbol_rate_hz),
                        amplitude=0.0,
                        seed=int(scenario.fsk.seed),
                    ),
                    thresholds=spur_thresholds,
                ),
                "modulation": replace(
                    scenario,
                    blockers=tuple(),
                    thresholds=modulation_thresholds,
                ),
            }

        if not result_modes:
            result_modes = list(mode_scenarios.keys())

        mode_input_files: Dict[str, Path] = {}
        mode_synth_meta: Dict[str, Dict[str, Any]] = {}
        for mode_name, mode_scenario in mode_scenarios.items():
            if external_input_file is not None:
                mode_input_files[mode_name] = external_input_file
                _, synth_meta = synthesize_composite_iq(mode_scenario)
                mode_synth_meta[mode_name] = synth_meta
                continue

            mode_input = (
                inputs_dir
                / mode_name
                / f"ddc_input_{int(input_rate)}.v49"
            )
            input_iq, synth_meta = synthesize_composite_iq(mode_scenario)
            write_v49_from_iq(mode_input, scenario=mode_scenario, iq=input_iq)
            mode_input_files[mode_name] = mode_input
            mode_synth_meta[mode_name] = synth_meta

        input_files[int(input_rate)] = {
            mode_name: str(path) for mode_name, path in sorted(mode_input_files.items())
        }
        print(
            f"Input {input_rate} Hz -> rates: {', '.join(str(r) for r in rates)} "
            f"| files={input_files[int(input_rate)]}"
        )

        for rate in rates:
            out_bw = float(decimator_paths[(int(input_rate), int(rate))].bandwidth_hz)
            case_modes: Dict[str, Dict[str, Any]] = {}
            case_failures: List[str] = []
            case_passed = True

            for mode_name, mode_scenario in mode_scenarios.items():
                out_dir_for_mode = outputs_dir / mode_name / f"in_{int(input_rate)}"
                out_file = out_dir_for_mode / f"ddc_{rate}.v49"

                summary = None
                out_iq = None
                out_fs = None
                try:
                    summary = convert_v49_ddc(
                        input_path=mode_input_files[mode_name],
                        output_path=out_file,
                        output_format_name=mode_scenario.output_format,
                        output_sample_rate_hz=int(rate),
                        chunk_samples=int(args.chunk_samples),
                        samples_per_packet=int(args.samples_per_packet),
                        config_path=ddc_config_path,
                        center_frequency_offset_hz=float(mode_scenario.center_frequency_offset_hz),
                    )

                    out_iq, out_fs, _ = read_v49_iq(
                        out_file, max_samples=int(args.max_analysis_samples)
                    )
                    metrics = evaluate_output_metrics(
                        iq=out_iq,
                        sample_rate_hz=float(out_fs),
                        output_bandwidth_hz=out_bw,
                        scenario=mode_scenario,
                        synthesis_meta=mode_synth_meta[mode_name],
                    )
                    mode_passed, mode_failures = check_metrics(metrics, mode_scenario.thresholds)
                except Exception as exc:
                    metrics = OutputMetrics(
                        tone_frequency_hz=float("nan"),
                        tone_frequency_error_hz=float("nan"),
                        am_correlation=float("nan"),
                        fsk_ber=float("nan"),
                        worst_blocker_db_relative_to_tone=float("nan"),
                        worst_unwanted_db_relative_to_tone=float("nan"),
                    )
                    mode_passed = False
                    mode_failures = (str(exc),)

                svg_path: Optional[Path] = None
                if not args.no_waterfall and out_iq is not None and out_fs is not None:
                    plot_dir_for_mode = plots_dir / mode_name / f"in_{int(input_rate)}"
                    candidate = plot_dir_for_mode / f"waterfall_{rate}.svg"
                    ok = write_waterfall_svg(
                        iq=out_iq,
                        sample_rate_hz=float(out_fs),
                        output_svg=candidate,
                        title=(
                            f"{mode_name} DDC {int(input_rate)} -> {rate} Hz (bw={int(out_bw)} Hz) | "
                            f"tone_err={metrics.tone_frequency_error_hz:.1f} Hz"
                        ),
                        output_bandwidth_hz=float(out_bw),
                    )
                    if ok:
                        svg_path = candidate

                case_modes[mode_name] = {
                    "output_file": str(out_file),
                    "waterfall_svg": str(svg_path) if svg_path else None,
                    "metrics": asdict(metrics),
                    "passed": bool(mode_passed),
                    "failures": list(mode_failures),
                    "data_packets_written": (
                        summary["data_packets_written"] if summary is not None else None
                    ),
                }

                if not mode_passed:
                    case_passed = False
                    case_failures.extend(f"{mode_name}: {failure}" for failure in mode_failures)

                status = "PASS" if mode_passed else "FAIL"
                packets_written = (
                    summary["data_packets_written"] if summary is not None else "n/a"
                )
                print(
                    f"[{status}] {mode_name:>10s} {int(input_rate):>8d} -> {rate:>8d} Hz "
                    f"| tone_err={metrics.tone_frequency_error_hz:8.2f} Hz "
                    f"| am_corr={metrics.am_correlation:6.3f} "
                    f"| fsk_ber={metrics.fsk_ber:6.3f} "
                    f"| blocker={metrics.worst_blocker_db_relative_to_tone:7.2f} dBc "
                    f"| unwanted={metrics.worst_unwanted_db_relative_to_tone:7.2f} dBc "
                    f"| packets={packets_written}"
                )
                if mode_failures:
                    for failure in mode_failures:
                        print(f"  - {failure}")

            any_failed = any_failed or (not case_passed)
            results.append(
                {
                    "input_sample_rate_hz": int(input_rate),
                    "output_sample_rate_hz": int(rate),
                    "output_bandwidth_hz": float(out_bw),
                    "passed": bool(case_passed),
                    "failures": case_failures,
                    "modes": case_modes,
                }
            )

    all_rates = sorted({int(res["output_sample_rate_hz"]) for res in results})

    report = {
        "scenario_path": str(scenario_path),
        "ddc_config_path": str(ddc_config_path),
        "input_file": str(external_input_file) if external_input_file is not None else None,
        "input_files": {str(k): v for k, v in sorted(input_files.items())},
        "output_dir": str(output_dir),
        "modes": result_modes,
        "input_rates": [int(r) for r in input_rates],
        "rates_by_input": {
            str(in_rate): [int(r) for r in rates]
            for in_rate, rates in sorted(rates_by_input.items())
        },
        "rates": all_rates,
        "overall_passed": not any_failed,
        "results": results,
    }
    report_path = output_dir / "validation_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    print(f"Report: {report_path}")
    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
