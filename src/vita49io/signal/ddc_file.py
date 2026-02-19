from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.signal import upfirdn
from scipy.signal._upfirdn import _output_len as _upfirdn_output_len

from vita49io.defaults.default_payload_formats import DefaultPayloadFormats
from vita49io.io.frequency import StreamingFrequencyShifter
from vita49io.io.iq_writer import IQStreamWriter
from vita49io.io.packet_reader import PacketReader, RawDataPacket
from vita49io.io.payload_codec import build_payload_decoder
from vita49io.protocol.cif0 import PayloadFormat, SampleType
from vita49io.protocol.context_packet import ContextPacket
from vita49io.protocol.enums import PacketType, TSI, TSF

_REPO_DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "examples" / "ddc_v49_file.toml"
)
DEFAULT_DECIMATOR_CONFIG_PATH = (
    _REPO_DEFAULT_CONFIG_PATH
    if _REPO_DEFAULT_CONFIG_PATH.is_file()
    else Path("examples/ddc_v49_file.toml")
)


@dataclass(frozen=True)
class DecimatorStageConfig:
    input_sample_rate_hz: int
    output_sample_rate_hz: int
    taps: Optional[np.ndarray]


@dataclass(frozen=True)
class DecimatorPath:
    bandwidth_hz: int
    stages: Tuple[DecimatorStageConfig, ...]


@dataclass
class RuntimeDecimatorStage:
    input_sample_rate_hz: int
    output_sample_rate_hz: int
    up: int
    down: int
    resampler: Optional["StreamingResampler"]


class StreamingResampler:
    def __init__(self, h: np.ndarray, up: int, down: int):
        if up <= 0 or down <= 0:
            raise ValueError("up and down must be positive integers")

        self.h = np.asarray(h, dtype=np.float32)
        if self.h.ndim != 1 or self.h.size == 0:
            raise ValueError("h must be a non-empty 1-D array")

        self.up = int(up)
        self.down = int(down)
        self._h_len = int(self.h.size)
        self._n_seen = 0

        # Fast exact path for pure decimation using chunked upfirdn.
        if self.up == 1:
            self._mode = "decimate"
            self._history_max = max(self._h_len + self.down - 2, 0)
            self._history = np.empty(0, dtype=np.complex64)
            self._flushed = False
            return

        # General exact rational path.
        self._mode = "polyphase"
        self._input = np.empty(0, dtype=np.complex64)
        self._input_start = 0
        self._m_emitted = 0
        self._phase_coeffs = tuple(
            np.asarray(self.h[p:: self.up][::-1], dtype=np.float32)
            for p in range(self.up)
        )
        self._max_phase_len = max((c.size for c in self._phase_coeffs), default=0)

    def _prefix_output_target(self, n_seen: int) -> int:
        if n_seen <= 0:
            return 0
        causal_target = (n_seen * self.up + self.down - 1) // self.down
        upfirdn_target = int(
            _upfirdn_output_len(self._h_len, n_seen, self.up, self.down)
        )
        return min(causal_target, upfirdn_target)

    def _get_tail_history(self, n_seen: int, hist_len: int) -> np.ndarray:
        if hist_len <= 0:
            return np.empty(0, dtype=np.complex64)

        if n_seen <= 0:
            return np.zeros(hist_len, dtype=np.complex64)

        have = min(int(self._history.size), int(n_seen))
        if have >= hist_len:
            return np.asarray(self._history[-hist_len:], dtype=np.complex64)

        pad = hist_len - have
        if have == 0:
            return np.zeros(hist_len, dtype=np.complex64)
        return np.concatenate(
            [np.zeros(pad, dtype=np.complex64), self._history[-have:]]
        )

    def _append_history(self, packet: np.ndarray) -> None:
        if packet.size == 0:
            return
        if self._history.size == 0:
            self._history = np.asarray(packet, dtype=np.complex64).reshape(-1)
        else:
            self._history = np.concatenate([self._history, packet])
        if self._history.size > self._history_max:
            self._history = self._history[-self._history_max :]

    def _process_decimate_upfirdn(self, packet: np.ndarray) -> np.ndarray:
        n0 = self._n_seen
        n = int(packet.size)
        if n == 0:
            return np.empty(0, dtype=np.complex64)

        align = int((n0 - (self._h_len - 1)) % self.down)
        hist_len = int((self._h_len - 1) + align)
        hist = self._get_tail_history(n_seen=n0, hist_len=hist_len)
        xin = np.concatenate([hist, packet]) if hist.size else np.asarray(packet, dtype=np.complex64)

        # start_idx is aligned to the global downsample grid.
        start_idx = int(n0 - hist_len)
        y = upfirdn(self.h, xin, up=1, down=self.down)

        t_lo = n0
        t_hi = n0 + n - 1
        p_lo = int((t_lo - start_idx + self.down - 1) // self.down)
        p_hi = int((t_hi - start_idx) // self.down)

        if p_hi < p_lo:
            out = np.empty(0, dtype=np.complex64)
        else:
            out = np.asarray(y[p_lo : p_hi + 1], dtype=np.complex64)

        self._n_seen += n
        self._append_history(np.asarray(packet, dtype=np.complex64).reshape(-1))
        return out

    def _flush_decimate_upfirdn(self) -> np.ndarray:
        if self._n_seen == 0 or self._flushed or self._h_len <= 1:
            return np.empty(0, dtype=np.complex64)

        n0 = self._n_seen
        tail_n = self._h_len - 1
        tail = np.zeros(tail_n, dtype=np.complex64)

        align = int((n0 - (self._h_len - 1)) % self.down)
        hist_len = int((self._h_len - 1) + align)
        hist = self._get_tail_history(n_seen=n0, hist_len=hist_len)
        xin = np.concatenate([hist, tail]) if hist.size else tail

        start_idx = int(n0 - hist_len)
        y = upfirdn(self.h, xin, up=1, down=self.down)

        t_lo = n0
        t_hi = n0 + tail_n - 1
        p_lo = int((t_lo - start_idx + self.down - 1) // self.down)
        p_hi = int((t_hi - start_idx) // self.down)

        self._flushed = True
        if p_hi < p_lo:
            return np.empty(0, dtype=np.complex64)
        return np.asarray(y[p_lo : p_hi + 1], dtype=np.complex64)

    def _emit_polyphase_until(self, target_outputs: int) -> np.ndarray:
        if target_outputs <= self._m_emitted:
            return np.empty(0, dtype=np.complex64)

        out = np.empty(target_outputs - self._m_emitted, dtype=np.complex64)
        out_idx = 0

        while self._m_emitted < target_outputs:
            t = self._m_emitted * self.down
            phase = t % self.up
            coeff = self._phase_coeffs[phase]

            if coeff.size == 0:
                y = 0.0j
            else:
                n = t // self.up
                start = int(n - coeff.size + 1)
                avail_start = max(start, self._input_start, 0)
                avail_end = min(
                    int(n + 1),
                    self._input_start + int(self._input.size),
                )

                if avail_end <= avail_start:
                    y = 0.0j
                else:
                    coeff_offset = avail_start - start
                    take = avail_end - avail_start
                    c = coeff[coeff_offset : coeff_offset + take]
                    x = self._input[
                        avail_start - self._input_start : avail_end - self._input_start
                    ]
                    y = np.dot(c, x)

            out[out_idx] = np.complex64(y)
            out_idx += 1
            self._m_emitted += 1

            if self._max_phase_len > 0:
                n_next = (self._m_emitted * self.down) // self.up
                keep_from = max(0, int(n_next - (self._max_phase_len - 1)))
                if keep_from > self._input_start:
                    drop = min(keep_from - self._input_start, int(self._input.size))
                    if drop > 0:
                        self._input = self._input[drop:]
                        self._input_start += drop

        return out

    def process(self, packet: np.ndarray) -> np.ndarray:
        packet = np.asarray(packet, dtype=np.complex64).reshape(-1)
        if packet.size == 0:
            return np.empty(0, dtype=np.complex64)

        if self._mode == "decimate":
            return self._process_decimate_upfirdn(packet)

        self._n_seen += int(packet.size)

        self._input = np.concatenate([self._input, packet])
        target_outputs = self._prefix_output_target(self._n_seen)
        return self._emit_polyphase_until(target_outputs)

    def flush(self) -> np.ndarray:
        if self._n_seen == 0:
            return np.empty(0, dtype=np.complex64)

        if self._mode == "decimate":
            return self._flush_decimate_upfirdn()

        total_outputs = int(
            _upfirdn_output_len(self._h_len, self._n_seen, self.up, self.down)
        )
        return self._emit_polyphase_until(total_outputs)


def _load_toml_module():
    try:
        import tomllib

        return tomllib
    except ModuleNotFoundError:
        try:
            import tomli

            return tomli
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "TOML parsing requires Python 3.11+ or the 'tomli' package on Python 3.8-3.10"
            ) from exc


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DDC a VITA 49 file (resample + re-pack).",
        epilog=(
            "Examples:\n"
            "  python examples/ddc_v49_file.py in.v49 out.v49 \\\n"
            "    --output-format S16_IQ \\\n"
            "    --output-sample-rate 2048000\n"
            "\n"
            "  python examples/ddc_v49_file.py in.v49 out_bw.v49 \\\n"
            "    --output-format S16_IQ \\\n"
            "    --bandwidth 10000000\n"
            "\n"
            "  python examples/ddc_v49_file.py in.v49 out_shifted.v49 \\\n"
            "    --output-format F32_IQ \\\n"
            "    --output-sample-rate 1024000 \\\n"
            "    --center-frequency-offset-hz -250000\n"
            "\n"
            "  python examples/ddc_v49_file.py in.v49 out_custom.v49 \\\n"
            "    --output-format S16_IQ \\\n"
            "    --output-sample-rate 1024000 \\\n"
            "    --config examples/ddc_v49_file.toml \\\n"
            "    --chunk-samples 61140 \\\n"
            "    --samples-per-packet 1024"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("input_file", help="Path to input .v49 file")
    parser.add_argument("output_file", help="Path to output .v49 file")
    parser.add_argument(
        "--output-format",
        required=True,
        help="Output payload format (F32_IQ, S32_IQ, S24_IQ, S16_IQ)",
    )
    output_selector_group = parser.add_mutually_exclusive_group(required=True)
    output_selector_group.add_argument(
        "--output-sample-rate",
        type=int,
        help="Output sample rate in Hz",
    )
    output_selector_group.add_argument(
        "--bandwidth",
        type=int,
        help="Select output path by configured output bandwidth in Hz",
    )
    tuning_group = parser.add_mutually_exclusive_group()
    tuning_group.add_argument(
        "--center-frequency-hz",
        type=float,
        default=None,
        help=(
            "Target RF center frequency in Hz (absolute). "
            "If input CIF0 rf_reference_frequency_hz is available, this is interpreted "
            "as absolute RF and converted to an offset from the input center."
        ),
    )
    tuning_group.add_argument(
        "--center-frequency-offset-hz",
        type=float,
        default=None,
        help=(
            "Target center frequency offset in Hz relative to input "
            "CIF0 rf_reference_frequency_hz."
        ),
    )
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Path to decimator TOML config. "
            f"Defaults to {DEFAULT_DECIMATOR_CONFIG_PATH}"
        ),
    )
    parser.add_argument(
        "--chunk-samples",
        type=int,
        default=61_140,
        help="Input samples per resampling block",
    )
    parser.add_argument(
        "--samples-per-packet",
        type=int,
        default=1024,
        help="Number of complex samples per output data packet",
    )
    parser.add_argument(
        "--timing",
        action="store_true",
        help="Print elapsed conversion time after completion",
    )
    parser.add_argument(
        "--start-time",
        default=None,
        help=(
            "Optional UTC start time for packet-level time slicing (ISO-8601, "
            "e.g. 2026-01-01T12:34:56Z). Packets overlapping the window are kept."
        ),
    )
    parser.add_argument(
        "--end-time",
        default=None,
        help=(
            "Optional UTC end time for packet-level time slicing (ISO-8601, "
            "e.g. 2026-01-01T12:35:56Z). Packets overlapping the window are kept."
        ),
    )
    return parser.parse_args(argv)


def _parse_iso_utc_time(value: str, *, option_name: str) -> float:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{option_name} must not be empty")
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(
            f"{option_name} must be ISO UTC time like YYYY-MM-DDThh:mm:ssZ"
        ) from exc
    if dt.tzinfo is None:
        raise ValueError(f"{option_name} must include timezone (use trailing 'Z')")
    return float(dt.astimezone(timezone.utc).timestamp())


def _packet_time_s(integer_seconds: Optional[int], fractional_seconds: Optional[int]) -> Optional[float]:
    if integer_seconds is None and fractional_seconds is None:
        return None
    sec = float(integer_seconds or 0)
    frac = float(fractional_seconds or 0)
    return sec + (frac / float(1 << 64))


def _resolve_input_format_name(pf, supported_formats: Dict[str, Any]) -> Optional[str]:
    for name, fmt in supported_formats.items():
        if pf == fmt:
            return name
    return None


def _resample_ratio(in_rate_hz: int, out_rate_hz: int) -> Tuple[int, int]:
    frac = Fraction(int(out_rate_hz), int(in_rate_hz))
    return frac.numerator, frac.denominator


def _packet_sample_count(payload: memoryview, payload_format: PayloadFormat) -> int:
    bits_per_component = int(payload_format.item_packing_field_size_bits)
    if bits_per_component <= 0 or bits_per_component % 8 != 0:
        raise ValueError(
            "Unsupported payload format for time slicing: "
            f"item_packing_field_size_bits={bits_per_component}"
        )
    components_per_sample = (
        2 if payload_format.sample_type == SampleType.COMPLEX_CARTESIAN else 1
    )
    bytes_per_sample = (bits_per_component // 8) * components_per_sample
    payload_nbytes = int(len(payload))
    if bytes_per_sample <= 0 or payload_nbytes % bytes_per_sample != 0:
        raise ValueError(
            "Payload length is not aligned to sample size for time slicing: "
            f"bytes={payload_nbytes}, bytes_per_sample={bytes_per_sample}"
        )
    return payload_nbytes // bytes_per_sample


def _parse_taps(taps_raw, label: str) -> Optional[np.ndarray]:
    if taps_raw is None:
        return None
    if not isinstance(taps_raw, list):
        raise ValueError(f"'taps' must be an array in {label}")
    taps = np.asarray(taps_raw, dtype=np.float32)
    if taps.ndim != 1 or taps.size == 0:
        raise ValueError(f"'taps' must be a non-empty 1-D array in {label}")
    return taps


def _parse_decimator_stage(
    stage_entry: dict,
    *,
    label: str,
) -> DecimatorStageConfig:
    try:
        in_rate = int(stage_entry["input_sample_rate"])
        out_rate = int(stage_entry["output_sample_rate"])
    except KeyError as exc:
        raise ValueError(f"Missing required key {exc} in {label}") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid numeric sample rate in {label}") from exc

    if in_rate <= 0 or out_rate <= 0:
        raise ValueError(f"Sample rates must be > 0 in {label}")

    taps = _parse_taps(stage_entry.get("taps"), label)
    if in_rate != out_rate and taps is None:
        raise ValueError(
            f"Non-identity stage requires 'taps' in {label}: {in_rate} -> {out_rate}"
        )

    return DecimatorStageConfig(
        input_sample_rate_hz=in_rate,
        output_sample_rate_hz=out_rate,
        taps=taps,
    )


def _load_decimator_paths(config_path: Path) -> Dict[Tuple[int, int], DecimatorPath]:
    config_path = config_path.expanduser()
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Decimator config file not found: {config_path}. "
            "Provide --config or create the default config file."
        )

    toml = _load_toml_module()
    with config_path.open("rb") as f:
        doc = toml.load(f)

    if not isinstance(doc, dict):
        raise ValueError(f"Invalid TOML root in decimator config: {config_path}")

    decimator = doc.get("decimator")
    if not isinstance(decimator, dict):
        raise ValueError(
            f"Missing [decimator] section in decimator config: {config_path}"
        )

    raw_paths = decimator.get("paths")
    if not isinstance(raw_paths, list) or not raw_paths:
        raise ValueError(
            f"Missing or empty [[decimator.paths]] entries in decimator config: {config_path}"
        )

    paths: Dict[Tuple[int, int], DecimatorPath] = {}
    for idx, entry in enumerate(raw_paths, start=1):
        if not isinstance(entry, dict):
            raise ValueError(
                f"Entry #{idx} in [[decimator.paths]] must be a table in {config_path}"
            )

        try:
            in_rate = int(entry["input_sample_rate"])
            out_rate = int(entry["output_sample_rate"])
            bandwidth_hz = int(entry["bandwidth"])
        except KeyError as exc:
            raise ValueError(
                f"Missing required key {exc} in [[decimator.paths]] entry #{idx}"
            ) from exc
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid numeric value in [[decimator.paths]] entry #{idx}"
            ) from exc

        if in_rate <= 0 or out_rate <= 0 or bandwidth_hz <= 0:
            raise ValueError(
                f"input_sample_rate, output_sample_rate, and bandwidth must be > 0 "
                f"in [[decimator.paths]] entry #{idx}"
            )

        raw_stages = entry.get("stages")
        path_label = f"[[decimator.paths]] entry #{idx}"
        if raw_stages is None:
            stage_entry = {
                "input_sample_rate": in_rate,
                "output_sample_rate": out_rate,
                "taps": entry.get("taps"),
            }
            stages = [_parse_decimator_stage(stage_entry, label=f"{path_label} stage #1")]
        else:
            if "taps" in entry:
                raise ValueError(
                    f"'taps' and 'stages' cannot both be set in {path_label}"
                )
            if not isinstance(raw_stages, list) or not raw_stages:
                raise ValueError(
                    f"'stages' must be a non-empty array in {path_label}"
                )
            stages = []
            for stage_idx, raw_stage in enumerate(raw_stages, start=1):
                if not isinstance(raw_stage, dict):
                    raise ValueError(
                        f"Stage #{stage_idx} must be a table in {path_label}"
                    )
                stages.append(
                    _parse_decimator_stage(
                        raw_stage,
                        label=f"{path_label} stage #{stage_idx}",
                    )
                )

            if stages[0].input_sample_rate_hz != in_rate:
                raise ValueError(
                    f"First stage input sample rate must match path input in {path_label}"
                )
            if stages[-1].output_sample_rate_hz != out_rate:
                raise ValueError(
                    f"Last stage output sample rate must match path output in {path_label}"
                )
            for stage_idx in range(len(stages) - 1):
                if (
                    stages[stage_idx].output_sample_rate_hz
                    != stages[stage_idx + 1].input_sample_rate_hz
                ):
                    raise ValueError(
                        f"Stage sample rates must chain continuously in {path_label}: "
                        f"stage #{stage_idx + 1} output does not match stage #{stage_idx + 2} input"
                    )

        key = (in_rate, out_rate)
        if key in paths:
            raise ValueError(
                f"Duplicate decimator path in config for {in_rate} -> {out_rate}"
            )

        paths[key] = DecimatorPath(
            bandwidth_hz=bandwidth_hz,
            stages=tuple(stages),
        )

    return paths


def load_decimator_paths(config_path: Path) -> Dict[Tuple[int, int], DecimatorPath]:
    """Load and validate decimator path definitions from a TOML file."""
    return _load_decimator_paths(config_path)


def _get_decimator_path(
    decimator_paths: Dict[Tuple[int, int], DecimatorPath],
    input_sample_rate_hz: int,
    output_sample_rate_hz: int,
) -> DecimatorPath:
    path = decimator_paths.get((input_sample_rate_hz, output_sample_rate_hz))
    if path is not None:
        return path

    configured_output_rates = sorted({out_rate for _, out_rate in decimator_paths})
    if output_sample_rate_hz not in configured_output_rates:
        raise ValueError(
            "Unsupported output sample rate. Configured outputs: "
            + ", ".join(str(x) for x in configured_output_rates)
        )

    supported_input_rates = sorted(
        {
            in_rate
            for (in_rate, out_rate) in decimator_paths
            if out_rate == output_sample_rate_hz
        }
    )
    raise ValueError(
        f"Unsupported input sample rate {input_sample_rate_hz} for output "
        f"{output_sample_rate_hz}. Configured inputs for this output: "
        + ", ".join(str(x) for x in supported_input_rates)
    )


def _get_decimator_path_for_request(
    decimator_paths: Dict[Tuple[int, int], DecimatorPath],
    input_sample_rate_hz: int,
    output_sample_rate_hz: Optional[int],
    target_bandwidth_hz: Optional[int],
) -> tuple[DecimatorPath, int]:
    if output_sample_rate_hz is None and target_bandwidth_hz is None:
        raise ValueError("Specify output_sample_rate_hz or target_bandwidth_hz")
    if output_sample_rate_hz is not None and target_bandwidth_hz is not None:
        raise ValueError("Specify only one of output_sample_rate_hz or target_bandwidth_hz")

    if output_sample_rate_hz is not None:
        path = _get_decimator_path(
            decimator_paths,
            input_sample_rate_hz,
            int(output_sample_rate_hz),
        )
        return path, int(output_sample_rate_hz)

    assert target_bandwidth_hz is not None
    matches: list[tuple[int, DecimatorPath]] = [
        (out_rate, path)
        for (in_rate, out_rate), path in decimator_paths.items()
        if in_rate == input_sample_rate_hz and path.bandwidth_hz == int(target_bandwidth_hz)
    ]
    if not matches:
        configured_bandwidths = sorted(
            {
                path.bandwidth_hz
                for (in_rate, _), path in decimator_paths.items()
                if in_rate == input_sample_rate_hz
            }
        )
        if not configured_bandwidths:
            configured_inputs = sorted({in_rate for in_rate, _ in decimator_paths})
            raise ValueError(
                f"Unsupported input sample rate {input_sample_rate_hz}. "
                "Configured inputs: "
                + ", ".join(str(x) for x in configured_inputs)
            )
        raise ValueError(
            f"Unsupported bandwidth {target_bandwidth_hz} for input {input_sample_rate_hz}. "
            "Configured bandwidths for this input: "
            + ", ".join(str(x) for x in configured_bandwidths)
        )
    if len(matches) > 1:
        matching_output_rates = sorted({out_rate for out_rate, _ in matches})
        raise ValueError(
            f"Ambiguous bandwidth {target_bandwidth_hz} for input {input_sample_rate_hz}. "
            "Matching output sample rates: "
            + ", ".join(str(x) for x in matching_output_rates)
            + ". Specify --output-sample-rate."
        )

    return matches[0][1], int(matches[0][0])


def _build_runtime_stage_chain(decimator_path: DecimatorPath) -> list[RuntimeDecimatorStage]:
    runtime_stages: list[RuntimeDecimatorStage] = []
    for idx, stage_cfg in enumerate(decimator_path.stages, start=1):
        up, down = _resample_ratio(
            stage_cfg.input_sample_rate_hz,
            stage_cfg.output_sample_rate_hz,
        )
        if (up != 1 or down != 1) and stage_cfg.taps is None:
            raise ValueError(
                f"Stage #{idx} is missing 'taps' for "
                f"{stage_cfg.input_sample_rate_hz} -> {stage_cfg.output_sample_rate_hz}"
            )

        runtime_stages.append(
            RuntimeDecimatorStage(
                input_sample_rate_hz=stage_cfg.input_sample_rate_hz,
                output_sample_rate_hz=stage_cfg.output_sample_rate_hz,
                up=up,
                down=down,
                resampler=(
                    None
                    if up == 1 and down == 1
                    else StreamingResampler(stage_cfg.taps, up, down)
                ),
            )
        )

    return runtime_stages


def _process_through_stages(
    runtime_stage_chain: list[RuntimeDecimatorStage],
    samples: np.ndarray,
) -> np.ndarray:
    y = np.asarray(samples, dtype=np.complex64).reshape(-1)
    for runtime_stage in runtime_stage_chain:
        if y.size == 0:
            break
        if runtime_stage.resampler is None:
            continue
        y = runtime_stage.resampler.process(y)
    return np.asarray(y, dtype=np.complex64).reshape(-1)


def _flush_stage_chain(runtime_stage_chain: list[RuntimeDecimatorStage]) -> np.ndarray:
    if not runtime_stage_chain:
        return np.empty(0, dtype=np.complex64)

    flushed_chunks: list[np.ndarray] = []
    for stage_idx, runtime_stage in enumerate(runtime_stage_chain):
        if runtime_stage.resampler is None:
            continue

        stage_tail = runtime_stage.resampler.flush()
        if stage_tail.size == 0:
            continue

        if stage_idx + 1 < len(runtime_stage_chain):
            stage_tail = _process_through_stages(
                runtime_stage_chain[stage_idx + 1 :],
                stage_tail,
            )

        if stage_tail.size > 0:
            flushed_chunks.append(np.asarray(stage_tail, dtype=np.complex64).reshape(-1))

    if not flushed_chunks:
        return np.empty(0, dtype=np.complex64)
    if len(flushed_chunks) == 1:
        return flushed_chunks[0]
    return np.concatenate(flushed_chunks)


def _resolve_center_frequency(
    center_frequency_hz: Optional[float],
    center_frequency_offset_hz: Optional[float],
    input_rf_reference_frequency_hz: Optional[float],
) -> Tuple[float, Optional[float]]:
    if center_frequency_hz is not None and center_frequency_offset_hz is not None:
        raise ValueError(
            "Specify either center_frequency_hz or center_frequency_offset_hz, not both"
        )

    if center_frequency_offset_hz is not None:
        if input_rf_reference_frequency_hz is None:
            raise ValueError(
                "center_frequency_offset_hz requires input CIF0 rf_reference_frequency_hz"
            )
        offset_hz = float(center_frequency_offset_hz)
        return offset_hz, float(input_rf_reference_frequency_hz + offset_hz)

    if center_frequency_hz is None:
        return 0.0, input_rf_reference_frequency_hz

    target_center_hz = float(center_frequency_hz)
    if input_rf_reference_frequency_hz is None:
        # Fall back to DC-referenced interpretation when RF reference is unavailable.
        return target_center_hz, target_center_hz

    return (
        float(target_center_hz - input_rf_reference_frequency_hz),
        target_center_hz,
    )


def _validate_output_band_within_input(
    input_bandwidth_hz: Optional[float],
    output_bandwidth_hz: float,
    center_frequency_offset_hz: float,
) -> None:
    if input_bandwidth_hz is None:
        raise ValueError(
            "Input CIF0 bandwidth_hz is missing; cannot verify requested center frequency"
        )

    in_bw = float(input_bandwidth_hz)
    out_bw = float(output_bandwidth_hz)
    if in_bw <= 0 or out_bw <= 0:
        raise ValueError("Input and output bandwidth must be > 0")
    if out_bw > in_bw:
        raise ValueError(
            f"Output bandwidth {out_bw} Hz exceeds input CIF0 bandwidth {in_bw} Hz"
        )

    max_offset = (in_bw - out_bw) / 2.0
    if abs(float(center_frequency_offset_hz)) > max_offset:
        raise ValueError(
            "Requested center frequency is outside input bandwidth: "
            f"|offset|={abs(center_frequency_offset_hz)} Hz, "
            f"max allowed={max_offset} Hz, input_bw={in_bw} Hz, output_bw={out_bw} Hz"
        )


def convert_v49_ddc(
    input_path: Path,
    output_path: Path,
    output_format_name: str,
    output_sample_rate_hz: Optional[int],
    chunk_samples: int,
    samples_per_packet: int,
    config_path: Optional[Path] = None,
    target_bandwidth_hz: Optional[int] = None,
    center_frequency_hz: Optional[float] = None,
    center_frequency_offset_hz: Optional[float] = None,
    start_time_epoch_s: Optional[float] = None,
    end_time_epoch_s: Optional[float] = None,
) -> Dict[str, int]:
    config_path = (config_path or DEFAULT_DECIMATOR_CONFIG_PATH).expanduser()
    decimator_paths = _load_decimator_paths(config_path)

    supported_formats = {
        "F32_IQ": DefaultPayloadFormats.F32_IQ,
        "S32_IQ": DefaultPayloadFormats.S32_IQ,
        "S24_IQ": DefaultPayloadFormats.S24_IQ,
        "S16_IQ": DefaultPayloadFormats.S16_IQ,
    }

    output_format_name = output_format_name.upper()
    if output_format_name not in supported_formats:
        raise ValueError(
            f"Unsupported output format '{output_format_name}'. "
            f"Supported: {', '.join(sorted(supported_formats))}"
        )
    output_payload_format = supported_formats[output_format_name]

    if output_sample_rate_hz is None and target_bandwidth_hz is None:
        raise ValueError("Specify output_sample_rate_hz or target_bandwidth_hz")
    if output_sample_rate_hz is not None and target_bandwidth_hz is not None:
        raise ValueError("Specify only one of output_sample_rate_hz or target_bandwidth_hz")
    if output_sample_rate_hz is not None:
        configured_output_rates = sorted({out_rate for _, out_rate in decimator_paths})
        if int(output_sample_rate_hz) not in configured_output_rates:
            raise ValueError(
                "Unsupported output sample rate. Configured outputs: "
                + ", ".join(str(x) for x in configured_output_rates)
            )

    if chunk_samples <= 0:
        raise ValueError("chunk_samples must be > 0")
    if samples_per_packet <= 0:
        raise ValueError("samples_per_packet must be > 0")
    if (
        start_time_epoch_s is not None
        and end_time_epoch_s is not None
        and float(end_time_epoch_s) < float(start_time_epoch_s)
    ):
        raise ValueError("end_time_epoch_s must be >= start_time_epoch_s")

    input_path = input_path.expanduser()
    output_path = output_path.expanduser()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # State from the first context packet
    input_sample_rate_hz: Optional[int] = None
    input_payload_format = None
    input_payload_name: Optional[str] = None
    input_bandwidth_hz: Optional[float] = None
    input_rf_ref_hz: Optional[float] = None
    input_rf_ref_offset_hz: Optional[float] = None
    input_if_ref_hz: Optional[float] = None
    input_if_band_offset_hz: Optional[float] = None
    input_reference_level_dbm: Optional[float] = None
    input_gain_db: Optional[Tuple[float, float]] = None
    input_device_identifier: Optional[Tuple[int, int]] = None
    input_state_event_indicators: Optional[int] = None
    input_context_tsm: bool = False
    input_stream_id: Optional[int] = None
    first_context_time_s: Optional[float] = None

    # Resampling parameters
    decimator_path: Optional[DecimatorPath] = None
    runtime_stage_chain: Optional[list[RuntimeDecimatorStage]] = None
    frequency_shifter: Optional[StreamingFrequencyShifter] = None
    center_frequency_offset_effective_hz: float = 0.0
    output_rf_reference_frequency_hz: Optional[float] = None

    # Output stream state
    writer: Optional[IQStreamWriter] = None
    payload_decoder = None

    in_chunks: list[np.ndarray] = []
    in_count = 0
    out_buffer = np.empty(0, dtype=np.complex64)

    total_in_samples = 0
    total_out_samples = 0
    data_packets_written = 0
    data_packets_seen = 0
    context_packets_seen = 0
    skipped_packets = 0
    output_bandwidth_hz: Optional[int] = None
    selected_output_sample_rate_hz: Optional[int] = (
        int(output_sample_rate_hz) if output_sample_rate_hz is not None else None
    )
    time_window_enabled = (start_time_epoch_s is not None) or (end_time_epoch_s is not None)
    next_packet_time_s: Optional[float] = None

    def emit_samples(samples: np.ndarray, out_f) -> None:
        nonlocal out_buffer, data_packets_written, total_out_samples
        if samples.size == 0:
            return
        if out_buffer.size == 0:
            out_buffer = samples
        else:
            out_buffer = np.concatenate([out_buffer, samples])
        while out_buffer.size >= samples_per_packet:
            chunk = out_buffer[:samples_per_packet]
            out_buffer = out_buffer[samples_per_packet:]
            out_f.write(writer.build_data_packet_bytes(chunk))
            data_packets_written += 1
            total_out_samples += samples_per_packet

    with input_path.open("rb") as f_in, output_path.open("wb") as f_out:
        reader = PacketReader(f_in)
        while True:
            pkt = reader.read_packet_fast()
            if pkt is None:
                break

            if isinstance(pkt, ContextPacket):
                context_packets_seen += 1
                context_time_s = _packet_time_s(pkt.integer_seconds, pkt.fractional_seconds)
                if first_context_time_s is None:
                    first_context_time_s = context_time_s
                if next_packet_time_s is None and context_time_s is not None:
                    next_packet_time_s = context_time_s
                if pkt.cif0 is not None:
                    cif0 = pkt.cif0
                    if input_payload_format is None and cif0.payload_format is not None:
                        input_payload_format = cif0.payload_format
                        input_payload_name = _resolve_input_format_name(
                            input_payload_format,
                            supported_formats,
                        )
                    if input_sample_rate_hz is None and cif0.sample_rate_hz is not None:
                        input_sample_rate_hz = int(round(float(cif0.sample_rate_hz)))
                    if input_bandwidth_hz is None and cif0.bandwidth_hz is not None:
                        input_bandwidth_hz = float(cif0.bandwidth_hz)
                    if input_rf_ref_hz is None and cif0.rf_reference_frequency_hz is not None:
                        input_rf_ref_hz = float(cif0.rf_reference_frequency_hz)
                    if input_rf_ref_offset_hz is None and cif0.rf_reference_frequency_offset_hz is not None:
                        input_rf_ref_offset_hz = float(cif0.rf_reference_frequency_offset_hz)
                    if input_if_ref_hz is None and cif0.if_reference_frequency_hz is not None:
                        input_if_ref_hz = float(cif0.if_reference_frequency_hz)
                    if input_if_band_offset_hz is None and cif0.if_band_offset_hz is not None:
                        input_if_band_offset_hz = float(cif0.if_band_offset_hz)
                    if input_reference_level_dbm is None and cif0.reference_level_dbm is not None:
                        input_reference_level_dbm = float(cif0.reference_level_dbm)
                    if input_gain_db is None and cif0.gain_db is not None:
                        input_gain_db = cif0.gain_db
                    if input_device_identifier is None and cif0.device_identifier is not None:
                        input_device_identifier = cif0.device_identifier
                    if input_state_event_indicators is None and cif0.state_event_indicators is not None:
                        input_state_event_indicators = cif0.state_event_indicators
                if input_stream_id is None and pkt.stream_id is not None:
                    input_stream_id = pkt.stream_id
                input_context_tsm = bool(pkt.header.indicators_24)
                continue

            if isinstance(pkt, RawDataPacket):
                data_packets_seen += 1
                if input_payload_format is None or input_sample_rate_hz is None:
                    skipped_packets += 1
                    continue
                if input_payload_name is None:
                    raise ValueError(
                        "Unsupported input payload format. "
                        "Supported: F32_IQ, S32_IQ, S24_IQ, S16_IQ"
                    )

                packet_start_time_s = _packet_time_s(pkt.integer_seconds, pkt.fractional_seconds)
                if packet_start_time_s is None and time_window_enabled:
                    packet_start_time_s = next_packet_time_s
                    if packet_start_time_s is None:
                        packet_start_time_s = first_context_time_s
                    if packet_start_time_s is None:
                        raise ValueError(
                            "Time slicing requires packet or context timestamps in the input stream"
                        )

                if time_window_enabled:
                    packet_n_samples = _packet_sample_count(pkt.payload, input_payload_format)
                    packet_end_time_s = packet_start_time_s + (
                        float(packet_n_samples) / float(input_sample_rate_hz)
                    )
                    next_packet_time_s = packet_end_time_s
                    if (
                        start_time_epoch_s is not None
                        and packet_end_time_s <= float(start_time_epoch_s)
                    ):
                        skipped_packets += 1
                        continue
                    if (
                        end_time_epoch_s is not None
                        and packet_start_time_s >= float(end_time_epoch_s)
                    ):
                        skipped_packets += 1
                        break

                if runtime_stage_chain is None:
                    decimator_path, selected_output_sample_rate_hz = _get_decimator_path_for_request(
                        decimator_paths,
                        input_sample_rate_hz,
                        selected_output_sample_rate_hz,
                        target_bandwidth_hz,
                    )
                    output_bandwidth_hz = decimator_path.bandwidth_hz
                    (
                        center_frequency_offset_effective_hz,
                        output_rf_reference_frequency_hz,
                    ) = _resolve_center_frequency(
                        center_frequency_hz=center_frequency_hz,
                        center_frequency_offset_hz=center_frequency_offset_hz,
                        input_rf_reference_frequency_hz=input_rf_ref_hz,
                    )
                    _validate_output_band_within_input(
                        input_bandwidth_hz=input_bandwidth_hz,
                        output_bandwidth_hz=float(output_bandwidth_hz),
                        center_frequency_offset_hz=center_frequency_offset_effective_hz,
                    )
                    runtime_stage_chain = _build_runtime_stage_chain(decimator_path)
                    if center_frequency_offset_effective_hz != 0.0:
                        frequency_shifter = StreamingFrequencyShifter(
                            sample_rate_hz=float(input_sample_rate_hz),
                            frequency_offset_hz=center_frequency_offset_effective_hz,
                        )

                if writer is None:
                    if input_stream_id is None:
                        input_stream_id = pkt.stream_id
                    if input_stream_id is None:
                        raise ValueError("Input stream_id is missing; cannot write output stream")
                    start_time_s = packet_start_time_s
                    if start_time_s is None:
                        start_time_s = first_context_time_s
                    writer = IQStreamWriter(
                        stream_id=input_stream_id,
                        sample_rate_hz=float(selected_output_sample_rate_hz),
                        payload_format=output_payload_format,
                        data_packet_type=PacketType(pkt.packet_type),
                        tsi=TSI(pkt.tsi),
                        tsf=TSF(pkt.tsf),
                        class_id=pkt.class_id,
                        requires_vita49_2=bool(pkt.indicators_25),
                        frequency_domain=bool(pkt.indicators_24),
                        start_time_epoch_s=start_time_s,
                        bandwidth_hz=float(output_bandwidth_hz),
                        rf_reference_frequency_hz=output_rf_reference_frequency_hz,
                        rf_reference_frequency_offset_hz=input_rf_ref_offset_hz,
                        if_reference_frequency_hz=input_if_ref_hz,
                        if_band_offset_hz=input_if_band_offset_hz,
                        reference_level_dbm=input_reference_level_dbm,
                        gain_db=input_gain_db,
                        device_identifier=input_device_identifier,
                        state_event_indicators=input_state_event_indicators,
                        context_timestamp_mode_general=input_context_tsm,
                    )
                    f_out.write(writer.build_context_packet().to_bytes())

                if payload_decoder is None:
                    payload_decoder = build_payload_decoder(input_payload_format)
                iq = payload_decoder(pkt.payload)

                in_chunks.append(iq)
                in_count += int(iq.size)
                total_in_samples += int(iq.size)

                while in_count >= chunk_samples:
                    combined = np.concatenate(in_chunks) if len(in_chunks) > 1 else in_chunks[0]
                    block = combined[:chunk_samples]
                    remainder = combined[chunk_samples:]
                    in_chunks = [remainder] if remainder.size else []
                    in_count = int(remainder.size)

                    if frequency_shifter is not None:
                        block = frequency_shifter.process(block)
                    resampled = _process_through_stages(runtime_stage_chain, block)
                    emit_samples(np.asarray(resampled, dtype=np.complex64).reshape(-1), f_out)
                continue

            skipped_packets += 1

        # Process remaining samples after loop ends
        if in_count > 0 and input_sample_rate_hz is not None and input_payload_format is not None:
            combined = np.concatenate(in_chunks) if len(in_chunks) > 1 else in_chunks[0]
            if runtime_stage_chain is None:
                decimator_path, selected_output_sample_rate_hz = _get_decimator_path_for_request(
                    decimator_paths,
                    input_sample_rate_hz,
                    selected_output_sample_rate_hz,
                    target_bandwidth_hz,
                )
                output_bandwidth_hz = decimator_path.bandwidth_hz
                (
                    center_frequency_offset_effective_hz,
                    output_rf_reference_frequency_hz,
                ) = _resolve_center_frequency(
                    center_frequency_hz=center_frequency_hz,
                    center_frequency_offset_hz=center_frequency_offset_hz,
                    input_rf_reference_frequency_hz=input_rf_ref_hz,
                )
                _validate_output_band_within_input(
                    input_bandwidth_hz=input_bandwidth_hz,
                    output_bandwidth_hz=float(output_bandwidth_hz),
                    center_frequency_offset_hz=center_frequency_offset_effective_hz,
                )
                runtime_stage_chain = _build_runtime_stage_chain(decimator_path)
                if center_frequency_offset_effective_hz != 0.0:
                    frequency_shifter = StreamingFrequencyShifter(
                        sample_rate_hz=float(input_sample_rate_hz),
                        frequency_offset_hz=center_frequency_offset_effective_hz,
                    )
            if frequency_shifter is not None:
                combined = frequency_shifter.process(combined)
            resampled = _process_through_stages(runtime_stage_chain, combined)
            emit_samples(np.asarray(resampled, dtype=np.complex64).reshape(-1), f_out)

        # Flush filter/carry state for all resampler stages at EOF
        if runtime_stage_chain is not None:
            tail = _flush_stage_chain(runtime_stage_chain)
            emit_samples(np.asarray(tail, dtype=np.complex64).reshape(-1), f_out)

        # Pad the last packet with zeros to reach samples_per_packet
        if writer is not None and out_buffer.size > 0:
            pad_len = samples_per_packet - int(out_buffer.size)
            padded = np.concatenate([out_buffer, np.zeros(pad_len, dtype=np.complex64)])
            f_out.write(writer.build_data_packet_bytes(padded))
            data_packets_written += 1
            total_out_samples += samples_per_packet

    return {
        "input_samples": total_in_samples,
        "output_samples": total_out_samples,
        "data_packets_written": data_packets_written,
        "data_packets_seen": data_packets_seen,
        "context_packets_seen": context_packets_seen,
        "skipped_packets": skipped_packets,
        "input_sample_rate_hz": int(input_sample_rate_hz or 0),
        "output_sample_rate_hz": int(selected_output_sample_rate_hz or 0),
        "input_payload_format": input_payload_name or "",
        "output_payload_format": output_format_name,
        "output_bandwidth_hz": int(output_bandwidth_hz or 0),
    }


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    start_time = time.perf_counter() if bool(args.timing) else None

    try:
        summary = convert_v49_ddc(
            input_path=Path(args.input_file),
            output_path=Path(args.output_file),
            output_format_name=args.output_format,
            output_sample_rate_hz=(
                int(args.output_sample_rate)
                if args.output_sample_rate is not None
                else None
            ),
            target_bandwidth_hz=(
                int(args.bandwidth)
                if args.bandwidth is not None
                else None
            ),
            chunk_samples=int(args.chunk_samples),
            samples_per_packet=int(args.samples_per_packet),
            center_frequency_hz=(
                float(args.center_frequency_hz)
                if args.center_frequency_hz is not None
                else None
            ),
            center_frequency_offset_hz=(
                float(args.center_frequency_offset_hz)
                if args.center_frequency_offset_hz is not None
                else None
            ),
            start_time_epoch_s=(
                _parse_iso_utc_time(args.start_time, option_name="--start-time")
                if args.start_time is not None
                else None
            ),
            end_time_epoch_s=(
                _parse_iso_utc_time(args.end_time, option_name="--end-time")
                if args.end_time is not None
                else None
            ),
            config_path=Path(args.config).expanduser() if args.config else None,
        )
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print(
        "DDC complete. "
        f"In={summary['input_sample_rate_hz']} Hz {summary['input_payload_format']} -> "
        f"Out={summary['output_sample_rate_hz']} Hz {summary['output_payload_format']} "
        f"(BW {summary['output_bandwidth_hz']}), "
        f"Input samples: {summary['input_samples']}, "
        f"Output samples: {summary['output_samples']}, "
        f"Data packets written: {summary['data_packets_written']}"
    )
    if start_time is not None:
        elapsed_s = time.perf_counter() - start_time
        print(f"Timing: elapsed {elapsed_s:.3f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
