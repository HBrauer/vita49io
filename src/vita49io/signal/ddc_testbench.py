from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from vita49io.defaults.default_payload_formats import DefaultPayloadFormats
from vita49io.io.iq_writer import IQStreamWriter
from vita49io.io.packet_reader import PacketReader
from vita49io.io.payload_codec import payload_as_numpy
from vita49io.protocol.context_packet import ContextPacket
from vita49io.protocol.data_packet import DataPacket

SAMPLES_PER_PACKET = 1024


@dataclass(frozen=True)
class ToneSpec:
    frequency_hz: float
    amplitude: float


@dataclass(frozen=True)
class AMSpec:
    carrier_frequency_hz: float
    amplitude: float
    modulation_frequency_hz: float
    modulation_index: float


@dataclass(frozen=True)
class FSKSpec:
    mark_frequency_hz: float
    space_frequency_hz: float
    symbol_rate_hz: float
    amplitude: float
    seed: int


@dataclass(frozen=True)
class BlockerSpec:
    frequency_hz: float
    amplitude: float


@dataclass(frozen=True)
class Thresholds:
    tone_freq_tolerance_hz: float
    am_min_correlation: float
    fsk_max_ber: float
    max_blocker_db_relative_to_tone: float
    max_unwanted_db_relative_to_tone: float


@dataclass(frozen=True)
class DDCScenario:
    sample_rate_hz: float
    bandwidth_hz: float
    rf_reference_frequency_hz: float
    duration_seconds: float
    output_format: str
    seed: int
    center_frequency_offset_hz: float
    noise_amplitude: float
    max_abs: float
    tone: ToneSpec
    am: AMSpec
    fsk: FSKSpec
    blockers: Tuple[BlockerSpec, ...]
    thresholds: Thresholds


@dataclass(frozen=True)
class OutputMetrics:
    tone_frequency_hz: float
    tone_frequency_error_hz: float
    am_correlation: float
    fsk_ber: float
    worst_blocker_db_relative_to_tone: float
    worst_unwanted_db_relative_to_tone: float


@dataclass(frozen=True)
class ValidationResult:
    output_sample_rate_hz: int
    output_bandwidth_hz: float
    output_file: Path
    waterfall_svg: Optional[Path]
    metrics: OutputMetrics
    passed: bool
    failures: Tuple[str, ...]


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
                "TOML parsing requires Python 3.11+ or 'tomli' on Python 3.8-3.10"
            ) from exc


def load_scenario(path: Path) -> DDCScenario:
    path = path.expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Scenario file not found: {path}")

    toml = _load_toml_module()
    with path.open("rb") as f:
        doc = toml.load(f)

    root = doc.get("scenario", doc)
    if not isinstance(root, dict):
        raise ValueError("Scenario root must be a table")

    tone_raw = root.get("tone")
    am_raw = root.get("am")
    fsk_raw = root.get("fsk")
    noise_raw = root.get("noise", {})
    blockers_raw = root.get("blockers", [])
    thresholds_raw = doc.get("thresholds", {})

    if not isinstance(tone_raw, dict):
        raise ValueError("Missing [scenario.tone] table")
    if not isinstance(am_raw, dict):
        raise ValueError("Missing [scenario.am] table")
    if not isinstance(fsk_raw, dict):
        raise ValueError("Missing [scenario.fsk] table")
    if not isinstance(noise_raw, dict):
        raise ValueError("[scenario.noise] must be a table")
    if not isinstance(blockers_raw, list):
        raise ValueError("[scenario.blockers] must be an array of tables")
    if not isinstance(thresholds_raw, dict):
        raise ValueError("[thresholds] must be a table")

    blockers: List[BlockerSpec] = []
    for idx, b in enumerate(blockers_raw, start=1):
        if not isinstance(b, dict):
            raise ValueError(f"Blocker #{idx} must be a table")
        blockers.append(
            BlockerSpec(
                frequency_hz=float(b["frequency_hz"]),
                amplitude=float(b["amplitude"]),
            )
        )

    return DDCScenario(
        sample_rate_hz=float(root.get("sample_rate_hz", 98_304_000.0)),
        bandwidth_hz=float(root.get("bandwidth_hz", 80_000_000.0)),
        rf_reference_frequency_hz=float(
            root.get("rf_reference_frequency_hz", 915_000_000.0)
        ),
        duration_seconds=float(root.get("duration_seconds", 0.05)),
        output_format=str(root.get("output_format", "S16_IQ")).upper(),
        seed=int(root.get("seed", 1337)),
        center_frequency_offset_hz=float(root.get("center_frequency_offset_hz", 24_000.0)),
        noise_amplitude=float(noise_raw.get("amplitude", 0.02)),
        max_abs=float(root.get("max_abs", 0.85)),
        tone=ToneSpec(
            frequency_hz=float(tone_raw["frequency_hz"]),
            amplitude=float(tone_raw["amplitude"]),
        ),
        am=AMSpec(
            carrier_frequency_hz=float(am_raw["carrier_frequency_hz"]),
            amplitude=float(am_raw["amplitude"]),
            modulation_frequency_hz=float(am_raw["modulation_frequency_hz"]),
            modulation_index=float(am_raw["modulation_index"]),
        ),
        fsk=FSKSpec(
            mark_frequency_hz=float(fsk_raw["mark_frequency_hz"]),
            space_frequency_hz=float(fsk_raw["space_frequency_hz"]),
            symbol_rate_hz=float(fsk_raw["symbol_rate_hz"]),
            amplitude=float(fsk_raw["amplitude"]),
            seed=int(fsk_raw.get("seed", 2026)),
        ),
        blockers=tuple(blockers),
        thresholds=Thresholds(
            tone_freq_tolerance_hz=float(
                thresholds_raw.get("tone_freq_tolerance_hz", 150.0)
            ),
            am_min_correlation=float(thresholds_raw.get("am_min_correlation", 0.8)),
            fsk_max_ber=float(thresholds_raw.get("fsk_max_ber", 0.05)),
            max_blocker_db_relative_to_tone=float(
                thresholds_raw.get("max_blocker_db_relative_to_tone", -30.0)
            ),
            max_unwanted_db_relative_to_tone=float(
                thresholds_raw.get("max_unwanted_db_relative_to_tone", -35.0)
            ),
        ),
    )


def _resolve_payload_format(name: str):
    if not hasattr(DefaultPayloadFormats, name):
        raise ValueError(
            f"Unsupported output_format '{name}'. Expected one of F32_IQ, S32_IQ, S24_IQ, S16_IQ"
        )
    return getattr(DefaultPayloadFormats, name)


def _synthesize_fsk(
    *,
    sample_rate_hz: float,
    n_samples: int,
    mark_frequency_hz: float,
    space_frequency_hz: float,
    symbol_rate_hz: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if symbol_rate_hz <= 0:
        raise ValueError("symbol_rate_hz must be > 0")
    n_symbols = max(1, int(math.ceil((n_samples / sample_rate_hz) * symbol_rate_hz)))

    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=n_symbols, dtype=np.int8)

    n = np.arange(n_samples, dtype=np.float64)
    sym_idx = np.floor((n * symbol_rate_hz) / sample_rate_hz).astype(np.int64)
    sym_idx = np.clip(sym_idx, 0, n_symbols - 1)
    symbols = bits[sym_idx]

    inst_freq = np.where(symbols > 0, mark_frequency_hz, space_frequency_hz).astype(np.float64)
    phase = 2.0 * np.pi * np.cumsum(inst_freq) / sample_rate_hz
    iq = np.exp(1j * phase).astype(np.complex64)
    return iq, bits


def synthesize_composite_iq(
    scenario: DDCScenario,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    n_samples = max(1, int(round(scenario.duration_seconds * scenario.sample_rate_hz)))
    t = np.arange(n_samples, dtype=np.float64) / scenario.sample_rate_hz

    iq = np.zeros(n_samples, dtype=np.complex64)

    tone = scenario.tone.amplitude * np.exp(1j * 2.0 * np.pi * scenario.tone.frequency_hz * t)
    iq += tone.astype(np.complex64)

    am_env = 1.0 + scenario.am.modulation_index * np.sin(
        2.0 * np.pi * scenario.am.modulation_frequency_hz * t
    )
    am_carrier = np.exp(1j * 2.0 * np.pi * scenario.am.carrier_frequency_hz * t)
    iq += (scenario.am.amplitude * am_env * am_carrier).astype(np.complex64)

    fsk_iq, fsk_bits = _synthesize_fsk(
        sample_rate_hz=scenario.sample_rate_hz,
        n_samples=n_samples,
        mark_frequency_hz=scenario.fsk.mark_frequency_hz,
        space_frequency_hz=scenario.fsk.space_frequency_hz,
        symbol_rate_hz=scenario.fsk.symbol_rate_hz,
        seed=scenario.fsk.seed,
    )
    iq += np.complex64(scenario.fsk.amplitude) * fsk_iq

    for blocker in scenario.blockers:
        blocker_iq = blocker.amplitude * np.exp(
            1j * 2.0 * np.pi * blocker.frequency_hz * t
        )
        iq += blocker_iq.astype(np.complex64)

    if scenario.noise_amplitude > 0.0:
        rng = np.random.default_rng(scenario.seed)
        sigma = np.float32(scenario.noise_amplitude / math.sqrt(2.0))
        noise = (
            rng.standard_normal(n_samples, dtype=np.float32)
            + 1j * rng.standard_normal(n_samples, dtype=np.float32)
        ) * sigma
        iq += noise.astype(np.complex64)

    peak = float(np.max(np.abs(iq))) if iq.size else 0.0
    scale = 1.0
    if peak > scenario.max_abs > 0.0:
        scale = float(scenario.max_abs / peak)
        iq = (iq * np.float32(scale)).astype(np.complex64)

    meta = {
        "fsk_bits": fsk_bits,
        "fsk_symbol_rate_hz": float(scenario.fsk.symbol_rate_hz),
        "fsk_n_symbols": int(fsk_bits.size),
        "scale_applied": scale,
        "n_samples": int(iq.size),
        "duration_seconds": float(iq.size / scenario.sample_rate_hz),
    }
    return iq, meta


def write_v49_from_iq(
    output_path: Path,
    *,
    scenario: DDCScenario,
    iq: np.ndarray,
) -> Dict[str, Any]:
    output_path = output_path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload_format = _resolve_payload_format(scenario.output_format)
    writer = IQStreamWriter(
        stream_id=0x13572468,
        sample_rate_hz=float(scenario.sample_rate_hz),
        payload_format=payload_format,
        bandwidth_hz=float(scenario.bandwidth_hz),
        rf_reference_frequency_hz=float(scenario.rf_reference_frequency_hz),
    )

    n_packets = 0
    with output_path.open("wb") as f:
        f.write(writer.build_context_packet().to_bytes())
        for start in range(0, int(iq.size), SAMPLES_PER_PACKET):
            block = np.asarray(iq[start : start + SAMPLES_PER_PACKET], dtype=np.complex64)
            if block.size < SAMPLES_PER_PACKET:
                pad = np.zeros(SAMPLES_PER_PACKET - block.size, dtype=np.complex64)
                block = np.concatenate([block, pad])
            f.write(writer.build_data_packet_bytes(block))
            n_packets += 1

    return {
        "path": str(output_path),
        "data_packets": int(n_packets),
        "samples_written": int(n_packets * SAMPLES_PER_PACKET),
    }


def read_v49_iq(path: Path, *, max_samples: Optional[int] = None) -> Tuple[np.ndarray, float, Optional[float]]:
    path = path.expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"V49 file not found: {path}")

    chunks: List[np.ndarray] = []
    sample_rate_hz: Optional[float] = None
    rf_reference_frequency_hz: Optional[float] = None
    payload_format = None
    total = 0

    with path.open("rb") as f:
        reader = PacketReader(f)
        while True:
            pkt = reader.read_packet()
            if pkt is None:
                break

            if isinstance(pkt, ContextPacket):
                if pkt.cif0 is not None:
                    if sample_rate_hz is None and pkt.cif0.sample_rate_hz is not None:
                        sample_rate_hz = float(pkt.cif0.sample_rate_hz)
                    if (
                        rf_reference_frequency_hz is None
                        and pkt.cif0.rf_reference_frequency_hz is not None
                    ):
                        rf_reference_frequency_hz = float(pkt.cif0.rf_reference_frequency_hz)
                    if payload_format is None and pkt.cif0.payload_format is not None:
                        payload_format = pkt.cif0.payload_format
                continue

            if isinstance(pkt, DataPacket):
                if payload_format is None:
                    continue
                payload = pkt.payload
                if isinstance(payload, memoryview):
                    payload = payload.tobytes()
                iq = payload_as_numpy(payload, payload_format)
                if max_samples is not None:
                    remaining = int(max_samples) - total
                    if remaining <= 0:
                        break
                    iq = iq[:remaining]
                chunks.append(np.asarray(iq, dtype=np.complex64))
                total += int(iq.size)

    if sample_rate_hz is None:
        raise ValueError(f"Could not determine sample rate from context in {path}")

    if not chunks:
        return np.empty(0, dtype=np.complex64), sample_rate_hz, rf_reference_frequency_hz

    iq = chunks[0] if len(chunks) == 1 else np.concatenate(chunks)
    return np.asarray(iq, dtype=np.complex64), sample_rate_hz, rf_reference_frequency_hz


def fold_frequency_hz(freq_hz: float, sample_rate_hz: float) -> float:
    fs = float(sample_rate_hz)
    return float(((float(freq_hz) + fs / 2.0) % fs) - fs / 2.0)


def _fft_db(iq: np.ndarray, sample_rate_hz: float, *, nfft: int = 262_144) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(iq, dtype=np.complex64).reshape(-1)
    if x.size == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    n = min(int(nfft), int(x.size))
    if n <= 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    seg = x[:n]
    w = np.hanning(n).astype(np.float32)
    spec = np.fft.fftshift(np.fft.fft(seg * w, n=n))
    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / float(sample_rate_hz)))
    db = 20.0 * np.log10(np.abs(spec) + 1e-12)
    return freqs.astype(np.float64), db.astype(np.float64)


def _peak_near(freqs: np.ndarray, db: np.ndarray, target_hz: float, *, half_bins: int = 1) -> float:
    if freqs.size == 0:
        return -300.0
    idx = int(np.argmin(np.abs(freqs - float(target_hz))))
    lo = max(0, idx - int(half_bins))
    hi = min(db.size, idx + int(half_bins) + 1)
    return float(np.max(db[lo:hi]))


def _dominant_frequency_hz(freqs: np.ndarray, db: np.ndarray) -> float:
    if db.size == 0:
        return 0.0
    return float(freqs[int(np.argmax(db))])


def _dominant_frequency_near(
    freqs: np.ndarray,
    db: np.ndarray,
    *,
    target_hz: float,
    half_span_hz: float,
) -> float:
    if db.size == 0:
        return 0.0
    mask = np.abs(freqs - float(target_hz)) <= float(half_span_hz)
    if not np.any(mask):
        return float(target_hz)
    idx_local = int(np.argmax(db[mask]))
    return float(freqs[mask][idx_local])


def _band_limited_iq(
    iq: np.ndarray,
    *,
    sample_rate_hz: float,
    center_hz: float,
    half_bw_hz: float,
) -> np.ndarray:
    x = np.asarray(iq, dtype=np.complex64).reshape(-1)
    if x.size == 0:
        return x
    if half_bw_hz <= 0.0:
        return np.zeros_like(x)

    spec = np.fft.fft(x)
    freqs = np.fft.fftfreq(x.size, d=1.0 / float(sample_rate_hz))
    mask = np.abs(freqs - float(center_hz)) <= float(half_bw_hz)
    filt = np.where(mask, spec, 0.0)
    return np.fft.ifft(filt).astype(np.complex64)


def _am_correlation(
    iq: np.ndarray,
    sample_rate_hz: float,
    *,
    carrier_hz: float,
    modulation_hz: float,
) -> float:
    x = np.asarray(iq, dtype=np.complex64).reshape(-1)
    if x.size < 8:
        return 0.0

    band = _band_limited_iq(
        x,
        sample_rate_hz=float(sample_rate_hz),
        center_hz=float(carrier_hz),
        half_bw_hz=max(500.0, 1.3 * float(modulation_hz)),
    )

    n = np.arange(x.size, dtype=np.float64)
    mix = np.exp(-1j * 2.0 * np.pi * carrier_hz * n / float(sample_rate_hz)).astype(np.complex64)
    bb = band * mix
    env = np.abs(bb).astype(np.float64)
    env = env - float(np.mean(env))

    smooth_n = max(3, int(round(float(sample_rate_hz) / max(1.0, 4.0 * float(modulation_hz)))))
    if smooth_n > 1:
        kernel = np.ones(smooth_n, dtype=np.float64) / float(smooth_n)
        env = np.convolve(env, kernel, mode="same")

    expected = np.sin(2.0 * np.pi * float(modulation_hz) * n / float(sample_rate_hz))
    denom = float(np.linalg.norm(env) * np.linalg.norm(expected))
    if denom <= 0.0:
        return 0.0
    corr = float(np.dot(env, expected) / denom)
    return abs(corr)


def _am_sideband_score(
    freqs: np.ndarray,
    db: np.ndarray,
    *,
    carrier_hz: float,
    modulation_hz: float,
) -> float:
    carrier_db = _peak_near(freqs, db, carrier_hz)
    lsb_db = _peak_near(freqs, db, carrier_hz - modulation_hz)
    usb_db = _peak_near(freqs, db, carrier_hz + modulation_hz)
    min_sideband_dbc = min(lsb_db - carrier_db, usb_db - carrier_db)
    asym_db = abs(lsb_db - usb_db)

    # Map sideband visibility and symmetry to [0, 1].
    sideband_score = (min_sideband_dbc + 40.0) / 30.0  # -40 dBc -> 0, -10 dBc -> 1
    sideband_score = float(np.clip(sideband_score, 0.0, 1.0))
    symmetry_score = float(np.clip((10.0 - asym_db) / 10.0, 0.0, 1.0))
    return float(sideband_score * symmetry_score)


def _decode_fsk_bits(
    iq: np.ndarray,
    sample_rate_hz: float,
    *,
    mark_hz: float,
    space_hz: float,
    symbol_rate_hz: float,
    n_symbols: int,
    symbol_offset_samples: int = 0,
) -> np.ndarray:
    x = np.asarray(iq, dtype=np.complex64).reshape(-1)
    if x.size < 16 or n_symbols <= 0:
        return np.empty(0, dtype=np.int8)

    center_hz = 0.5 * (float(mark_hz) + float(space_hz))
    deviation_hz = 0.5 * abs(float(mark_hz) - float(space_hz))
    band = _band_limited_iq(
        x,
        sample_rate_hz=float(sample_rate_hz),
        center_hz=float(center_hz),
        half_bw_hz=max(500.0, deviation_hz + 0.35 * float(symbol_rate_hz)),
    )
    n = np.arange(x.size, dtype=np.float64)
    mix = np.exp(-1j * 2.0 * np.pi * center_hz * n / float(sample_rate_hz)).astype(np.complex64)
    bb = band * mix

    phase_step = np.angle(bb[1:] * np.conj(bb[:-1]))
    inst_freq = (phase_step * float(sample_rate_hz) / (2.0 * np.pi)).astype(np.float64)

    sps = float(sample_rate_hz) / float(symbol_rate_hz)
    bits: List[int] = []
    offset = max(0, int(symbol_offset_samples))
    for k in range(int(n_symbols)):
        start = offset + int(round(k * sps))
        end = offset + int(round((k + 1) * sps))
        if end <= start:
            continue
        if start >= inst_freq.size:
            break
        end = min(end, inst_freq.size)
        mean_f = float(np.mean(inst_freq[start:end]))
        bits.append(1 if mean_f >= 0.0 else 0)

    return np.asarray(bits, dtype=np.int8)


def _bit_error_rate(expected: np.ndarray, got: np.ndarray) -> float:
    if expected.size == 0 or got.size == 0:
        return 1.0
    n = min(int(expected.size), int(got.size))
    if n <= 0:
        return 1.0

    guard = max(2, int(round(0.05 * n)))
    lo = guard
    hi = n - guard
    if hi <= lo:
        lo = 0
        hi = n

    ref = expected[lo:hi]
    obs = got[lo:hi]
    if ref.size == 0 or obs.size == 0:
        return 1.0

    err = np.count_nonzero(ref != obs)
    return float(err) / float(ref.size)


def _best_fsk_ber(
    *,
    iq: np.ndarray,
    sample_rate_hz: float,
    mark_hz: float,
    space_hz: float,
    symbol_rate_hz: float,
    expected_bits: np.ndarray,
    n_symbols: int,
) -> float:
    sps = max(1, int(round(float(sample_rate_hz) / float(symbol_rate_hz))))
    max_offset = min(sps, 32)
    best = 1.0
    for offset in range(max_offset):
        got = _decode_fsk_bits(
            iq,
            sample_rate_hz,
            mark_hz=mark_hz,
            space_hz=space_hz,
            symbol_rate_hz=symbol_rate_hz,
            n_symbols=n_symbols,
            symbol_offset_samples=offset,
        )
        ber = _bit_error_rate(expected_bits, got)
        inv = _bit_error_rate(expected_bits, (1 - got).astype(np.int8)) if got.size else 1.0
        best = min(best, ber, inv)
    return float(best)


def _worst_unwanted_dbc(
    freqs: np.ndarray,
    db: np.ndarray,
    *,
    tone_db: float,
    half_bw_hz: float,
    protected_hz: Sequence[float],
    protected_spans_hz: Sequence[Tuple[float, float]] = (),
    guard_hz: float,
) -> float:
    if freqs.size == 0 or db.size == 0:
        return float("-inf")

    mask = np.abs(freqs) <= float(half_bw_hz)
    for f0 in protected_hz:
        mask &= np.abs(freqs - float(f0)) > float(guard_hz)
    for center_hz, half_span_hz in protected_spans_hz:
        if half_span_hz <= 0.0:
            continue
        mask &= np.abs(freqs - float(center_hz)) > float(half_span_hz)

    if not np.any(mask):
        return float("-inf")
    return float(np.max(db[mask]) - float(tone_db))


def evaluate_output_metrics(
    *,
    iq: np.ndarray,
    sample_rate_hz: float,
    output_bandwidth_hz: float,
    scenario: DDCScenario,
    synthesis_meta: Dict[str, Any],
) -> OutputMetrics:
    freqs, db = _fft_db(iq, sample_rate_hz)

    tone_expected_hz = float(scenario.tone.frequency_hz - scenario.center_frequency_offset_hz)
    bin_hz = float(abs(freqs[1] - freqs[0])) if freqs.size > 1 else 1.0
    tone_meas_hz = _dominant_frequency_near(
        freqs,
        db,
        target_hz=tone_expected_hz,
        half_span_hz=max(200.0, 6.0 * bin_hz),
    )
    tone_err_hz = float(abs(tone_meas_hz - tone_expected_hz))
    tone_db = _peak_near(freqs, db, tone_expected_hz)

    am_carrier_out_hz = float(
        scenario.am.carrier_frequency_hz - scenario.center_frequency_offset_hz
    )
    am_demod_corr = _am_correlation(
        iq,
        sample_rate_hz,
        carrier_hz=am_carrier_out_hz,
        modulation_hz=float(scenario.am.modulation_frequency_hz),
    )
    am_sideband_corr = _am_sideband_score(
        freqs,
        db,
        carrier_hz=am_carrier_out_hz,
        modulation_hz=float(scenario.am.modulation_frequency_hz),
    )
    am_corr = max(float(am_demod_corr), float(am_sideband_corr))

    fsk_mark_out_hz = float(
        scenario.fsk.mark_frequency_hz - scenario.center_frequency_offset_hz
    )
    fsk_space_out_hz = float(
        scenario.fsk.space_frequency_hz - scenario.center_frequency_offset_hz
    )
    expected_bits = np.asarray(synthesis_meta["fsk_bits"], dtype=np.int8)
    fsk_ber = _best_fsk_ber(
        iq=iq,
        sample_rate_hz=sample_rate_hz,
        mark_hz=fsk_mark_out_hz,
        space_hz=fsk_space_out_hz,
        symbol_rate_hz=float(scenario.fsk.symbol_rate_hz),
        expected_bits=expected_bits,
        n_symbols=int(synthesis_meta["fsk_n_symbols"]),
    )

    half_bw = float(output_bandwidth_hz) / 2.0
    blocker_dbc: List[float] = []
    protected = [
        tone_expected_hz,
        am_carrier_out_hz,
        am_carrier_out_hz - float(scenario.am.modulation_frequency_hz),
        am_carrier_out_hz + float(scenario.am.modulation_frequency_hz),
        fsk_mark_out_hz,
        fsk_space_out_hz,
    ]
    # If a blocker is intentionally inside the output passband, exclude it from
    # "unwanted spur" scoring. It is part of the test stimulus, not an artifact.
    for blocker in scenario.blockers:
        blocker_shifted = float(blocker.frequency_hz - scenario.center_frequency_offset_hz)
        if abs(blocker_shifted) <= half_bw:
            protected.append(blocker_shifted)

    protect_hz = max(250.0, 4.0 * bin_hz)
    protected_spans_hz: List[Tuple[float, float]] = []
    tone_span_hz = max(protect_hz, 8.0 * bin_hz)
    protected_spans_hz.append((tone_expected_hz, tone_span_hz))
    am_span_hz = max(protect_hz, 3.0 * float(scenario.am.modulation_frequency_hz))
    protected_spans_hz.append((am_carrier_out_hz, am_span_hz))
    fsk_center_out_hz = 0.5 * (fsk_mark_out_hz + fsk_space_out_hz)
    fsk_dev_hz = 0.5 * abs(fsk_mark_out_hz - fsk_space_out_hz)
    fsk_span_hz = max(
        protect_hz,
        float(fsk_dev_hz + 6.0 * float(scenario.fsk.symbol_rate_hz)),
    )
    protected_spans_hz.append((fsk_center_out_hz, fsk_span_hz))

    for blocker in scenario.blockers:
        blocker_shifted = float(blocker.frequency_hz - scenario.center_frequency_offset_hz)
        if abs(blocker_shifted) <= half_bw:
            protected_spans_hz.append((blocker_shifted, protect_hz))
        alias_hz = fold_frequency_hz(blocker_shifted, sample_rate_hz)
        if abs(alias_hz) > half_bw:
            blocker_dbc.append(float("-inf"))
            continue
        if abs(blocker_shifted) <= half_bw:
            blocker_dbc.append(float("-inf"))
            continue
        if any(abs(alias_hz - pf) <= protect_hz for pf in protected):
            blocker_dbc.append(float("-inf"))
            continue

        blocker_db = _peak_near(freqs, db, alias_hz)
        blocker_dbc.append(float(blocker_db - tone_db))

    worst_blocker = max(blocker_dbc) if blocker_dbc else float("-inf")
    unwanted_guard_hz = max(
        1200.0,
        24.0 * bin_hz,
        2.0 * float(scenario.fsk.symbol_rate_hz),
        3.0 * float(scenario.am.modulation_frequency_hz),
    )
    worst_unwanted = _worst_unwanted_dbc(
        freqs,
        db,
        tone_db=tone_db,
        half_bw_hz=half_bw,
        protected_hz=protected,
        protected_spans_hz=protected_spans_hz,
        guard_hz=unwanted_guard_hz,
    )

    return OutputMetrics(
        tone_frequency_hz=float(tone_meas_hz),
        tone_frequency_error_hz=tone_err_hz,
        am_correlation=float(am_corr),
        fsk_ber=float(fsk_ber),
        worst_blocker_db_relative_to_tone=float(worst_blocker),
        worst_unwanted_db_relative_to_tone=float(worst_unwanted),
    )


def check_metrics(
    metrics: OutputMetrics,
    thresholds: Thresholds,
) -> Tuple[bool, Tuple[str, ...]]:
    failures: List[str] = []

    if metrics.tone_frequency_error_hz > thresholds.tone_freq_tolerance_hz:
        failures.append(
            f"tone frequency error {metrics.tone_frequency_error_hz:.3f} Hz exceeds {thresholds.tone_freq_tolerance_hz:.3f} Hz"
        )
    if metrics.am_correlation < thresholds.am_min_correlation:
        failures.append(
            f"AM correlation {metrics.am_correlation:.4f} below {thresholds.am_min_correlation:.4f}"
        )
    if metrics.fsk_ber > thresholds.fsk_max_ber:
        failures.append(
            f"FSK BER {metrics.fsk_ber:.4f} exceeds {thresholds.fsk_max_ber:.4f}"
        )
    if metrics.worst_blocker_db_relative_to_tone > thresholds.max_blocker_db_relative_to_tone:
        failures.append(
            "worst blocker alias level "
            f"{metrics.worst_blocker_db_relative_to_tone:.2f} dBc exceeds "
            f"{thresholds.max_blocker_db_relative_to_tone:.2f} dBc"
        )
    if metrics.worst_unwanted_db_relative_to_tone > thresholds.max_unwanted_db_relative_to_tone:
        failures.append(
            "worst unwanted in-band spur "
            f"{metrics.worst_unwanted_db_relative_to_tone:.2f} dBc exceeds "
            f"{thresholds.max_unwanted_db_relative_to_tone:.2f} dBc"
        )

    return len(failures) == 0, tuple(failures)


def _stft_waterfall(iq: np.ndarray, fs: float, fft_size: int, hop: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(iq, dtype=np.complex64).reshape(-1)
    if x.size < fft_size:
        return np.empty((0, 0), dtype=np.float32), np.array([]), np.array([])

    n_frames = 1 + (x.size - fft_size) // hop
    idx = np.expand_dims(np.arange(fft_size), 0) + np.expand_dims(np.arange(n_frames) * hop, 1)
    frames = x[idx]

    win = np.hanning(fft_size).astype(np.float32)
    frames = frames * win

    spec = np.fft.fftshift(np.fft.fft(frames, n=fft_size, axis=1), axes=1)
    power = np.abs(spec) ** 2
    db = 10.0 * np.log10(power + 1e-12)

    freqs = np.linspace(-fs / 2.0, fs / 2.0, fft_size, endpoint=False)
    times = (np.arange(n_frames) * hop) / fs
    return db.astype(np.float32), freqs.astype(np.float64), times.astype(np.float64)


def write_waterfall_svg(
    *,
    iq: np.ndarray,
    sample_rate_hz: float,
    output_svg: Path,
    title: str,
    output_bandwidth_hz: Optional[float] = None,
    fft_size: int = 1024,
    overlap: float = 0.75,
    db_span: float = 90.0,
) -> bool:
    hop = max(1, int(round(float(fft_size) * (1.0 - float(overlap)))))
    db, freqs, times = _stft_waterfall(iq, float(sample_rate_hz), int(fft_size), hop)
    if db.size == 0:
        return False

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    hi = float(np.max(db))
    lo = float(hi - max(20.0, float(db_span)))
    img = np.clip(db, lo, hi)

    output_svg = output_svg.expanduser()
    output_svg.parent.mkdir(parents=True, exist_ok=True)

    extent = [freqs[0], freqs[-1], times[0] if times.size else 0.0, times[-1] if times.size else 0.0]
    plt.figure(figsize=(10, 5.5))
    plt.imshow(
        img,
        origin="upper",
        aspect="auto",
        extent=extent,
        interpolation="nearest",
        cmap="viridis",
        vmin=lo,
        vmax=hi,
    )
    if output_bandwidth_hz is not None and output_bandwidth_hz > 0:
        half_bw = min(float(output_bandwidth_hz) / 2.0, float(sample_rate_hz) / 2.0)
        if half_bw > 0:
            ax = plt.gca()
            ax.axvspan(freqs[0], -half_bw, color="white", alpha=0.08, lw=0)
            ax.axvspan(half_bw, freqs[-1], color="white", alpha=0.08, lw=0)
            ax.axvline(-half_bw, color="white", linestyle="--", linewidth=1.0, alpha=0.8)
            ax.axvline(half_bw, color="white", linestyle="--", linewidth=1.0, alpha=0.8)
    plt.colorbar(label="Power (dB)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Time (s)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_svg, format="svg")
    plt.close()
    return True


__all__ = [
    "SAMPLES_PER_PACKET",
    "ToneSpec",
    "AMSpec",
    "FSKSpec",
    "BlockerSpec",
    "Thresholds",
    "DDCScenario",
    "OutputMetrics",
    "ValidationResult",
    "load_scenario",
    "synthesize_composite_iq",
    "write_v49_from_iq",
    "read_v49_iq",
    "fold_frequency_hz",
    "evaluate_output_metrics",
    "check_metrics",
    "write_waterfall_svg",
]
