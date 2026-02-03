from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.fft import fft, fftfreq, fftshift
from scipy.signal import windows


def _default_output_bins(
    sample_rate_hz: float,
    bandwidth_hz: float,
    band_mode: str,
    fft_size: int,
) -> int:
    if band_mode == "full":
        return int(fft_size)
    freqs = fftshift(fftfreq(fft_size, d=1.0 / float(sample_rate_hz)))
    half_bw = float(bandwidth_hz) / 2.0
    mask = (freqs >= -half_bw) & (freqs <= half_bw)
    return int(np.count_nonzero(mask))


@dataclass
class SpectrumFrame:
    timestamp: float
    spectrum_db: np.ndarray
    freqs_hz: np.ndarray
    meta: dict


class SpectrumProcessor:
    def __init__(
        self,
        sample_rate_hz: float,
        bandwidth_hz: float,
        band_mode: str,
        fft_size: int,
        hop_size: int,
        window_type: str,
        averaging_mode: str,
        averaging_param: float | int,
        output_fps: float,
        output_bins: int | None,
        fft_kwargs: dict | None = None,
        *,
        window_param: float | None = None,
        dc_block: bool = False,
        power_scale: str = "dbfs",
    ) -> None:
        if sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be > 0")
        if bandwidth_hz <= 0 or bandwidth_hz > sample_rate_hz:
            raise ValueError("bandwidth_hz must be in (0, sample_rate_hz]")
        if fft_size <= 0:
            raise ValueError("fft_size must be > 0")
        if hop_size <= 0:
            raise ValueError("hop_size must be > 0")
        if output_fps <= 0:
            raise ValueError("output_fps must be > 0")
        if output_bins is None:
            output_bins = _default_output_bins(
                sample_rate_hz=sample_rate_hz,
                bandwidth_hz=bandwidth_hz,
                band_mode=band_mode,
                fft_size=fft_size,
            )
        if output_bins <= 0:
            raise ValueError("output_bins must be > 0")
        if band_mode not in {"full", "inband"}:
            raise ValueError("band_mode must be 'full' or 'inband'")
        if window_type not in {"hann", "rect", "blackmanharris", "kaiser"}:
            raise ValueError("window_type must be 'hann', 'rect', 'blackmanharris', or 'kaiser'")
        if window_type == "kaiser" and window_param is not None and window_param <= 0:
            raise ValueError("window_param must be > 0 for kaiser")
        if not isinstance(dc_block, bool):
            raise ValueError("dc_block must be a bool")
        if averaging_mode not in {"none", "mean", "frame_mean", "exponential", "exponential_tau", "peak_hold"}:
            raise ValueError(
                "averaging_mode must be one of: "
                "'none', 'mean', 'frame_mean', 'exponential', 'exponential_tau', 'peak_hold'"
            )
        if averaging_mode == "mean":
            if not isinstance(averaging_param, int) or averaging_param <= 0:
                raise ValueError("averaging_param must be a positive int for mean")
        if averaging_mode == "exponential":
            if not isinstance(averaging_param, (float, int)):
                raise ValueError("averaging_param must be a float in (0, 1] for exponential")
            if not (0 < float(averaging_param) <= 1.0):
                raise ValueError("averaging_param must be in (0, 1] for exponential")
        if averaging_mode == "exponential_tau":
            if not isinstance(averaging_param, (float, int)):
                raise ValueError("averaging_param must be a float > 0 (seconds) for exponential_tau")
            if not (float(averaging_param) > 0.0):
                raise ValueError("averaging_param must be > 0 (seconds) for exponential_tau")
        if power_scale not in {"raw", "dbfs"}:
            raise ValueError("power_scale must be 'raw' or 'dbfs'")
        if fft_kwargs is not None and not isinstance(fft_kwargs, dict):
            raise ValueError("fft_kwargs must be a dict or None")

        self.sample_rate_hz = float(sample_rate_hz)
        self.bandwidth_hz = float(bandwidth_hz)
        self.band_mode = band_mode
        self.fft_size = int(fft_size)
        self.hop_size = int(hop_size)
        self.window_type = window_type
        self.window_param = window_param
        self.dc_block = dc_block
        self.power_scale = power_scale
        self.averaging_mode = averaging_mode
        self.averaging_param = averaging_param
        self.output_fps = float(output_fps)
        self.output_bins = int(output_bins)
        self.fft_kwargs = dict(fft_kwargs) if fft_kwargs is not None else {}

        if window_type == "hann":
            self._window = np.hanning(self.fft_size).astype(np.float32)
        elif window_type == "rect":
            self._window = np.ones(self.fft_size, dtype=np.float32)
        elif window_type == "blackmanharris":
            self._window = windows.blackmanharris(self.fft_size).astype(np.float32)
        else:
            beta = 8.0 if window_param is None else float(window_param)
            self._window = np.kaiser(self.fft_size, beta).astype(np.float32)

        # Coherent gain (sum of window weights). This normalizes a bin-centered tone's
        # magnitude so results don't change with fft_size/window choice.
        self._coherent_gain = float(np.sum(self._window, dtype=np.float64))
        if self._coherent_gain <= 0.0:
            raise ValueError("Internal error: window coherent gain must be > 0")

        # Keep float64 here so the inband mask and default output bin count are consistent
        # across platforms/precisions (avoids accidental "default interpolation" off-by-one).
        self._freqs = fftshift(fftfreq(self.fft_size, d=1.0 / self.sample_rate_hz))
        if self.band_mode == "full":
            self._band_mask: np.ndarray | None = None
        else:
            half_bw = self.bandwidth_hz / 2.0
            self._band_mask = (self._freqs >= -half_bw) & (self._freqs <= half_bw)

        self._reset_state()

    def reset(self) -> None:
        self._reset_state()

    def _reset_state(self) -> None:
        self._buffer = np.empty(0, dtype=np.complex64)
        self._buffer_start = 0
        self._total_samples_processed = 0
        self._next_frame_time = 1.0 / self.output_fps
        self._last_power: np.ndarray | None = None
        self._mean_sum: np.ndarray | None = None
        self._mean_count = 0
        self._ema_state: np.ndarray | None = None
        self._ema_count = 0
        self._peak_hold: np.ndarray | None = None
        self._peak_hold_count = 0
        self._frame_sum: np.ndarray | None = None
        self._frame_count = 0

    def push(self, iq: np.ndarray) -> list[SpectrumFrame]:
        iq_array = np.asarray(iq)
        if iq_array.ndim != 1:
            raise ValueError("iq must be a 1-D complex array")
        if not np.iscomplexobj(iq_array):
            raise ValueError("iq must be complex-valued")
        if iq_array.dtype != np.complex64:
            iq_array = iq_array.astype(np.complex64, copy=False)

        if iq_array.size:
            self._buffer = np.concatenate((self._buffer, iq_array))

        frames: list[SpectrumFrame] = []
        while self._buffer_start + self.fft_size <= self._buffer.size:
            segment = self._buffer[self._buffer_start : self._buffer_start + self.fft_size]
            if self.dc_block:
                mean = segment.mean(dtype=np.complex64)
                segment = segment - mean
            windowed = segment * self._window
            spectrum = fft(windowed, n=self.fft_size, **self.fft_kwargs)
            power = (np.abs(spectrum) ** 2).astype(np.float32)

            self._update_averaging(power)

            self._buffer_start += self.hop_size
            self._total_samples_processed += self.hop_size

            if self._buffer_start > self.fft_size * 4:
                self._buffer = self._buffer[self._buffer_start :]
                self._buffer_start = 0

            current_time = self._total_samples_processed / self.sample_rate_hz
            while current_time >= self._next_frame_time:
                frame = self._emit_frame(current_time)
                if frame is not None:
                    frames.append(frame)
                self._next_frame_time += 1.0 / self.output_fps

        return frames

    def _update_averaging(self, power: np.ndarray) -> None:
        if self.averaging_mode == "none":
            self._last_power = power
            return

        if self.averaging_mode == "mean":
            if self._mean_sum is None:
                self._mean_sum = power.astype(np.float32)
                self._mean_count = 1
                return
            if self._mean_count < int(self.averaging_param):
                self._mean_sum += power
                self._mean_count += 1
            return
        if self.averaging_mode == "peak_hold":
            if self._peak_hold is None:
                self._peak_hold = power.astype(np.float32)
            else:
                self._peak_hold = np.maximum(self._peak_hold, power)
            self._peak_hold_count += 1
            return

        if self.averaging_mode == "frame_mean":
            if self._frame_sum is None:
                self._frame_sum = power.astype(np.float32)
                self._frame_count = 1
            else:
                self._frame_sum += power
                self._frame_count += 1
            return

        if self.averaging_mode == "exponential":
            alpha = float(self.averaging_param)
        else:
            tau = float(self.averaging_param)
            dt_s = float(self.hop_size) / float(self.sample_rate_hz)
            alpha = 1.0 - float(np.exp(-dt_s / tau))

        if self._ema_state is None:
            self._ema_state = power.astype(np.float32)
            self._ema_count = 1
        else:
            self._ema_state = (1.0 - alpha) * self._ema_state + alpha * power
            self._ema_count += 1

    def _select_power(self) -> tuple[np.ndarray | None, int]:
        if self.averaging_mode == "none":
            if self._last_power is None:
                return None, 0
            return self._last_power, 1

        if self.averaging_mode == "mean":
            if self._mean_sum is None or self._mean_count == 0:
                return None, 0
            mean_power = self._mean_sum / float(self._mean_count)
            count = self._mean_count
            self._mean_sum = None
            self._mean_count = 0
            return mean_power, count

        if self.averaging_mode == "peak_hold":
            if self._peak_hold is None:
                return None, 0
            return self._peak_hold, max(self._peak_hold_count, 1)

        if self.averaging_mode == "frame_mean":
            if self._frame_sum is None or self._frame_count == 0:
                return None, 0
            mean_power = self._frame_sum / float(self._frame_count)
            count = self._frame_count
            self._frame_sum = None
            self._frame_count = 0
            return mean_power, count

        if self._ema_state is None:
            return None, 0
        return self._ema_state, self._ema_count

    def _emit_frame(self, timestamp: float) -> SpectrumFrame | None:
        power, num_averaged = self._select_power()
        if power is None:
            return None

        power_shifted = fftshift(power)

        if self._band_mask is None:
            freqs_band = self._freqs
            power_band = power_shifted
        else:
            freqs_band = self._freqs[self._band_mask]
            power_band = power_shifted[self._band_mask]

        if freqs_band.size == 0:
            return None

        if self.output_bins != freqs_band.size:
            freq_out = np.linspace(freqs_band[0], freqs_band[-1], self.output_bins)
            power_out = np.interp(freq_out, freqs_band, power_band).astype(np.float32)
            freqs_out = freq_out.astype(np.float32)
        else:
            power_out = power_band.astype(np.float32)
            freqs_out = freqs_band.astype(np.float32)

        if self.power_scale == "dbfs":
            power_out = power_out / float(self._coherent_gain**2)

        spectrum_db = (10.0 * np.log10(power_out + 1e-20)).astype(np.float32)

        meta = {
            "fft_size": self.fft_size,
            "hop_size": self.hop_size,
            "rbw_hz": self.sample_rate_hz / self.fft_size,
            "band_mode": self.band_mode,
            "bandwidth_hz": self.bandwidth_hz,
            "dc_block": self.dc_block,
            "power_scale": self.power_scale,
            "coherent_gain": self._coherent_gain,
            "num_ffts_averaged": num_averaged,
            "fft_kwargs": dict(self.fft_kwargs),
        }

        return SpectrumFrame(
            timestamp=timestamp,
            spectrum_db=spectrum_db,
            freqs_hz=freqs_out,
            meta=meta,
        )


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    sample_rate = 1.0e6
    fft_size = 1024
    hop_size = 256

    processor = SpectrumProcessor(
        sample_rate_hz=sample_rate,
        bandwidth_hz=400e3,
        band_mode="inband",
        fft_size=fft_size,
        hop_size=hop_size,
        window_type="hann",
        window_param=None,
        dc_block=False,
        averaging_mode="mean",
        averaging_param=4,
        output_fps=10.0,
        output_bins=256,
        fft_kwargs={"norm": None},
    )

    t = np.arange(0, 5000) / sample_rate
    tone = np.exp(2j * np.pi * 50e3 * t)
    noise = (rng.standard_normal(t.size) + 1j * rng.standard_normal(t.size)) * 0.1
    iq = (tone + noise).astype(np.complex64)

    frames = processor.push(iq)
    for frame in frames:
        print(
            f"t={frame.timestamp:.6f}s spectrum={frame.spectrum_db.shape} freqs={frame.freqs_hz.shape}"
        )
