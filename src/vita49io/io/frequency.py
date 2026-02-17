"""Frequency-domain helpers for streaming IQ processing."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


class StreamingFrequencyShifter:
    """Stateful complex frequency shifter with reusable oscillator templates.

    The shifter keeps phase continuity across calls and caches per-length oscillator
    templates so repeated block sizes avoid rebuilding `exp()` vectors.
    """

    def __init__(
        self,
        sample_rate_hz: float,
        frequency_offset_hz: float,
        *,
        cache_templates: bool = True,
    ) -> None:
        if sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be > 0")
        self.sample_rate_hz = float(sample_rate_hz)
        self.frequency_offset_hz = float(frequency_offset_hz)
        self.cache_templates = bool(cache_templates)

        self._step_rad = float(
            -2.0 * np.pi * self.frequency_offset_hz / self.sample_rate_hz
        )
        # Keep phase state in complex128 for long-run numerical stability.
        self._phase = np.complex128(1.0 + 0.0j)
        self._template_cache: Dict[int, Tuple[np.ndarray, np.complex128]] = {}

    def reset(self, *, phase_rad: float = 0.0) -> None:
        """Reset internal phase state."""
        self._phase = np.complex128(np.exp(1j * float(phase_rad)))

    def _build_template(self, n: int) -> Tuple[np.ndarray, np.complex128]:
        if n <= 0:
            return np.empty(0, dtype=np.complex64), np.complex128(1.0 + 0.0j)
        phase = self._step_rad * np.arange(n, dtype=np.float64)
        template = np.exp(1j * phase).astype(np.complex64)
        phase_advance = np.complex128(np.exp(1j * self._step_rad * n))
        return template, phase_advance

    def _template_for(self, n: int) -> Tuple[np.ndarray, np.complex128]:
        if not self.cache_templates:
            return self._build_template(n)
        cached = self._template_cache.get(n)
        if cached is None:
            cached = self._build_template(n)
            self._template_cache[n] = cached
        return cached

    def process(self, samples: np.ndarray) -> np.ndarray:
        """Frequency-shift a block while preserving phase continuity."""
        x = np.asarray(samples, dtype=np.complex64).reshape(-1)
        if x.size == 0 or self.frequency_offset_hz == 0.0:
            return x

        template, phase_advance = self._template_for(int(x.size))
        y = x * template
        y *= np.complex64(self._phase)

        self._phase *= phase_advance
        mag = abs(self._phase)
        if mag != 0.0:
            self._phase /= mag
        return y


__all__ = ["StreamingFrequencyShifter"]

