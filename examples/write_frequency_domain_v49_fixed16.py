"""
Example: Build VITA 49 IQ packets with 16-bit signed fixed-point samples.

Generates synthetic IQ with drifting/pulsing tones plus noise, then encodes it
as complex cartesian 16-bit fixed-point (I/Q interleaved) in the payload.

Usage:
  python examples/write_frequency_domain_v49_fixed16.py out_iq_fixed16.v49
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import numpy as np

from vita49io.protocol.cif0 import PayloadFormat, PackingMethod, SampleType, DataItemFormat


def _ensure_src_on_path() -> None:
    # Allow running the example from the repo root without installation
    here = os.path.dirname(__file__)
    src = os.path.normpath(os.path.join(here, "..", "src"))
    if os.path.isdir(src) and src not in sys.path:
        sys.path.insert(0, src)


def build_payload_format() -> "PayloadFormat":
    return PayloadFormat(
        packing_method=PackingMethod.PROCESSING_EFFICIENT,
        sample_type=SampleType.COMPLEX_CARTESIAN,
        data_item_format=DataItemFormat.SIGNED_FIXED_POINT,
        sample_component_repeat=False,
        event_tag_size_bits=0,
        channel_tag_size_bits=0,
        data_item_fraction_size_bits=0,
        item_packing_field_size_bits=16,
        data_item_size_bits=16,
        repeat_count=1,
        vector_size=1,
    )


def synthesize_iq(
    num_samples: int,
    sample_rate_hz: float,
    sim_time: float,
    *,
    phase1: float,
    phase2: float,
) -> tuple[np.ndarray, float, float]:
    """Generate synthetic IQ with drifting/pulsing tones plus noise (continuous phase)."""
    # Target levels (dBFS)
    noise_db = -100.0
    signal_db = -60.0
    noise_amp = 10.0 ** (noise_db / 20.0)
    signal_amp = 10.0 ** (signal_db / 20.0)

    # Tone 1: drifting frequency (normalized to [-1, 1] like the JS bins)
    freq1 = 0.2 + np.sin(sim_time * 0.1) * 0.25
    freq1 = max(min(freq1, 0.45), -0.45)
    f1_hz = freq1 * (sample_rate_hz / 2.0)
    w1 = 2.0 * np.pi * f1_hz / sample_rate_hz

    # Tone 2: pulsing amplitude, move off Nyquist
    freq2 = -0.45
    f2_hz = freq2 * (sample_rate_hz / 2.0)
    w2 = 2.0 * np.pi * f2_hz / sample_rate_hz

    idx = np.arange(num_samples, dtype=np.float32)
    ph1 = phase1 + w1 * idx
    ph2 = phase2 + w2 * idx

    tone1 = signal_amp * np.exp(1j * ph1)
    amp2 = (np.sin(sim_time * 5.0) * 0.5 + 0.5) * signal_amp
    tone2 = amp2 * np.exp(1j * ph2)

    noise_sigma = noise_amp / np.sqrt(2.0)
    noise = (np.random.standard_normal(num_samples) + 1j * np.random.standard_normal(num_samples)) * noise_sigma

    iq = (tone1 + tone2 + noise).astype(np.complex64)

    # Advance phases for continuity
    phase1 = float((phase1 + w1 * num_samples) % (2.0 * np.pi))
    phase2 = float((phase2 + w2 * num_samples) % (2.0 * np.pi))
    return iq, phase1, phase2


def main(argv: list[str]) -> int:
    _ensure_src_on_path()

    from vita49io import ContextPacket, DataPacket, PacketType, TSI, TSF, CIF0Fields
    from vita49io.protocol.core import Header
    from vita49io.io.payload_codec import payload_from_numpy

    marker_only = "--marker-only" in argv
    argv = [arg for arg in argv if arg != "--marker-only"]
    out_path = Path(argv[1]) if len(argv) > 1 else Path("iq_fixed16.v49")
    stream_id = 0x2468ACE1
    frame_period_s = 0.020
    frame_size = 2048
    duration_s = 30.0
    sample_rate_hz = frame_size / frame_period_s
    num_frames = int(round(duration_s / frame_period_s))
    sim_time = 0.0

    payload_format = build_payload_format()

    # Context packet (no S-bit here; bit 24 on context is TSM, not spectrum)
    ctx_header = Header(
        packet_type=PacketType.CONTEXT_PACKET,
        class_id_present=False,
        indicators_26=False,
        indicators_25=False,
        indicators_24=False,
        tsi=TSI.UTC,
        tsf=TSF.FRACTIONAL,
        packet_count=0,
        packet_size=0,
    )
    cif0 = CIF0Fields(
        sample_rate_hz=sample_rate_hz,
        bandwidth_hz=sample_rate_hz,
        payload_format=payload_format,
    )
    ctx = ContextPacket(
        header=ctx_header,
        stream_id=stream_id,
        integer_seconds=1_700_000_000,
        fractional_seconds=0,
        cif0=cif0,
    )

    out = bytearray(ctx.to_bytes())
    packet_count = 1
    start_epoch_s = 1_700_000_000.0

    if marker_only:
        data_header = Header(
            packet_type=PacketType.IF_DATA_WITH_STREAM_ID,
            class_id_present=False,
            indicators_26=False,
            indicators_25=False,
            indicators_24=False,
            tsi=TSI.UTC,
            tsf=TSF.FRACTIONAL,
            packet_count=packet_count & 0xF,
            packet_size=0,
        )
        packet_count += 1

        words = np.array([0x0123, 0x4567, 0x89AB, 0xCDEF], dtype=np.uint16)
        signed = words.view(np.int16)
        scaled = (signed.astype(np.float32) / float(1 << 15)).reshape(-1, 2)
        marker_iq = (scaled[:, 0] + 1j * scaled[:, 1]).astype(np.complex64)
        payload = payload_from_numpy(marker_iq, payload_format)
        data_pkt = DataPacket(
            header=data_header,
            stream_id=stream_id,
            integer_seconds=int(start_epoch_s),
            fractional_seconds=0,
            payload=payload,
        )
        out += data_pkt.to_bytes()
        out_path.write_bytes(bytes(out))
        print(f"Wrote marker-only IQ file to {out_path} ({len(out)} bytes)")
        print("Marker samples (I/Q from hex words):")
        for val in marker_iq:
            print(f"  {val.real:+.6f} + {val.imag:+.6f}j")
        return 0

    first_iq: np.ndarray | None = None
    phase1 = 0.0
    phase2 = 0.0
    for frame_idx in range(num_frames):
        sim_time += frame_period_s
        frame_time = start_epoch_s + frame_idx * frame_period_s
        integer_seconds = int(frame_time)
        fractional_seconds = int((frame_time - integer_seconds) * (1 << 64))

        data_header = Header(
            packet_type=PacketType.IF_DATA_WITH_STREAM_ID,
            class_id_present=False,
            indicators_26=False,
            indicators_25=False,
            indicators_24=False,
            tsi=TSI.UTC,
            tsf=TSF.FRACTIONAL,
            packet_count=packet_count & 0xF,
            packet_size=0,
        )
        packet_count += 1

        iq, phase1, phase2 = synthesize_iq(
            frame_size,
            sample_rate_hz,
            sim_time,
            phase1=phase1,
            phase2=phase2,
        )
        if first_iq is None:
            first_iq = iq.copy()
        payload = payload_from_numpy(iq, payload_format)
        data_pkt = DataPacket(
            header=data_header,
            stream_id=stream_id,
            integer_seconds=integer_seconds,
            fractional_seconds=fractional_seconds,
            payload=payload,
        )
        out += data_pkt.to_bytes()

    out_path.write_bytes(bytes(out))

    print(f"Wrote fixed-point IQ to {out_path} ({len(out)} bytes)")
    print(f"Frames: {num_frames}, frame_size: {frame_size}, sample_rate_hz: {sample_rate_hz:.1f}")
    if first_iq is not None:
        print("First 4 IQ samples (complex):")
        for val in first_iq[:4]:
            print(f"  {val.real:+.4f} + {val.imag:+.4f}j")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
