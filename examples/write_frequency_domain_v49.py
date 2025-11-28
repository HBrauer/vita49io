"""
Example: Build VITA 49 frequency-domain packets (context + spectral data) and write to a file.

The data header sets the spectrum (S) bit (header bit 24) to indicate Signal Spectral Data.
Context uses CIF0 to describe the payload format and sample rate.

Usage:
  python examples/write_frequency_domain_v49.py out_spectrum.v49
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

from vita49io import ContextPacket, DataPacket, PacketType, TSI, TSF, CIF0Fields
from vita49io.protocol.cif0 import PayloadFormat, PackingMethod, SampleType, DataItemFormat
from vita49io.protocol.core import Header


def build_payload_format() -> PayloadFormat:
    return PayloadFormat(
        packing_method=PackingMethod.PROCESSING_EFFICIENT,
        sample_type=SampleType.COMPLEX_CARTESIAN,
        data_item_format_code=int(DataItemFormat.IEEE754_SINGLE),
        data_item_format=DataItemFormat.IEEE754_SINGLE,
        sample_component_repeat=False,
        event_tag_size_bits=0,
        channel_tag_size_bits=0,
        data_item_fraction_size_bits=0,
        item_packing_field_size_bits=32,
        data_item_size_bits=32,
        repeat_count=1,
        vector_size=0,
    )


def synthesize_frequency_bins(sample_rate_hz: float, n_fft: int) -> np.ndarray:
    """Create a simple spectrum with two tones and noise."""
    t = np.arange(n_fft, dtype=np.float32) / np.float32(sample_rate_hz)
    tone_a = np.exp(1j * 2 * np.pi * 75_000.0 * t)
    tone_b = 0.6 * np.exp(1j * 2 * np.pi * -180_000.0 * t)
    noise = (np.random.randn(n_fft) + 1j * np.random.randn(n_fft)).astype(np.complex64) * 0.02
    time_signal = tone_a + tone_b + noise

    spectrum = np.fft.fftshift(np.fft.fft(time_signal))
    spectrum /= spectrum.size  # normalize amplitude to stay within [-1, 1]
    return spectrum.astype(np.complex64)


def main(argv: list[str]) -> int:
    out_path = Path(argv[1]) if len(argv) > 1 else Path("frequency_domain.v49")
    stream_id = 0x2468ACE0
    sample_rate_hz = 1_000_000.0
    n_fft = 2048

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
        bandwidth_hz=sample_rate_hz / 2,
        payload_format=payload_format,
        data_packet_payload_format=payload_format.pack_words(),
    )
    ctx = ContextPacket(
        header=ctx_header,
        stream_id=stream_id,
        integer_seconds=1_700_000_000,
        fractional_seconds=0,
        cif0=cif0,
    )
    out = bytearray(ctx.to_bytes())

    # Build one spectral data packet with the S-bit asserted
    data_header = Header(
        packet_type=PacketType.IF_DATA_WITH_STREAM_ID,
        class_id_present=False,
        indicators_26=False,
        indicators_25=False,
        indicators_24=True,  # S-bit: Signal Spectral Data
        tsi=TSI.UTC,
        tsf=TSF.FRACTIONAL,
        packet_count=1,
        packet_size=0,
    )

    spectrum = synthesize_frequency_bins(sample_rate_hz, n_fft)
    data_pkt = DataPacket(
        header=data_header,
        stream_id=stream_id,
        integer_seconds=1_700_000_000,
        fractional_seconds=0,
        iq=spectrum,  # treat frequency bins as complex samples; S-bit marks them spectral
    )
    out += data_pkt.to_bytes(payload_format=payload_format)

    out_path.write_bytes(bytes(out))
    print(f"Wrote frequency-domain context + data to {out_path} ({len(out)} bytes)")
    print("First 4 spectral bins (complex):")
    for val in spectrum[:4]:
        print(f"  {val.real:+.4f} + {val.imag:+.4f}j")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
