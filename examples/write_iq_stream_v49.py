"""
Example: Build a VITA 49 IQ stream (context + data packets) and write to a file.

This uses IQStreamWriter with defaults:
- TSI.UTC + TSF.FRACTIONAL
- PayloadFormat: 32-bit IEEE754 float, complex I/Q, processing-efficient

Usage:
  python examples/write_iq_stream_v49.py out.v49
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

from vita49io.io import IQStreamWriter


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Usage: python examples/write_iq_stream_v49.py <out.v49>")
        return 2

    out_path = Path(argv[1])

    # Stream configuration
    fs = 1_000_000.0  # 1 MS/s
    stream_id = 0x13572468

    w = IQStreamWriter(
        stream_id=stream_id,
        sample_rate_hz=fs,
        # reference_level_dbm=-3.0,  # Uncomment to encode voltage-domain samples
        # normalize_iq_to_reference_level=True,
        # Defaults already set to 32-bit float complex I/Q
    )

    # Build a context packet first
    ctx = w.build_context_packet()
    data = bytearray(ctx.to_bytes())

    # Generate a test tone and emit in blocks
    tone_hz = 50_000.0
    duration_s = 0.01  # 10 ms
    N = int(fs * duration_s)
    t = np.arange(N, dtype=np.float32) / np.float32(fs)
    sig = np.exp(1j * 2 * np.pi * tone_hz * t).astype(np.complex64)

    # Chunk into frames (e.g., 1024 samples per packet)
    block = 1024
    for i in range(0, N, block):
        iq = sig[i : i + block]
        if iq.size == 0:
            break
        pkt_bytes = w.build_data_packet_bytes(iq)
        data += pkt_bytes

    out_path.write_bytes(bytes(data))
    print(f"Wrote {len(data)} bytes to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
