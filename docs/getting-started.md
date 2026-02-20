# Getting Started

Follow these steps to decode or synthesize VITA 49 packets with vita49io.

## 1. Install the library

```bash
pip install git+https://github.com/HBrauer/vita49io.git
```

If you plan to build the docs locally, install the extras listed in `requirements-docs.txt`.

## 2. Parse a captured data packet

```python
from vita49io.protocol.data_packet import DataPacket

# Hex string copied from a capture utility
hex_payload = (
    "40000004"  # header
    "00001001"  # stream_id
    "00000000"  # payload words (zero padding)
)
packet_bytes = bytes.fromhex(hex_payload)
packet = DataPacket.from_bytes(packet_bytes)
print(packet.stream_id)  # 0x1001
```

## 3. Create Context Packet

```python
from vita49io.protocol.context_packet import ContextPacket
from vita49io.protocol.cif0 import CIF0Fields
from vita49io.protocol.enums import PacketType, TSI, TSF

cif0 = CIF0Fields(sample_rate_hz=1e6, bandwidth_hz=20e6)
context = ContextPacket(
    packet_type=PacketType.CONTEXT_PACKET,
    stream_id=0x2002,
    tsi=TSI.UTC,
    tsf=TSF.FRACTIONAL,
    cif0=cif0,
)
print(len(context.to_bytes()))
```

## 4. Read Packets From File

The examples follow a streaming parser that reads a header, uses `packet_size` to read the full packet, and decodes context/data accordingly. It maintains the last seen `PayloadFormat` from context to decode subsequent data packets into IQ arrays.

```python
from vita49io.protocol.core import Header
from vita49io.protocol.enums import PacketType
from vita49io import DataPacket, ContextPacket
from vita49io.protocol.cif0 import PayloadFormat
from vita49io.io.payload_codec import payload_as_numpy

last_pf: PayloadFormat | None = None
with open(path, "rb") as f:
    while True:
        h = f.read(4)
        if not h:
            break
        w0 = int.from_bytes(h, "big")
        header = Header.parse(w0)
        rest = f.read((header.packet_size - 1) * 4)
        pkt_bytes = h + rest
        if header.packet_type is PacketType.CONTEXT_PACKET:
            ctx = ContextPacket.from_bytes(pkt_bytes)
            if ctx.cif0 and ctx.cif0.payload_format:
                last_pf = ctx.cif0.payload_format
            handle(ctx)
        else:
            data = DataPacket.from_bytes(pkt_bytes)
            if last_pf is not None:
                payload = data.payload
                payload_bytes = payload.tobytes() if isinstance(payload, memoryview) else payload
                iq = payload_as_numpy(payload_bytes, last_pf)
                handle(iq)
```

See: `src/vita49io/scripts/read_v49_file.py` or run `vita-read-v49`.

## 5. Write Data Into A File

`IQStreamWriter` produces context and data packets for a configured IQ stream and advances timestamps based on the sample rate.

```python
import numpy as np
from vita49io.io import IQStreamWriter

w = IQStreamWriter(
    stream_id=0x13572468,
    sample_rate_hz=1_000_000.0,
    # Optional: customize payload format (defaults to 32‑bit float I/Q)
)

# Context first
ctx = w.build_context_packet()
out = bytearray(ctx.to_bytes())

# Emit samples in blocks; accepts complex or shape (N,2)
tone = 50_000.0
N = 10_000
t = np.arange(N, dtype=np.float32) / np.float32(w.sample_rate_hz)
sig = np.exp(1j * 2 * np.pi * tone * t).astype(np.complex64)
for i in range(0, N, 1024):
    out += w.build_data_packet_bytes(sig[i:i+1024])

open("out.v49", "wb").write(out)
```

- Time handling: timestamps start at `start_time_epoch_s` (default now, UTC) and advance by `len(iq)/sample_rate_hz` for each packet.
- Timestamps are emitted using `tsi`/`tsf` (defaults: `UTC` + `FRACTIONAL`).
- To use fixed‑point payloads, build a `PayloadFormat` and pass it to the writer via constructor fields; it propagates to context and data encoding.

## 6. Explore the API reference

The [API reference](reference/vita49io.md) is generated directly from the codebase using mkdocstrings, so it always reflects the latest implementation.

## 7. Spectrum Processor defaults

For `SpectrumStreamProcessor` and `SpectrumProcessor` in `processing_mode="continuous"`:

- If `hop_size` is omitted, it defaults to `fft_size` (no overlap).
- Set `hop_size < fft_size` when you want overlap.
- In `processing_mode="snapshot"`, `hop_size` does not control FFT cadence.
  One FFT is computed per output frame from the latest `fft_size` samples.

GNU Radio QT Frequency Sink-like snapshot settings:

```python
from vita49io.io.spectrum_processor import SpectrumStreamProcessor

processor = SpectrumStreamProcessor(
    stream=f,
    fft_size=1024,
    processing_mode="snapshot",
    output_fps=10.0,
    averaging_mode="none",
    band_mode="full",
    power_scale="raw",
    window_type="hann",  # set to match GNU Radio sink window
)
```
