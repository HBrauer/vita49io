This project is not affiliated with or endorsed by the VITA Standards Organization.

VITA 49 (VRT) Python Library
============================

Lightweight read/write utilities for VITA 49.x (VRT) packets with a focus on:
- IF/Extension Data packets with optional Stream ID
- Context packets with CIF0
- IQ payload encode/decode using NumPy

Includes an IQ stream helper (`IQStreamWriter`) and end‑to‑end examples for reading and writing `.v49` files.

Install
-------

```bash
pip install . # from source
# or
pip install git+https://github.com/HBrauer/vita49io.git # directly from GitHub
```

Or use directly from source (examples add `src/` to `PYTHONPATH`).

Direct Packet Usage
-------------------

You can directly build and parse context/data packets without the stream helper.

Example: build a Context packet (CIF0 with sample rate and payload format)
```python
from vita49io import ContextPacket, PacketType, TSI, TSF, CIF0Fields
from vita49io.protocol.cif0 import PayloadFormat, PackingMethod, SampleType, DataItemFormat

stream_id = 0x12345678

# Define how subsequent data packet payloads are encoded (32‑bit float I/Q)
pf = PayloadFormat(
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

# Define the Context Indicator Fields 
cif0 = CIF0Fields(
    sample_rate_hz=1_000_000.0,
    payload_format=pf,  # library writes words for CIF0 bit 15
)

# Create the Context Packet
ctx = ContextPacket(
    packet_type=PacketType.CONTEXT_PACKET,
    stream_id=stream_id,
    tsi=TSI.UTC,
    tsf=TSF.FRACTIONAL,
    integer_seconds=1_700_000_000,
    fractional_seconds=0,
    cif0=cif0,
)
ctx_bytes = ctx.to_bytes()
ctx_same = ContextPacket.from_bytes(ctx_bytes)
```

Example: build a Data packet using raw payload bytes
```python
import numpy as np
from vita49io import DataPacket, PacketType, TSI, TSF

stream_id = 0x12345678

# Two complex samples: (I0,Q0)=(1.0,0.0), (I1,Q1)=(0.0,1.0)
# Encode as big‑endian IEEE754 float32 words: I0,Q0,I1,Q1
payload = np.array([1.0, 0.0, 0.0, 1.0], dtype=">f4").tobytes()

pkt = DataPacket(
    packet_type=PacketType.IF_DATA_WITH_STREAM_ID,
    stream_id=stream_id,
    tsi=TSI.UTC,
    tsf=TSF.FRACTIONAL,
    integer_seconds=1_700_000_000,
    fractional_seconds=0,
    payload=payload,  # raw payload goes here
)
raw = pkt.to_bytes()

# If you know the payload format (e.g., from the last Context packet),
# pass it to decode IQ back to a complex64 NumPy array.
from vita49io.protocol.cif0 import PayloadFormat, PackingMethod, SampleType, DataItemFormat
pf = PayloadFormat(
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
same = DataPacket.from_bytes(raw, payload_format=pf)
iq = same.iq  # complex64 array of shape (2,)
```

Notes
- Data payload bytes must be 32‑bit aligned; the library pads to a word boundary as needed.
- For numeric payloads, use big‑endian dtypes (e.g., `">f4"`, `">i2"`) to match VRT network byte order.
- When you already have encoded bytes, set `payload` and omit `payload_format`.
- When you want the library to encode/decode IQ, provide `payload_format` and set/use the `iq` field.


Read Packets From File
-------------------------

The examples follow a streaming parser that reads a header, uses `packet_size` to read the full packet, and decodes context/data accordingly. It maintains the last seen `PayloadFormat` from context to decode subsequent data packets into IQ arrays.

```python
from vita49io.protocol.core import Header
from vita49io.protocol.enums import PacketType
from vita49io import DataPacket, ContextPacket
from vita49io.protocol.cif0 import PayloadFormat

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
            data = DataPacket.from_bytes(pkt_bytes, payload_format=last_pf)
            # data.iq is a complex64 NumPy array when last_pf is compatible
            handle(data)
```

See: `examples/read_v49_file.py`.

Write Data Into A File
---------------------------

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

# Emit IQ in blocks; accepts complex or shape (N,2)
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
- To scale voltage-domain IQ automatically, set `reference_level_dbm` and `normalize_iq_to_reference_level=True` on `IQStreamWriter`. The writer converts the reference level (dBmFS) into a 50-ohm peak voltage before normalizing samples. You can call `DataPacket.to_bytes(..., reference_level_dbm=...)` directly when working outside the helper.

See: `examples/write_iq_stream_v49.py`.


Examples
--------

- `examples/read_v49_file.py`: parse and print packets from a file
- `examples/write_iq_stream_v49.py`: synthesize a tone and write IQ packets
- `examples/waterfall_v49_file.py`: visualize spectrogram

License
-------

MIT License. See `LICENSE`.
