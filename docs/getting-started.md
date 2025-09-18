# Getting Started

Follow these steps to decode or synthesize VITA 49 packets with vita49io.

## 1. Install the library

```bash
pip install vita49io
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

## 3. Emit a context packet with CIF0 metadata

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

## 4. Produce synchronized IQ packets

```python
import numpy as np
from vita49io.io.iq_writer import IQStreamWriter

writer = IQStreamWriter(stream_id=0x3003, sample_rate_hz=1e6)
iq_block = np.zeros(8, dtype=np.complex64)
packet_bytes = writer.build_data_packet_bytes(iq_block)
print(writer.current_time())
```

## 5. Explore the API reference

The [API reference](reference/vita49io.md) is generated directly from the codebase using mkdocstrings, so it always reflects the latest implementation.
