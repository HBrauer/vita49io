# Common Tasks

## Decode a stream of packets from disk

```python
from pathlib import Path
from vita49io.protocol.data_packet import DataPacket

capture = Path('capture.bin').read_bytes()
packets = []
for offset in range(0, len(capture), 32):
    chunk = capture[offset:offset + 32]
    if len(chunk) < 32:
        break
    try:
        packets.append(DataPacket.from_bytes(chunk))
    except ValueError:
        continue
print(f"decoded {len(packets)} packets")
```

## Build alternating data and context packets

```python
import numpy as np
from vita49io.io.iq_writer import IQStreamWriter

writer = IQStreamWriter(stream_id=0x5005, sample_rate_hz=2e6)
context_packet = writer.build_context_packet()
data_packets = []
for _ in range(4):
    iq = (np.arange(4, dtype=np.float32) * (1 + 1j)).astype(np.complex64)
    data_packets.append(writer.build_data_packet(iq))
```

## Regenerate CIF0 payload words from friendly values

```python
from vita49io.protocol.cif0 import CIF0Fields

fields = CIF0Fields(sample_rate_hz=1e6, bandwidth_hz=10e6)
raw_bytes = fields.pack()
print(len(raw_bytes))
```
