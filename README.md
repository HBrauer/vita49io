# vita49io
Small Python Packet for reading and writing Vita 49 streams or files

VITA 49 (VRT) Python Library
============================

This package provides minimal read/write support for VITA 49.2 (VRT) packets:

- Parse VRT headers, presence flags, stream ID, class ID, timestamps, payload, trailer
- Serialize the same fields back to bytes

Status: early, focusing on core fields commonly used by IF Data and Context packets.

Usage
-----

```python
from vita49 import Packet, PacketType, TSI, TSF

# Build a minimal IF Data packet with stream ID and timestamps
p = Packet(
    packet_type=PacketType.IF_DATA,
    stream_id=0x12345678,
    tsi=TSI.UTC,
    tsf=TSF.FRACTIONAL,
    integer_seconds=1700000000,
    fractional_seconds=0x01020304,
    payload=b"\x01\x02\x03\x04"
)

data = p.pack()
same = Packet.parse(data)
assert same.stream_id == 0x12345678
assert same.payload == b"\x01\x02\x03\x04"
```

Notes
-----

- Packet size is computed automatically from present fields.
- Class ID is represented as (oui, information_class, packet_class).
- Trailer is parsed/serialized as a raw 32-bit word for now.
- This library targets VITA 49.2 common header layout. Some advanced/optional fields may be added later.

