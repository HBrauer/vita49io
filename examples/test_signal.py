from vita49io import ContextPacket, PacketType, TSI, TSF, CIF0Fields
from vita49io.protocol.cif0 import PayloadFormat, PackingMethod, SampleType, DataItemFormat

stream_id = 0x12345678

# Define how subsequent data packet payloads are encoded (32‑bit float I/Q)
pf = PayloadFormat(
    packing_method=PackingMethod.PROCESSING_EFFICIENT,
    sample_type=SampleType.COMPLEX_CARTESIAN,
    data_item_format=DataItemFormat.IEEE754_SINGLE,
    sample_component_repeat=False,
    event_tag_size_bits=0,
    channel_tag_size_bits=0,
    data_item_fraction_size_bits=0,
    item_packing_field_size_bits=32,
    data_item_size_bits=32,
    repeat_count=1,
    vector_size=1,
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