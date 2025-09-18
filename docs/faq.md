# Frequently Asked Questions

**How do I decode IQ payloads into NumPy arrays?**

Call `DataPacket.from_bytes` with the optional `payload_format` argument populated. When the payload format matches a supported complex encoding, the returned packet includes an `iq` ndarray.

**Can I use vita49io with streaming sockets?**

Yes. Feed fixed-size chunks into `DataPacket.from_bytes` or `ContextPacket.from_bytes` as they arrive. The parsing helpers accept `bytes` objects and perform validation before decoding.

**Does the library handle VITA 49.2 indicator bits?**

Indicator bits 25 and 24 are exposed through the `Header` dataclass. `IQStreamWriter` can set them via the `requires_vita49_2` and `context_timestamp_mode_general` flags.
