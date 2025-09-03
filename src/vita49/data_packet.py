from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List

from .core import (
    Header,
    _Common,
    _finalize_words_to_bytes,
    _pack_common_prefix,
    _parse_common_from_words,
    _payload_bytes_to_words,
    _payload_words_to_bytes,
    _unpack_u32_be,
    _u32,
)
from .enums import PacketType, TSI, TSF
from .vrt_types import ClassID
from .cif0 import PayloadFormat, PackingMethod, SampleType, DataItemFormat


@dataclass
class DataPacket:
    packet_type: PacketType
    stream_id: Optional[int] = None
    class_id: Optional[ClassID] = None
    tsi: TSI = TSI.NONE
    tsf: TSF = TSF.NONE
    integer_seconds: Optional[int] = None
    fractional_seconds: Optional[int] = None
    payload: bytes = b""
    trailer: Optional[int] = None
    packet_count: int = 0
    # Optional decoded IQ samples (complex64) when a compatible PayloadFormat
    # is provided to parse(). Not serialized unless pack() is called with a
    # compatible payload_format.
    iq: Optional["np.ndarray"] = None

    def __repr__(self) -> str:  # pragma: no cover - human-facing formatting
        def _hex32(v: int) -> str:
            return f"0x{v & 0xFFFFFFFF:08X}"

        parts: List[str] = []
        parts.append(f"packet_type={self.packet_type.name}")
        if self.stream_id is not None:
            parts.append(f"stream_id={_hex32(self.stream_id)}")
        if self.class_id is not None:
            oui, ic, pc = self.class_id
            parts.append(
                f"class_id=(0x{oui & 0xFFFFFF:06X}, 0x{ic & 0xFFFF:04X}, 0x{pc & 0xFFFF:04X})"
            )
        if self.tsi != TSI.NONE:
            parts.append(f"tsi={self.tsi.name}")
        if self.tsf != TSF.NONE:
            parts.append(f"tsf={self.tsf.name}")
        if self.integer_seconds is not None:
            parts.append(f"integer_seconds={self.integer_seconds}")
        if self.fractional_seconds is not None:
            parts.append(f"fractional_seconds={int(self.fractional_seconds)}")
        # Keep payload concise; show length only
        parts.append(f"payload_len={len(self.payload)}")
        if self.trailer is not None:
            parts.append(f"trailer={_hex32(self.trailer)}")
        # Show packet count always for debugging sequences
        parts.append(f"packet_count={self.packet_count}")
        # If IQ present, summarize size without importing numpy
        if self.iq is not None:
            n = None
            try:
                n = int(getattr(self.iq, "size", None))
            except Exception:
                pass
            if n is None:
                try:
                    n = len(self.iq)  # type: ignore[arg-type]
                except Exception:
                    n = 0
            parts.append(f"iq_len={n}")
        return f"DataPacket({', '.join(parts)})"

    def pack(self, payload_format: Optional[PayloadFormat] = None) -> bytes:
        if self.packet_type not in (
            PacketType.IF_DATA_WITHOUT_STREAM_ID,
            PacketType.IF_DATA_WITH_STREAM_ID,
            PacketType.EXTENSION_DATA_WITHOUT_STREAM_ID,
            PacketType.EXTENSION_DATA_WITH_STREAM_ID,
        ):
            raise ValueError(
                "DataPacket must be IF/EXT data (with/without Stream ID)"
            )

        # Enforce consistency between packet type and Stream ID presence
        if self.packet_type in (
            PacketType.IF_DATA_WITH_STREAM_ID,
            PacketType.EXTENSION_DATA_WITH_STREAM_ID,
        ) and self.stream_id is None:
            raise ValueError("Packet type requires a Stream ID, but none provided")
        if self.packet_type in (
            PacketType.IF_DATA_WITHOUT_STREAM_ID,
            PacketType.EXTENSION_DATA_WITHOUT_STREAM_ID,
        ) and self.stream_id is not None:
            raise ValueError("Packet type forbids a Stream ID, but one was provided")

        # Build common prefix using _Common helper
        common = _Common(
            header=Header(
                packet_type=self.packet_type,
                class_id_present=self.class_id is not None,
                trailer_present=self.trailer is not None,
                packet_specific_indicators=0,
                tsi=self.tsi,
                tsf=self.tsf,
                packet_count=self.packet_count,
                packet_size=0,
            ),
            stream_id=self.stream_id,
            class_id=self.class_id,
            integer_seconds=self.integer_seconds,
            fractional_seconds=self.fractional_seconds,
            trailer=self.trailer,
        )
        words: List[int] = _pack_common_prefix(common)

        # If IQ samples are provided and a payload_format is given, encode IQ
        # into payload bytes according to the supported subset. Otherwise use
        # raw payload bytes as-is for backward compatibility.
        if self.iq is not None and payload_format is not None:
            payload_bytes = _encode_iq_payload(self.iq, payload_format)
        else:
            payload_bytes = self.payload

        words.extend(_payload_bytes_to_words(payload_bytes))
        if self.trailer is not None:
            words.append(_u32(self.trailer))
        return _finalize_words_to_bytes(words)

    @staticmethod
    def parse(data: bytes, payload_format: Optional[PayloadFormat] = None) -> "DataPacket":
        if len(data) < 4 or len(data) % 4 != 0:
            raise ValueError("Invalid VRT packet length")
        words = [_unpack_u32_be(data[i : i + 4]) for i in range(0, len(data), 4)]
        common, idx, end_idx = _parse_common_from_words(words)
        header = common.header
        if header.packet_type not in (
            PacketType.IF_DATA_WITHOUT_STREAM_ID,
            PacketType.IF_DATA_WITH_STREAM_ID,
            PacketType.EXTENSION_DATA_WITHOUT_STREAM_ID,
            PacketType.EXTENSION_DATA_WITH_STREAM_ID,
        ):
            raise ValueError("Not a Data packet type")
        payload = _payload_words_to_bytes(words[idx:end_idx])
        trailer = common.trailer

        iq = None
        if payload_format is not None:
            iq = _decode_iq_payload(payload, payload_format)

        return DataPacket(
            packet_type=header.packet_type,
            stream_id=common.stream_id,
            class_id=common.class_id,
            tsi=header.tsi,
            tsf=header.tsf,
            integer_seconds=common.integer_seconds,
            fractional_seconds=common.fractional_seconds,
            payload=payload,
            trailer=trailer,
            packet_count=header.packet_count,
            iq=iq,
        )

__all__ = ["DataPacket"]


# --------------------------
# Internal helpers (IQ I/O)
# --------------------------

def _validate_supported(pf: PayloadFormat) -> Tuple[int, int, DataItemFormat]:
    if pf.packing_method != PackingMethod.PROCESSING_EFFICIENT:
        raise ValueError("Unsupported packing method: only Processing-efficient is supported")
    if pf.sample_type != SampleType.COMPLEX_CARTESIAN:
        raise ValueError("Unsupported sample type: only Complex Cartesian (I/Q) is supported")
    if pf.sample_component_repeat:
        raise ValueError("Unsupported sample-component repeat: must be false")
    if pf.event_tag_size_bits != 0 or pf.channel_tag_size_bits != 0:
        raise ValueError("Unsupported tag sizes: event and channel tag sizes must be 0")
    if pf.data_item_fraction_size_bits != 0:
        raise ValueError("Unsupported data item fraction size: must be 0")
    if pf.vector_size != 0:
        raise ValueError("Unsupported vector size: only vector size 0 is supported")
    if pf.repeat_count != 1:
        raise ValueError(f"Unsupported repeat count of {pf.repeat_count}, only 1 is supported")

    try:
        fmt = DataItemFormat(pf.data_item_format_code)
    except ValueError as e:
        raise ValueError(f"Unsupported data item format code: {pf.data_item_format_code}") from e

    # Validate combinations
    ipf = pf.item_packing_field_size_bits
    di = pf.data_item_size_bits

    valid = False
    if fmt in (DataItemFormat.SIGNED_FIXED_POINT, DataItemFormat.UNSIGNED_FIXED_POINT):
        if (ipf == 16 and di == 16) or (
            ipf == 32 and di in (16, 24, 32)
        ):
            valid = True
    elif fmt == DataItemFormat.IEEE754_SINGLE:
        if ipf == 32 and di == 32:
            valid = True

    if not valid:
        raise ValueError(
            f"Unsupported item packing/data size combination: item_packing={ipf}, data_item={di}, format={fmt.name}"
        )

    return ipf, di, fmt


def _decode_iq_payload(payload: bytes, pf: PayloadFormat):
    import numpy as np  # lazy import to avoid hard dependency when unused

    ipf, di, fmt = _validate_supported(pf)

    # Field extraction according to packing field size
    if ipf == 16:
        # Big-endian 16-bit fields
        fields = np.frombuffer(payload, dtype=">u2")  # type: ignore[arg-type]
        uvals = fields.astype(np.uint32)
    else:
        # ipf == 32: big-endian 32-bit fields
        if fmt == DataItemFormat.IEEE754_SINGLE and di == 32:
            # Directly interpret as float32 fields
            floats = np.frombuffer(payload, dtype=">f4")  # type: ignore[arg-type]
            if floats.size % 2 != 0:
                raise ValueError("Payload does not contain an even number of components for I/Q")
            iq = floats.reshape(-1, 2).astype(np.float32)
            return (iq[:, 0] + 1j * iq[:, 1]).astype(np.complex64)
        # Otherwise treat as u32 and mask lower di bits
        fields32 = np.frombuffer(payload, dtype=">u4")  # type: ignore[arg-type]
        uvals = fields32

    # Extract data item from lower di bits (right-justified assumption)
    if di < 32:
        mask = (1 << di) - 1
        uvals = uvals & mask

    # Convert to normalized float
    import numpy as _np  # local alias for calculations
    if fmt == DataItemFormat.SIGNED_FIXED_POINT:
        sign_bit = 1 << (di - 1)
        signed = _np.where(uvals & sign_bit, uvals - (1 << di), uvals).astype(_np.int64)
        scale = float(1 << (di - 1))
        vals = (signed.astype(_np.float32) / scale).astype(_np.float32)
    elif fmt == DataItemFormat.UNSIGNED_FIXED_POINT:
        scale = float(1 << di)
        vals = (uvals.astype(_np.float32) / scale).astype(_np.float32)
    else:
        raise ValueError("Internal error: unexpected format in fixed-point decoder")

    if vals.size % 2 != 0:
        raise ValueError("Payload does not contain an even number of components for I/Q")

    vec = vals.reshape(-1, 2)
    return (vec[:, 0] + 1j * vec[:, 1]).astype(_np.complex64)


def _encode_iq_payload(iq: "np.ndarray", pf: PayloadFormat) -> bytes:
    import numpy as np  # lazy import

    ipf, di, fmt = _validate_supported(pf)

    # Normalize input to (N, 2) float32 array of I and Q
    arr = np.asarray(iq)
    if arr.dtype.kind == "c":
        I = arr.real.astype(np.float32)
        Q = arr.imag.astype(np.float32)
        vec = np.stack([I, Q], axis=1)
    else:
        vec = arr.astype(np.float32)
        if vec.ndim == 1:
            raise ValueError("IQ array must be complex or shape (N,2)")
        if vec.shape[-1] != 2:
            raise ValueError("IQ array last dimension must be 2 (I,Q)")
    vals = vec.reshape(-1).astype(np.float32)

    if fmt == DataItemFormat.IEEE754_SINGLE and di == 32 and ipf == 32:
        # Pack as big-endian float32 fields in I,Q order
        return vals.astype(">f4").tobytes()

    # Fixed-point encoding
    if fmt == DataItemFormat.SIGNED_FIXED_POINT:
        scale = float(1 << (di - 1))
        # Clip to representable range [-1, 1 - 2^(1-di)]
        max_val = (float((1 << (di - 1)) - 1)) / scale
        vals = np.clip(vals, -1.0, max_val)
        ints = np.rint(vals * scale).astype(np.int64)
        # Convert to two's complement unsigned of di bits
        uvals = (ints & ((1 << di) - 1)).astype(np.uint32)
    else:  # UNSIGNED_FIXED_POINT
        scale = float(1 << di)
        max_val = (float((1 << di) - 1)) / scale
        vals = np.clip(vals, 0.0, max_val)
        uvals = np.rint(vals * scale).astype(np.uint32)

    if ipf == 16 and di == 16:
        return uvals.astype(">u2").tobytes()

    # ipf == 32
    if di == 32:
        fields32 = uvals.astype(">u4").tobytes()
        return fields32

    # di in (16, 24) packed into lower bits of 32-bit field
    fields32 = (uvals & ((1 << di) - 1)).astype(np.uint32)
    return fields32.astype(">u4").tobytes()


