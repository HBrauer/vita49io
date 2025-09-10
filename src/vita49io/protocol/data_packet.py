from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np

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


@dataclass(init=False)
class DataPacket:
    header: Header
    stream_id: Optional[int] = None
    class_id: Optional[ClassID] = None
    integer_seconds: Optional[int] = None
    fractional_seconds: Optional[int] = None
    payload: bytes = b""
    trailer: Optional[int] = None
    # Optional decoded IQ samples (complex64) when a compatible PayloadFormat
    # is provided to parse(). Not serialized unless pack() is called with a
    # compatible payload_format.
    iq: Optional["np.ndarray"] = None

    def __init__(
        self,
        *,
        # Either provide a ready Header or supply header fields below
        header: Optional[Header] = None,
        packet_type: Optional[PacketType] = None,
        tsi: TSI = TSI.NONE,
        tsf: TSF = TSF.NONE,
        packet_count: int = 0,
        # Common fields
        stream_id: Optional[int] = None,
        class_id: Optional[ClassID] = None,
        integer_seconds: Optional[int] = None,
        fractional_seconds: Optional[int] = None,
        payload: bytes = b"",
        trailer: Optional[int] = None,
        # Optional decoded IQ
        iq: Optional["np.ndarray"] = None,
        # If true, set header.indicators_25 (V49.2-only packet)
        requiresVita49_2: bool = False,
    ) -> None:
        if header is None:
            if packet_type is None:
                raise TypeError("Either header or packet_type must be provided")
            # packet_size is computed during pack()
            header = Header(
                packet_type=packet_type,
                class_id_present=(class_id is not None),
                indicators_26=(trailer is not None),  
                indicators_25=bool(requiresVita49_2),
                tsi=tsi,
                tsf=tsf,
                packet_count=int(packet_count),
                packet_size=0,
            )
        self.header = header
        self.stream_id = stream_id
        self.class_id = class_id
        self.integer_seconds = integer_seconds
        self.fractional_seconds = fractional_seconds
        self.payload = payload
        self.trailer = trailer
        self.iq = iq

    # Thin convenience accessors expected by tests/users
    @property
    def packet_type(self) -> PacketType:
        return self.header.packet_type

    @property
    def tsi(self) -> TSI:
        return self.header.tsi

    @property
    def tsf(self) -> TSF:
        return self.header.tsf

    @property
    def packet_count(self) -> int:
        return self.header.packet_count

    def __repr__(self) -> str:  # pragma: no cover - human-facing formatting
        def _hex32(v: int) -> str:
            return f"0x{v & 0xFFFFFFFF:08X}"

        parts: List[str] = []
        # Summarize header inline
        parts.append(f"packet_type={self.header.packet_type.name}")
        if self.stream_id is not None:
            parts.append(f"stream_id={_hex32(self.stream_id)}")
        if self.class_id is not None:
            oui, ic, pc = self.class_id
            parts.append(
                f"class_id=(0x{oui & 0xFFFFFF:06X}, 0x{ic & 0xFFFF:04X}, 0x{pc & 0xFFFF:04X})"
            )
        if self.header.tsi != TSI.NONE:
            parts.append(f"tsi={self.header.tsi.name}")
        if self.header.tsf != TSF.NONE:
            parts.append(f"tsf={self.header.tsf.name}")
        if self.integer_seconds is not None:
            parts.append(f"integer_seconds={self.integer_seconds}")
        if self.fractional_seconds is not None:
            parts.append(f"fractional_seconds={int(self.fractional_seconds)}")
        # Keep payload concise; show length only
        parts.append(f"payload_len={len(self.payload)}")
        if self.trailer is not None:
            parts.append(f"trailer={_hex32(self.trailer)}")
        # Show packet count always for debugging sequences
        parts.append(f"packet_count={self.header.packet_count}")
        parts.append(f"requiresVita49_2={self.header.indicators_25}")
        if self.header.indicators_24: # Indicator bits (for debugging)
            parts.append("indicators_24=True")
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
        if self.header.packet_type not in (
            PacketType.IF_DATA_WITHOUT_STREAM_ID,
            PacketType.IF_DATA_WITH_STREAM_ID,
            PacketType.EXTENSION_DATA_WITHOUT_STREAM_ID,
            PacketType.EXTENSION_DATA_WITH_STREAM_ID,
        ):
            raise ValueError(
                "DataPacket must be IF/EXT data (with/without Stream ID)"
            )

        # Enforce consistency between packet type and Stream ID presence
        if self.header.packet_type in (
            PacketType.IF_DATA_WITH_STREAM_ID,
            PacketType.EXTENSION_DATA_WITH_STREAM_ID,
        ) and self.stream_id is None:
            raise ValueError("Packet type requires a Stream ID, but none provided")
        if self.header.packet_type in (
            PacketType.IF_DATA_WITHOUT_STREAM_ID,
            PacketType.EXTENSION_DATA_WITHOUT_STREAM_ID,
        ) and self.stream_id is not None:
            raise ValueError("Packet type forbids a Stream ID, but one was provided")
        if self.header.class_id_present and self.class_id is None:
            raise ValueError("Packet type requires a Class ID, but none provided")
        if self.header.indicators_26 and self.trailer is None:
            raise ValueError("Packet type requires a trailer, but none provided")

        common = _Common(
            header=self.header,
            stream_id=self.stream_id,
            class_id=self.class_id,
            integer_seconds=self.integer_seconds,
            fractional_seconds=self.fractional_seconds,
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
        # Determine trailer presence from header.indicators_26
        trailer: Optional[int] = None
        if header.indicators_26:
            if end_idx <= idx:
                raise ValueError("Truncated packet: trailer indicated but no words present")
            trailer = words[-1]
            end_idx = len(words) - 1
        payload = _payload_words_to_bytes(words[idx:end_idx])

        iq = None
        if payload_format is not None:
            iq = _decode_iq_payload(payload, payload_format)

        return DataPacket(
            header=header,
            stream_id=common.stream_id,
            class_id=common.class_id,
            integer_seconds=common.integer_seconds,
            fractional_seconds=common.fractional_seconds,
            payload=payload,
            trailer=trailer,
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

    ipf, di, fmt = _validate_supported(pf)

    # Fast paths for common full-width cases
    if ipf == 32 and di == 32:
        if fmt == DataItemFormat.IEEE754_SINGLE:
            floats = np.frombuffer(payload, dtype=">f4")  # type: ignore[arg-type]
            if floats.size % 2 != 0:
                raise ValueError("Payload does not contain an even number of components for I/Q")
            iq = floats.reshape(-1, 2).astype(np.float32)
            return (iq[:, 0] + 1j * iq[:, 1]).astype(np.complex64)
        if fmt == DataItemFormat.SIGNED_FIXED_POINT:
            s32 = np.frombuffer(payload, dtype=">i4")  # type: ignore[arg-type]
            # Use float64 for arithmetic, then cast to float32 to avoid precision loss
            vals = (s32.astype(np.float64) / float(1 << 31)).astype(np.float32)
        else:  # UNSIGNED_FIXED_POINT
            u32 = np.frombuffer(payload, dtype=">u4")  # type: ignore[arg-type]
            # Use float64 for arithmetic, then cast to float32 to avoid precision loss
            vals = (u32.astype(np.float64) / float(1 << 32)).astype(np.float32)
    elif ipf == 16 and di == 16:
        if fmt == DataItemFormat.SIGNED_FIXED_POINT:
            s16 = np.frombuffer(payload, dtype=">i2")  # type: ignore[arg-type]
            vals = (s16.astype(np.float32) / float(1 << 15)).astype(np.float32)
        else:  # UNSIGNED_FIXED_POINT
            u16 = np.frombuffer(payload, dtype=">u2")  # type: ignore[arg-type]
            vals = (u16.astype(np.float32) / float(1 << 16)).astype(np.float32)
    else:
        # Generic paths: extract fields as big-endian and unpack lower di bits
        if ipf == 16:
            fields = np.frombuffer(payload, dtype=">u2")  # type: ignore[arg-type]
            uvals = fields.astype(np.uint32)
        else:  # ipf == 32
            fields32 = np.frombuffer(payload, dtype=">u4")  # type: ignore[arg-type]
            uvals = fields32

        if di < 32:
            mask = (1 << di) - 1
            uvals = (uvals & mask).astype(np.uint32, copy=False)

        if fmt == DataItemFormat.SIGNED_FIXED_POINT:
            # Sign-extend using arithmetic shifts for speed
            if di < 32:
                s = (uvals.astype(np.int32) << (32 - di)) >> (32 - di)
            else:
                s = uvals.astype(np.int32)
            scale = float(1 << (di - 1))
            vals = (s.astype(np.float32) / scale).astype(np.float32)
        elif fmt == DataItemFormat.UNSIGNED_FIXED_POINT:
            scale = float(1 << di)
            vals = (uvals.astype(np.float32) / scale).astype(np.float32)
        else:
            raise ValueError(f"Internal error: unexpected format in fixed-point decoder: {pf}")

    # Convert interleaved I,Q values to complex
    if vals.size % 2 != 0:
        raise ValueError("Payload does not contain an even number of components for I/Q")
    vec = vals.reshape(-1, 2)
    return (vec[:, 0] + 1j * vec[:, 1]).astype(np.complex64)


def _encode_iq_payload(iq: "np.ndarray", pf: PayloadFormat) -> bytes:

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

    # Fast path: 32-bit fields
    if ipf == 32 and di == 32:
        if fmt == DataItemFormat.IEEE754_SINGLE:
            return vals.astype(">f4").tobytes()
        if fmt == DataItemFormat.SIGNED_FIXED_POINT:
            # Perform arithmetic in float64 to reduce rounding artifacts
            scale = float(1 << 31)
            max_val = (float((1 << 31) - 1)) / scale
            vals64 = vals.astype(np.float64)
            vals_c = np.clip(vals64, -1.0, max_val)
            ints = np.rint(vals_c * scale).astype(np.int32)
            return ints.astype(">i4").tobytes()
        else:  # UNSIGNED_FIXED_POINT
            # Use float64 for clipping and scaling to avoid rounding 0.999.. to 1.0
            scale = float(1 << 32)
            max_val = (float((1 << 32) - 1)) / scale
            vals64 = vals.astype(np.float64)
            vals_c = np.clip(vals64, 0.0, max_val)
            u32 = np.rint(vals_c * scale).astype(np.uint32)
            return u32.astype(">u4").tobytes()

    # Fast path: 16-bit fields
    if ipf == 16 and di == 16:
        if fmt == DataItemFormat.SIGNED_FIXED_POINT:
            scale = float(1 << 15)
            max_val = (float((1 << 15) - 1)) / scale
            vals_c = np.clip(vals, -1.0, max_val)
            i16 = np.rint(vals_c * scale).astype(np.int16)
            return i16.astype(">i2").tobytes()
        else:  # UNSIGNED_FIXED_POINT
            scale = float(1 << 16)
            max_val = (float((1 << 16) - 1)) / scale
            vals_c = np.clip(vals, 0.0, max_val)
            u16 = np.rint(vals_c * scale).astype(np.uint16)
            return u16.astype(">u2").tobytes()

    # Generic fixed-point encoding into 32-bit fields (16 or 24-bit in lower bits)
    if fmt == DataItemFormat.SIGNED_FIXED_POINT:
        scale = float(1 << (di - 1))
        max_val = (float((1 << (di - 1)) - 1)) / scale
        vals_c = np.clip(vals, -1.0, max_val)
        ints = np.rint(vals_c * scale).astype(np.int64)
        uvals = (ints & ((1 << di) - 1)).astype(np.uint32)
    else:  # UNSIGNED_FIXED_POINT
        scale = float(1 << di)
        max_val = (float((1 << di) - 1)) / scale
        vals_c = np.clip(vals, 0.0, max_val)
        uvals = np.rint(vals_c * scale).astype(np.uint32)

    # ipf == 32 here; pack lower di bits of 32-bit field
    fields32 = (uvals & ((1 << di) - 1)).astype(np.uint32)
    return fields32.astype(">u4").tobytes()

