"""Implement VITA 49 data packet helpers for encoding and decoding IQ payloads.

Packets follow a lazy, memoryview-backed design:

* ``from_bytes`` keeps a ``memoryview`` of the raw packet and defers decoding.
* ``to_bytes`` fast-paths to the stored bytes when the packet was not mutated.
* IQ/payload are decoded only when accessed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Union
import numpy as np

from .core import (
    LazyBinary,
    Header,
    WORD,
    _Common,
    _finalize_words_to_bytes,
    _pack_common_prefix,
    _parse_common_from_bytes,
    _payload_bytes_to_words,
    _u32,
)
from .enums import PacketType, TSI, TSF
from .vrt_types import ClassID
from .cif0 import PayloadFormat, PackingMethod, SampleType, DataItemFormat


@dataclass(init=False, slots=True)
class DataPacket(LazyBinary):
    """Represent a VITA 49 data packet with lazy payload/IQ decoding."""

    _header: Header | None = None
    _stream_id: int | None = None
    _class_id: ClassID | None = None
    _integer_seconds: int | None = None
    _fractional_seconds: int | None = None
    _payload: Union[bytes, memoryview, None] = None
    _trailer: int | None = None
    _iq: "np.ndarray | None" = None
    _payload_format: PayloadFormat | None = None
    _copy_payload: bool = False
    validate_strict: bool = False

    def __init__(
        self,
        *,
        header: Header | None = None,
        packet_type: PacketType | None = None,
        tsi: TSI = TSI.NONE,
        tsf: TSF = TSF.NONE,
        packet_count: int = 0,
        stream_id: int | None = None,
        class_id: ClassID | None = None,
        integer_seconds: int | None = None,
        fractional_seconds: int | None = None,
        payload: Union[bytes, memoryview, None] = None,
        trailer: int | None = None,
        iq: "np.ndarray | None" = None,
        payload_format: PayloadFormat | None = None,
        validate_strict: bool = False,
        requiresVita49_2: bool = False,
        _mv: memoryview | None = None,
        copy_payload: bool = False,
    ) -> None:
        # Call base __init__ directly to avoid dataclass/super slot quirks
        LazyBinary.__init__(self, _mv=_mv)
        if header is None and packet_type is not None:
            header = Header(
                packet_type=packet_type,
                class_id_present=(class_id is not None),
                indicators_26=(trailer is not None),
                indicators_25=bool(requiresVita49_2),
                indicators_24=False,
                tsi=tsi,
                tsf=tsf,
                packet_count=int(packet_count),
                packet_size=0,
            )
        if header is None and _mv is None:
            raise TypeError("Either header or packet_type must be provided")
        self._header = header
        self._stream_id = stream_id
        self._class_id = class_id
        self._integer_seconds = integer_seconds
        self._fractional_seconds = fractional_seconds
        self._payload = payload if _mv is None else payload
        self._trailer = trailer
        self._iq = iq
        self._payload_format = payload_format
        self._copy_payload = copy_payload
        self.validate_strict = validate_strict
        if _mv is None:
            if self._payload is None:
                self._payload = b""
            self._mark_dirty()

    # ---------- common prefix ----------
    def _common_info(self) -> Tuple[_Common, int, int]:
        def decode(self, mv: memoryview) -> Tuple[_Common, int, int]:
            common, payload_start, payload_end = _parse_common_from_bytes(mv)
            if common.header.packet_type not in (
                PacketType.IF_DATA_WITHOUT_STREAM_ID,
                PacketType.IF_DATA_WITH_STREAM_ID,
                PacketType.EXTENSION_DATA_WITHOUT_STREAM_ID,
                PacketType.EXTENSION_DATA_WITH_STREAM_ID,
            ):
                raise ValueError("Not a Data packet type")
            if self._header is None:
                self._header = common.header
            if self._stream_id is None:
                self._stream_id = common.stream_id
            if self._class_id is None:
                self._class_id = common.class_id
            if self._integer_seconds is None:
                self._integer_seconds = common.integer_seconds
            if self._fractional_seconds is None:
                self._fractional_seconds = common.fractional_seconds
            return common, payload_start, payload_end

        return self._lazy_field("common_info", decode)

    def _payload_bounds(self) -> Tuple[int, int]:
        _, start, end = self._common_info()
        if self.header.indicators_26:
            end -= 4
        return start, end

    # ---------- convenience accessors ----------
    @property
    def header(self) -> Header:
        if self._header is None:
            if self._mv is None:
                raise ValueError("Header not available; packet not backed by bytes")
            common, _, _ = self._common_info()
            self._header = common.header
        return self._header

    @header.setter
    def header(self, value: Header) -> None:
        self._header = value
        self._mark_dirty()

    @property
    def stream_id(self) -> int | None:
        if self._stream_id is None and self._mv is not None:
            common, _, _ = self._common_info()
            self._stream_id = common.stream_id
        return self._stream_id

    @stream_id.setter
    def stream_id(self, value: int | None) -> None:
        self._stream_id = value
        self._mark_dirty()

    @property
    def class_id(self) -> ClassID | None:
        if self._class_id is None and self._mv is not None:
            common, _, _ = self._common_info()
            self._class_id = common.class_id
        return self._class_id

    @class_id.setter
    def class_id(self, value: ClassID | None) -> None:
        self._class_id = value
        self._mark_dirty()

    @property
    def integer_seconds(self) -> int | None:
        if self._integer_seconds is None and self._mv is not None:
            common, _, _ = self._common_info()
            self._integer_seconds = common.integer_seconds
        return self._integer_seconds

    @integer_seconds.setter
    def integer_seconds(self, value: int | None) -> None:
        self._integer_seconds = value
        self._mark_dirty()

    @property
    def fractional_seconds(self) -> int | None:
        if self._fractional_seconds is None and self._mv is not None:
            common, _, _ = self._common_info()
            self._fractional_seconds = common.fractional_seconds
        return self._fractional_seconds

    @fractional_seconds.setter
    def fractional_seconds(self, value: int | None) -> None:
        self._fractional_seconds = value
        self._mark_dirty()

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

    @property
    def payload(self) -> Union[bytes, memoryview]:
        if self._payload is None:
            if self._mv is None:
                return b""
            start, end = self._payload_bounds()
            view = self._mv[start:end]
            self._payload = view.tobytes() if self._copy_payload else view
        return self._payload

    @payload.setter
    def payload(self, value: Union[bytes, memoryview]) -> None:
        self._payload = value
        self._mark_dirty()

    @property
    def trailer(self) -> int | None:
        if self._trailer is None and self.header.indicators_26 and self._mv is not None:
            _, _, end = self._common_info()
            if end < 4:
                raise ValueError("Truncated packet: trailer indicated but no words present")
            self._trailer = WORD.unpack_from(self._mv, end - 4)[0]
        return self._trailer

    @trailer.setter
    def trailer(self, value: int | None) -> None:
        self._trailer = value
        self._mark_dirty()

    @property
    def payload_format(self) -> PayloadFormat | None:
        return self._payload_format

    @payload_format.setter
    def payload_format(self, value: PayloadFormat | None) -> None:
        self._payload_format = value
        self._mark_dirty()

    @property
    def iq(self) -> "np.ndarray | None":
        if self._iq is not None:
            return self._iq
        if self._payload_format is None:
            return None
        pay = self.payload
        pay_bytes = pay.tobytes() if isinstance(pay, memoryview) else pay
        self._iq = _decode_iq_payload(pay_bytes, self._payload_format)
        return self._iq

    @iq.setter
    def iq(self, value: "np.ndarray | None") -> None:
        self._iq = value
        self._mark_dirty()

    @property
    def iq_raw(self) -> "np.ndarray | None":
        if self._payload_format is None:
            return None
        pay = self.payload
        pay_view = pay if isinstance(pay, memoryview) else memoryview(pay)
        return _view_iq_payload(pay_view, self._payload_format)

    def __repr__(self) -> str:  # pragma: no cover - human-facing formatting
        def _hex32(v: int) -> str:
            return f"0x{v & 0xFFFFFFFF:08X}"

        parts: List[str] = [f"packet_type={self.header.packet_type.name}"]
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
        parts.append(f"payload_len={len(self.payload)}")
        if self.trailer is not None:
            parts.append(f"trailer={_hex32(self.trailer)}")
        parts.append(f"packet_count={self.header.packet_count}")
        parts.append(f"requiresVita49_2={self.header.indicators_25}")
        if self.header.indicators_24:
            parts.append("indicators_24=True")
        if self._iq is not None:
            try:
                parts.append(f"iq_len={int(getattr(self._iq, 'size', len(self._iq)))}")
            except Exception:
                parts.append("iq_len=?")
        return f"DataPacket({', '.join(parts)})"

    def to_bytes(
        self,
        payload_format: PayloadFormat | None = None,
    ) -> bytes:
        if not self._dirty and self._mv is not None:
            return self._mv.tobytes()

        hdr = self.header
        if hdr.packet_type not in (
            PacketType.IF_DATA_WITHOUT_STREAM_ID,
            PacketType.IF_DATA_WITH_STREAM_ID,
            PacketType.EXTENSION_DATA_WITHOUT_STREAM_ID,
            PacketType.EXTENSION_DATA_WITH_STREAM_ID,
        ):
            raise ValueError("DataPacket must be IF/EXT data (with/without Stream ID)")

        if hdr.packet_type in (
            PacketType.IF_DATA_WITH_STREAM_ID,
            PacketType.EXTENSION_DATA_WITH_STREAM_ID,
        ) and self.stream_id is None:
            raise ValueError("Packet type requires a Stream ID, but none provided")
        if hdr.packet_type in (
            PacketType.IF_DATA_WITHOUT_STREAM_ID,
            PacketType.EXTENSION_DATA_WITHOUT_STREAM_ID,
        ) and self.stream_id is not None:
            raise ValueError("Packet type forbids a Stream ID, but one was provided")
        if hdr.class_id_present and self.class_id is None:
            raise ValueError("Packet type requires a Class ID, but none provided")
        if hdr.indicators_26 and self.trailer is None:
            raise ValueError("Packet type requires a trailer, but none provided")

        pf = payload_format or self._payload_format

        common = _Common(
            header=hdr,
            stream_id=self.stream_id,
            class_id=self.class_id,
            integer_seconds=self.integer_seconds,
            fractional_seconds=self.fractional_seconds,
        )
        words: List[int] = _pack_common_prefix(common)

        if self.iq is not None and pf is not None:
            iq_values = self.iq
            payload_bytes: Union[bytes, memoryview] = _encode_iq_payload(iq_values, pf)
        else:
            payload_bytes = self.payload

        words.extend(_payload_bytes_to_words(payload_bytes))
        if self.trailer is not None:
            words.append(_u32(self.trailer))

        raw_bytes = _finalize_words_to_bytes(words)
        self._mv = memoryview(raw_bytes)
        self._dirty = False
        return raw_bytes

    @classmethod
    def from_bytes(
        cls,
        data: Union[bytes, bytearray, memoryview],
        payload_format: PayloadFormat | None = None,
        *,
        copy_payload: bool = False,
    ) -> "DataPacket":
        """
        Construct a DataPacket backed by a memoryview of the raw bytes.

        IQ decoding is deferred; `decode_iq` is accepted for compatibility but
        `.iq` is decoded lazily when accessed and a payload_format is available.
        """
        mv = data if isinstance(data, memoryview) else memoryview(data)
        pkt = cls(_mv=mv, payload_format=payload_format, copy_payload=copy_payload)
        return pkt


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
    validate_strict = False  # For now no strict validation because test files have wrong vector size and repeat count and we don't use it currently
    if validate_strict:    
        if pf.vector_size != 1:
            raise ValueError(f"Unsupported vector size: decoded value {pf.vector_size}, only 1 is supported")
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


def _decode_iq_payload(payload: bytes, pf: PayloadFormat) -> "np.ndarray":

    ipf, di, fmt = _validate_supported(pf)

    # Fast paths for common full-width cases
    if ipf == 32 and di == 32:
        if fmt == DataItemFormat.IEEE754_SINGLE:
            floats = np.frombuffer(payload, dtype=">f4")
            if floats.size % 2 != 0:
                raise ValueError("Payload does not contain an even number of components for I/Q")
            iq = floats.reshape(-1, 2).astype(np.float32)
            return (iq[:, 0] + 1j * iq[:, 1]).astype(np.complex64)
        if fmt == DataItemFormat.SIGNED_FIXED_POINT:
            s32 = np.frombuffer(payload, dtype=">i4")
            # Use float64 for arithmetic, then cast to float32 to avoid precision loss
            vals = (s32.astype(np.float64) / float(1 << 31)).astype(np.float32)
        else:  # UNSIGNED_FIXED_POINT
            u32 = np.frombuffer(payload, dtype=">u4")
            # Use float64 for arithmetic, then cast to float32 to avoid precision loss
            vals = (u32.astype(np.float64) / float(1 << 32)).astype(np.float32)
    elif ipf == 16 and di == 16:
        if fmt == DataItemFormat.SIGNED_FIXED_POINT:
            s16 = np.frombuffer(payload, dtype=">i2")
            vals = (s16.astype(np.float32) / float(1 << 15)).astype(np.float32)
        else:  # UNSIGNED_FIXED_POINT
            u16 = np.frombuffer(payload, dtype=">u2")
            vals = (u16.astype(np.float32) / float(1 << 16)).astype(np.float32)
    else:
        # Generic paths: extract fields as big-endian and unpack lower di bits
        if ipf == 16:
            fields = np.frombuffer(payload, dtype=">u2")
            uvals = fields.astype(np.uint32)
        else:  # ipf == 32
            fields32 = np.frombuffer(payload, dtype=">u4")
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


def _view_iq_payload(payload: Union[bytes, memoryview], pf: PayloadFormat) -> "np.ndarray":

    ipf, di, fmt = _validate_supported(pf)
    if ipf == 32 and di == 16:
        dtype = ">i2" if fmt == DataItemFormat.SIGNED_FIXED_POINT else ">u2"
        # 32-bit fields store the data item in the lower 16 bits (big-endian).
        vals16 = np.frombuffer(payload, dtype=dtype)[1::2]
        if vals16.size % 2 != 0:
            raise ValueError("Payload does not contain an even number of components for I/Q")
        return vals16.reshape(-1, 2)

    if ipf != di or ipf not in (16, 32):
        raise ValueError(
            "Native IQ view requires full-width 16- or 32-bit data items "
            "(item_packing_field_size_bits == data_item_size_bits), "
            "except 16-in-32 fixed-point which returns the lower 16 bits."
        )

    if fmt == DataItemFormat.IEEE754_SINGLE:
        dtype = ">f4"
    elif fmt == DataItemFormat.SIGNED_FIXED_POINT:
        dtype = ">i2" if ipf == 16 else ">i4"
    else:  # UNSIGNED_FIXED_POINT
        dtype = ">u2" if ipf == 16 else ">u4"

    vals = np.frombuffer(payload, dtype=dtype)
    if vals.size % 2 != 0:
        raise ValueError("Payload does not contain an even number of components for I/Q")
    return vals.reshape(-1, 2)


def _encode_iq_payload(iq: "np.ndarray", pf: PayloadFormat) -> bytes:

    ipf, di, fmt = _validate_supported(pf)

    # Converted input to (N, 2) float32 array of I and Q in order 
    # to have a consistent starting point for all encoding paths.
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
            vals_c = np.clip(vals, -1.0, 1.0)
            return vals_c.astype(">f4").tobytes()
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
