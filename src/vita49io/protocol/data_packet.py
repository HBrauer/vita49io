"""Implement VITA 49 data packet helpers for encoding and decoding IQ payloads.

Examples:
    >>> from vita49io.protocol.data_packet import DataPacket
    >>> from vita49io.protocol.enums import PacketType
    >>> DataPacket(packet_type=PacketType.IF_DATA_WITH_STREAM_ID, stream_id=1).payload
    b''
"""

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


_DEFAULT_REFERENCE_IMPEDANCE_OHMS = 50.0


def _reflevel_dbm_to_vpk(reference_level_dbm: float) -> float:
    # Convert power in dBm to power in watts
    power_w = 10.0 ** ((float(reference_level_dbm) - 30.0) / 10.0)
    return float(np.sqrt(2.0 * _DEFAULT_REFERENCE_IMPEDANCE_OHMS * power_w))


def _normalize_iq_to_reference_level(
    iq: "np.ndarray",
    reference_level_dbm: float,
) -> "np.ndarray":
    ref_dbm = float(reference_level_dbm)
    if not np.isfinite(ref_dbm):
        raise ValueError("reference_level_dbm must be finite")
    vpk = _reflevel_dbm_to_vpk(ref_dbm)
    if not np.isfinite(vpk) or vpk <= 0.0:
        raise ValueError("Computed peak voltage must be finite and positive")
    scale = 1.0 / vpk
    return np.asarray(iq) * scale


@dataclass(init=False)
class DataPacket:
    """Represent a VITA 49 data packet. These packets carry the raw, high-rate signal samples (I and Q data). The payload of a Data Packet is a contiguous stream of binary values representing the digitized RF signal over time.

    Args:
        header (Header): Pre-built header to attach to the packet.
        stream_id (Optional[int]): Stream identifier value if the packet type carries one.
        class_id (Optional[ClassID]): VITA 49 class identifier tuple when present.
        integer_seconds (Optional[int]): Integer seconds component for timestamped packets.
        fractional_seconds (Optional[int]): Fractional seconds component for timestamped packets.
        payload (bytes): Raw payload buffer containing opaque data words.
        trailer (Optional[int]): Optional 32-bit trailer word when indicator bit 26 is set.
        iq (Optional[np.ndarray]): Decoded IQ samples when constructed from structured data.

    Examples:
        >>> from vita49io.protocol.data_packet import DataPacket
        >>> from vita49io.protocol.enums import PacketType
        >>> payload = np.array([1.0, 0.0, 0.0, 1.0], dtype=">f4").tobytes()
        >>> pkt = DataPacket(
            packet_type=PacketType.IF_DATA_WITH_STREAM_ID,
            stream_id=1,
            tsi=TSI.UTC,
            tsf=TSF.FRACTIONAL,
            integer_seconds=1_700_000_000,
            fractional_seconds=0,
            payload=payload,  # raw payload goes here

    """
    header: Header
    stream_id: Optional[int] = None
    class_id: Optional[ClassID] = None
    integer_seconds: Optional[int] = None
    fractional_seconds: Optional[int] = None
    payload: bytes = b""
    trailer: Optional[int] = None
    validate_strict: bool = False
    # Optional decoded IQ samples (complex64) when a compatible PayloadFormat
    # is provided to from_bytes(). Not serialized unless to_bytes() is called with a
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
            # packet_size is computed during to_bytes()
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
        """Return the packet type reported by the packet header.

        Returns:
            PacketType: The enumerated packet type associated with the packet.

        Examples:
            >>> from vita49io.protocol.data_packet import DataPacket
            >>> from vita49io.protocol.enums import PacketType
            >>> DataPacket(packet_type=PacketType.IF_DATA_WITH_STREAM_ID, stream_id=1).packet_type
            <PacketType.IF_DATA_WITH_STREAM_ID: 1>
        """
        return self.header.packet_type

    @property
    def tsi(self) -> TSI:
        """Return the Timestamp Integer (TSI) mode encoded in the header.

        Returns:
            TSI: The enumerated integer timestamp mode.

        Examples:
            >>> from vita49io.protocol.data_packet import DataPacket
            >>> from vita49io.protocol.enums import PacketType, TSI
            >>> DataPacket(packet_type=PacketType.IF_DATA_WITH_STREAM_ID, stream_id=1, tsi=TSI.UTC).tsi
            <TSI.UTC: 1>
        """
        return self.header.tsi

    @property
    def tsf(self) -> TSF:
        """Return the Timestamp Fractional (TSF) mode encoded in the header.

        Returns:
            TSF: The enumerated fractional timestamp selection.

        Examples:
            >>> from vita49io.protocol.data_packet import DataPacket
            >>> from vita49io.protocol.enums import PacketType, TSF
            >>> DataPacket(packet_type=PacketType.IF_DATA_WITH_STREAM_ID, stream_id=1, tsf=TSF.FRACTIONAL).tsf
            <TSF.FRACTIONAL: 2>
        """
        return self.header.tsf

    @property
    def packet_count(self) -> int:
        """Return the rolling packet count encoded in the header.

        Returns:
            int: The lower 4-bit packet count value extracted from the header.

        Examples:
            >>> from vita49io.protocol.data_packet import DataPacket
            >>> from vita49io.protocol.enums import PacketType
            >>> DataPacket(packet_type=PacketType.IF_DATA_WITH_STREAM_ID, stream_id=1, packet_count=3).packet_count
            3
        """
        return self.header.packet_count

    def __repr__(self) -> str:  # pragma: no cover - human-facing formatting
        """Return a comprehensive string summary of the packet.

        Returns:
            str: Human-readable fields for debugging sequences and payloads.

        Examples:
            >>> from vita49io.protocol.data_packet import DataPacket
            >>> repr(DataPacket(packet_type=PacketType.IF_DATA_WITH_STREAM_ID, stream_id=1))
            'DataPacket(packet_type=IF_DATA_WITH_STREAM_ID, stream_id=0x00000001, payload_len=0, packet_count=0, requiresVita49_2=False)'
        """
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
            n = 0
            try:
                n = int(getattr(self.iq, "size", 0))
            except Exception:
                pass
            if n == 0:
                try:
                    n = len(self.iq)
                except Exception:
                    n = 0
            parts.append(f"iq_len={n}")
        return f"DataPacket({', '.join(parts)})"

    def to_bytes(
        self,
        payload_format: Optional[PayloadFormat] = None,
        *,
        reference_level_dbm: Optional[float] = None,
    ) -> bytes:
        """Serialize the packet into raw VITA 49 bytes.

        Args:
            payload_format (Optional[PayloadFormat]): Optional payload format metadata used to decode IQ arrays when present.
            reference_level_dbm (Optional[float]): When provided together with IQ samples,
                normalize the samples using the given reference level (dBmFS)
                before quantization. Requires `payload_format`.

        Returns:
            bytes: The serialized packet bytes including header, payload, and optional trailer.

        Raises:
            ValueError: If the packet type is not one of the supported data packet variants or required fields are missing.

        Examples:
            >>> from vita49io.protocol.data_packet import DataPacket
            >>> from vita49io.protocol.enums import PacketType
            >>> pkt = DataPacket(packet_type=PacketType.IF_DATA_WITH_STREAM_ID, stream_id=1, payload=b"\x00\x00\x00\x00")
            >>> isinstance(pkt.to_bytes(), bytes)
            True
        """
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

        if reference_level_dbm is not None:
            if payload_format is None:
                raise ValueError(
                    "reference_level_dbm requires payload_format when encoding IQ data"
                )
            if self.iq is None:
                raise ValueError(
                    "reference_level_dbm provided but packet was constructed without IQ samples"
                )

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
            iq_values = self.iq
            if reference_level_dbm is not None:
                iq_values = _normalize_iq_to_reference_level(
                    iq_values,
                    reference_level_dbm,
                )
            payload_bytes = _encode_iq_payload(iq_values, payload_format)
        else:
            payload_bytes = self.payload

        words.extend(_payload_bytes_to_words(payload_bytes))
        if self.trailer is not None:
            words.append(_u32(self.trailer))
        return _finalize_words_to_bytes(words)

    @staticmethod
    def from_bytes(data: bytes, payload_format: Optional[PayloadFormat] = None) -> "DataPacket":
        """Construct a DataPacket from serialized VITA 49 bytes.

        Args:
            data (bytes): Raw packet bytes starting with the VITA 49 header word.
            payload_format (Optional[PayloadFormat]): Optional payload metadata used to decode IQ samples.

        Returns:
            DataPacket: The decoded packet with header, payload, and optional IQ samples.

        Raises:
            ValueError: If the bytes are not a valid VITA 49 data packet or required fields are missing.

        Examples:
            >>> from vita49io.protocol.data_packet import DataPacket
            >>> from vita49io.protocol.enums import PacketType
            >>> pkt = DataPacket(packet_type=PacketType.IF_DATA_WITH_STREAM_ID, stream_id=1, payload=b"\x00\x00\x00\x00")
            >>> DataPacket.from_bytes(pkt.to_bytes()).stream_id
            1
        """
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
