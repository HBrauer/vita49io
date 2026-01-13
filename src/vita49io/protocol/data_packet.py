"""Implement VITA 49 data packet helpers for parsing and serializing raw payloads.

Packets follow a lazy, memoryview-backed design:

* ``from_bytes`` keeps a ``memoryview`` of the raw packet and defers decoding.
* ``to_bytes`` fast-paths to the stored bytes when the packet was not mutated.
* Payload bytes are provided raw; sample decoding is handled by helpers in
  ``vita49io.io.payload_codec``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Union

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

@dataclass(init=False, slots=True)
class DataPacket(LazyBinary):
    """Represent a VITA 49 data packet with lazy payload parsing.

    Attributes:
        _header (Header | None): Parsed or user-provided header.
        _stream_id (int | None): Stream identifier if present.
        _class_id (ClassID | None): Class identifier if present.
        _integer_seconds (int | None): Integer timestamp if present.
        _fractional_seconds (int | None): Fractional timestamp if present.
        _payload (bytes | memoryview | None): Raw on-wire payload bytes.
        _trailer (int | None): Trailer word if present.
        _copy_payload (bool): Copy payload bytes out of `_mv` on first access.

    Notes:
        - `payload` exposes the raw on-wire bytes; sample interpretation is external.
    """

    _header: Header | None = None
    _stream_id: int | None = None
    _class_id: ClassID | None = None
    _integer_seconds: int | None = None
    _fractional_seconds: int | None = None
    _payload: Union[bytes, memoryview, None] = None
    _trailer: int | None = None
    _copy_payload: bool = False

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
        requiresVita49_2: bool = False,
        _mv: memoryview | None = None,
        copy_payload: bool = False,
    ) -> None:
        """Initialize a DataPacket for user construction or memoryview-backed parsing.

        Args:
            header (Header | None): Pre-built header; required if `packet_type` is omitted.
            packet_type (PacketType | None): Convenience for creating a header in-place.
            tsi (TSI): Timestamp integer selection (if building a new header).
            tsf (TSF): Timestamp fractional selection (if building a new header).
            packet_count (int): Rolling 4-bit sequence counter (if building a new header).
            stream_id (int | None): Optional stream identifier; required/forbidden by packet type.
            class_id (ClassID | None): Optional class identifier when `header.class_id_present`.
            integer_seconds (int | None): Integer seconds timestamp (if present).
            fractional_seconds (int | None): Fractional seconds timestamp (if present).
            payload (bytes | memoryview | None): Raw on-wire payload bytes.
            trailer (int | None): Trailer word (present when header indicator 26 is set).
            requiresVita49_2 (bool): Sets header indicator 25 when building a new header.
            _mv (memoryview | None): Internal raw-bytes backing; set by `from_bytes()`.
            copy_payload (bool): If true, copy payload bytes out of `_mv` on first access.

        Notes:
            - For user-created packets, provide `header` or `packet_type`; `_mv` is internal.
            - If `_mv` is provided, fields are decoded lazily and `to_bytes()` fast-paths
              to the original bytes unless mutated.

        Examples:
            >>> from vita49io.protocol.data_packet import DataPacket
            >>> from vita49io.protocol.enums import PacketType
            >>> pkt = DataPacket(packet_type=PacketType.IF_DATA_WITHOUT_STREAM_ID, payload=b"")
            >>> pkt.packet_type == PacketType.IF_DATA_WITHOUT_STREAM_ID
            True
        """
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
        self._copy_payload = copy_payload
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
        return f"DataPacket({', '.join(parts)})"

    def to_bytes(self) -> bytes:
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

        common = _Common(
            header=hdr,
            stream_id=self.stream_id,
            class_id=self.class_id,
            integer_seconds=self.integer_seconds,
            fractional_seconds=self.fractional_seconds,
        )
        words: List[int] = _pack_common_prefix(common)

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
        *,
        copy_payload: bool = False,
    ) -> "DataPacket":
        """
        Construct a DataPacket backed by a memoryview of the raw bytes.
        """
        mv = data if isinstance(data, memoryview) else memoryview(data)
        pkt = cls(_mv=mv, copy_payload=copy_payload)
        return pkt


__all__ = ["DataPacket"]
