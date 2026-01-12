"""Provide helpers for constructing and parsing VITA 49 context packets.

The class now follows a lazy, memoryview-backed design:

* ``from_bytes`` keeps a ``memoryview`` of the raw packet and defers decoding.
* ``to_bytes`` returns the cached bytes quickly when the packet was not mutated.
* Optional/large blocks (CIF) are parsed only when accessed.

Examples:
    >>> from vita49io.protocol.context_packet import ContextPacket
    >>> from vita49io.protocol.cif0 import CIF0Fields
    >>> from vita49io.protocol.enums import PacketType
    >>> ctx = ContextPacket(packet_type=PacketType.CONTEXT_PACKET, stream_id=1, cif0=CIF0Fields())
    >>> isinstance(ctx.to_bytes(), bytes)
    True
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import List, Tuple

from .cif0 import CIF0Fields
from .core import (
    Header,
    LazyBinary,
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
class ContextPacket(LazyBinary):
    """Represent a VITA 49 context packet including CIF0 metadata.

    Attributes:
        _header (Header | None): Parsed or user-provided header.
        _stream_id (int | None): Stream identifier if present.
        _class_id (ClassID | None): Class identifier if present.
        _integer_seconds (int | None): Integer timestamp if present.
        _fractional_seconds (int | None): Fractional timestamp if present.
        _cif0 (CIF0Fields | None): CIF0 metadata block if present.
    """

    _header: Header | None = None
    _stream_id: int | None = None
    _class_id: ClassID | None = None
    _integer_seconds: int | None = None
    _fractional_seconds: int | None = None
    _cif0: CIF0Fields | None = None

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
        cif0: CIF0Fields | None = None,
        # If true, set header.indicators_25 (V49.2-only packet)
        requiresVita49_2: bool = False,
        # If true, set header.indicators_24 (Timestamp Mode bit / TSM)
        timestamp_mode: bool = False,
        _mv: memoryview | None = None,
    ) -> None:
        """Initialize a ContextPacket for user construction or memoryview-backed parsing.

        Args:
            header (Header | None): Pre-built header; required if `packet_type` is omitted.
            packet_type (PacketType | None): Convenience for creating a header in-place.
            tsi (TSI): Timestamp integer selection (if building a new header).
            tsf (TSF): Timestamp fractional selection (if building a new header).
            packet_count (int): Rolling 4-bit sequence counter (if building a new header).
            stream_id (int | None): Stream identifier (required for context packets).
            class_id (ClassID | None): Optional class identifier when `header.class_id_present`.
            integer_seconds (int | None): Integer seconds timestamp (if present).
            fractional_seconds (int | None): Fractional seconds timestamp (if present).
            cif0 (CIF0Fields | None): CIF0 context metadata block.
            requiresVita49_2 (bool): Sets header indicator 25 when building a new header.
            timestamp_mode (bool): Sets header indicator 24 (timestamp mode/TSM).
            _mv (memoryview | None): Internal raw-bytes backing; set by `from_bytes()`.

        Notes:
            - For user-created packets, provide `header` or `packet_type`; `_mv` is internal.
            - If `_mv` is provided, fields are decoded lazily and `to_bytes()` fast-paths
              to the original bytes unless mutated.

        Examples:
            >>> from vita49io.protocol.context_packet import ContextPacket
            >>> from vita49io.protocol.enums import PacketType
            >>> pkt = ContextPacket(packet_type=PacketType.CONTEXT_PACKET, stream_id=1)
            >>> pkt.stream_id
            1
        """
        # Call base __init__ directly to avoid dataclass/super slot quirks
        LazyBinary.__init__(self, _mv=_mv)
        if header is None and packet_type is not None:
            header = Header(
                packet_type=packet_type,
                class_id_present=(class_id is not None),
                indicators_25=bool(requiresVita49_2),
                indicators_24=bool(timestamp_mode),
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
        self._cif0 = cif0
        # Newly built objects must be encoded; memoryview-backed packets can fast-path
        if _mv is None:
            self._mark_dirty()

    # ---------- lazy common prefix ----------
    def _common_info(self) -> Tuple[_Common, int, int]:
        def decode(self, mv: memoryview) -> Tuple[_Common, int, int]:
            common, payload_start, payload_end = _parse_common_from_bytes(mv)
            if common.header.packet_type is not PacketType.CONTEXT_PACKET:
                raise ValueError("Not a Context packet type")
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

    # Convenience accessors expected by tests/users
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

    # ---------- CIF handling ----------
    def _decode_cif0(self) -> CIF0Fields:
        if self._cif0 is not None:
            return self._cif0
        if self._mv is None:
            raise ValueError("Cannot decode CIF0 without backing bytes")
        _, payload_start, payload_end = self._common_info()
        payload = self._mv[payload_start:payload_end]
        if len(payload) < 4:
            raise ValueError("Context packet missing CIF0 mask word")
        cif0, used_cif0_bytes = CIF0Fields.parse(payload)
        self._cif0 = cif0
        return cif0

    @property
    def cif0(self) -> CIF0Fields:
        if self._cif0 is None:
            return self._decode_cif0()
        return self._cif0

    @cif0.setter
    def cif0(self, value: CIF0Fields) -> None:
        self._cif0 = value
        self._mark_dirty()

    def __repr__(self) -> str:  # pragma: no cover - human-facing formatting
        def _hex32(v: int) -> str:
            return f"0x{v & 0xFFFFFFFF:08X}"

        parts = [f"packet_type={self.header.packet_type.name}"]
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
        parts.append(f"cif0={self._cif0 if self._cif0 is not None else 'lazy'}")
        parts.append(f"packet_count={self.header.packet_count}")
        if self.header.indicators_25:
            parts.append("indicators_25=True")
        if self.header.indicators_24:
            parts.append("indicators_24=True")
        return f"ContextPacket({', '.join(parts)})"

    def to_bytes(self) -> bytes:
        """Serialize the context packet into bytes, with zero-copy fast-path."""
        if not self._dirty and self._mv is not None:
            return self._mv.tobytes()

        hdr = self.header
        if hdr.packet_type is not PacketType.CONTEXT_PACKET:
            raise ValueError("ContextPacket must have CONTEXT_PACKET packet_type")
        if self.stream_id is None:
            raise ValueError("ContextPacket requires a Stream ID")

        common = _Common(
            header=hdr,
            stream_id=self.stream_id,
            class_id=self.class_id,
            integer_seconds=self.integer_seconds,
            fractional_seconds=self.fractional_seconds,
        )
        words = _pack_common_prefix(common)

        if self.cif0 is None:
            raise TypeError("cif0 is required for ContextPacket")
        cif0_words: List[int] = _payload_bytes_to_words(self.cif0.pack())

        words.extend(cif0_words)

        raw_bytes = _finalize_words_to_bytes(words)
        self._mv = memoryview(raw_bytes)
        self._dirty = False
        return raw_bytes

    @classmethod
    def from_bytes(cls, data: bytes | bytearray | memoryview) -> "ContextPacket":
        """Keep a memoryview of the packet and decode lazily."""
        mv = data if isinstance(data, memoryview) else memoryview(data)
        return cls(_mv=mv)


__all__ = ["ContextPacket"]
