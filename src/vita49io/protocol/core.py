"""Implement low-level helpers for VITA 49 header and payload encoding.

Examples:
    >>> from vita49io.protocol.core import Header
    >>> from vita49io.protocol.enums import PacketType, TSI, TSF
    >>> Header(packet_type=PacketType.CONTEXT_PACKET, tsi=TSI.UTC, tsf=TSF.FRACTIONAL).packet_type
    <PacketType.CONTEXT_PACKET: 4>
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .enums import PacketType, TSI, TSF
from .vrt_types import ClassID


# Header bit masks (based on VITA 49.x)
_HDR_PACKET_TYPE_MASK = 0xF0000000
_HDR_C_MASK = 0x08000000  # Class ID present
# Packet Specific Indicators (PSI) are 3 bits (26-24). Treat bits 26, 25 and 24 uniformly.
_HDR_PSI_MASK = 0x07000000  # Bits 26-24
_HDR_TSI_MASK = 0x00C00000  # 2 bits
_HDR_TSF_MASK = 0x00300000  # 2 bits
_HDR_PKT_CNT_MASK = 0x000F0000  # 4 bits
_HDR_PKT_SIZE_MASK = 0x0000FFFF  # 16 bits (32-bit words)


def _u32(v: int) -> int:
    return v & 0xFFFFFFFF


def _pack_u32_le(v: int) -> bytes:
    return _u32(v).to_bytes(4, byteorder="big")


def _unpack_u32_be(b: bytes) -> int:
    return int.from_bytes(b, byteorder="big")


def _payload_bytes_to_words(payload: bytes) -> List[int]:
    data = payload or b""
    if len(data) % 4 != 0:
        data += b"\x00" * (4 - (len(data) % 4))
    return [_unpack_u32_be(data[i : i + 4]) for i in range(0, len(data), 4)]


def _payload_words_to_bytes(words: List[int]) -> bytes:
    b = bytearray()
    for w in words:
        b += _pack_u32_le(w)
    return bytes(b)


@dataclass
class Header:
    """Represent the 32-bit VITA 49 header word.

    Args:
        packet_type (PacketType): Packet type value encoded into the header.
        class_id_present (bool): Indicates the presence of Class ID fields.
        indicators_26 (bool): Packet Specific Indicator bit 26 flag.
        indicators_25 (bool): Packet Specific Indicator bit 25 flag.
        indicators_24 (bool): Packet Specific Indicator bit 24 flag.
        tsi (TSI): Timestamp Integer mode selection.
        tsf (TSF): Timestamp Fractional mode selection.
        packet_count (int): Rolling 4-bit sequence counter.
        packet_size (int): Packet size in 32-bit words including the header.

    Examples:
        >>> from vita49io.protocol.core import Header
        >>> from vita49io.protocol.enums import PacketType
        >>> Header(packet_type=PacketType.CONTEXT_PACKET).packet_size
        0
    """
    packet_type: PacketType
    class_id_present: bool = False
    # Packet Specific Indicator bits (26, 25, 24)
    indicators_26: bool = False  # If true, data packet has trailer, no meaning for context packet
    indicators_25: bool = False  # If true, this is a V49.2 packet and not a valid V49.0 Packet
    indicators_24: bool = False  # Data packet: False - Time Data, True - Spectrum Data; Context packet (Timestamp Mode (TSM) bit): False - precise timing, True - general timing 
    tsi: TSI = TSI.NONE
    tsf: TSF = TSF.NONE
    packet_count: int = 0  # 4 bits
    packet_size: int = 0  # total words

    def pack(self) -> int:
        """Serialize the header fields into a 32-bit word.

        Returns:
            int: The packed 32-bit header word encoded in big-endian bit order.

        Raises:
            ValueError: If `packet_count` is outside 0..15 or `packet_size` outside 0..65535.

        Examples:
            >>> from vita49io.protocol.core import Header
            >>> from vita49io.protocol.enums import PacketType
            >>> Header(packet_type=PacketType.CONTEXT_PACKET).pack() >> 28
            4
        """
        if not (0 <= self.packet_count <= 0xF):
            raise ValueError("packet_count must be in 0..15")
        if not (0 <= self.packet_size <= 0xFFFF):
            raise ValueError("packet_size must be in 0..65535 (words)")

        w0 = 0
        w0 |= (int(self.packet_type) & 0xF) << 28
        if self.class_id_present:
            w0 |= _HDR_C_MASK
        # Packet Specific Indicators (bits 26-24) packed uniformly
        psi = (
            (1 if self.indicators_26 else 0) << 2
            | (1 if self.indicators_25 else 0) << 1
            | (1 if self.indicators_24 else 0)
        )
        w0 |= (psi & 0x7) << 24
        w0 |= (int(self.tsi) & 0x3) << 22
        w0 |= (int(self.tsf) & 0x3) << 20
        w0 |= (self.packet_count & 0xF) << 16
        w0 |= (self.packet_size & 0xFFFF)
        return _u32(w0)

    def __repr__(self) -> str:  # pragma: no cover - human-facing formatting
        """Return a debug-friendly string representation of the header.

        Returns:
            str: A descriptive string detailing header fields.

        Examples:
            >>> from vita49io.protocol.core import Header
            >>> from vita49io.protocol.enums import PacketType
            >>> repr(Header(packet_type=PacketType.CONTEXT_PACKET))  # doctest: +NORMALIZE_WHITESPACE
            'Header(packet_type=CONTEXT_PACKET, class_id_present=False, packet_count=0, packet_size=0)'
        """
        parts: list[str] = []
        parts.append(f"packet_type={self.packet_type.name}")
        parts.append(f"class_id_present={self.class_id_present}")
        if self.indicators_26:
            parts.append("indicators_26=True")
        if self.indicators_25:
            parts.append("indicators_25=True")
        if self.indicators_24:
            parts.append("indicators_24=True")
        if self.tsi != TSI.NONE:
            parts.append(f"tsi={self.tsi.name}")
        if self.tsf != TSF.NONE:
            parts.append(f"tsf={self.tsf.name}")
        parts.append(f"packet_count={self.packet_count}")
        parts.append(f"packet_size={self.packet_size}")
        return f"Header({', '.join(parts)})"

    @staticmethod
    def parse(w0: int) -> "Header":
        """Build a Header instance by parsing a 32-bit word.

        Args:
            w0 (int): The 32-bit header word encoded in big-endian bit order.

        Returns:
            Header: A populated header instance reflecting the encoded values.

        Raises:
            ValueError: If the header bits map to an unknown PacketType, TSI, or TSF value.

        Examples:
            >>> from vita49io.protocol.core import Header
            >>> Header.parse(0x40000000).packet_type
            <PacketType.CONTEXT_PACKET: 4>
        """
        pkt_type = PacketType((w0 & _HDR_PACKET_TYPE_MASK) >> 28)
        c_present = bool(w0 & _HDR_C_MASK)
        # Parse Packet Specific Indicators uniformly from bits 26-24
        psi_bits = (w0 & _HDR_PSI_MASK) >> 24
        ind26 = bool(psi_bits & 0b100)
        ind25 = bool(psi_bits & 0b010)
        ind24 = bool(psi_bits & 0b001)
        tsi = TSI((w0 & _HDR_TSI_MASK) >> 22)
        tsf = TSF((w0 & _HDR_TSF_MASK) >> 20)
        pkt_cnt = (w0 & _HDR_PKT_CNT_MASK) >> 16
        pkt_size_words = w0 & _HDR_PKT_SIZE_MASK
        return Header(
            packet_type=pkt_type,
            class_id_present=c_present,
            indicators_26=ind26,
            indicators_25=ind25,
            indicators_24=ind24,
            tsi=tsi,
            tsf=tsf,
            packet_count=int(pkt_cnt),
            packet_size=int(pkt_size_words),
        )


@dataclass
class _Common:
    header: Header
    stream_id: Optional[int]
    class_id: Optional[ClassID]
    integer_seconds: Optional[int]
    fractional_seconds: Optional[int]


def _pack_common_prefix(c: _Common) -> List[int]:
    words: List[int] = []
    # Build header word using provided header with presence flags synced
    hdr = Header(
        packet_type=c.header.packet_type,
        class_id_present=c.class_id is not None,
        indicators_26=c.header.indicators_26,
        indicators_25=c.header.indicators_25,
        indicators_24=c.header.indicators_24,
        tsi=c.header.tsi,
        tsf=c.header.tsf,
        packet_count=c.header.packet_count,
        packet_size=0,
    )
    words.append(hdr.pack())

    # Stream ID presence driven by packet type
    if c.header.packet_type in (
        PacketType.IF_DATA_WITH_STREAM_ID,
        PacketType.EXTENSION_DATA_WITH_STREAM_ID,
        PacketType.CONTEXT_PACKET,  # Context packet requires Stream ID in this model
    ):
        if c.stream_id is None:
            raise ValueError("Packet type requires a Stream ID, but none provided")
        words.append(_u32(c.stream_id))

    if c.class_id is not None:
        oui, ic, pc = c.class_id
        words.append(_u32((oui & 0xFFFFFF) << 8))
        words.append(_u32(((ic & 0xFFFF) << 16) | (pc & 0xFFFF)))

    if c.header.tsi != TSI.NONE:
        if c.integer_seconds is None:
            raise ValueError("TSI set but integer_seconds is None")
        words.append(_u32(c.integer_seconds))

    if c.header.tsf != TSF.NONE:
        if c.fractional_seconds is None:
            raise ValueError("TSF set but fractional_seconds is None")
        fs = int(c.fractional_seconds) & 0xFFFFFFFFFFFFFFFF
        words.append(_u32((fs >> 32) & 0xFFFFFFFF))
        words.append(_u32(fs & 0xFFFFFFFF))

    return words


def _finalize_words_to_bytes(words: List[int]) -> bytes:
    # Update size in-place while preserving other bits
    words[0] = (words[0] & ~_HDR_PKT_SIZE_MASK) | (len(words) & _HDR_PKT_SIZE_MASK)
    out = bytearray()
    for w in words:
        out += _pack_u32_le(w)
    return bytes(out)


def _parse_common_from_words(words: List[int]) -> tuple[_Common, int, int]:
    if not words:
        raise ValueError("Empty words for VRT packet")
    hdr = Header.parse(words[0])
    pkt_type = hdr.packet_type
    c_present = hdr.class_id_present
    tsi = hdr.tsi
    tsf = hdr.tsf
    pkt_size_words = hdr.packet_size

    if pkt_size_words != len(words):
        raise ValueError("Packet size mismatch")

    idx = 1
    stream_id: Optional[int] = None
    class_id: Optional[ClassID] = None
    integer_seconds: Optional[int] = None
    fractional_seconds: Optional[int] = None

    # Stream ID presence is determined by packet type (with/without) in this model
    if pkt_type in (
        PacketType.IF_DATA_WITH_STREAM_ID,
        PacketType.EXTENSION_DATA_WITH_STREAM_ID,
        PacketType.CONTEXT_PACKET,  # Context packet requires Stream ID in this model
    ):
        if idx >= len(words):
            raise ValueError("Truncated after header: missing Stream ID")
        stream_id = words[idx]
        idx += 1

    if c_present:
        if idx + 1 >= len(words):
            raise ValueError("Truncated: missing Class ID words")
        w_a = words[idx]
        w_b = words[idx + 1]
        idx += 2
        oui = (w_a >> 8) & 0xFFFFFF
        information_class = (w_b >> 16) & 0xFFFF
        packet_class = w_b & 0xFFFF
        class_id = (oui, information_class, packet_class)

    if tsi != TSI.NONE:
        if idx >= len(words):
            raise ValueError("Truncated: missing integer seconds")
        integer_seconds = words[idx]
        idx += 1

    if tsf != TSF.NONE:
        if idx + 1 >= len(words):
            raise ValueError("Truncated: missing fractional seconds")
        msw = words[idx]
        lsw = words[idx + 1]
        fractional_seconds = ((msw & 0xFFFFFFFF) << 32) | (lsw & 0xFFFFFFFF)
        idx += 2

    end_idx = len(words)

    common = _Common(
        header=hdr,
        stream_id=stream_id,
        class_id=class_id,
        integer_seconds=integer_seconds,
        fractional_seconds=fractional_seconds,
    )
    return common, idx, end_idx


__all__ = [
    "PacketType",
    "TSI",
    "TSF",
    "ClassID",
    "_u32",
    "_pack_u32_le",
    "_unpack_u32_be",
    "_payload_bytes_to_words",
    "_payload_words_to_bytes",
    "Header",
    "_Common",
    "_pack_common_prefix",
    "_finalize_words_to_bytes",
    "_parse_common_from_words",
]

