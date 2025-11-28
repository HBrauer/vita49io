"""Provide helpers for constructing and parsing VITA 49 context packets.

Examples:
    >>> from vita49io.protocol.context_packet import ContextPacket
    >>> from vita49io.protocol.cif0 import CIF0Fields
    >>> ContextPacket(packet_type=PacketType.CONTEXT_PACKET, stream_id=1, cif0=CIF0Fields()).cif0 is not None
    True
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Tuple

from .cif0 import CIF0Fields
from .core import (
    Header,
    _Common,
    _finalize_words_to_bytes,
    _pack_common_prefix,
    _parse_common_from_bytes,
    _payload_bytes_to_words,
    _u32,
)
from .enums import PacketType, TSI, TSF
from .vrt_types import ClassID


@dataclass(init=False)
class ContextPacket:
    """Represent a VITA 49 context packet including CIF0 metadata.

    Args:
        header (Header): Pre-built header describing the packet layout.
        cif0 (CIF0Fields): Mandatory CIF0 field structure describing context.
        stream_id (Optional[int]): Stream identifier if the header indicates one.
        class_id (Optional[ClassID]): Class identifier tuple attached to the packet.
        integer_seconds (Optional[int]): Integer seconds timestamp component when TSI is set.
        fractional_seconds (Optional[int]): Fractional seconds timestamp component when TSF is set.
        cif_extra_masks (Optional[List[Tuple[int, int]]]): Additional CIF mask words beyond CIF0.
        raw_cif_fields (Optional[List[int]]): Raw CIF field words not decoded into structures.

    Examples:
        >>> from vita49io.protocol.context_packet import ContextPacket
        >>> from vita49io.protocol.cif0 import CIF0Fields
        >>> from vita49io.protocol.enums import PacketType
        >>> stream_id = 0x12345678
        >>> pf = PayloadFormat(
            packing_method=PackingMethod.PROCESSING_EFFICIENT,
            sample_type=SampleType.COMPLEX_CARTESIAN,
            data_item_format_code=int(DataItemFormat.IEEE754_SINGLE),
            data_item_format=DataItemFormat.IEEE754_SINGLE,
            sample_component_repeat=False,
            event_tag_size_bits=0,
            channel_tag_size_bits=0,
            data_item_fraction_size_bits=0,
            item_packing_field_size_bits=32,
            data_item_size_bits=32,
            repeat_count=1,
            vector_size=0,
        )
        >>> cif0 = CIF0Fields(
            sample_rate_hz=1_000_000.0,
            payload_format=pf,
        )
        >>> ctx = ContextPacket(
            packet_type=PacketType.CONTEXT_PACKET,
            stream_id=stream_id,
            tsi=TSI.UTC,
            tsf=TSF.FRACTIONAL,
            integer_seconds=1_700_000_000,
            fractional_seconds=0,
            cif0=cif0,
        )
        >>> ctx_bytes = ctx.to_bytes()
        >>> ctx_same = ContextPacket.from_bytes(ctx_bytes)
        
    """
    header: Header
    # Required structured CIF0 describing context information. Set in __init__.
    cif0: Optional[CIF0Fields] = None
    stream_id: Optional[int] = None
    class_id: Optional[ClassID] = None
    integer_seconds: Optional[int] = None
    fractional_seconds: Optional[int] = None
    # Optional list of additional CIF masks found after CIF0 mask.
    # Each entry is (cif_index, mask) for CIF1..CIF6 when present.
    cif_extra_masks: Optional[List[Tuple[int, int]]] = None
    # Raw CIF field words (after CIF0 fields), as 32-bit words
    raw_cif_fields: Optional[List[int]] = None

    def __init__(
        self,
        *,
        header: Optional[Header] = None,
        packet_type: Optional[PacketType] = None,
        tsi: TSI = TSI.NONE,
        tsf: TSF = TSF.NONE,
        packet_count: int = 0,
        stream_id: Optional[int] = None,
        class_id: Optional[ClassID] = None,
        integer_seconds: Optional[int] = None,
        fractional_seconds: Optional[int] = None,
        cif0: CIF0Fields,
        cif_extra_masks: Optional[List[Tuple[int, int]]] = None,
        raw_cif_fields: Optional[List[int]] = None,
        # If true, set header.indicators_25 (V49.2-only packet)
        requiresVita49_2: bool = False,
        # If true, set header.indicators_24 (Timestamp Mode bit / TSM)
        timestamp_mode: bool = False,
    ) -> None:
        if header is None:
            if packet_type is None:
                raise TypeError("Either header or packet_type must be provided")
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
        self.header = header
        self.stream_id = stream_id
        self.class_id = class_id
        self.integer_seconds = integer_seconds
        self.fractional_seconds = fractional_seconds
        if cif0 is None:
            raise TypeError("cif0 is required for ContextPacket")
        self.cif0 = cif0
        self.cif_extra_masks = cif_extra_masks
        self.raw_cif_fields = raw_cif_fields 


    # Convenience accessors expected by tests/users
    @property
    def packet_type(self) -> PacketType:
        """Return the packet type reported by the context packet header.

        Returns:
            PacketType: The header packet type enumeration.

        Examples:
            >>> from vita49io.protocol.context_packet import ContextPacket
            >>> from vita49io.protocol.cif0 import CIF0Fields
            >>> from vita49io.protocol.enums import PacketType
            >>> ContextPacket(packet_type=PacketType.CONTEXT_PACKET, stream_id=1, cif0=CIF0Fields()).packet_type
            <PacketType.CONTEXT_PACKET: 4>
        """
        return self.header.packet_type

    @property
    def tsi(self) -> TSI:
        """Return the Timestamp Integer (TSI) selection stored in the header.

        Returns:
            TSI: The integer timestamp mode determining the presence of integer seconds.

        Examples:
            >>> from vita49io.protocol.context_packet import ContextPacket
            >>> from vita49io.protocol.cif0 import CIF0Fields
            >>> from vita49io.protocol.enums import PacketType, TSI
            >>> ContextPacket(packet_type=PacketType.CONTEXT_PACKET, stream_id=1, cif0=CIF0Fields(), tsi=TSI.UTC).tsi
            <TSI.UTC: 1>
        """
        return self.header.tsi

    @property
    def tsf(self) -> TSF:
        """Return the Timestamp Fractional (TSF) selection stored in the header.

        Returns:
            TSF: The fractional timestamp mode controlling fractional second interpretation.

        Examples:
            >>> from vita49io.protocol.context_packet import ContextPacket
            >>> from vita49io.protocol.cif0 import CIF0Fields
            >>> from vita49io.protocol.enums import PacketType, TSF
            >>> ContextPacket(packet_type=PacketType.CONTEXT_PACKET, stream_id=1, cif0=CIF0Fields(), tsf=TSF.FRACTIONAL).tsf
            <TSF.FRACTIONAL: 2>
        """
        return self.header.tsf

    @property
    def packet_count(self) -> int:
        """Return the rolling packet count from the context packet header.

        Returns:
            int: The 4-bit packet count sequence number.

        Examples:
            >>> from vita49io.protocol.context_packet import ContextPacket
            >>> from vita49io.protocol.cif0 import CIF0Fields
            >>> from vita49io.protocol.enums import PacketType
            >>> ContextPacket(packet_type=PacketType.CONTEXT_PACKET, stream_id=1, cif0=CIF0Fields(), packet_count=5).packet_count
            5
        """
        return self.header.packet_count

    def __repr__(self) -> str:  # pragma: no cover - human-facing formatting
        """Generate a detailed string for debugging context packets.

        Returns:
            str: A formatted string enumerating key header and CIF details.

        Examples:
            >>> from vita49io.protocol.context_packet import ContextPacket
            >>> from vita49io.protocol.cif0 import CIF0Fields
            >>> from vita49io.protocol.enums import PacketType
            >>> 'ContextPacket(' in repr(ContextPacket(packet_type=PacketType.CONTEXT_PACKET, stream_id=1, cif0=CIF0Fields()))
            True
        """
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
        # CIF summary
        parts.append(f"cif0={self.cif0}")
        if self.cif_extra_masks:
            masks_summ = ", ".join(f"CIF{i}:{m & 0xFFFFFFFF:#010x}" for i, m in self.cif_extra_masks)
            parts.append(f"extra_masks=[{masks_summ}]")
        parts.append(f"packet_count={self.header.packet_count}")
        # Indicator bits (for debugging)
        if self.header.indicators_25:
            parts.append("indicators_25=True")
        if self.header.indicators_24:
            parts.append("indicators_24=True")
        return f"ContextPacket({', '.join(parts)})"

    def to_bytes(self) -> bytes:
        """Serialize the context packet into bytes.

        Returns:
            bytes: The serialized packet including header and CIF fields.

        Raises:
            ValueError: If required fields such as packet type or Stream ID are inconsistent.

        Examples:
            >>> from vita49io.protocol.context_packet import ContextPacket
            >>> from vita49io.protocol.cif0 import CIF0Fields
            >>> from vita49io.protocol.enums import PacketType
            >>> pkt = ContextPacket(packet_type=PacketType.CONTEXT_PACKET, stream_id=1, cif0=CIF0Fields())
            >>> isinstance(pkt.to_bytes(), bytes)
            True
        """
        if self.header.packet_type is not PacketType.CONTEXT_PACKET:
            raise ValueError("ContextPacket must have CONTEXT_PACKET packet_type")
        if self.stream_id is None:
            raise ValueError("ContextPacket requires a Stream ID")

        # Build common prefix via _Common helper (stream_id required for context)
        common = _Common(
            header=self.header,
            stream_id=self.stream_id,
            class_id=self.class_id,
            integer_seconds=self.integer_seconds,
            fractional_seconds=self.fractional_seconds,
        )
        words = _pack_common_prefix(common)

        # Build payload words: combined CIF0 mask, extra masks, CIF0 fields, then raw CIF fields
        if self.cif0 is None:
            raise TypeError("cif0 is required for ContextPacket")
        cif0 = self.cif0
        cif0_words: List[int] = _payload_bytes_to_words(cif0.pack())
        cif0_mask = cif0_words[0] & 0xFFFFFFFF
        if self.cif_extra_masks:
            for i, _m in self.cif_extra_masks:
                cif0_mask |= (1 << i)
        words.append(_u32(cif0_mask))
        for _i, m in (self.cif_extra_masks or []):
            words.append(_u32(m))
        words.extend(cif0_words[1:])
        if self.raw_cif_fields:
            words.extend(self.raw_cif_fields)

        return _finalize_words_to_bytes(words)

    @staticmethod
    def from_bytes(data: bytes) -> "ContextPacket":
        """Parse a ContextPacket instance from serialized bytes.

        Args:
            data (bytes): Raw VITA 49 packet bytes beginning at the header.

        Returns:
            ContextPacket: A decoded context packet with parsed CIF0 fields.

        Raises:
            ValueError: If the bytes do not contain a valid context packet structure.

        Examples:
            >>> from vita49io.protocol.context_packet import ContextPacket
            >>> from vita49io.protocol.cif0 import CIF0Fields
            >>> from vita49io.protocol.enums import PacketType
            >>> pkt = ContextPacket(packet_type=PacketType.CONTEXT_PACKET, stream_id=1, cif0=CIF0Fields())
            >>> ContextPacket.from_bytes(pkt.to_bytes()).stream_id
            1
        """
        mv = memoryview(data)
        common, payload_start, payload_end = _parse_common_from_bytes(mv)
        header = common.header
        if header.packet_type is not PacketType.CONTEXT_PACKET:
            raise ValueError("Not a Context packet type")

        payload_mv = mv[payload_start:payload_end]
        if len(payload_mv) < 4:
            raise ValueError("Context packet missing CIF0 mask word")

        parsed_cif0: Optional[CIF0Fields] = None
        extra_masks: List[Tuple[int, int]] = []
        raw_cif_fields: Optional[List[int]] = None

        cif0_mask = int.from_bytes(payload_mv[0:4], byteorder="big") & 0xFFFFFFFF
        pos = 4
        for i in range(1, 7):
            if (cif0_mask >> i) & 1:
                if pos + 4 > len(payload_mv):
                    break
                extra_masks.append((i, int.from_bytes(payload_mv[pos : pos + 4], byteorder="big")))
                pos += 4

        remaining = payload_mv[pos:]
        remaining_words = [
            int.from_bytes(remaining[i : i + 4], byteorder="big")
            for i in range(0, len(remaining), 4)
        ]

        parsed_cif0, used_cif0_words = CIF0Fields.parse_from_mask(cif0_mask, remaining_words)
        raw_cif_fields = remaining_words[used_cif0_words:]

        return ContextPacket(
            header=header,
            stream_id=common.stream_id,
            class_id=common.class_id,
            integer_seconds=common.integer_seconds,
            fractional_seconds=common.fractional_seconds,
            cif0=parsed_cif0,
            cif_extra_masks=extra_masks if extra_masks else None,
            raw_cif_fields=raw_cif_fields,
        )

__all__ = ["ContextPacket"]
