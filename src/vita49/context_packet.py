from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Tuple

from .cif0 import CIF0Fields
from .core import (
    Header,
    _Common,
    _finalize_words_to_bytes,
    _pack_common_prefix,
    _parse_common_from_words,
    _payload_bytes_to_words,
    _unpack_u32_be,
    _u32,
)
from .enums import PacketType, TSI, TSF
from .vrt_types import ClassID


@dataclass(init=False)
class ContextPacket:
    header: Header
    stream_id: Optional[int] = None
    class_id: Optional[ClassID] = None
    integer_seconds: Optional[int] = None
    fractional_seconds: Optional[int] = None
    # Required structured CIF0 describing context information.
    cif0: CIF0Fields
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

    def pack(self) -> bytes:
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
        
        cif0_words: List[int] = _payload_bytes_to_words(self.cif0.pack())
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
    def parse(data: bytes) -> "ContextPacket":
        if len(data) < 4 or len(data) % 4 != 0:
            raise ValueError("Invalid VRT packet length")
        words = [_unpack_u32_be(data[i : i + 4]) for i in range(0, len(data), 4)]
        common, idx, end_idx = _parse_common_from_words(words)
        header = common.header
        if header.packet_type is not PacketType.CONTEXT_PACKET:
            raise ValueError("Not a Context packet type")

        # Work with payload as words directly to avoid redundant conversions.
        p_words = words[idx:end_idx]

        # Best-effort parse of CIF masks and CIF0 fields; capture remaining raw CIF fields.
        parsed_cif0: Optional[CIF0Fields] = None
        extra_masks: List[Tuple[int, int]] = []
        raw_cif_fields: Optional[List[int]] = None

        cif0_mask = p_words[0] & 0xFFFFFFFF
        w_idx = 1
        # Collect any additional CIF mask words (CIF1..CIF6)
        for i in range(1, 7):
            if (cif0_mask >> i) & 1:
                if w_idx >= len(p_words):
                    # Do not raise; just stop collecting if truncated
                    break
                extra_masks.append((i, p_words[w_idx] & 0xFFFFFFFF))
                w_idx += 1

        # Parse CIF0 fields from remaining words
        parsed_cif0, used_cif0_words = CIF0Fields.parse_from_mask(cif0_mask, p_words[w_idx:])

        
        raw_cif_fields = p_words[w_idx + used_cif0_words :]

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
