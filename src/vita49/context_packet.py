from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .cif0 import CIF0Fields
from .cif1 import CIF1Fields
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
    # Optional structured CIF0. If provided when packing, it takes precedence
    # over the raw `payload` field and will be encoded as the payload.
    cif0: Optional[CIF0Fields] = None
    # Optional parsed CIF1 (same format as CIF0). Values are currently ignored
    # by the library; this is provided to indicate presence and raw capture.
    cif1: Optional[CIF1Fields] = None
    trailer: Optional[int] = None

    def __init__(
        self,
        *,
        header: Optional[Header] = None,
        packet_type: Optional[PacketType] = None,
        packet_specific_indicators: int = 0,
        tsi: TSI = TSI.NONE,
        tsf: TSF = TSF.NONE,
        packet_count: int = 0,
        stream_id: Optional[int] = None,
        class_id: Optional[ClassID] = None,
        integer_seconds: Optional[int] = None,
        fractional_seconds: Optional[int] = None,
        cif0: Optional[CIF0Fields] = None,
        cif1: Optional[CIF1Fields] = None,
        trailer: Optional[int] = None,
        psi: Optional[int] = None,
    ) -> None:
        if header is None:
            if packet_type is None:
                raise TypeError("Either header or packet_type must be provided")
            if psi is not None:
                packet_specific_indicators = int(psi)
            header = Header(
                packet_type=packet_type,
                class_id_present=(class_id is not None),
                trailer_present=(trailer is not None),
                packet_specific_indicators=int(packet_specific_indicators),
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
        self.cif0 = cif0
        self.cif1 = cif1
        self.trailer = trailer

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
        # CIF0/CIF1 summary
        if self.cif0 is not None:
            parts.append(f"cif0={self.cif0}")
        if self.cif1 is not None:
            parts.append("cif1=True")
        if self.trailer is not None:
            parts.append(f"trailer={_hex32(self.trailer)}")
        parts.append(f"packet_count={self.header.packet_count}")
        return f"ContextPacket({', '.join(parts)})"

    def pack(self) -> bytes:
        if self.header.packet_type is not PacketType.CONTEXT_PACKET:
            raise ValueError("ContextPacket must have CONTEXT_PACKET packet_type")
        if self.stream_id is None:
            raise ValueError("ContextPacket requires a Stream ID")

        # Build common prefix via _Common helper (stream_id ignored for context)
        common = _Common(
            header=self.header,
            stream_id=self.stream_id,
            class_id=self.class_id,
            integer_seconds=self.integer_seconds,
            fractional_seconds=self.fractional_seconds,
            trailer=self.trailer,
        )
        words = _pack_common_prefix(common)

        # Encode payload from CIF0 when provided; otherwise no payload.
        if self.cif0 is not None:
            payload_bytes = self.cif0.pack()
        else:
            payload_bytes = b""
        words.extend(_payload_bytes_to_words(payload_bytes))
        if self.trailer is not None:
            words.append(_u32(self.trailer))
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
        trailer = common.trailer

        # Best-effort parse of CIF0/CIF1 where possible.
        parsed_cif0: Optional[CIF0Fields] = None
        parsed_cif1: Optional[CIF1Fields] = None

        if p_words:
            cif0_mask = p_words[0]
            w_idx = 1
            # Skip unsupported CIF1..CIF6 mask-only flags if present
            for i in range(1, 7):  # CIF1 - CIF6 are not supported, we just ignore them
                if (cif0_mask & (1 << i)) != 0:
                    w_idx += 1
                    print(f"Ignore CIF{i} because currently not supported")
            # Parse CIF0 fields from remaining words
            parsed_cif0, _used = CIF0Fields.parse_from_mask(cif0_mask, p_words[w_idx:])

        return ContextPacket(
            header=header,
            stream_id=common.stream_id,
            class_id=common.class_id,
            integer_seconds=common.integer_seconds,
            fractional_seconds=common.fractional_seconds,
            cif0=parsed_cif0,
            cif1=parsed_cif1,
            trailer=trailer,
        )

__all__ = ["ContextPacket"]


