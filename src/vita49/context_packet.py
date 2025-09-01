from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .cif0 import CIF0Fields
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


@dataclass
class ContextPacket:
    packet_type: PacketType
    stream_id: Optional[int] = None
    class_id: Optional[ClassID] = None
    tsi: TSI = TSI.NONE
    tsf: TSF = TSF.NONE
    integer_seconds: Optional[int] = None
    fractional_seconds: Optional[int] = None
    payload: bytes = b""
    # Optional structured CIF0. If provided when packing, it takes precedence
    # over the raw `payload` field and will be encoded as the payload.
    cif0: Optional[CIF0Fields] = None
    trailer: Optional[int] = None
    packet_count: int = 0

    def __repr__(self) -> str:  # pragma: no cover - human-facing formatting
        def _hex32(v: int) -> str:
            return f"0x{v & 0xFFFFFFFF:08X}"

        parts = [f"packet_type={self.packet_type.name}"]
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
        # Payload and CIF0 summary
        parts.append(f"payload_len={len(self.payload)}")
        if self.cif0 is not None:
            parts.append("cif0=True")
        if self.trailer is not None:
            parts.append(f"trailer={_hex32(self.trailer)}")
        parts.append(f"packet_count={self.packet_count}")
        return f"ContextPacket({', '.join(parts)})"

    def pack(self) -> bytes:
        if self.packet_type is not PacketType.CONTEXT_PACKET:
            raise ValueError("ContextPacket must have CONTEXT_PACKET packet_type")

        # Build common prefix via _Common helper (stream_id ignored for context)
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
        words = _pack_common_prefix(common)

        payload_bytes = self.cif0.pack() if self.cif0 is not None else self.payload
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

        payload = _payload_words_to_bytes(words[idx:end_idx])
        trailer = common.trailer

        return ContextPacket(
            packet_type=header.packet_type,
            stream_id=None,
            class_id=common.class_id,
            tsi=header.tsi,
            tsf=header.tsf,
            integer_seconds=common.integer_seconds,
            fractional_seconds=common.fractional_seconds,
            payload=payload,
            cif0=None,
            trailer=trailer,
            packet_count=header.packet_count,
        )

__all__ = ["ContextPacket"]


