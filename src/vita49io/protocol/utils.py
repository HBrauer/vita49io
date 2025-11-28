"""Shared low-level helpers for VITA 49 packing and fixed-point conversions."""

from __future__ import annotations

import struct
from typing import List, Tuple, Union


def _u32(v: int) -> int:
    return v & 0xFFFFFFFF


def _pack_u32_le(v: int) -> bytes:
    return _u32(v).to_bytes(4, byteorder="big")


def _unpack_u32_be(b: bytes) -> int:
    return int.from_bytes(b, byteorder="big")


def _payload_bytes_to_words(payload: Union[bytes, memoryview]) -> List[int]:
    if not payload:
        return []
    mv = memoryview(payload)
    if len(mv) % 4 != 0:
        data = mv.tobytes() + b"\x00" * (4 - (len(mv) % 4))
    else:
        data = mv
    return [w[0] for w in struct.iter_unpack(">I", data)]


def _payload_words_to_bytes(words: List[int]) -> bytes:
    if not words:
        return b""
    out = bytearray(len(words) * 4)
    for i, w in enumerate(words):
        struct.pack_into(">I", out, i * 4, _u32(w))
    return bytes(out)


def _to_s64_fixed20(v: float) -> Tuple[int, int]:
    scale = 1 << 20
    i = int(round(v * scale)) & ((1 << 64) - 1)
    hi = (i >> 32) & 0xFFFFFFFF
    lo = i & 0xFFFFFFFF
    return hi, lo


def _from_s64_fixed20(hi: int, lo: int) -> float:
    i = ((hi & 0xFFFFFFFF) << 32) | (lo & 0xFFFFFFFF)
    if i & (1 << 63):
        i -= 1 << 64
    return i / float(1 << 20)


def _to_u64_fixed20(v: float) -> Tuple[int, int]:
    if v < 0:
        raise ValueError("Unsigned fixed-point cannot be negative")
    scale = 1 << 20
    i = int(round(v * scale)) & ((1 << 64) - 1)
    hi = (i >> 32) & 0xFFFFFFFF
    lo = i & 0xFFFFFFFF
    return hi, lo


def _from_u64_fixed20(hi: int, lo: int) -> float:
    i = ((hi & 0xFFFFFFFF) << 32) | (lo & 0xFFFFFFFF)
    return i / float(1 << 20)


def _to_s16_fixed7(v: float) -> int:
    scale = 1 << 7
    i = int(round(v * scale))
    if i < -32768:
        i = -32768
    if i > 32767:
        i = 32767
    return i & 0xFFFF


def _from_s16_fixed7(w: int) -> float:
    v = w & 0xFFFF
    if v & 0x8000:
        v -= 0x10000
    return v / float(1 << 7)


def _encode_fixed_point(value: float, scale: int) -> int:
    raw = int(round(value * scale))
    return _u32(raw)


def _decode_fixed_point(raw: int, scale: int) -> float:
    v = raw & 0xFFFFFFFF
    if v & 0x80000000:
        v -= 0x100000000
    return v / float(scale)


__all__ = [
    "_u32",
    "_pack_u32_le",
    "_unpack_u32_be",
    "_payload_bytes_to_words",
    "_payload_words_to_bytes",
    "_to_s64_fixed20",
    "_from_s64_fixed20",
    "_to_u64_fixed20",
    "_from_u64_fixed20",
    "_to_s16_fixed7",
    "_from_s16_fixed7",
    "_encode_fixed_point",
    "_decode_fixed_point",
]
