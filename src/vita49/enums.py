from __future__ import annotations

from enum import IntEnum


class PacketType(IntEnum):
    IF_DATA_WITHOUT_STREAM_ID = 0x0
    IF_DATA_WITH_STREAM_ID = 0x1
    EXTENSION_DATA_WITHOUT_STREAM_ID = 0x2
    EXTENSION_DATA_WITH_STREAM_ID = 0x3
    CONTEXT_PACKET = 0x4
    # 0x5–0xF reserved


class TSI(IntEnum):
    NONE = 0
    UTC = 1
    GPS = 2
    OTHER = 3


class TSF(IntEnum):
    NONE = 0
    SAMPLE_COUNT = 1  # Or Real-time fractional seconds depending on type
    FRACTIONAL = 2
    FREE_RUNNING = 3

