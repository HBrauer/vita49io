"""Define enumerations for common VITA 49 header fields.

Args:
    None.

Returns:
    None.

Raises:
    None.

Side Effects:
    None.

Examples:
    >>> from vita49io.protocol.enums import PacketType
    >>> PacketType.CONTEXT_PACKET.value
    4
"""

from __future__ import annotations

from enum import IntEnum


class PacketType(IntEnum):
    """Enumerate the VITA 49 packet type identifiers.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.

    Side Effects:
        None.

    Examples:
        >>> from vita49io.protocol.enums import PacketType
        >>> PacketType.IF_DATA_WITH_STREAM_ID.value
        1
    """

    IF_DATA_WITHOUT_STREAM_ID = 0x0
    IF_DATA_WITH_STREAM_ID = 0x1
    EXTENSION_DATA_WITHOUT_STREAM_ID = 0x2
    EXTENSION_DATA_WITH_STREAM_ID = 0x3
    CONTEXT_PACKET = 0x4
    # 0x5-0xF reserved


class TSI(IntEnum):
    """Enumerate Timestamp Integer (TSI) selection modes for VITA 49 packets.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.

    Side Effects:
        None.

    Examples:
        >>> from vita49io.protocol.enums import TSI
        >>> TSI.UTC.name
        'UTC'
    """

    NONE = 0
    UTC = 1
    GPS = 2
    OTHER = 3


class TSF(IntEnum):
    """Enumerate Timestamp Fractional (TSF) selection modes for VITA 49 packets.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.

    Side Effects:
        None.

    Examples:
        >>> from vita49io.protocol.enums import TSF
        >>> TSF.FRACTIONAL.value
        2
    """

    NONE = 0
    SAMPLE_COUNT = 1  # Or Real-time fractional seconds depending on type
    FRACTIONAL = 2
    FREE_RUNNING = 3


