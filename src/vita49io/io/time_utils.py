"""Convert between POSIX epoch time and VITA 49 timestamp fields.

This module centralizes timestamp conversion logic used by I/O helpers and
applications that bridge between VITA timestamps and `time.time()` style epoch
seconds.

All `TSI` and `TSF` modes are supported. Some modes require extra context:

- `TSI.GPS`: requires `gps_utc_offset_s`.
- `TSI.OTHER`: requires `other_epoch_s`.
- `TSI.NONE` with fractional modes: requires `tsi_none_reference_s`.
- `TSF.SAMPLE_COUNT`: requires `sample_count_rate_hz`.
- `TSF.FREE_RUNNING`: requires `free_running_rate_hz`.

Examples:
    >>> from vita49io.io.time_utils import epoch_time_to_vita_timestamp
    >>> from vita49io.protocol.enums import TSI, TSF
    >>> sec, frac = epoch_time_to_vita_timestamp(
    ...     1700000000.25, tsi=TSI.UTC, tsf=TSF.FRACTIONAL
    ... )
    >>> sec is not None and frac is not None
    True
"""

from __future__ import annotations

import math
from typing import Protocol

from ..protocol.enums import TSI, TSF

_U32_MAX = (1 << 32) - 1
_U64_MAX = (1 << 64) - 1
_FRAC_SCALE = float(1 << 64)
_GPS_EPOCH_UNIX_S = 315964800.0  # 1980-01-06T00:00:00Z in POSIX seconds.


class _HeaderTimestampLike(Protocol):
    """Minimal timestamp-related header shape required by this module."""

    tsi: TSI
    tsf: TSF


class PacketTimestampLike(Protocol):
    """Protocol for packet objects that carry VITA timestamp fields.

    Any object exposing this shape is accepted by
    `packet_vita_time_to_epoch_time()`. In practice this includes:

    - `ContextPacket`
    - `DataPacket`

    Attributes:
        header (_HeaderTimestampLike): Header object exposing `tsi` and `tsf`.
        integer_seconds (int | None): Integer seconds field from the packet.
        fractional_seconds (int | None): Fractional field from the packet.
    """

    header: _HeaderTimestampLike
    integer_seconds: int | None
    fractional_seconds: int | None


def _require_positive(value: float | None, name: str) -> float:
    if value is None or float(value) <= 0.0:
        raise ValueError(f"{name} is required and must be > 0 for this timestamp mode")
    return float(value)


def _round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def _epoch_to_tsi_seconds(
    epoch_time_s: float,
    *,
    tsi: TSI,
    gps_utc_offset_s: int | None,
    other_epoch_s: float | None,
) -> float:
    if tsi in (TSI.UTC, TSI.NONE):
        return float(epoch_time_s)
    if tsi == TSI.GPS:
        if gps_utc_offset_s is None:
            raise ValueError("gps_utc_offset_s is required when tsi=TSI.GPS")
        return float(epoch_time_s) - _GPS_EPOCH_UNIX_S + float(gps_utc_offset_s)
    if tsi == TSI.OTHER:
        if other_epoch_s is None:
            raise ValueError("other_epoch_s is required when tsi=TSI.OTHER")
        return float(epoch_time_s) - float(other_epoch_s)
    raise ValueError(f"Unsupported TSI mode: {tsi!r}")


def _tsi_seconds_to_epoch(
    tsi_seconds: float,
    *,
    tsi: TSI,
    gps_utc_offset_s: int | None,
    other_epoch_s: float | None,
) -> float:
    if tsi in (TSI.UTC, TSI.NONE):
        return float(tsi_seconds)
    if tsi == TSI.GPS:
        if gps_utc_offset_s is None:
            raise ValueError("gps_utc_offset_s is required when tsi=TSI.GPS")
        return float(tsi_seconds) + _GPS_EPOCH_UNIX_S - float(gps_utc_offset_s)
    if tsi == TSI.OTHER:
        if other_epoch_s is None:
            raise ValueError("other_epoch_s is required when tsi=TSI.OTHER")
        return float(tsi_seconds) + float(other_epoch_s)
    raise ValueError(f"Unsupported TSI mode: {tsi!r}")


def _encode_counter(offset_s: float, *, rate_hz: float, rate_name: str) -> int:
    if offset_s < 0.0:
        raise ValueError(f"{rate_name} conversion produced a negative offset; check reference values")
    ticks = _round_half_up(offset_s * rate_hz)
    if ticks < 0 or ticks > _U64_MAX:
        raise ValueError(f"Converted {rate_name} counter is out of 64-bit unsigned range")
    return int(ticks)


def epoch_time_to_vita_timestamp(
    epoch_time_s: float,
    *,
    tsi: TSI,
    tsf: TSF,
    gps_utc_offset_s: int | None = None,
    other_epoch_s: float | None = None,
    tsi_none_reference_s: float | None = None,
    sample_count_rate_hz: float | None = None,
    free_running_rate_hz: float | None = None,
) -> tuple[int | None, int | None]:
    """Convert POSIX epoch seconds into VITA integer/fractional timestamp fields.

    Args:
        epoch_time_s (float): POSIX epoch seconds (same basis as `time.time()`).
        tsi (TSI): Integer timestamp interpretation mode.
        tsf (TSF): Fractional timestamp interpretation mode.
        gps_utc_offset_s (int | None): GPS-UTC leap second offset when `tsi=TSI.GPS`.
        other_epoch_s (float | None): Epoch origin when `tsi=TSI.OTHER`.
        tsi_none_reference_s (float | None): Reference epoch used when `tsi=TSI.NONE`
            and `tsf != TSF.NONE`.
        sample_count_rate_hz (float | None): Tick rate for `tsf=TSF.SAMPLE_COUNT`.
        free_running_rate_hz (float | None): Tick rate for `tsf=TSF.FREE_RUNNING`.

    Returns:
        tuple[int | None, int | None]: `(integer_seconds, fractional_seconds)` in
            on-wire field form. Each element may be `None` when that field is not
            present for the selected `TSI`/`TSF` combination.

    Raises:
        ValueError: If the selected mode requires missing context, or converted
            values overflow packet field ranges.

    Examples:
        >>> from vita49io.io.time_utils import epoch_time_to_vita_timestamp
        >>> from vita49io.protocol.enums import TSI, TSF
        >>> epoch_time_to_vita_timestamp(1700000000.5, tsi=TSI.UTC, tsf=TSF.FRACTIONAL)  # doctest: +ELLIPSIS
        (1700000000, ...)
    """
    tsi_seconds = _epoch_to_tsi_seconds(
        epoch_time_s,
        tsi=tsi,
        gps_utc_offset_s=gps_utc_offset_s,
        other_epoch_s=other_epoch_s,
    )

    integer_seconds: int | None = None
    if tsi == TSI.NONE:
        if tsf == TSF.NONE:
            return None, None
        if tsi_none_reference_s is None:
            raise ValueError("tsi_none_reference_s is required when tsi=TSI.NONE and tsf!=TSF.NONE")
        base_seconds = float(tsi_none_reference_s)
    else:
        integer_seconds = int(math.floor(tsi_seconds))
        if integer_seconds < 0 or integer_seconds > _U32_MAX:
            raise ValueError("Converted integer_seconds is out of 32-bit unsigned range")
        base_seconds = float(integer_seconds)

    if tsf == TSF.NONE:
        return integer_seconds, None

    offset_s = tsi_seconds - base_seconds
    if tsf == TSF.FRACTIONAL:
        if not (0.0 <= offset_s < 1.0 + 1e-15):
            raise ValueError(
                "TSF.FRACTIONAL requires an offset in [0, 1). For tsi=TSI.NONE, "
                "provide tsi_none_reference_s close to the target time."
            )
        frac = _round_half_up(offset_s * _FRAC_SCALE)
        if frac >= (1 << 64):
            # Rounded up to exactly 1 second.
            if integer_seconds is None:
                raise ValueError(
                    "TSF.FRACTIONAL rounded to 1 second but tsi=TSI.NONE has no "
                    "integer field for carry. Adjust tsi_none_reference_s."
                )
            if integer_seconds >= _U32_MAX:
                raise ValueError("Integer second carry would overflow 32-bit range")
            integer_seconds += 1
            frac = 0
        return integer_seconds, int(frac)

    if tsf == TSF.SAMPLE_COUNT:
        rate_hz = _require_positive(sample_count_rate_hz, "sample_count_rate_hz")
        return integer_seconds, _encode_counter(offset_s, rate_hz=rate_hz, rate_name="sample_count_rate_hz")

    if tsf == TSF.FREE_RUNNING:
        rate_hz = _require_positive(free_running_rate_hz, "free_running_rate_hz")
        return integer_seconds, _encode_counter(offset_s, rate_hz=rate_hz, rate_name="free_running_rate_hz")

    raise ValueError(f"Unsupported TSF mode: {tsf!r}")


def vita_timestamp_to_epoch_time(
    *,
    tsi: TSI,
    tsf: TSF,
    integer_seconds: int | None,
    fractional_seconds: int | None,
    gps_utc_offset_s: int | None = None,
    other_epoch_s: float | None = None,
    tsi_none_reference_s: float | None = None,
    sample_count_rate_hz: float | None = None,
    free_running_rate_hz: float | None = None,
) -> float | None:
    """Convert VITA integer/fractional timestamp fields into POSIX epoch seconds.

    Args:
        tsi (TSI): Integer timestamp interpretation mode.
        tsf (TSF): Fractional timestamp interpretation mode.
        integer_seconds (int | None): Packet integer timestamp field.
        fractional_seconds (int | None): Packet fractional timestamp field.
        gps_utc_offset_s (int | None): GPS-UTC leap second offset when `tsi=TSI.GPS`.
        other_epoch_s (float | None): Epoch origin when `tsi=TSI.OTHER`.
        tsi_none_reference_s (float | None): Reference epoch used when `tsi=TSI.NONE`
            and `tsf != TSF.NONE`.
        sample_count_rate_hz (float | None): Tick rate for `tsf=TSF.SAMPLE_COUNT`.
        free_running_rate_hz (float | None): Tick rate for `tsf=TSF.FREE_RUNNING`.

    Returns:
        float | None: POSIX epoch seconds. Returns `None` only for the
            `TSI.NONE` + `TSF.NONE` combination.

    Raises:
        ValueError: If required fields/context are missing, or provided fields are
            invalid for the selected timestamp modes.

    Examples:
        >>> from vita49io.io.time_utils import vita_timestamp_to_epoch_time
        >>> from vita49io.protocol.enums import TSI, TSF
        >>> t = vita_timestamp_to_epoch_time(
        ...     tsi=TSI.UTC,
        ...     tsf=TSF.FRACTIONAL,
        ...     integer_seconds=1700000000,
        ...     fractional_seconds=0,
        ... )
        >>> t == 1700000000.0
        True
    """
    if tsi == TSI.NONE:
        if integer_seconds is not None:
            raise ValueError("integer_seconds must be None when tsi=TSI.NONE")
        if tsf == TSF.NONE:
            if fractional_seconds is not None:
                raise ValueError("fractional_seconds must be None when tsf=TSF.NONE")
            return None
        if tsi_none_reference_s is None:
            raise ValueError("tsi_none_reference_s is required when tsi=TSI.NONE and tsf!=TSF.NONE")
        tsi_seconds = float(tsi_none_reference_s)
    else:
        if integer_seconds is None:
            raise ValueError("integer_seconds is required when tsi!=TSI.NONE")
        if integer_seconds < 0 or integer_seconds > _U32_MAX:
            raise ValueError("integer_seconds must be in 0..0xFFFFFFFF")
        tsi_seconds = float(integer_seconds)

    if tsf == TSF.NONE:
        if fractional_seconds is not None:
            raise ValueError("fractional_seconds must be None when tsf=TSF.NONE")
        offset_s = 0.0
    else:
        if fractional_seconds is None:
            raise ValueError("fractional_seconds is required when tsf!=TSF.NONE")
        if fractional_seconds < 0 or fractional_seconds > _U64_MAX:
            raise ValueError("fractional_seconds must be in 0..0xFFFFFFFFFFFFFFFF")

        if tsf == TSF.FRACTIONAL:
            offset_s = float(fractional_seconds) / _FRAC_SCALE
        elif tsf == TSF.SAMPLE_COUNT:
            rate_hz = _require_positive(sample_count_rate_hz, "sample_count_rate_hz")
            offset_s = float(fractional_seconds) / rate_hz
        elif tsf == TSF.FREE_RUNNING:
            rate_hz = _require_positive(free_running_rate_hz, "free_running_rate_hz")
            offset_s = float(fractional_seconds) / rate_hz
        else:
            raise ValueError(f"Unsupported TSF mode: {tsf!r}")

    return _tsi_seconds_to_epoch(
        tsi_seconds + offset_s,
        tsi=tsi,
        gps_utc_offset_s=gps_utc_offset_s,
        other_epoch_s=other_epoch_s,
    )


def packet_vita_time_to_epoch_time(
    packet: PacketTimestampLike,
    *,
    gps_utc_offset_s: int | None = None,
    other_epoch_s: float | None = None,
    tsi_none_reference_s: float | None = None,
    sample_count_rate_hz: float | None = None,
    free_running_rate_hz: float | None = None,
) -> float | None:
    """Convert a packet's VITA timestamp fields directly into POSIX epoch seconds.

    Args:
        packet (PacketTimestampLike): Packet-like object with
            `header.tsi`/`header.tsf`, `integer_seconds`, and
            `fractional_seconds`. Typical inputs are `ContextPacket` and
            `DataPacket`.
        gps_utc_offset_s (int | None): GPS-UTC leap second offset when `tsi=TSI.GPS`.
        other_epoch_s (float | None): Epoch origin when `tsi=TSI.OTHER`.
        tsi_none_reference_s (float | None): Reference epoch used when `tsi=TSI.NONE`
            and `tsf != TSF.NONE`.
        sample_count_rate_hz (float | None): Tick rate for `tsf=TSF.SAMPLE_COUNT`.
        free_running_rate_hz (float | None): Tick rate for `tsf=TSF.FREE_RUNNING`.

    Returns:
        float | None: POSIX epoch seconds, or `None` for `TSI.NONE` + `TSF.NONE`.

    Raises:
        ValueError: Propagated from `vita_timestamp_to_epoch_time()` when packet
            fields and mode-specific context are inconsistent.

    Examples:
        >>> from vita49io.protocol.data_packet import DataPacket
        >>> from vita49io.protocol.enums import PacketType, TSI, TSF
        >>> pkt = DataPacket(
        ...     packet_type=PacketType.IF_DATA_WITHOUT_STREAM_ID,
        ...     tsi=TSI.UTC,
        ...     tsf=TSF.FRACTIONAL,
        ...     integer_seconds=1700000000,
        ...     fractional_seconds=0,
        ...     payload=b"",
        ... )
        >>> packet_vita_time_to_epoch_time(pkt)
        1700000000.0
    """
    return vita_timestamp_to_epoch_time(
        tsi=packet.header.tsi,
        tsf=packet.header.tsf,
        integer_seconds=packet.integer_seconds,
        fractional_seconds=packet.fractional_seconds,
        gps_utc_offset_s=gps_utc_offset_s,
        other_epoch_s=other_epoch_s,
        tsi_none_reference_s=tsi_none_reference_s,
        sample_count_rate_hz=sample_count_rate_hz,
        free_running_rate_hz=free_running_rate_hz,
    )


__all__ = [
    "epoch_time_to_vita_timestamp",
    "vita_timestamp_to_epoch_time",
    "packet_vita_time_to_epoch_time",
    "PacketTimestampLike",
]
