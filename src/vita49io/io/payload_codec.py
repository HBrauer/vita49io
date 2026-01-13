"""Helpers for encoding and decoding VITA 49 sample payloads."""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np

from ..protocol.cif0 import PayloadFormat, PackingMethod, SampleType, DataItemFormat


def _validate_supported(
    pf: PayloadFormat,
    *,
    validate_strict: bool = False,
) -> Tuple[int, int, DataItemFormat]:
    """Validate and summarize a payload format for sample conversion.

    Args:
        pf: Payload format metadata describing the on-wire sample layout.
        validate_strict: When True, enforce vector/repeat constraints that some
            real-world captures may violate.

    Returns:
        Tuple of (item_packing_field_size_bits, data_item_size_bits, data_item_format).

    Raises:
        ValueError: If the payload format is unsupported for sample conversion.
    """
    if pf.packing_method != PackingMethod.PROCESSING_EFFICIENT:
        raise ValueError("Unsupported packing method: only Processing-efficient is supported")
    if pf.sample_type not in (SampleType.COMPLEX_CARTESIAN, SampleType.REAL):
        raise ValueError(
            "Unsupported sample type: only Complex Cartesian (I/Q) and Real are supported"
        )
    if pf.sample_component_repeat:
        raise ValueError("Unsupported sample-component repeat: must be false")
    if pf.event_tag_size_bits != 0 or pf.channel_tag_size_bits != 0:
        raise ValueError("Unsupported tag sizes: event and channel tag sizes must be 0")
    if pf.data_item_fraction_size_bits != 0:
        raise ValueError("Unsupported data item fraction size: must be 0")
    if validate_strict:
        if pf.vector_size != 1:
            raise ValueError(
                f"Unsupported vector size: decoded value {pf.vector_size}, only 1 is supported"
            )
        if pf.repeat_count != 1:
            raise ValueError(
                f"Unsupported repeat count of {pf.repeat_count}, only 1 is supported"
            )

    fmt = pf.data_item_format

    # Validate combinations
    ipf = pf.item_packing_field_size_bits
    di = pf.data_item_size_bits

    valid = False
    if fmt in (DataItemFormat.SIGNED_FIXED_POINT, DataItemFormat.UNSIGNED_FIXED_POINT):
        if (ipf == 16 and di == 16) or (ipf == 32 and di in (16, 24, 32)):
            valid = True
    elif fmt == DataItemFormat.IEEE754_SINGLE:
        if ipf == 32 and di == 32:
            valid = True

    if not valid:
        raise ValueError(
            f"Unsupported item packing/data size combination: item_packing={ipf}, "
            f"data_item={di}, format={fmt.name}"
        )

    return ipf, di, fmt


def payload_as_numpy(
    payload: bytes,
    pf: PayloadFormat,
    *,
    validate_strict: bool = False,
) -> "np.ndarray":
    """Decode payload bytes into complex64 NumPy samples.

    Args:
        payload: Raw payload bytes in on-wire big-endian ordering.
        pf: Payload format describing how to interpret the payload.
        validate_strict: When True, enforce vector/repeat constraints.

    Returns:
        A complex64 NumPy array of I/Q samples.

    Raises:
        ValueError: If the payload format is unsupported or the payload length
            does not contain an even number of I/Q components.

    Supported Formats:
        - Packing method: Processing-efficient only.
        - Sample type: Complex Cartesian (I/Q) or Real.
        - Data item formats:
            - IEEE754 single (32-bit float, 32-bit field).
            - Signed fixed-point (16/24/32-bit items in 16/32-bit fields).
            - Unsigned fixed-point (16/24/32-bit items in 16/32-bit fields).

    """
    ipf, di, fmt = _validate_supported(pf, validate_strict=validate_strict)

    sample_type = pf.sample_type

    # Fast paths for common full-width cases
    if ipf == 32 and di == 32:
        if fmt == DataItemFormat.IEEE754_SINGLE:
            floats = np.frombuffer(payload, dtype=">f4")
            if sample_type == SampleType.REAL:
                return floats.astype(np.float32)
            if floats.size % 2 != 0:
                raise ValueError("Payload does not contain an even number of components for I/Q")
            iq = floats.reshape(-1, 2).astype(np.float32)
            return (iq[:, 0] + 1j * iq[:, 1]).astype(np.complex64)
        if fmt == DataItemFormat.SIGNED_FIXED_POINT:
            s32 = np.frombuffer(payload, dtype=">i4")
            # Use float64 for arithmetic, then cast to float32 to avoid precision loss
            vals = (s32.astype(np.float64) / float(1 << 31)).astype(np.float32)
        else:  # UNSIGNED_FIXED_POINT
            u32 = np.frombuffer(payload, dtype=">u4")
            # Use float64 for arithmetic, then cast to float32 to avoid precision loss
            vals = (u32.astype(np.float64) / float(1 << 32)).astype(np.float32)
    elif ipf == 16 and di == 16:
        if fmt == DataItemFormat.SIGNED_FIXED_POINT:
            s16 = np.frombuffer(payload, dtype=">i2")
            vals = (s16.astype(np.float32) / float(1 << 15)).astype(np.float32)
        else:  # UNSIGNED_FIXED_POINT
            u16 = np.frombuffer(payload, dtype=">u2")
            vals = (u16.astype(np.float32) / float(1 << 16)).astype(np.float32)
    else:
        # Generic paths: extract fields as big-endian and unpack lower di bits
        if ipf == 16:
            fields = np.frombuffer(payload, dtype=">u2")
            uvals = fields.astype(np.uint32)
        else:  # ipf == 32
            fields32 = np.frombuffer(payload, dtype=">u4")
            uvals = fields32

        if di < 32:
            mask = (1 << di) - 1
            uvals = (uvals & mask).astype(np.uint32, copy=False)

        if fmt == DataItemFormat.SIGNED_FIXED_POINT:
            # Sign-extend using arithmetic shifts for speed
            if di < 32:
                s = (uvals.astype(np.int32) << (32 - di)) >> (32 - di)
            else:
                s = uvals.astype(np.int32)
            scale = float(1 << (di - 1))
            vals = (s.astype(np.float32) / scale).astype(np.float32)
        elif fmt == DataItemFormat.UNSIGNED_FIXED_POINT:
            scale = float(1 << di)
            vals = (uvals.astype(np.float32) / scale).astype(np.float32)
        else:
            raise ValueError(f"Internal error: unexpected format in fixed-point decoder: {pf}")

    if sample_type == SampleType.REAL:
        return vals.astype(np.float32)

    # Convert interleaved I/Q components to complex samples.
    if vals.size % 2 != 0:
        raise ValueError("Payload does not contain an even number of components for I/Q")
    vec = vals.reshape(-1, 2)
    return (vec[:, 0] + 1j * vec[:, 1]).astype(np.complex64)


def payload_as_numpy_view(
    payload: Union[bytes, memoryview],
    pf: PayloadFormat,
    *,
    validate_strict: bool = False,
) -> "np.ndarray":
    """Return a zero-copy NumPy view of the raw payload in on-wire dtype.

    Args:
        payload: Raw payload bytes or memoryview.
        pf: Payload format describing how to interpret the payload.
        validate_strict: When True, enforce vector/repeat constraints.

    Returns:
        A NumPy array view into the payload data. For IEEE754 single, the view
        is complex64 (big-endian). For fixed-point formats, the view is the raw
        integer components reshaped as (N, 2) for I/Q.

    Raises:
        ValueError: If the payload format is unsupported or incompatible with a
            native view (e.g., non-full-width items).

    Supported Formats:
        - Packing method: Processing-efficient only.
        - Sample type: Complex Cartesian (I/Q) or Real.
        - Data item formats:
            - IEEE754 single (32-bit float, 32-bit field) yields a `>c8` view.
            - Signed fixed-point (16/24/32-bit items in 16/32-bit fields).
            - Unsigned fixed-point (16/24/32-bit items in 16/32-bit fields).
        - Fixed-point yields integer views; 16-in-32 returns lower 16 bits.
    """
    ipf, di, fmt = _validate_supported(pf, validate_strict=validate_strict)
    sample_type = pf.sample_type
    if fmt == DataItemFormat.IEEE754_SINGLE:
        if sample_type == SampleType.REAL:
            return np.frombuffer(payload, dtype=">f4")
        if len(payload) % 8 != 0:
            raise ValueError("Payload does not contain an even number of components for I/Q")
        return np.frombuffer(payload, dtype=">c8")
    if ipf == 32 and di == 16:
        dtype = ">i2" if fmt == DataItemFormat.SIGNED_FIXED_POINT else ">u2"
        # 32-bit fields store the data item in the lower 16 bits (big-endian).
        vals16 = np.frombuffer(payload, dtype=dtype)[1::2]
        if sample_type == SampleType.REAL:
            return vals16
        if vals16.size % 2 != 0:
            raise ValueError("Payload does not contain an even number of components for I/Q")
        return vals16.reshape(-1, 2)

    if ipf != di or ipf not in (16, 32):
        raise ValueError(
            "Native sample view requires full-width 16- or 32-bit data items "
            "(item_packing_field_size_bits == data_item_size_bits), "
            "except 16-in-32 fixed-point which returns the lower 16 bits."
        )

    if fmt == DataItemFormat.SIGNED_FIXED_POINT:
        dtype = ">i2" if ipf == 16 else ">i4"
    else:  # UNSIGNED_FIXED_POINT
        dtype = ">u2" if ipf == 16 else ">u4"

    vals = np.frombuffer(payload, dtype=dtype)
    if sample_type == SampleType.REAL:
        return vals
    if vals.size % 2 != 0:
        raise ValueError("Payload does not contain an even number of components for I/Q")
    return vals.reshape(-1, 2)


def payload_from_numpy(
    samples: "np.ndarray",
    pf: PayloadFormat,
    *,
    validate_strict: bool = False,
) -> bytes:
    """Encode complex or interleaved I/Q samples into payload bytes.

    Args:
        samples: Complex array or real array shaped (N, 2) containing I/Q.
        pf: Payload format describing how to encode the samples.
        validate_strict: When True, enforce vector/repeat constraints.

    Returns:
        Payload bytes in on-wire big-endian ordering.

    Raises:
        ValueError: If the payload format is unsupported or the sample array
            shape is incompatible.

    Supported Formats:
        - Packing method: Processing-efficient only.
        - Sample type: Complex Cartesian (I/Q) or Real.
        - Data item formats:
            - IEEE754 single (32-bit float, 32-bit field).
            - Signed fixed-point (16/24/32-bit items in 16/32-bit fields).
            - Unsigned fixed-point (16/24/32-bit items in 16/32-bit fields).

    Notes:
        - Input samples are converted to float32 internally before encoding.
        - For IEEE754 single, inputs (including float16/int16) are cast to float32
          and encoded as big-endian 32-bit floats.
        - For fixed-point formats, inputs are scaled, clipped, and rounded to the
          target bit width before packing.
    """
    ipf, di, fmt = _validate_supported(pf, validate_strict=validate_strict)

    sample_type = pf.sample_type

    # Normalize input to float32 components for encoding paths.
    arr = np.asarray(samples)
    if sample_type == SampleType.REAL:
        if arr.dtype.kind == "c":
            raise ValueError("Sample array must be real-valued for SampleType.REAL")
        vals = arr.astype(np.float32).reshape(-1)
    else:
        if arr.dtype.kind == "c":
            I = arr.real.astype(np.float32)
            Q = arr.imag.astype(np.float32)
            vec = np.stack([I, Q], axis=1)
        else:
            vec = arr.astype(np.float32)
            if vec.ndim == 1:
                raise ValueError("Sample array must be complex or shape (N,2)")
            if vec.shape[-1] != 2:
                raise ValueError("Sample array last dimension must be 2 (I,Q)")
        vals = vec.reshape(-1).astype(np.float32)

    # Fast path: 32-bit fields
    if ipf == 32 and di == 32:
        if fmt == DataItemFormat.IEEE754_SINGLE:
            vals_c = np.clip(vals, -1.0, 1.0)
            return vals_c.astype(">f4").tobytes()
        if fmt == DataItemFormat.SIGNED_FIXED_POINT:
            # Perform arithmetic in float64 to reduce rounding artifacts
            scale = float(1 << 31)
            max_val = (float((1 << 31) - 1)) / scale
            vals64 = vals.astype(np.float64)
            vals_c = np.clip(vals64, -1.0, max_val)
            ints = np.rint(vals_c * scale).astype(np.int32)
            return ints.astype(">i4").tobytes()
        else:  # UNSIGNED_FIXED_POINT
            # Use float64 for clipping and scaling to avoid rounding 0.999.. to 1.0
            scale = float(1 << 32)
            max_val = (float((1 << 32) - 1)) / scale
            vals64 = vals.astype(np.float64)
            vals_c = np.clip(vals64, 0.0, max_val)
            u32 = np.rint(vals_c * scale).astype(np.uint32)
            return u32.astype(">u4").tobytes()

    # Fast path: 16-bit fields
    if ipf == 16 and di == 16:
        if fmt == DataItemFormat.SIGNED_FIXED_POINT:
            scale = float(1 << 15)
            max_val = (float((1 << 15) - 1)) / scale
            vals_c = np.clip(vals, -1.0, max_val)
            i16 = np.rint(vals_c * scale).astype(np.int16)
            return i16.astype(">i2").tobytes()
        else:  # UNSIGNED_FIXED_POINT
            scale = float(1 << 16)
            max_val = (float((1 << 16) - 1)) / scale
            vals_c = np.clip(vals, 0.0, max_val)
            u16 = np.rint(vals_c * scale).astype(np.uint16)
            return u16.astype(">u2").tobytes()

    # Generic fixed-point encoding into 32-bit fields (16 or 24-bit in lower bits)
    if fmt == DataItemFormat.SIGNED_FIXED_POINT:
        scale = float(1 << (di - 1))
        max_val = (float((1 << (di - 1)) - 1)) / scale
        vals_c = np.clip(vals, -1.0, max_val)
        ints = np.rint(vals_c * scale).astype(np.int64)
        uvals = (ints & ((1 << di) - 1)).astype(np.uint32)
    else:  # UNSIGNED_FIXED_POINT
        scale = float(1 << di)
        max_val = (float((1 << di) - 1)) / scale
        vals_c = np.clip(vals, 0.0, max_val)
        uvals = np.rint(vals_c * scale).astype(np.uint32)

    # ipf == 32 here; pack lower di bits of 32-bit field
    fields32 = (uvals & ((1 << di) - 1)).astype(np.uint32)
    return fields32.astype(">u4").tobytes()


__all__ = ["payload_as_numpy", "payload_from_numpy", "payload_as_numpy_view"]
