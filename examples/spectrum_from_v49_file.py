from __future__ import annotations

import os
import sys
import argparse
from typing import Optional, Tuple


def _ensure_src_on_path() -> None:
    # Allow running the example from the repo root without installation
    here = os.path.dirname(__file__)
    src = os.path.normpath(os.path.join(here, "..", "src"))
    if os.path.isdir(src) and src not in sys.path:
        sys.path.insert(0, src)


def _resolve_center_hz(cif0) -> Optional[float]:
    center = None
    if cif0 is None:
        return None
    if cif0.rf_reference_frequency_hz is not None:
        center = cif0.rf_reference_frequency_hz
        if cif0.rf_reference_frequency_offset_hz is not None:
            center += cif0.rf_reference_frequency_offset_hz
        if cif0.if_band_offset_hz is not None:
            center += cif0.if_band_offset_hz
    elif cif0.if_reference_frequency_hz is not None:
        center = cif0.if_reference_frequency_hz
    return float(center) if center is not None else None


def _packet_time_s(integer_seconds: int | None, fractional_seconds: int | None) -> Optional[float]:
    if integer_seconds is None and fractional_seconds is None:
        return None
    sec = float(integer_seconds or 0)
    frac = float(fractional_seconds or 0)
    return sec + (frac / float(1 << 64))


def _read_first_context_info(path: str) -> Tuple[Optional[float], Optional[float]]:
    from vita49io.protocol.context_packet import ContextPacket
    from vita49io.io.packet_reader import PacketReader

    with open(path, "rb") as f:
        reader = PacketReader(f)
        while True:
            pkt = reader.read_packet()
            if pkt is None:
                break
            if isinstance(pkt, ContextPacket):
                cif0 = pkt.cif0
                if cif0 is None:
                    continue
                sample_rate = float(cif0.sample_rate_hz) if cif0.sample_rate_hz is not None else None
                center_hz = _resolve_center_hz(cif0)
                if sample_rate is not None or center_hz is not None:
                    return sample_rate, center_hz

    return None, None


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute spectra from a VITA 49 file and print (spectrum, sample_rate, center_hz, time).",
    )
    parser.add_argument("input_file", help="Path to input VITA49 binary file")
    parser.add_argument("--fft-size", type=int, default=1024)
    parser.add_argument("--hop-size", type=int, default=256)
    parser.add_argument("--window", choices=["hann", "rect"], default="hann")
    parser.add_argument("--averaging", choices=["none", "mean", "exponential"], default="mean")
    parser.add_argument("--averaging-param", type=float, default=4)
    parser.add_argument("--output-fps", type=float, default=10.0)
    parser.add_argument("--output-bins", type=int, default=None)
    parser.add_argument("--band-mode", choices=["inband", "full"], default="full")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    _ensure_src_on_path()

    import numpy as np

    from vita49io.protocol.data_packet import DataPacket
    from vita49io.io.spectrum_processor import SpectrumStreamProcessor

    args = _parse_args(sys.argv[1:] if argv is None else argv)

    sample_rate_hz, center_hz = _read_first_context_info(args.input_file)

    output_bins = args.output_bins if args.output_bins is not None else args.fft_size

    with open(args.input_file, "rb") as f:
        processor = SpectrumStreamProcessor(
            stream=f,
            fft_size=args.fft_size,
            hop_size=args.hop_size,
            window_type=args.window,
            averaging_mode=args.averaging,
            averaging_param=int(args.averaging_param)
            if args.averaging == "mean"
            else float(args.averaging_param),
            output_fps=args.output_fps,
            output_bins=output_bins,
            band_mode=args.band_mode,
        )

        for pkt in processor.read_packets():
            if not isinstance(pkt, DataPacket):
                continue
            payload = pkt.payload
            payload_bytes = payload.tobytes() if isinstance(payload, memoryview) else payload
            spectrum = np.frombuffer(payload_bytes, dtype=">f4")
            timestamp = _packet_time_s(pkt.integer_seconds, pkt.fractional_seconds)

            print((spectrum, sample_rate_hz, center_hz, timestamp))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
