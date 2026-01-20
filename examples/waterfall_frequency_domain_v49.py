"""
Example: Read a VITA 49 file containing spectral (frequency-domain) data packets and
display a waterfall plot.

This expects packets with header bit 24 (S-bit) set on data packets to denote
Signal Spectral Data. It will parse headers, collect spectral payloads, and
display them as a time/frequency waterfall.

Usage:
  python examples/waterfall_frequency_domain_v49.py frequency_domain.v49
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt

from vita49io.io.packet_reader import PacketReader
from vita49io.protocol.enums import PacketType
from vita49io.protocol.data_packet import DataPacket


def read_packets(path: Path):
    with path.open("rb") as f:
        reader = PacketReader(f)
        index = 0
        while True:
            pkt = reader.read_packet()
            if pkt is None:
                break
            yield pkt.header, pkt.to_bytes()
            index += 1


def collect_spectra(path: Path):
    spectra = []
    for header, raw in read_packets(path):
        if header.packet_type is not PacketType.IF_DATA_WITH_STREAM_ID:
            continue
        if not header.indicators_24:
            continue  # skip non-spectral
        pkt = DataPacket.from_bytes(raw)
        payload_words = len(pkt.payload) // 4
        if payload_words == 0 or payload_words % 2 != 0:
            continue
        vals = np.frombuffer(pkt.payload, dtype=">f4")  # interleaved I,Q float32
        if vals.size % 2 != 0:
            continue
        spectrum = vals.reshape(-1, 2)
        complex_bins = spectrum[:, 0] + 1j * spectrum[:, 1]
        spectra.append(complex_bins)
    return spectra


def plot_waterfall(spectra: list[np.ndarray]) -> None:
    if not spectra:
        print("No spectral packets found.")
        return
    # Stack into magnitude (dB) waterfall
    mags = [20.0 * np.log10(np.abs(s) + 1e-12) for s in spectra]
    waterfall = np.vstack(mags)
    plt.figure(figsize=(10, 6))
    plt.imshow(
        waterfall,
        aspect="auto",
        origin="lower",
        cmap="magma",
    )
    plt.colorbar(label="Magnitude (dB)")
    plt.xlabel("Frequency bin")
    plt.ylabel("Packet index (time)")
    plt.title("Spectral Waterfall")
    plt.tight_layout()
    plt.show()


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Usage: python examples/waterfall_frequency_domain_v49.py <path.v49>")
        return 2
    path = Path(argv[1])
    spectra = collect_spectra(path)
    print(f"Loaded {len(spectra)} spectral packets")
    plot_waterfall(spectra)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
