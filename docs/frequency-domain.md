# Frequency-Domain (Spectral) Data in VITA 49.2

VITA 49 (VRT) can carry frequency-domain data as Signal Spectral Data. This guide explains how spectral payloads are encoded, what metadata describes them, and how to use them with `vita49io`. It assumes familiarity with FFT-based analysis and RF front-ends.

## What makes a packet “spectral”

- **Header S-bit (bit 24)**: Set to `1` in the data packet header to indicate frequency-domain (spectral) data. In `vita49io`, this is `Header.indicators_24=True`.
- **Packet type**: Spectral data uses the same Signal Data packet types as time-domain IQ (IF/Extension, with or without Stream ID). Only the S-bit differentiates spectral vs time-domain payloads.
- **Sample Frame**: Rule 6.3.1-3 requires a spectral packet to hold data for a single Sample Frame. Multi-packet frames are allowed; each packet repeats the S-bit.

## Spectral payload structure

Spectral payloads are a contiguous list of FFT (or similar transform) bins. Ordering is typically **low-to-high frequency** (per Fig. 6.3.1-1), but “high-to-low” is permitted if documented in context. There is no delimiter between bins; all framing comes from packet size and payload format metadata.

- **Complex bins**: Most spectral payloads are complex (real/imag). Values are interleaved: `Re0, Im0, Re1, Im1, ...`.
- **Magnitude/phase**: The standard allows other formats (e.g., log power). Use the payload format to declare the numeric type; consumers must decode accordingly.
- **Numeric width**: Common widths are 16-bit or 32-bit signed fixed-point, or 32-bit IEEE754 float. The library supports these via `PayloadFormat`.
- **Packing method**: Use Processing-efficient packing (bit 31 = 0) for straightforward I/Q/bin layouts.

### Bin frequency mapping

To map a bin index `k` to frequency:

```
f_k = f_start + k * Δf
```

- **f_start**: The start (or center of the first) bin frequency. Convey via RF/IF reference frequency + band offset in CIF0.
- **Δf (bin width)**: Typically `sample_rate / N_fft` for a radix-2 FFT; encode sample rate in CIF0 and the transform size in spectral context/control (Section 9.6 fields).
- **Nonlinear spacing**: If using logarithmic or wavelet spacing, document it and use the spectral context fields that indicate non-linear indexing.

## Metadata that makes spectra usable

Context packets (CIF0 + optional spectral fields) supply the parameters needed to turn bins into real frequencies and levels:

- **Sample rate (CIF0 bit 21)**: Required to derive bin width for FFT-based spectra.
- **RF/IF reference frequency, RF offset, IF band offset (CIF0 bits 27–25)**: Set the tuning reference and offset the bin map.
- **Bandwidth (CIF0 bit 29)**: Document total analyzed bandwidth if it differs from sample rate.
- **Reference level and gain (CIF0 bits 24–23)**: Needed to interpret absolute power.
- **Payload format (CIF0 bit 15)**: Declares numeric encoding: packing method, sample type, data item format, item size, etc. For spectra, use complex Cartesian and a float or fixed-point format.
- **State/event indicators, device identifiers, timestamps**: Maintain provenance and time alignment (TSI/TSF). Spectral timestamps normally mark the end of the time window used to compute the transform (Rule 6.3.1.2-2).
- **Spectral-specific controls (Section 9.6, not all implemented in this library)**: Transform size (Npoints), start/stop frequency, resolution bandwidth, window type, overlap, zoom indexing, etc. Use these where interoperability requires explicit spectral configuration.

## How frequency payloads differ from time-domain IQ

| Aspect | Time-domain IQ | Frequency-domain (spectral) |
| --- | --- | --- |
| Header bit 24 | 0 | 1 (S-bit) |
| Payload meaning | Time samples | FFT/transform bins |
| Timestamp meaning | Time of first sample | Time of final sample in window (Rule 6.3.1.2-2) |
| Ordering | Time order | Bin order (usually low→high) |
| Numeric formats | Same payload format system; often float32 or fixed-point | Same; often float32 complex bins or log-power fixed-point |

## Using `vita49io` to write spectral data

```python
import numpy as np
from vita49io import ContextPacket, DataPacket, PacketType, TSI, TSF, CIF0Fields
from vita49io.protocol.cif0 import PayloadFormat, PackingMethod, SampleType, DataItemFormat
from vita49io.protocol.core import Header
from vita49io.io.payload_codec import payload_from_numpy

stream_id = 0x2468ACE0
sample_rate_hz = 1_000_000.0

pf = PayloadFormat(
    packing_method=PackingMethod.PROCESSING_EFFICIENT,
    sample_type=SampleType.COMPLEX_CARTESIAN,
    data_item_format=DataItemFormat.IEEE754_SINGLE,
    sample_component_repeat=False,
    event_tag_size_bits=0,
    channel_tag_size_bits=0,
    data_item_fraction_size_bits=0,
    item_packing_field_size_bits=32,
    data_item_size_bits=32,
    repeat_count=1,
    vector_size=0,
)

# Context (no S-bit here; bit 24 on context is TSM)
ctx_hdr = Header(
    packet_type=PacketType.CONTEXT_PACKET,
    indicators_24=False,
    tsi=TSI.UTC,
    tsf=TSF.FRACTIONAL,
    packet_count=0,
    packet_size=0,
)
cif0 = CIF0Fields(
    sample_rate_hz=sample_rate_hz,
    bandwidth_hz=sample_rate_hz / 2,
    payload_format=pf,
    data_packet_payload_format=pf.pack_words(),
)
ctx_pkt = ContextPacket(
    header=ctx_hdr,
    stream_id=stream_id,
    integer_seconds=1_700_000_000,
    fractional_seconds=0,
    cif0=cif0,
)
ctx_bytes = ctx_pkt.to_bytes()

# Make spectral bins (FFT of two tones + noise)
n_fft = 2048
t = np.arange(n_fft, dtype=np.float32) / np.float32(sample_rate_hz)
sig = np.exp(1j * 2 * np.pi * 75_000.0 * t) + 0.6 * np.exp(1j * 2 * np.pi * -180_000.0 * t)
sig += (np.random.randn(n_fft) + 1j * np.random.randn(n_fft)) * 0.02
spectrum = np.fft.fftshift(np.fft.fft(sig)).astype(np.complex64) / n_fft

# Data packet with S-bit set
data_hdr = Header(
    packet_type=PacketType.IF_DATA_WITH_STREAM_ID,
    indicators_24=True,  # S-bit: spectral data
    tsi=TSI.UTC,
    tsf=TSF.FRACTIONAL,
    packet_count=1,
    packet_size=0,
)
data_pkt = DataPacket(
    header=data_hdr,
    stream_id=stream_id,
    integer_seconds=1_700_000_000,
    fractional_seconds=0,
    payload=payload_from_numpy(spectrum, pf),  # complex bins
)
data_bytes = data_pkt.to_bytes()
open("out_spectrum.v49", "wb").write(ctx_bytes + data_bytes)
```

See `scripts/write_frequency_domain_v49.py` for a complete script.

## Using `vita49io` to read spectral data and plot a waterfall

```python
import numpy as np
from matplotlib import pyplot as plt
from vita49io.protocol.core import Header
from vita49io.protocol.enums import PacketType
from vita49io.protocol.data_packet import DataPacket

def iter_packets(path):
    with open(path, "rb") as f:
        while True:
            hdr = f.read(4)
            if not hdr:
                break
            w0 = int.from_bytes(hdr, "big")
            header = Header.parse(w0)
            rest = f.read((header.packet_size - 1) * 4)
            yield header, hdr + rest

spectra = []
for header, raw in iter_packets("out_spectrum.v49"):
    if header.packet_type is not PacketType.IF_DATA_WITH_STREAM_ID:
        continue
    if not header.indicators_24:
        continue  # not spectral
    pkt = DataPacket.from_bytes(raw)
    vals = np.frombuffer(pkt.payload, dtype=">f4")
    if vals.size % 2 != 0:
        continue
    bins = vals.reshape(-1, 2)
    spectra.append(bins[:, 0] + 1j * bins[:, 1])

if spectra:
    waterfall = 20 * np.log10(np.abs(np.vstack(spectra)) + 1e-12)
    plt.imshow(waterfall, aspect="auto", origin="lower", cmap="magma")
    plt.colorbar(label="Magnitude (dB)")
    plt.xlabel("Frequency bin")
    plt.ylabel("Packet index")
    plt.title("Spectral Waterfall")
    plt.show()
```

See `scripts/waterfall_frequency_domain_v49.py` for a runnable version.

## Generating illustrative plots (spectrum + waterfall)

You can produce PNGs directly from the synthetic spectrum above:

```python
import numpy as np
from matplotlib import pyplot as plt

# spectrum is the complex array from the write example
freq_bins = np.arange(len(spectrum)) - len(spectrum) // 2
plt.figure()
plt.plot(freq_bins, 20 * np.log10(np.abs(spectrum) + 1e-12))
plt.xlabel("Bin")
plt.ylabel("Magnitude (dB)")
plt.title("Synthetic Spectrum")
plt.tight_layout()
plt.savefig("spectrum_example.png", dpi=150)

# Suppose you collected multiple spectra in a list called spectra_list
waterfall = 20 * np.log10(np.abs(np.vstack([spectrum]*40)) + 1e-12)
plt.figure()
plt.imshow(waterfall, aspect="auto", origin="lower", cmap="magma")
plt.colorbar(label="Magnitude (dB)")
plt.xlabel("Frequency bin")
plt.ylabel("Packet index")
plt.title("Waterfall Example")
plt.tight_layout()
plt.savefig("waterfall_example.png", dpi=150)
```

These images are not part of the repository but show how to visualize spectral packets you generate or ingest.

## Key takeaways for implementers

- Set the S-bit (header bit 24) on every spectral data packet.
- Declare numeric encoding with CIF0 payload format (bit 15) and keep it consistent across packets in a stream.
- Provide spectral metadata: sample rate, RF/IF frequency, offsets, bandwidth, reference level, gain, and (when needed) spectral control fields (Npoints, window, start/stop/zoom).
- Timestamps for spectra normally mark the end of the time window that produced the FFT.
- Use consistent bin ordering and document if you depart from low→high ordering.
