# CIF0 fields not yet implemented

Derived from AV49.2-2017 + Errata-3, §9.1. This notes how the remaining CIF0 bits (14..0) are specified so they can be implemented later. Bits 31..15 are already handled in `src/vita49io/protocol/cif0.py`.

- Bit 14 — Formatted GPS Geolocation (9.4.5)  
  - Multiword field carrying a GPS position fix. Contains a GPS/INS manufacturer OUI and TSI/TSF subfields, integer- and fractional-second timestamps of the position fix, then latitude, longitude, altitude, speed over ground, heading angle, track angle, and magnetic variation.  
  - Angles use the “Geolocation Angle Format”: 32-bit two’s-complement with the radix to the right of bit 22 (≈2.38e-7° resolution, ±512° span). Altitude is a 32-bit signed fixed-point value (≈3.1 cm resolution, ±67,108 km span). Speed over ground is 32-bit signed fixed-point with radix at bit 16 (≈1.5e-5 m/s resolution).  
  - TSI/TSF use the Signal Data Packet encodings except code `00` means “included but unspecified”; when unspecified, corresponding timestamp words are `0xFFFFFFFF`. Unspecified geo subfields use `0x7FFFFFFF`.

- Bit 13 — Formatted INS Geolocation (9.4.6)  
  - Identical format and rules as the Formatted GPS field; may coexist with the GPS version in the same packet.

- Bit 12 — ECEF Ephemeris (9.4.3)  
  - Thirteen-word structure: integer-second timestamp; TSI/TSF + GPS/INS OUI word; two words for fractional-second timestamp; nine words for position (X/Y/Z), attitude (alpha/beta/phi), velocity (dX/dY/dZ), and a reserved word.  
  - Position uses 32-bit signed fixed-point meters with radix at bit 5 (~0.03125 m resolution). Attitude uses 32-bit signed fixed-point degrees with radix at bit 22. Velocity uses 32-bit signed fixed-point m/s with radix at bit 16. Unknown position/attitude/velocity values are `0x7FFFFFFF`. Timestamp fields follow the same rules as Formatted GPS (TSI/TSF and default values).

- Bit 11 — Relative Ephemeris (9.4.9)  
  - Same word layout as ECEF Ephemeris but expressed in a system-defined reference frame (e.g., platform-centric). Commonly paired with the Ephemeris Reference Identifier (bit 10) to tie the relative frame back to an ECEF origin.

- Bit 10 — Ephemeris Reference Identifier (9.4.4)  
  - Single 32-bit word containing the Stream ID of the Context packet whose ECEF Ephemeris provides the origin for a relative ephemeris.

- Bit 9 — GPS ASCII (9.4.7)  
  - Word 1: GPS/INS manufacturer OUI. Word 2: 32-bit “Number of Words” giving the ASCII payload length. Words 3..N+2: packed ASCII sentences (e.g., NMEA-0183), padded with `0x00` bytes to the declared length. Multiple sentences may be concatenated. Recommended to accompany the formatted GPS field.

- Bit 8 — Context Association Lists (9.13.2)  
  - Section begins with two 32-bit size words: Source list size and System list size in word 1; Vector-component list size and Asynchronous-channel list size plus an “A” flag in word 2. List size limits: Source/System 0–511 entries, Vector-component 0–65,535, Asynchronous-channel 0–32,767.  
  - Lists of Stream IDs follow in that order; if the A flag is set, an asynchronous-channel tag list (same length as the async list) is appended. Reserved bits are zero; lists with size 0 are omitted.

- Bit 7 — Field Attributes Enable / CIF7 (9.12)  
  - When set, a CIF7 word follows the other CIF words to select per-field attributes (current, average/median, standard deviation, max/min, precision/accuracy, derivatives, probability, belief). CIF7 selections override/augment the default value type for each selected field.

- Bits 6, 5, 4 — Reserved for CIF expansion  
  - Placeholder enables for undefined future CIF words (CIF6..CIF4). Keep cleared for V49.2.

- Bit 3 — CIF3 Enable  
  - When set, CIF3 is present after CIF2 (if any). Per §9.1-6, a CIF<n> word may appear even with no bits set.

- Bit 2 — CIF2 Enable  
  - When set, CIF2 is present after CIF1 (if any).

- Bit 1 — CIF1 Enable  
  - When set, CIF1 is present immediately after CIF0.

- Bit 0 — Reserved  
  - Unused in V49.2; must remain zero.
