# Changelog

## Unreleased

- `IQStreamWriter` and `SpectrumStreamProcessor` packet times are now exact. They added the sample offset to a float epoch, which near today's epoch resolves only ~238 ns, so packet spacing jittered by up to that much. `epoch_time_to_vita_timestamp` now computes with exact rationals and also accepts a `fractions.Fraction`.

- **Breaking:** `TSF.FRACTIONAL` (Real-Time) timestamps are now encoded and decoded as picoseconds (0..999,999,999,999), as VITA 49.2 specifies. Earlier versions stored the fractional part as a fraction of 2^64, so other VITA 49 readers saw wrong times for any packet whose fraction was not zero. Files written by vita49io <= 0.1.9 with `TSF.FRACTIONAL` will now decode to the wrong times or raise `ValueError`; re-convert them. `convert_v49_ddc` now decodes packet times according to each packet's TSI/TSF instead of always assuming a 2^64 fraction.

- `payload_from_numpy` no longer clips IEEE-754 float samples to [-1, 1]. VITA 49.2 Rule 6.1.1.4-13 gives IEEE floats the full IEEE 754 range; 1.0 is only the Unit-Scale Sinusoid that the Reference Level refers to (§9.5.9). Fixed-point formats still saturate at their normalized range.

- Fixed Class ID OUI encoding and decoding to use bits 23–0 of the first Class ID word, as specified by VITA 49.2 §5.1.3.
- Added comprehensive docstrings across the public API.
- Introduced MkDocs documentation site powered by mkdocstrings.
- Published JSON schemas for the top vita49io helper functions in `ai_tools/`.
