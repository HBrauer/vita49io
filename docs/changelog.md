# Changelog

## Unreleased

- `payload_from_numpy` no longer clips IEEE-754 float samples to [-1, 1]. VITA 49.2 Rule 6.1.1.4-13 gives IEEE floats the full IEEE 754 range; 1.0 is only the Unit-Scale Sinusoid that the Reference Level refers to (§9.5.9). Fixed-point formats still saturate at their normalized range.

- Fixed Class ID OUI encoding and decoding to use bits 23–0 of the first Class ID word, as specified by VITA 49.2 §5.1.3.
- Added comprehensive docstrings across the public API.
- Introduced MkDocs documentation site powered by mkdocstrings.
- Published JSON schemas for the top vita49io helper functions in `ai_tools/`.
