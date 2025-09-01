from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

# CIF1 shares the exact on-wire format with CIF0. To avoid duplicating
# decoding logic, we reuse CIF0's parser to determine the length and simply
# capture the raw bytes for now (values intentionally ignored).
from .cif0 import CIF0Fields


@dataclass
class CIF1Fields:
    # Raw CIF1 payload bytes (presence mask + any following words)
    raw: bytes

    @staticmethod
    def parse(payload: bytes) -> Tuple["CIF1Fields", int]:
        # Use CIF0 parser to determine how many bytes CIF1 occupies.
        # Intentionally ignore decoded field values for now.
        _parsed, used = CIF0Fields.parse(payload)
        return CIF1Fields(raw=payload[:used]), used


__all__ = ["CIF1Fields"]

