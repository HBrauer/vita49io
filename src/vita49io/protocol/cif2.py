"""Implement VITA 49 CIF2 helpers for identifier-oriented fields (Table 9.1-1, Section 9.8)."""

from __future__ import annotations

from dataclasses import dataclass
import struct
from enum import IntFlag
from typing import List, Sequence, Tuple
from uuid import UUID

from .utils import _u32


class CIF2Flags(IntFlag):
    """Bit positions for CIF2 presence mask (identifier-oriented fields)."""

    NONE = 0
    BIND = 1 << 31
    CITED_SID = 1 << 30
    SIBLING_SID = 1 << 29
    PARENT_SID = 1 << 28
    CHILD_SID = 1 << 27
    CITED_MESSAGE_ID = 1 << 26
    CONTROLLEE_ID = 1 << 25
    CONTROLLEE_UUID = 1 << 24
    CONTROLLER_ID = 1 << 23
    CONTROLLER_UUID = 1 << 22
    INFORMATION_SOURCE_ID = 1 << 21
    TRACK_ID = 1 << 20
    COUNTRY_CODE = 1 << 19
    OPERATOR_ID = 1 << 18
    PLATFORM_CLASS = 1 << 17
    PLATFORM_INSTANCE = 1 << 16
    PLATFORM_DISPLAY_TYPE = 1 << 15
    EMS_DEVICE_CLASS = 1 << 14
    EMS_DEVICE_TYPE = 1 << 13
    EMS_DEVICE_INSTANCE = 1 << 12
    MODULATION_CLASS = 1 << 11
    MODULATION_TYPE = 1 << 10
    FUNCTION_ID = 1 << 9
    MODE_ID = 1 << 8
    EVENT_ID = 1 << 7
    FUNCTION_PRIORITY_ID = 1 << 6
    COMMS_PRIORITY_ID = 1 << 5
    RF_FOOTPRINT = 1 << 4
    RF_FOOTPRINT_RANGE = 1 << 3


def _pack_uuid(u: UUID) -> List[int]:
    data = u.bytes
    return list(struct.unpack(">IIII", data))


def _take_words(seq: Sequence[int], start: int, count: int) -> Tuple[List[int], int]:
    end = start + count
    if end > len(seq):
        raise ValueError("Truncated CIF2 payload")
    return list(seq[start:end]), end


def _words_to_uuid(words: Sequence[int]) -> UUID:
    if len(words) != 4:
        raise ValueError("UUID requires exactly four words")
    data = struct.pack(">IIII", *[w & 0xFFFFFFFF for w in words])
    return UUID(bytes=data)


@dataclass
class CIF2Fields:
    """Represent the CIF2 identifier fields."""

    bind: int | None = None
    cited_sid: int | None = None
    sibling_sid: int | None = None
    parent_sid: int | None = None
    child_sid: int | None = None
    cited_message_id: int | None = None
    controllee_id: int | None = None
    controllee_uuid: UUID | None = None
    controller_id: int | None = None
    controller_uuid: UUID | None = None
    information_source_id: int | None = None
    track_id: int | None = None
    country_code: int | None = None
    operator_id: int | None = None
    platform_class: int | None = None
    platform_instance: int | None = None
    platform_display_type: int | None = None
    ems_device_class: int | None = None
    ems_device_type: int | None = None
    ems_device_instance: int | None = None
    modulation_class: int | None = None
    modulation_type: int | None = None
    function_id: int | None = None
    mode_id: int | None = None
    event_id: int | None = None
    function_priority_id: int | None = None
    comms_priority_id: int | None = None
    rf_footprint_range: int | None = None
    rf_footprint: int | None = None

    SUPPORTED_MASK = (
        CIF2Flags.BIND
        | CIF2Flags.CITED_SID
        | CIF2Flags.SIBLING_SID
        | CIF2Flags.PARENT_SID
        | CIF2Flags.CHILD_SID
        | CIF2Flags.CITED_MESSAGE_ID
        | CIF2Flags.CONTROLLEE_ID
        | CIF2Flags.CONTROLLEE_UUID
        | CIF2Flags.CONTROLLER_ID
        | CIF2Flags.CONTROLLER_UUID
        | CIF2Flags.INFORMATION_SOURCE_ID
        | CIF2Flags.TRACK_ID
        | CIF2Flags.COUNTRY_CODE
        | CIF2Flags.OPERATOR_ID
        | CIF2Flags.PLATFORM_CLASS
        | CIF2Flags.PLATFORM_INSTANCE
        | CIF2Flags.PLATFORM_DISPLAY_TYPE
        | CIF2Flags.EMS_DEVICE_CLASS
        | CIF2Flags.EMS_DEVICE_TYPE
        | CIF2Flags.EMS_DEVICE_INSTANCE
        | CIF2Flags.MODULATION_CLASS
        | CIF2Flags.MODULATION_TYPE
        | CIF2Flags.FUNCTION_ID
        | CIF2Flags.MODE_ID
        | CIF2Flags.EVENT_ID
        | CIF2Flags.FUNCTION_PRIORITY_ID
        | CIF2Flags.COMMS_PRIORITY_ID
        | CIF2Flags.RF_FOOTPRINT
        | CIF2Flags.RF_FOOTPRINT_RANGE
    )

    def _presence_mask(self) -> int:
        m = CIF2Flags.NONE
        if self.bind is not None:
            m |= CIF2Flags.BIND
        if self.cited_sid is not None:
            m |= CIF2Flags.CITED_SID
        if self.sibling_sid is not None:
            m |= CIF2Flags.SIBLING_SID
        if self.parent_sid is not None:
            m |= CIF2Flags.PARENT_SID
        if self.child_sid is not None:
            m |= CIF2Flags.CHILD_SID
        if self.cited_message_id is not None:
            m |= CIF2Flags.CITED_MESSAGE_ID
        if self.controllee_id is not None:
            m |= CIF2Flags.CONTROLLEE_ID
        if self.controllee_uuid is not None:
            m |= CIF2Flags.CONTROLLEE_UUID
        if self.controller_id is not None:
            m |= CIF2Flags.CONTROLLER_ID
        if self.controller_uuid is not None:
            m |= CIF2Flags.CONTROLLER_UUID
        if self.information_source_id is not None:
            m |= CIF2Flags.INFORMATION_SOURCE_ID
        if self.track_id is not None:
            m |= CIF2Flags.TRACK_ID
        if self.country_code is not None:
            m |= CIF2Flags.COUNTRY_CODE
        if self.operator_id is not None:
            m |= CIF2Flags.OPERATOR_ID
        if self.platform_class is not None:
            m |= CIF2Flags.PLATFORM_CLASS
        if self.platform_instance is not None:
            m |= CIF2Flags.PLATFORM_INSTANCE
        if self.platform_display_type is not None:
            m |= CIF2Flags.PLATFORM_DISPLAY_TYPE
        if self.ems_device_class is not None:
            m |= CIF2Flags.EMS_DEVICE_CLASS
        if self.ems_device_type is not None:
            m |= CIF2Flags.EMS_DEVICE_TYPE
        if self.ems_device_instance is not None:
            m |= CIF2Flags.EMS_DEVICE_INSTANCE
        if self.modulation_class is not None:
            m |= CIF2Flags.MODULATION_CLASS
        if self.modulation_type is not None:
            m |= CIF2Flags.MODULATION_TYPE
        if self.function_id is not None:
            m |= CIF2Flags.FUNCTION_ID
        if self.mode_id is not None:
            m |= CIF2Flags.MODE_ID
        if self.event_id is not None:
            m |= CIF2Flags.EVENT_ID
        if self.function_priority_id is not None:
            m |= CIF2Flags.FUNCTION_PRIORITY_ID
        if self.comms_priority_id is not None:
            m |= CIF2Flags.COMMS_PRIORITY_ID
        if self.rf_footprint_range is not None:
            m |= CIF2Flags.RF_FOOTPRINT_RANGE
        if self.rf_footprint is not None:
            m |= CIF2Flags.RF_FOOTPRINT
        return int(m)

    def pack(self) -> bytes:
        """Serialize CIF2 fields (without the mask word)."""
        words: List[int] = []
        if self.bind is not None:
            words.append(_u32(self.bind))
        if self.cited_sid is not None:
            words.append(_u32(self.cited_sid))
        if self.sibling_sid is not None:
            words.append(_u32(self.sibling_sid))
        if self.parent_sid is not None:
            words.append(_u32(self.parent_sid))
        if self.child_sid is not None:
            words.append(_u32(self.child_sid))
        if self.cited_message_id is not None:
            words.append(_u32(self.cited_message_id))
        if self.controllee_id is not None:
            words.append(_u32(self.controllee_id))
        if self.controllee_uuid is not None:
            words.extend(_pack_uuid(self.controllee_uuid))
        if self.controller_id is not None:
            words.append(_u32(self.controller_id))
        if self.controller_uuid is not None:
            words.extend(_pack_uuid(self.controller_uuid))
        if self.information_source_id is not None:
            words.append(_u32(self.information_source_id))
        if self.track_id is not None:
            words.append(_u32(self.track_id))
        if self.country_code is not None:
            words.append(_u32(self.country_code))
        if self.operator_id is not None:
            words.append(_u32(self.operator_id))
        if self.platform_class is not None:
            words.append(_u32(self.platform_class))
        if self.platform_instance is not None:
            words.append(_u32(self.platform_instance))
        if self.platform_display_type is not None:
            words.append(_u32(self.platform_display_type))
        if self.ems_device_class is not None:
            words.append(_u32(self.ems_device_class))
        if self.ems_device_type is not None:
            words.append(_u32(self.ems_device_type))
        if self.ems_device_instance is not None:
            words.append(_u32(self.ems_device_instance))
        if self.modulation_class is not None:
            words.append(_u32(self.modulation_class))
        if self.modulation_type is not None:
            words.append(_u32(self.modulation_type))
        if self.function_id is not None:
            words.append(_u32(self.function_id))
        if self.mode_id is not None:
            words.append(_u32(self.mode_id))
        if self.event_id is not None:
            words.append(_u32(self.event_id))
        if self.function_priority_id is not None:
            words.append(_u32(self.function_priority_id))
        if self.comms_priority_id is not None:
            words.append(_u32(self.comms_priority_id))
        if self.rf_footprint is not None:
            words.append(_u32(self.rf_footprint))
        if self.rf_footprint_range is not None:
            words.append(_u32(self.rf_footprint_range))

        out = bytearray(len(words) * 4)
        for i, w in enumerate(words):
            struct.pack_into(">I", out, i * 4, _u32(w))
        return bytes(out)

    @staticmethod
    def parse_from_mask(mask: int, field_words: memoryview | bytes | bytearray) -> Tuple["CIF2Fields", int]:
        """Parse CIF2 fields based on a mask and return (fields, words_consumed)."""
        mv = field_words if isinstance(field_words, memoryview) else memoryview(field_words)
        if mv.format != "B":
            mv = mv.cast("B")
        if len(mv) % 4 != 0:
            raise ValueError("CIF2 payload must be a whole number of 32-bit words")

        flags = CIF2Flags(mask)
        unsupported = int(flags & ~CIF2Fields.SUPPORTED_MASK)
        if unsupported:
            raise ValueError(f"Unsupported CIF2 bits set: 0x{unsupported:08X}")

        idx = 0  # word index
        nwords = len(mv) // 4
        words = struct.unpack_from(f">{nwords}I", mv, 0)

        def next_word() -> int:
            nonlocal idx
            if idx >= len(words):
                raise ValueError("Truncated CIF2 payload")
            w = words[idx]
            idx += 1
            return w

        bind = cited_sid = sibling_sid = parent_sid = child_sid = None
        cited_message_id = controllee_id = controller_id = None
        controllee_uuid = controller_uuid = None
        information_source_id = track_id = None
        country_code = operator_id = None
        platform_class = platform_instance = platform_display_type = None
        ems_device_class = ems_device_type = ems_device_instance = None
        modulation_class = modulation_type = None
        function_id = mode_id = event_id = None
        function_priority_id = comms_priority_id = None
        rf_footprint_range = rf_footprint = None

        if flags & CIF2Flags.BIND:
            bind = next_word()
        if flags & CIF2Flags.CITED_SID:
            cited_sid = next_word()
        if flags & CIF2Flags.SIBLING_SID:
            sibling_sid = next_word()
        if flags & CIF2Flags.PARENT_SID:
            parent_sid = next_word()
        if flags & CIF2Flags.CHILD_SID:
            child_sid = next_word()
        if flags & CIF2Flags.CITED_MESSAGE_ID:
            cited_message_id = next_word()
        if flags & CIF2Flags.CONTROLLEE_ID:
            controllee_id = next_word()
        if flags & CIF2Flags.CONTROLLEE_UUID:
            uuid_words, idx = _take_words(words, idx, 4)
            controllee_uuid = _words_to_uuid(uuid_words)
        if flags & CIF2Flags.CONTROLLER_ID:
            controller_id = next_word()
        if flags & CIF2Flags.CONTROLLER_UUID:
            uuid_words, idx = _take_words(words, idx, 4)
            controller_uuid = _words_to_uuid(uuid_words)
        if flags & CIF2Flags.INFORMATION_SOURCE_ID:
            information_source_id = next_word()
        if flags & CIF2Flags.TRACK_ID:
            track_id = next_word()
        if flags & CIF2Flags.COUNTRY_CODE:
            country_code = next_word()
        if flags & CIF2Flags.OPERATOR_ID:
            operator_id = next_word()
        if flags & CIF2Flags.PLATFORM_CLASS:
            platform_class = next_word()
        if flags & CIF2Flags.PLATFORM_INSTANCE:
            platform_instance = next_word()
        if flags & CIF2Flags.PLATFORM_DISPLAY_TYPE:
            platform_display_type = next_word()
        if flags & CIF2Flags.EMS_DEVICE_CLASS:
            ems_device_class = next_word()
        if flags & CIF2Flags.EMS_DEVICE_TYPE:
            ems_device_type = next_word()
        if flags & CIF2Flags.EMS_DEVICE_INSTANCE:
            ems_device_instance = next_word()
        if flags & CIF2Flags.MODULATION_CLASS:
            modulation_class = next_word()
        if flags & CIF2Flags.MODULATION_TYPE:
            modulation_type = next_word()
        if flags & CIF2Flags.FUNCTION_ID:
            function_id = next_word()
        if flags & CIF2Flags.MODE_ID:
            mode_id = next_word()
        if flags & CIF2Flags.EVENT_ID:
            event_id = next_word()
        if flags & CIF2Flags.FUNCTION_PRIORITY_ID:
            function_priority_id = next_word()
        if flags & CIF2Flags.COMMS_PRIORITY_ID:
            comms_priority_id = next_word()
        if flags & CIF2Flags.RF_FOOTPRINT:
            rf_footprint = next_word()
        if flags & CIF2Flags.RF_FOOTPRINT_RANGE:
            rf_footprint_range = next_word()

        fields = CIF2Fields(
            bind=bind,
            cited_sid=cited_sid,
            sibling_sid=sibling_sid,
            parent_sid=parent_sid,
            child_sid=child_sid,
            cited_message_id=cited_message_id,
            controllee_id=controllee_id,
            controllee_uuid=controllee_uuid,
            controller_id=controller_id,
            controller_uuid=controller_uuid,
            information_source_id=information_source_id,
            track_id=track_id,
            country_code=country_code,
            operator_id=operator_id,
            platform_class=platform_class,
            platform_instance=platform_instance,
            platform_display_type=platform_display_type,
            ems_device_class=ems_device_class,
            ems_device_type=ems_device_type,
            ems_device_instance=ems_device_instance,
            modulation_class=modulation_class,
            modulation_type=modulation_type,
            function_id=function_id,
            mode_id=mode_id,
            event_id=event_id,
            function_priority_id=function_priority_id,
            comms_priority_id=comms_priority_id,
            rf_footprint_range=rf_footprint_range,
            rf_footprint=rf_footprint,
        )
        return fields, idx


__all__ = ["CIF2Flags", "CIF2Fields"]
