import os
import io
import tempfile
import unittest

from vita49io.protocol.core import Header
from vita49io.protocol.enums import PacketType
from vita49io.protocol.data_packet import DataPacket
from vita49io.protocol.context_packet import ContextPacket


def _iter_packets(path: str):
    """Yield parsed packet objects from a VITA49 file.

    Uses the header word to determine total packet size, then parses into
    ContextPacket or DataPacket as appropriate.
    """
    with open(path, "rb") as f:
        index = 0
        while True:
            w0_bytes = f.read(4)
            if not w0_bytes:
                break  # EOF
            if len(w0_bytes) != 4:
                raise ValueError(
                    f"Truncated header at packet {index}: expected 4 bytes, got {len(w0_bytes)}"
                )

            w0 = int.from_bytes(w0_bytes, byteorder="big")
            header = Header.parse(w0)
            total_words = header.packet_size
            if total_words <= 0:
                raise ValueError(
                    f"Invalid packet size (words) at packet {index}: {total_words}"
                )

            remaining_bytes = (total_words - 1) * 4
            rest = f.read(remaining_bytes)
            if len(rest) != remaining_bytes:
                raise ValueError(
                    f"Truncated packet {index}: expected {remaining_bytes} bytes after header, got {len(rest)}"
                )
            packet_bytes = w0_bytes + rest

            if header.packet_type == PacketType.CONTEXT_PACKET:
                pkt = ContextPacket.from_bytes(packet_bytes)
                yield pkt
            elif header.packet_type in (
                PacketType.IF_DATA_WITHOUT_STREAM_ID,
                PacketType.IF_DATA_WITH_STREAM_ID,
                PacketType.EXTENSION_DATA_WITHOUT_STREAM_ID,
                PacketType.EXTENSION_DATA_WITH_STREAM_ID,
            ):
                yield DataPacket.from_bytes(packet_bytes)
            else:
                raise ValueError(
                    f"Unsupported packet type at index {index}: {header.packet_type}"
                )

            index += 1


class TestVitaExampleFilesRoundtrip(unittest.TestCase):
    def test_roundtrip_all_example_files(self):
        root = os.path.dirname(os.path.dirname(__file__))
        examples_dir = os.path.join(root, "vita_example_files")
        self.assertTrue(os.path.isdir(examples_dir), "Missing vita_example_files directory")

        # Collect files (any regular files under the directory)
        # Only process raw VITA 49 files, not PCAP containers
        files = [
            os.path.join(examples_dir, name)
            for name in os.listdir(examples_dir)
            if os.path.isfile(os.path.join(examples_dir, name)) and (name.endswith(".vita49")) or (name.endswith(".v49"))
        ]

        self.assertTrue(files, "No files found in vita_example_files")

        for path in files:
            print(f"Testing roundtrip for {os.path.basename(path)}")
            with self.subTest(path=path):
                with open(path, "rb") as rf:
                    original_bytes = rf.read()

                # Repack all packets into memory buffer
                out = io.BytesIO()
                for pkt in _iter_packets(path):
                    # Force a logical rebuild: decode fields then re-encode, not
                    # just forwarding the original memoryview-backed bytes.
                    if isinstance(pkt, DataPacket):
                        payload = pkt.payload
                        pkt.payload = payload.tobytes() if isinstance(payload, memoryview) else payload
                    elif isinstance(pkt, ContextPacket) and pkt.cif0 is not None:
                        pkt.cif0 = pkt.cif0
                    out.write(pkt.to_bytes())
                repacked = out.getvalue()

                # Also write to a temp file to satisfy the requirement
                with tempfile.TemporaryDirectory() as td:
                    temp_path = os.path.join(td, os.path.basename(path) + ".roundtrip")
                    with open(temp_path, "wb") as wf:
                        wf.write(repacked)
                    with open(temp_path, "rb") as cf:
                        check_bytes = cf.read()

                # Compare byte-for-byte
                if original_bytes != check_bytes:
                    # Persist the repacked bytes for debugging
                    repo_root = os.path.dirname(os.path.dirname(__file__))
                    artifacts_dir = os.path.join(repo_root, "tests", "_artifacts")
                    os.makedirs(artifacts_dir, exist_ok=True)
                    artifact_name = os.path.basename(path) + ".roundtrip"
                    artifact_path = os.path.join(artifacts_dir, artifact_name)
                    try:
                        with open(artifact_path, "wb") as af:
                            af.write(repacked)
                    except Exception:
                        # Do not mask the original assertion; saving is best-effort
                        artifact_path = None

                    # Build helpful diff context
                    msg_parts = [f"Roundtrip mismatch for {os.path.basename(path)}"]
                    if len(original_bytes) != len(check_bytes):
                        msg_parts.append(
                            f"size {len(original_bytes)} != {len(check_bytes)}"
                        )
                    # Find first differing offset
                    n = min(len(original_bytes), len(check_bytes))
                    off = None
                    for i in range(n):
                        if original_bytes[i] != check_bytes[i]:
                            off = i
                            break
                    if off is not None:
                        o = original_bytes[off]
                        c = check_bytes[off]
                        msg_parts.append(
                            f"first diff at offset {off}: src=0x{o:02X} dst=0x{c:02X}"
                        )
                    if artifact_path:
                        msg_parts.append(f"saved roundtrip to {artifact_path}")
                    self.fail("; ".join(msg_parts))
