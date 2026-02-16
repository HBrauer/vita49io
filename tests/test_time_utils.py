import math
import unittest

from vita49io.io.time_utils import (
    epoch_time_to_vita_timestamp,
    packet_vita_time_to_epoch_time,
    vita_timestamp_to_epoch_time,
)
from vita49io.protocol.data_packet import DataPacket
from vita49io.protocol.enums import PacketType, TSI, TSF


class TestTimeUtils(unittest.TestCase):
    def _mode_kwargs(self, tsi: TSI, tsf: TSF, t_epoch_s: float) -> dict[str, float | int]:
        kwargs: dict[str, float | int] = {}
        if tsi == TSI.GPS:
            kwargs["gps_utc_offset_s"] = 18
        elif tsi == TSI.OTHER:
            kwargs["other_epoch_s"] = 1_600_000_000.0

        if tsi == TSI.NONE and tsf != TSF.NONE:
            kwargs["tsi_none_reference_s"] = float(math.floor(t_epoch_s))

        if tsf == TSF.SAMPLE_COUNT:
            kwargs["sample_count_rate_hz"] = 1_000_000.0
        elif tsf == TSF.FREE_RUNNING:
            kwargs["free_running_rate_hz"] = 2_500_000.0
        return kwargs

    def _mode_tolerance_s(self, tsf: TSF, kwargs: dict[str, float | int]) -> float:
        base_float_tol = 2e-6
        if tsf == TSF.NONE:
            return 1.0
        if tsf == TSF.SAMPLE_COUNT:
            rate = float(kwargs["sample_count_rate_hz"])
            return max(base_float_tol, 0.5 / rate)
        if tsf == TSF.FREE_RUNNING:
            rate = float(kwargs["free_running_rate_hz"])
            return max(base_float_tol, 0.5 / rate)
        return base_float_tol

    def test_roundtrip_all_tsi_tsf(self) -> None:
        t_epoch_s = 1_700_000_123.456789

        for tsi in TSI:
            for tsf in TSF:
                kwargs = self._mode_kwargs(tsi, tsf, t_epoch_s)
                integer_seconds, fractional_seconds = epoch_time_to_vita_timestamp(
                    t_epoch_s,
                    tsi=tsi,
                    tsf=tsf,
                    **kwargs,
                )
                self.assertEqual(integer_seconds is None, tsi == TSI.NONE)
                self.assertEqual(fractional_seconds is None, tsf == TSF.NONE)

                t_roundtrip = vita_timestamp_to_epoch_time(
                    tsi=tsi,
                    tsf=tsf,
                    integer_seconds=integer_seconds,
                    fractional_seconds=fractional_seconds,
                    **kwargs,
                )

                if tsi == TSI.NONE and tsf == TSF.NONE:
                    self.assertIsNone(t_roundtrip)
                    continue

                self.assertIsNotNone(t_roundtrip)
                assert t_roundtrip is not None
                tol_s = self._mode_tolerance_s(tsf, kwargs)
                self.assertLessEqual(
                    abs(t_roundtrip - t_epoch_s),
                    tol_s,
                    msg=f"Roundtrip mismatch for tsi={tsi.name}, tsf={tsf.name}",
                )

    def test_missing_required_context_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "gps_utc_offset_s"):
            epoch_time_to_vita_timestamp(
                1_700_000_000.0,
                tsi=TSI.GPS,
                tsf=TSF.FRACTIONAL,
            )

        with self.assertRaisesRegex(ValueError, "sample_count_rate_hz"):
            vita_timestamp_to_epoch_time(
                tsi=TSI.UTC,
                tsf=TSF.SAMPLE_COUNT,
                integer_seconds=1,
                fractional_seconds=100,
            )

        with self.assertRaisesRegex(ValueError, "tsi_none_reference_s"):
            vita_timestamp_to_epoch_time(
                tsi=TSI.NONE,
                tsf=TSF.FRACTIONAL,
                integer_seconds=None,
                fractional_seconds=123,
            )

    def test_packet_helper(self) -> None:
        t_epoch_s = 1_700_000_200.125432
        kwargs = {
            "gps_utc_offset_s": 18,
            "sample_count_rate_hz": 1_000_000.0,
        }
        integer_seconds, fractional_seconds = epoch_time_to_vita_timestamp(
            t_epoch_s,
            tsi=TSI.GPS,
            tsf=TSF.SAMPLE_COUNT,
            **kwargs,
        )
        self.assertIsNotNone(integer_seconds)
        self.assertIsNotNone(fractional_seconds)

        pkt = DataPacket(
            packet_type=PacketType.IF_DATA_WITHOUT_STREAM_ID,
            tsi=TSI.GPS,
            tsf=TSF.SAMPLE_COUNT,
            integer_seconds=integer_seconds,
            fractional_seconds=fractional_seconds,
            payload=b"",
        )

        t_from_pkt = packet_vita_time_to_epoch_time(pkt, **kwargs)
        self.assertIsNotNone(t_from_pkt)
        assert t_from_pkt is not None
        self.assertLessEqual(abs(t_from_pkt - t_epoch_s), 2e-6)

    def test_packet_helper_data_packet_without_kwargs(self) -> None:
        t_epoch_s = 1_700_000_300.875125
        integer_seconds, fractional_seconds = epoch_time_to_vita_timestamp(
            t_epoch_s,
            tsi=TSI.UTC,
            tsf=TSF.FRACTIONAL,
        )
        self.assertIsNotNone(integer_seconds)
        self.assertIsNotNone(fractional_seconds)

        pkt = DataPacket(
            packet_type=PacketType.IF_DATA_WITHOUT_STREAM_ID,
            tsi=TSI.UTC,
            tsf=TSF.FRACTIONAL,
            integer_seconds=integer_seconds,
            fractional_seconds=fractional_seconds,
            payload=b"",
        )

        t_from_pkt = packet_vita_time_to_epoch_time(pkt)
        self.assertIsNotNone(t_from_pkt)
        assert t_from_pkt is not None
        self.assertLessEqual(abs(t_from_pkt - t_epoch_s), 2e-6)


if __name__ == "__main__":
    unittest.main()
