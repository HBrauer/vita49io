import pytest

from vita49io.protocol.cif0 import (
    CIF0Fields,
    ContextAssociationLists,
    Ephemeris,
    FormattedGeolocation,
    GPSASCIIField,
)
from vita49io.protocol.enums import TSI, TSF


def test_cif0_new_fields_roundtrip():
    lat_deg = 12.345678
    lon_deg = -98.765432
    alt_m = 1234.56
    speed_m_s = 250.25
    heading_deg = 33.25
    track_deg = 5.5
    mag_var_deg = -7.0

    gps_geo = FormattedGeolocation(
        tsi=TSI.UTC,
        tsf=TSF.FRACTIONAL,
        manufacturer_oui=0x123456,
        integer_seconds=0x01020304,
        fractional_seconds=0x1122334455667788,
        latitude_deg=lat_deg,
        longitude_deg=lon_deg,
        altitude_m=alt_m,
        speed_over_ground_m_s=speed_m_s,
        heading_angle_deg=heading_deg,
        track_angle_deg=track_deg,
        magnetic_variation_deg=mag_var_deg,
    )
    ins_geo = FormattedGeolocation(
        tsi=TSI.GPS,
        tsf=TSF.SAMPLE_COUNT,
        manufacturer_oui=0xABCDEF,
        integer_seconds=0x0A0B0C0D,
        fractional_seconds=0x99AABBCCDDEEFF00,
        latitude_deg=-1.0,
        longitude_deg=2.0,
        altitude_m=-10.0,
        speed_over_ground_m_s=1.0,
        heading_angle_deg=45.0,
        track_angle_deg=90.0,
        magnetic_variation_deg=1.5,
    )
    ecef = Ephemeris(
        tsi=TSI.UTC,
        tsf=TSF.FRACTIONAL,
        manufacturer_oui=0x00A0A0,
        integer_seconds=0x11121314,
        fractional_seconds=0xAABBCCDDEEFF0011,
        position_x_m=100.0,
        position_y_m=-200.0,
        position_z_m=300.5,
        attitude_alpha_deg=1.5,
        attitude_beta_deg=-2.5,
        attitude_phi_deg=3.0,
        velocity_dx_m_s=10.0,
        velocity_dy_m_s=20.5,
        velocity_dz_m_s=-5.25,
    )
    relative = Ephemeris(
        tsi=TSI.NONE,
        tsf=TSF.NONE,
        manufacturer_oui=0x00B0B0,
        integer_seconds=0x21222324,
        fractional_seconds=0x0102030405060708,
        position_x_m=-10.0,
        position_y_m=20.0,
        position_z_m=-30.0,
        attitude_alpha_deg=-1.0,
        attitude_beta_deg=2.0,
        attitude_phi_deg=-3.0,
        velocity_dx_m_s=-1.5,
        velocity_dy_m_s=2.5,
        velocity_dz_m_s=-3.5,
    )
    gps_ascii = GPSASCIIField(
        manufacturer_oui=0x00C0FFEE, sentences="$GPGGA,TEST*00\r\n"
    )
    cal = ContextAssociationLists(
        source_list=[0xAAAA0001, 0xAAAA0002],
        system_list=[0xBBBB0001],
        vector_component_list=[0xCCCC0001, 0xCCCC0002, 0xCCCC0003],
        async_channel_list=[0xDDDD0001, 0xDDDD0002],
        async_channel_tags=[0xEEEE0001, 0xEEEE0002],
    )

    cif = CIF0Fields(
        formatted_gps_geolocation=gps_geo,
        formatted_ins_geolocation=ins_geo,
        ecef_ephemeris=ecef,
        relative_ephemeris=relative,
        ephemeris_reference_identifier=0xCAFEBABE,
        gps_ascii=gps_ascii,
        context_association_lists=cal,
    )

    payload = cif.pack()
    parsed, used = CIF0Fields.parse(payload)

    assert used == len(payload)
    assert pytest.approx(parsed.formatted_gps_geolocation.latitude_deg, rel=1e-6) == lat_deg  # type: ignore[union-attr]
    assert pytest.approx(parsed.formatted_gps_geolocation.longitude_deg, rel=1e-6) == lon_deg  # type: ignore[union-attr]
    assert pytest.approx(parsed.formatted_gps_geolocation.heading_angle_deg, rel=1e-6) == heading_deg  # type: ignore[union-attr]
    assert pytest.approx(parsed.formatted_gps_geolocation.altitude_m, abs=1 / (1 << 5)) == alt_m  # type: ignore[union-attr]
    assert pytest.approx(parsed.formatted_gps_geolocation.speed_over_ground_m_s, rel=1e-6) == speed_m_s  # type: ignore[union-attr]

    assert pytest.approx(parsed.ecef_ephemeris.position_x_m, rel=1e-6) == 100.0  # type: ignore[union-attr]
    assert pytest.approx(parsed.ecef_ephemeris.position_y_m, rel=1e-6) == -200.0  # type: ignore[union-attr]
    assert pytest.approx(parsed.ecef_ephemeris.velocity_dy_m_s, rel=1e-6) == 20.5  # type: ignore[union-attr]

    assert parsed.ephemeris_reference_identifier == 0xCAFEBABE
    assert parsed.gps_ascii == gps_ascii
    assert parsed.context_association_lists == cal
