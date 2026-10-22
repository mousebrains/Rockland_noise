"""The standalone P-file reader.

The reference values for the template file were read with microstructure-tpw
0.4.0 (odas_tpw.rsi.p_file.PFile, deconvolve=False), the reader this one was
checked against sample for sample on 23 files from MicroRiders and VMPs
(formats 6.0 to 6.3, 512 and 1024 Hz, 32-bit channels, real bad-buffer runs).
"""

from datetime import UTC, datetime

import numpy as np
import pytest

from rockland_noise.pfile import BAD_BUFFER_VALUE, PFile, PFileError, parse_config

from .conftest import T0, TEMPLATE

# name: (samples, sum, first, last), raw counts
REFERENCE = {
    "Ax": (10240, -14470, -2449, -577),
    "Ay": (10240, -25163, 10, -967),
    "EMC_Cur": (10240, -5967109, -3498, -3849),
    "Gnd": (1280, -1274, -1, -1),
    "Incl_T": (1280, 1664000, 1300, 1300),
    "Incl_X": (1280, -574696, -458, -435),
    "Incl_Y": (1280, -1070590, -642, -1055),
    "P": (1280, 11106848, 8686, 8666),
    "PV": (1280, -2191917, -1712, -1713),
    "P_dP": (1280, 11087282, 8681, 8642),
    "T1": (1280, -3239892, -2532, -2530),
    "T1_dT1": (10240, -25936687, -2537, -2535),
    "T2": (1280, -4017776, -3141, -3137),
    "T2_dT2": (10240, -32149356, -3147, -3131),
    "U_EM": (1280, 4447986, 0, 3609),
    "V_Bat": (1280, 27896882, 21853, 21735),
    "sh1": (10240, -4208, 538, -265),
    "sh2": (10240, -25068, 767, 1148),
}


def test_header_and_timing():
    pf = PFile(TEMPLATE)
    assert pf.start_time == datetime(2025, 2, 12, 6, 43, 1, 853000, tzinfo=UTC)
    assert pf.fs_fast == 512.0327777777778
    assert pf.fs_slow == 64.00409722222223
    assert pf.n_records == 20
    assert pf.instrument == "MR1000RDL-EM SN 435"
    assert not pf.notes


def test_every_channel_matches_the_reference():
    pf = PFile(TEMPLATE)
    assert sorted(pf.channel_names) == sorted(REFERENCE)
    for name, (n, total, first, last) in REFERENCE.items():
        x = pf.raw(name).astype(np.int64)
        assert (x.size, int(x.sum()), int(x[0]), int(x[-1])) == (n, total, first, last), name
        assert pf.is_fast(name) == (n == 10240), name


@pytest.mark.parametrize(("name", "median", "units"), [
    ("T1", 12.097173169808286, "deg_C"),
    ("T2", 11.109155136284869, "deg_C"),
    ("P", 506.20719558128803, "dBar"),
    ("Incl_X", 11.25, "deg"),
    ("Incl_Y", -20.6625, "deg"),
    ("Incl_T", 13.0, "deg_C"),
])
def test_physical_units_match_the_reference(name, median, units):
    values, got_units = PFile(TEMPLATE).physical(name)
    assert float(np.median(values)) == pytest.approx(median, rel=1e-12)
    assert got_units == units


def test_shear_is_not_converted():
    with pytest.raises(PFileError, match="no conversion"):
        PFile(TEMPLATE).physical("sh1")


def test_bad_buffer_runs_but_not_lone_values(maker, tmp_path):
    n = maker.fast_samples(10)
    x = np.zeros(n)
    x[1000:1010] = BAD_BUFFER_VALUE  # a dropout
    x[3000] = BAD_BUFFER_VALUE  # a value that just happens to match
    pf = PFile(maker.write(tmp_path / "bad.p", start=T0, seconds=10, channels={"T1_dT1": x}))
    bad = pf.bad("T1_dT1")
    assert bad[1000:1010].all() and bad.sum() == 10


def test_config_parsing():
    cfg = parse_config("""; a comment
    [Root]
    recsize = 2 ; trailing comment
    [cruise info]
    Ship = Sikuliaq
    [matrix]
    row01 = 1 2
    [channel]
    id = 1
    NAME = sh1
    """)
    assert cfg["root"]["recsize"] == "2"
    assert cfg["cruise_info"]["ship"] == "Sikuliaq"
    assert cfg["matrix"] == [[1, 2]]
    assert cfg["channels"] == [{"id": "1", "name": "sh1"}]


def test_refuses_what_it_cannot_read(tmp_path):
    small = tmp_path / "small.p"
    small.write_bytes(b"\x00" * 50)
    with pytest.raises(PFileError, match="too small"):
        PFile(small)
    raw = bytearray(TEMPLATE.read_bytes())
    old = tmp_path / "v1.p"
    raw[20:22] = (1).to_bytes(2, "big")  # header word 11: format version 1
    old.write_bytes(raw)
    with pytest.raises(PFileError, match="format-1"):
        PFile(old)
    header_only = tmp_path / "empty.p"
    header_only.write_bytes(TEMPLATE.read_bytes()[:128 + 9032])
    with pytest.raises(PFileError, match="no data records"):
        PFile(header_only)


def test_partial_last_record_is_noted(tmp_path):
    cut = tmp_path / "cut.p"
    cut.write_bytes(TEMPLATE.read_bytes()[:-100])
    pf = PFile(cut)
    assert pf.n_records == 19 and "partial record" in pf.notes[0]
