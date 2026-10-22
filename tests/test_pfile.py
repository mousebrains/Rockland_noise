"""The standalone P-file reader.

The reference values were read with microstructure-tpw 0.4.0
(odas_tpw.rsi.p_file.PFile, deconvolve=False), the reader this one was checked
against sample for sample on 23 files from MicroRiders and VMPs. Four small
real files cover the code paths:

* MR_SL435_template.p: MicroRider SN 435, format 6.3, 512 Hz (20 records).
* MR_SN429_v60_1024Hz.p: MicroRider SN 429, format 6.0, 1024 Hz (the first 6
  records of palau2023_0006.p).
* VMP_SN479_dropout.p: VMP SN 479, format 6.1, with a 32-bit channel (JAC_C),
  unsigned channels (JAC_T, Chlorophyll, Turbidity) and a real 64-sample
  dropout on DO_T (records 57 to 66 of ARCTERX_Thompson_2025_SN479_0002.p).
* VMP_SN409_deck.p: VMP SN 409 with test probes on the deck of the Ocean
  Hunter III, ARCTERX 2023 Wake (records 2000 to 2009 of OH3409_0004.p; the
  operator's initials in the setup text are blanked).

Each cut keeps record 0 (the header and setup) and whole data records, so it
is a valid P-file whose start time is that of the original file.

Gnd is the exception: it fills 2 to 6 spare cells of each matrix cycle, and
microstructure-tpw keeps only the first. Its values here are from the ODAS
MATLAB library's read_odas.m, which reads every cell in the order sampled.
"""

from datetime import UTC, datetime

import numpy as np
import pytest

from rockland_noise import pfile
from rockland_noise.pfile import BAD_BUFFER_VALUE, EmptyPFileError, PFile, PFileError, parse_config

from .conftest import DATA, T0, TEMPLATE

# file: (start, fs_fast, records, {channel: (samples, sum, first, last, bad samples)},
#        {channel: median in physical units})
FIXTURES = {
    "MR_SN429_v60_1024Hz.p": (
        datetime(2023, 5, 7, 3, 52, 32, 474000, tzinfo=UTC), 1023.6577777777778, 6, {
            "Ax": (6144, 5132, 2036, 8468, 0),
            "Ay": (6144, -31743, 357, -7486, 0),
            "EMC_Cur": (6144, -6577048, -7355, -6949, 0),
            "Gnd": (4608, -5036, -1, -1, 0),
            "Incl_T": (768, 960000, 1250, 1250, 0),
            "Incl_X": (768, 2679015, 3600, 3600, 0),
            "Incl_Y": (768, 28236, 31, 20, 0),
            "P": (768, 143416, 187, 186, 0),
            "P_dP": (768, 141949, 183, 186, 0),
            "PV": (768, -2409054, -3137, -3136, 0),
            "sh1": (6144, -19514, -19, 196, 0),
            "sh2": (6144, -22904, 87, 187, 0),
            "T1": (768, 11023691, 14306, 14306, 0),
            "T1_dT1": (6144, 88176293, 14236, 14540, 0),
            "T2": (768, 9743051, 12637, 12765, 0),
            "T2_dT2": (6144, 78059392, 12968, 12823, 0),
            "U_EM": (768, 2136936, 0, 2590, 0),
            "V_Bat": (768, 18353777, 23862, 23952, 0),
        }, {"Incl_T": 36.5, "Incl_X": -90.0, "Incl_Y": 0.9500000000000001,
            "P": 0.6486451729100011, "T1": 43.23761879993248, "T2": 39.43437360286043}),
    "VMP_SN479_dropout.p": (
        datetime(2025, 1, 14, 15, 30, 16, 445000, tzinfo=UTC), 511.94539999999995, 10, {
            "Ax": (5120, -16943, 2955, 331, 0),
            "Ay": (5120, -23126, 6812, 1350, 0),
            "Chlorophyll": (5120, 7585743, 1466, 1499, 0),
            "DO": (640, 15029786, 23458, 23464, 0),
            "DO_T": (640, -20952849, -32740, -32725, 64),
            "Gnd": (1280, -1939, -1, -1, 0),
            "Incl_T": (640, 822400, 1285, 1285, 0),
            "Incl_X": (640, -2089851, -3286, -3600, 0),
            "Incl_Y": (640, -16743, -20, -14, 0),
            "JAC_C": (640, 152822185, 211752, 277290, 0),
            "JAC_T": (640, 24545794, 38359, 38375, 0),
            "P": (640, 16201, 25, 26, 0),
            "P_dP": (640, 15774, 56, 23, 0),
            "PV": (640, -1209690, -1892, -1889, 0),
            "sh1": (5120, 1445, 1665, 281, 0),
            "sh2": (5120, -13365, 2513, 1304, 0),
            "T1": (640, 4167541, 6511, 6505, 0),
            "T1_dT1": (5120, 33331684, 6534, 6552, 0),
            "T2": (640, 7096961, 11086, 11079, 0),
            "T2_dT2": (5120, 56780220, 11054, 11101, 0),
            "Turbidity": (5120, 7032921, 1376, 1377, 0),
            "V_Bat": (640, 16783130, 26228, 26225, 0),
        }, {"Incl_T": 20.050000000000068, "Incl_X": 86.125, "Incl_Y": -0.525,
            "JAC_T": 27.988511537863445, "P": 0.22687384062499993, "T1": 27.27139201213032,
            "T2": 36.054012277553}),
    "VMP_SN409_deck.p": (
        datetime(2023, 5, 31, 4, 6, 12, 157000, tzinfo=UTC), 512.03275, 10, {
            "Ax": (5120, 46341, -692, -307, 0),
            "Ay": (5120, 28969, 273, 782, 0),
            "Gnd": (2560, 12342, 5, 5, 0),
            "Incl_T": (640, 803840, 1256, 1256, 0),
            "Incl_X": (640, 309101, 487, 469, 0),
            "Incl_Y": (640, -2008046, -3170, -3140, 0),
            "JAC_C": (640, 198824543, 341993, 342005, 0),
            "JAC_T": (640, 23182452, 36065, 36922, 0),
            "P": (640, 43347, 68, 67, 0),
            "P_dP": (640, 41037, 65, 64, 0),
            "PV": (640, -115161, -180, -180, 0),
            "sh1": (5120, 24772, 5, 4, 0),
            "sh2": (5120, 26481, 5, 5, 0),
            "T1": (640, -10080, -16, -16, 0),
            "T1_dT1": (5120, -20397, -6, 0, 0),
            "T2": (640, -7023, -11, -11, 0),
            "T2_dT2": (5120, 3564, 3, 5, 0),
            "V_Bat": (640, 15884279, 24819, 24818, 0),
        }, {"Incl_T": 33.680000000000064, "Incl_X": -12.100000000000001,
            "Incl_Y": -78.28750000000001, "JAC_T": 25.527585043869337,
            "P": -0.30018611545599994, "T1": 16.146275475627817, "T2": 16.157512043231407}),
}

# name: (samples, sum, first, last), raw counts
REFERENCE = {
    "Ax": (10240, -14470, -2449, -577),
    "Ay": (10240, -25163, 10, -967),
    "EMC_Cur": (10240, -5967109, -3498, -3849),
    "Gnd": (7680, -7587, -1, 0),
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
    header_only = tmp_path / "header_only.p"
    header_only.write_bytes(TEMPLATE.read_bytes()[:128 + 9032])
    with pytest.raises(EmptyPFileError, match="header and setup but no data"):
        PFile(header_only)
    nothing = tmp_path / "nothing.p"
    nothing.write_bytes(b"")
    with pytest.raises(EmptyPFileError, match="0 bytes"):
        PFile(nothing)
    blank = tmp_path / "blank.p"
    blank.write_bytes(b"\xe9\xa6\x49\x63" + bytes(200_000))  # seen in a real unfinished file
    with pytest.raises(PFileError, match="header is blank"):
        PFile(blank)


def test_partial_last_record_is_noted(tmp_path):
    cut = tmp_path / "cut.p"
    cut.write_bytes(TEMPLATE.read_bytes()[:-100])
    pf = PFile(cut)
    assert pf.n_records == 19 and "partial record" in pf.notes[0]


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_real_files_match_the_reference(name):
    start, fs_fast, records, raw, medians = FIXTURES[name]
    pf = PFile(DATA / name)
    assert (pf.start_time, pf.fs_fast, pf.n_records) == (start, fs_fast, records)
    assert not pf.notes
    assert sorted(pf.channel_names) == sorted(raw)
    for ch, (n, total, first, last, n_bad) in raw.items():
        x = pf.raw(ch)
        assert (x.size, int(np.sum(x, dtype=np.int64)), int(x[0]), int(x[-1])) == \
            (n, total, first, last), ch
        assert int(pf.bad(ch).sum()) == n_bad, ch
    for ch, median in medians.items():
        assert float(np.median(pf.physical(ch)[0])) == pytest.approx(median, rel=1e-12), ch


def test_word_sizes_and_limits():
    pf = PFile(DATA / "VMP_SN479_dropout.p")
    assert (pf.bits("JAC_C"), pf.limits("JAC_C")) == (32, None)
    assert pf.limits("JAC_T") == (0, 65535) and pf.raw("JAC_T").min() >= 0
    assert pf.limits("T1_dT1") == (-32768, 32767)


def test_bad_buffer_run_in_a_real_file():
    pf = PFile(DATA / "VMP_SN479_dropout.p")
    bad = np.flatnonzero(pf.bad("DO_T"))
    assert bad.size == 64 and np.all(np.diff(bad) == 1)
    assert np.all(pf.raw("DO_T")[bad] == BAD_BUFFER_VALUE)
    assert not pf.bad("JAC_T").any()


def test_reading_a_few_records_at_a_time_gives_the_same_samples(monkeypatch):
    whole = PFile(DATA / "VMP_SN479_dropout.p")
    expected = {n: whole.raw(n) for n in whole.channel_names}
    monkeypatch.setattr(pfile, "READ_CHUNK_BYTES", 1)  # one record per read
    piecemeal = PFile(DATA / "VMP_SN479_dropout.p")
    piecemeal.load(["sh1", "JAC_C"])
    for n, x in expected.items():
        assert np.array_equal(piecemeal.raw(n), x), n


def test_odd_header_values_are_noted(tmp_path):
    raw = bytearray(TEMPLATE.read_bytes())
    raw[6:8] = (1980).to_bytes(2, "big")  # header word 4: the year
    odd_date = tmp_path / "odd_date.p"
    odd_date.write_bytes(raw)
    assert any("date (1980-" in n for n in PFile(odd_date).notes)
    raw = bytearray(TEMPLATE.read_bytes())
    raw[40:42] = (50).to_bytes(2, "big")  # header word 21: the clock, in Hz
    slow_clock = tmp_path / "slow_clock.p"
    slow_clock.write_bytes(raw)
    assert any("sample rate" in n for n in PFile(slow_clock).notes)


def test_an_id_in_several_cells_is_read_from_every_cell_in_order():
    """As read_odas.m: Gnd fills 6 spare cells of the template's 8 x 9 matrix."""
    pf = PFile(TEMPLATE)
    cells = np.flatnonzero(pf.matrix.ravel() == pf.setup("Gnd").ids[0])
    assert cells.size == 6
    assert pf.rate("Gnd") == pf.fs_fast * 6 / 8
    assert (pf.rate("sh1"), pf.rate("P")) == (pf.fs_fast, pf.fs_slow)
    dtype = np.dtype([("hdr", ">u2", (64,)), ("data", ">i2", (pf.data_words,))])
    records = np.fromfile(TEMPLATE, dtype=dtype, offset=pf.first_record)
    expected = records["data"].reshape(-1, pf.matrix.size)[:, cells].ravel()
    assert np.array_equal(pf.raw("Gnd"), expected)


def test_record_numbers_wrap_after_65535(tmp_path):
    """The record number is a 16-bit word: 65535 is followed by 0, not a gap."""
    pf = PFile(TEMPLATE)

    def numbered(name, numbers):
        raw = bytearray(TEMPLATE.read_bytes())
        for k, number in enumerate(numbers):
            at = pf.first_record + k * pf.record_size + 2  # header word 2
            raw[at:at + 2] = (number % 2**16).to_bytes(2, "big")
        (tmp_path / name).write_bytes(raw)
        out = PFile(tmp_path / name)
        out.load(["sh1"])
        return out.notes

    n = pf.n_records
    assert numbered("wrapped.p", [65530 + k for k in range(n)]) == []
    # Records 0 and 1 are missing, just after the wrap.
    assert numbered("gap.p", [65530 + k + 2 * (k >= 6) for k in range(n)]) == [
        "2 record(s) are missing from the file; later times are shifted earlier"]
