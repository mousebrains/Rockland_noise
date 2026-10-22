"""Telling a bench test from real probes or a deployed instrument."""

import math

from rockland_noise.reader import Evidence
from rockland_noise.setup import classify, run_mode

NAN = math.nan
# Numbers like those measured on the MR433 tank bench tests (T_0 = 16.151 C).
BENCH = Evidence(t1=16.13, t2=16.13, t1_ref=16.151, t2_ref=16.151, t1_spread=0.0016,
                 t2_spread=0.0016, tilt_spread=0.025, shake=3.9, pressure=0.1,
                 pressure_spread=0.12)


def with_(**changes) -> Evidence:
    return Evidence(**{**{f: getattr(BENCH, f) for f in BENCH.__slots__}, **changes})


def test_bench_test():
    s = classify(BENCH)
    assert s.probes == "test" and s.steady and s.bench


def test_bench_test_on_a_ship_rolls_and_shakes():
    """Ocean Hunter III, ARCTERX 2023 Wake: still a valid bench test."""
    s = classify(with_(tilt_spread=9.8, shake=3251))
    assert s.bench and "tilt varied" in s.detail and "shaking" in s.detail


def test_test_probes_read_their_own_reference_temperature():
    """VMP SN132: T_0 is 7.88 C on T1 and 14.27 C on T2, and its test probes read that."""
    s = classify(with_(t1=7.876, t1_ref=7.876, t2=14.30, t2_ref=14.269))
    assert s.probes == "test" and s.bench


def test_real_probes():
    s = classify(with_(t1=12.1, t2=11.1, t1_spread=0.3, t2_spread=0.3))
    assert s.probes == "real" and not s.bench


def test_real_probes_in_water_at_the_reference_temperature_are_not_steady():
    assert classify(with_(t1=16.0, t2=16.2, t1_spread=0.4, t2_spread=0.5)).probes == "real"


def test_test_probes_on_a_deployed_instrument_is_not_a_bench_test():
    """ARCTERX 2023 Wake: a VMP used as a CTD, with test probes."""
    s = classify(with_(pressure_spread=124, tilt_spread=9.9, shake=131))
    assert s.probes == "test" and s.steady is False and not s.bench


def test_one_probe_missing():
    """VMP SN465: T1 had no probe (reads -17.09 C), T2 a test probe."""
    s = classify(with_(t1=-17.09, t1_spread=0.0))
    assert s.probes == "test" and s.bench


def test_a_faulty_test_probe_channel_is_still_a_test_probe():
    assert classify(with_(t1_spread=0.2)).probes == "test"


def test_without_pressure_tilt_and_shaking_decide():
    assert classify(with_(pressure_spread=NAN)).bench
    assert not classify(with_(pressure_spread=NAN, tilt_spread=20.0)).bench


def test_no_thermistors():
    assert classify(with_(t1=NAN, t2=NAN, t1_spread=NAN, t2_spread=NAN)).probes == "unknown"


def test_run_mode():
    bench, moved = classify(BENCH), classify(with_(pressure_spread=4.9))
    real = classify(with_(t1=20.0, t2=19.0, t1_spread=2.0, t2_spread=2.0))
    assert run_mode([bench] * 10)[0] == "bench"
    mode, why = run_mode([bench] * 9 + [moved])  # handled during one file
    assert mode == "bench" and "in 1 file(s) the pressure changed" in why
    assert run_mode([moved] * 5)[0] == "live"
    mode, why = run_mode([bench] * 5 + [real])
    assert mode == "live" and "separately" in why
    assert run_mode([real] * 3)[0] == "live"


def test_one_unsteady_channel_next_to_a_test_probe_is_flagged_not_called_real():
    """Bad temperature electronics: seen on one channel at a time, with test probes on."""
    s = classify(with_(t2=17.3, t2_spread=0.8))
    assert s.probes == "test" and s.bench
    assert s.warnings == ("T2 is unsteady, unlike a test probe: bad temperature electronics?",)


def test_jac_t_tells_bad_electronics_from_a_real_probe():
    bad = classify(with_(t2=20.0, t2_spread=3.0, t2_follows=0.05, water_spread=5.0))
    assert "does not follow the water temperature" in bad.warnings[0]
    real = classify(with_(t2=20.0, t2_spread=3.0, t2_follows=0.97, water_spread=5.0))
    assert "follows the water temperature (JAC_T): a real probe?" in real.warnings[0]
    # Too little change in the water to tell either way.
    unsure = classify(with_(t2=20.0, t2_spread=0.3, t2_follows=0.1, water_spread=0.5))
    assert "unlike a test probe" in unsure.warnings[0]
