"""When a channel's spikes and noise cannot be measured, on numbers seen in real files."""

from rockland_noise.health import problem, slow_thermistor
from rockland_noise.reader import Evidence
from rockland_noise.setup import classify
from rockland_noise.spikes import ChannelResult

HOUR = 3600.0


def channel(name="sh1", *, noise=1.1, largest=6.0, saturated=0, threshold=None, kind="shear",
            fast=True, usable_s=HOUR, fs=512.0) -> ChannelResult:
    return ChannelResult(name, usable_s, noise, 6 * noise if threshold is None else threshold, 0,
                         0, float("nan"), largest, saturated, 0, kind=kind, fs=fs, fast=fast)


def bench_setup(t2=16.15):
    return classify(Evidence(t1=16.15, t2=t2, t1_ref=16.15, t2_ref=16.15, t1_spread=0.0,
                             t2_spread=0.0, pressure=0.1, pressure_spread=0.01,
                             pressure_high=0.2))


def test_healthy_channels_can_be_measured():
    assert problem(channel(), bench_setup()) is None
    assert problem(channel("T2_dT2", noise=3.2, largest=17, kind="therm"), bench_setup()) is None
    # VMP SN 132's quiet sh1: 0.67 counts, never more than 2 from its running median.
    assert problem(channel(noise=0.67, largest=2), bench_setup()) is None
    # A slow thermistor channel on a test probe hardly moves; that is not a fault.
    assert problem(channel("T1", noise=0.3, largest=1, kind="therm", fast=False, fs=64.0),
                   bench_setup()) is None
    # A glider thermistor channel clipping in 1% of its samples is still measured.
    assert problem(channel("T2_dT2", noise=77, kind="therm", saturated=int(0.0098 * HOUR * 512)),
                   bench_setup()) is None


def test_no_probe_comes_from_the_slow_thermistor_channel():
    """Hydrographer MR330: T2 read -16.05 C from file 107; T2_dT2 pinned near -32,347."""
    setup = bench_setup(t2=-16.054)
    assert setup.no_probe == {"T2": -16.054}
    assert "T2 no probe (-16.05 C is out of range)" in setup.detail
    p = problem(channel("T2_dT2", noise=0.4, largest=2, kind="therm"), setup)
    assert (p.kind, p.detail) == ("no probe", "T2 reads -16.05 C")
    # Floating instead of pinned (MR330 T1_dT1 at the end): still no probe.
    assert problem(channel("t2_dt2", noise=290, largest=5449, kind="therm"), setup).kind == \
        "no probe"
    assert problem(channel("T1_dT1", noise=3.2, largest=17, kind="therm"), setup) is None
    assert (slow_thermistor("T1_dT1"), slow_thermistor("T2"), slow_thermistor("sh1")) == \
        ("T1", "T2", None)


def test_railed_at_the_end_of_its_range():
    """ARCTERX Wake VMP SN 465 sh1 in the overnight hangs: 9-16% of samples at +-32767."""
    p = problem(channel(noise=22000, saturated=int(0.12 * HOUR * 512)), None)
    assert (p.kind, p.detail) == ("railed", "12% of its samples at the end of its range")


def test_railed_short_of_the_converters_limit():
    """VMP SN 479: rail to rail at -32,385 and +31,800, so nothing counts as 'at its limit'."""
    p = problem(channel(noise=20000, threshold=160000), None)
    assert (p.kind, p.detail) == ("railed", "threshold 160,000 counts, over half its range")
    assert problem(channel(noise=3000, threshold=24000), None) is None  # a glider at the surface


def test_flat():
    assert problem(channel(noise=0.0, largest=0), None).kind == "flat"
    p = problem(channel("T1_dT1", noise=0.41, largest=2, kind="therm"), None)
    assert (p.kind, p.detail) == ("flat", "noise 0.41 counts")
