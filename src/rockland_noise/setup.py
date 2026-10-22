"""Tell a bench test from data taken with real probes, from the file itself.

A bench test has Rockland TEST PROBES on, and the instrument stays at the
surface: on a bench, on a ship's deck, or in a shallow test tank. Both show in
the data.

Test probes
    The temperature test probe is a fixed resistor equal to the thermistor
    calibration's reference resistance, so T1 and T2 read a steady temperature
    equal to the reference temperature T_0 in the instrument's setup (usually
    16.15 deg C, which is why "about 16.3" is the rule of thumb; but 7.88, 9.45
    or 14.27 on some probes). Measured on bench tests of MicroRiders SN 433,
    435, 533 and VMPs SN 132, 142, 409, 465: 99% of channels read within 0.03
    deg C of T_0 (0.2 at most, on a channel full of fault spikes), the middle
    half of the samples within 0.005 deg C (0.2 with the fault). None of 376
    channel-files with real probes, from gliders, moorings and profilers,
    came within 0.5 deg C of T_0 while steady.

At the surface
    The pressure stays near zero and steady: 95% of the readings at most 1.2
    dbar, and the middle half within 0.5 dbar. Bench tests read -0.9 to +0.7
    dbar (sensor offsets and a shallow test tank) with the middle half within
    0.12 dbar, or 0.42 on the deck of the Ocean Hunter III (ARCTERX 2023
    Wake). Not bench tests: VMPs hung overnight at 2 to 5 m in a lagoon (1.9
    dbar or more, and swinging with wind and ship motion), casts (a VMP used
    as a CTD with test probes: tens of dbar), and a file that is mostly on
    deck but includes a cast. Handling at the start and end of a file does
    not count. Tilt (Incl_X, Incl_Y) and shaking (Ax, Ay)
    are reported but do not decide: a bench test on a ship (Ocean Hunter III,
    ARCTERX 2023 Wake) rolls and shakes and is still a valid test of the
    temperature and shear channels. They are used only if a file has no
    pressure channel.

One unsteady temperature channel next to a test probe usually means bad
temperature electronics (seen in practice on one channel, never on both), so one
channel reading its T_0 is enough to call the file a test-probe file; the other
is flagged. On a VMP the CT sensor's temperature (JAC_T) tells a real probe from
bad electronics: in every VMP cast where the water temperature varied by more
than 2 deg C, real probes followed JAC_T with a rank correlation of at least
0.52 (157 channel-files from SN 132, 409, 465, 479; median 1.00). The
inclinometer's temperature (Incl_T) cannot: it is the housing temperature and
lags the water (correlation about 0 on VMP casts, -0.8 to 1.0 on gliders).

Spreads use the middle of the samples, so handling at the start and end of a
file does not count.
"""

import math
from dataclasses import dataclass

from .reader import Evidence

TEST_PROBE_TOLERANCE = 0.5  # deg C from T_0
TEST_PROBE_MAX_SPREAD = 0.25  # deg C, middle half of the samples
# Readings this far out mean no probe is connected (the electronics hit a limit).
NO_PROBE_LOW, NO_PROBE_HIGH = -10.0, 50.0
NO_PROBE_MAX_SPREAD = 0.01
STEADY_PRESSURE = 0.5  # dbar, interquartile range
AT_SURFACE = 1.2  # dbar, 95th percentile
FOLLOWS = 0.5  # rank correlation with JAC_T that a real probe reaches...
WATER_VARIES = 2.0  # ...whenever the water temperature varies this much, deg C
# Only when there is no pressure channel:
STILL_TILT = 1.0  # degrees, middle half of Incl_X / Incl_Y
STILL_SHAKE = 300.0  # counts, typical 1-s standard deviation of Ax / Ay


@dataclass(frozen=True, slots=True)
class Setup:
    probes: str  # "test" (at least one test probe), "real" or "unknown" (no thermistors)
    steady: bool | None  # at the surface, pressure steady; None when nothing tells
    detail: str  # the evidence in a few words
    warnings: tuple[str, ...] = ()  # e.g. a temperature channel that looks faulty

    @property
    def bench(self) -> bool:
        return self.probes == "test" and self.steady is not False


def _thermistor(t: float, ref: float, spread: float) -> str | None:
    if math.isnan(t):
        return None
    ref = 16.15 if math.isnan(ref) else ref
    if abs(t - ref) <= TEST_PROBE_TOLERANCE and spread <= TEST_PROBE_MAX_SPREAD:
        return "test"
    if (t < NO_PROBE_LOW or t > NO_PROBE_HIGH) and spread <= NO_PROBE_MAX_SPREAD:
        return "none"
    return "real"


def _odd_channel(name: str, follows: float, water_spread: float) -> str:
    """Why a temperature channel next to a test probe is not steady at its T_0."""
    if not math.isnan(follows) and water_spread >= WATER_VARIES:
        if follows >= FOLLOWS:
            return (f"{name} is not a test probe and follows the water temperature (JAC_T):"
                    " a real probe?")
        return (f"{name} is unsteady and does not follow the water temperature (JAC_T):"
                " bad temperature electronics?")
    return f"{name} is unsteady, unlike a test probe: bad temperature electronics?"


def classify(ev: Evidence) -> Setup:
    channels = [("T1", _thermistor(ev.t1, ev.t1_ref, ev.t1_spread), ev.t1_follows),
                ("T2", _thermistor(ev.t2, ev.t2_ref, ev.t2_spread), ev.t2_follows)]
    kinds = [k for _, k, _ in channels if k is not None]
    warnings = []
    if "test" in kinds:
        probes = "test"
        warnings = [_odd_channel(name, follows, ev.water_spread)
                    for name, k, follows in channels if k == "real"]
    elif "real" in kinds:
        probes = "real"
    else:
        probes = "unknown"

    context = []
    if ev.tilt_spread > STILL_TILT:
        context.append(f"tilt varied {ev.tilt_spread:.1f} deg")
    if ev.shake > STILL_SHAKE:
        context.append(f"shaking {ev.shake:,.0f} counts")
    if not math.isnan(ev.pressure_spread):
        if ev.pressure_spread > STEADY_PRESSURE:
            steady = False
            context.insert(0, f"pressure changing (middle half spans"
                              f" {ev.pressure_spread:,.1f} dbar)")
        elif not ev.pressure_high <= AT_SURFACE:
            steady = False
            where = (f"at {ev.pressure:,.1f} dbar" if ev.pressure > AT_SURFACE
                     else f"down to {ev.pressure_high:,.1f} dbar")
            context.insert(0, f"in the water {where} (hung or deployed)")
        else:
            steady = True
            context.insert(0, f"pressure steady near the surface ({ev.pressure:,.1f} dbar)")
    elif not (math.isnan(ev.tilt_spread) and math.isnan(ev.shake)):
        steady = not context
    else:
        steady = None

    temps = ", ".join(f"{name} {t:.2f} C" for name, t in (("T1", ev.t1), ("T2", ev.t2))
                      if not math.isnan(t))
    return Setup(probes, steady, "; ".join(([temps] if temps else []) + context),
                 tuple(warnings))


def run_mode(setups: list[Setup]) -> tuple[str, str]:
    """The mode for a whole set of files, and why, in plain words."""
    n = len(setups)
    test = sum(s.probes == "test" for s in setups)
    real = sum(s.probes == "real" for s in setups)
    bench = sum(s.bench for s in setups)
    if n == 0:
        return "live", "no files to judge by"
    if test and real:
        return "live", (f"{test} file(s) look like test probes and {real} like real probes;"
                        " using live, the cautious choice. Run the two kinds separately.")
    if test == n and bench * 2 >= n:
        moved = n - bench
        extra = (f"; {moved} file(s) were in the water or moving (handled or deployed?)"
                 if moved else "")
        return "bench", (f"T1 and T2 read their steady reference temperature (test probes)"
                         f" and the instrument stayed at the surface in {bench} of {n}"
                         f" files{extra}")
    if test == n:
        return "live", (f"test probes, but the instrument was in the water or moving in"
                        f" {n - bench} of {n} files (hung or deployed, e.g. a VMP used as"
                        " a CTD?)")
    if real:
        return "live", f"real probes ({real} of {n} files)"
    return "live", "no thermistor channels to tell test probes from real ones"
