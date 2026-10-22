"""Agreement with the AEM1-G analysis on real MR433 files, where those files exist.

AEM1-G (results/mr_t1_spike_history_raw.csv) counted MR433 T1 fault bursts in
the 2026 tank bench tests (test probes, in a water tank) with a threshold that
came out at 23.7 counts, skipping the first 600 s. With the same fixed
threshold this program must give the same counts. The P-files are large and
not part of this repository; set
ROCKLAND_NOISE_AEM1G to the AEM1-G Tank_Test folder to run these tests.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from rockland_noise.reader import read_channels
from rockland_noise.spikes import MODES, SpikeSettings, analyze_channel

FOLDER = Path(os.environ.get("ROCKLAND_NOISE_AEM1G",
                             Path(__file__).parents[2] / "AEM1-G" / "data" / "Tank_Test"))
TANK = FOLDER / "Western" / "Config7" / "MR433"

pytestmark = pytest.mark.skipif(not TANK.is_dir(), reason="AEM1-G tank files not available")

AEM1G = SpikeSettings(threshold_sigma=8, min_threshold_counts=10, fixed_threshold_counts=23.7)


@pytest.mark.parametrize(("name", "t1", "t2"), [
    ("RIOT_SL684_0520.p", 6781, 0),
    ("RIOT_SL684_0511.p", 63, 0),
    ("RIOT_SL684_0524.p", 312, 0),
])
def test_same_counts_as_aem1g(name, t1, t2):
    data = read_channels(TANK / name, ["T1_dT1", "T2_dT2"], skip_start_s=600)
    got = [analyze_channel(c.name, c.counts, c.fs, c.usable, AEM1G, bad=c.bad).spikes
           for c in data.channels.values()]
    assert got == [t1, t2]


def test_healthy_channels_have_no_bench_spikes():
    """RIOT_SL684_0511 is a quiet 4-hour bench file: its healthy channels never reach 6 x
    their noise. (Busier bench sessions do have some, from handling; see spikes.py.)"""
    data = read_channels(TANK / "RIOT_SL684_0511.p", ["T2_dT2", "sh1", "sh2"], skip_start_s=600)
    for c in data.channels.values():
        r = analyze_channel(c.name, c.counts, c.fs, c.usable, MODES["bench"], bad=c.bad)
        assert r.spikes == 0, c.name
        assert np.isfinite(r.noise)


def test_tank_files_are_recognised_as_bench_tests():
    from rockland_noise.setup import classify

    data = read_channels(TANK / "RIOT_SL684_0511.p", ["T1_dT1"])
    s = classify(data.evidence)
    assert s.probes == "test" and s.bench
    assert abs(data.evidence.t1 - data.evidence.t1_ref) < 0.05


def test_the_t1_fault_is_on_t1_alone():
    """MR433's T1 fault bursts are electrical: T2 shares almost none of them."""
    from rockland_noise.spikes import SHARED_WINDOW_S, analyze_channel_times, shared_spikes

    data = read_channels(TANK / "RIOT_SL684_0520.p", ["T1_dT1", "T2_dT2"], skip_start_s=600)
    (r1, t1), (_, t2) = (analyze_channel_times(c.name, c.counts, c.fs, c.usable,
                                               [MODES["bench"]], bad=c.bad)[0]
                         for c in data.channels.values())
    assert r1.spikes > 5000
    assert shared_spikes(t1, t2, SHARED_WINDOW_S) <= 0.01 * r1.spikes
