"""The spike counter and noise measure, on synthetic channels with known answers."""

import numpy as np
import pytest

from rockland_noise.spikes import MODES, SpikeSettings, analyze_channel, noise_level

from .conftest import add_bursts, noise

FS = 512.0
TEN_MIN = int(600 * FS)


def counts(x):
    """Whole numbers, as an analog-to-digital converter records them."""
    return np.round(x)


@pytest.mark.parametrize("sigma", [1.1, 3.2, 10.0])
def test_noise_level_matches_known_noise(sigma):
    x = counts(noise(TEN_MIN, sigma, seed=1))
    hp = x - np.median(x)
    # Rounding to whole counts adds 1/12 count^2 of variance.
    assert noise_level(hp) == pytest.approx(np.sqrt(sigma**2 + 1 / 12), rel=0.03)


def test_noise_level_changes_smoothly_on_whole_counts():
    """The median absolute deviation jumps in steps on whole counts; this must not."""
    sigmas = np.linspace(2.0, 3.0, 11)
    levels = [noise_level(counts(noise(TEN_MIN, s, seed=2))) for s in sigmas]
    assert np.all(np.diff(levels) > 0)
    assert np.max(np.diff(levels)) < 0.2


def test_noise_level_ignores_spikes():
    x = counts(noise(TEN_MIN, 3.0, seed=3))
    spiky = add_bursts(x, list(range(1000, TEN_MIN - 100, 5000)), size=300)
    assert noise_level(spiky) == pytest.approx(noise_level(x), rel=0.02)


def test_noise_level_flat_and_empty():
    assert noise_level(np.zeros(1000)) == 0
    assert np.isnan(noise_level(np.array([])))


@pytest.mark.parametrize("mode", ["bench", "live"])
def test_counts_every_injected_burst(mode):
    starts = list(range(2000, TEN_MIN - 2000, 6000))  # about 50 bursts, 11.7 s apart
    x = counts(add_bursts(noise(TEN_MIN, 3.0, -20, seed=4), starts, size=80))
    r = analyze_channel("T1_dT1", x, FS, np.ones(x.size, bool), MODES[mode])
    assert r.spikes == len(starts)
    assert r.positive_first == len(starts)
    assert r.rate_per_hour == pytest.approx(len(starts) * 6, rel=0.01)
    assert r.noise == pytest.approx(3.0, rel=0.05)


@pytest.mark.parametrize("mode", ["bench", "live"])
def test_no_false_spikes_in_clean_noise(mode):
    x = counts(noise(6 * TEN_MIN, 3.2, seed=5))
    r = analyze_channel("T2_dT2", x, FS, np.ones(x.size, bool), MODES[mode])
    assert r.spikes == 0


def test_bench_is_more_sensitive_than_live():
    starts = list(range(2000, TEN_MIN - 2000, 6000))
    # 7 x the noise: mostly a spike on the bench (6 x), mostly not in live data
    # (8 x). The noise riding on each burst moves some across either line.
    x = counts(add_bursts(noise(TEN_MIN, 3.0, seed=6), starts, size=21))
    ones = np.ones(x.size, bool)
    bench = analyze_channel("x", x, FS, ones, MODES["bench"])
    live = analyze_channel("x", x, FS, ones, MODES["live"])
    assert bench.threshold == pytest.approx(6 * bench.noise)
    assert live.threshold == pytest.approx(8 * live.noise)
    # A 7 x burst crosses 6 x unless its noise is below -1 sd (16% of the time)
    # and crosses 8 x only when its noise is above +1 sd (also 16%).
    assert bench.spikes >= 0.7 * len(starts)
    assert live.spikes <= 0.3 * len(starts)
    assert bench.spikes > 3 * live.spikes


def test_close_bursts_count_once():
    x = counts(noise(TEN_MIN, 3.0, seed=7))
    near = add_bursts(x, [10_000, 10_000 + int(0.05 * FS)])
    apart = add_bursts(x, [10_000, 10_000 + int(0.3 * FS)])
    ones = np.ones(x.size, bool)
    assert analyze_channel("x", near, FS, ones, MODES["live"]).spikes == 1
    assert analyze_channel("x", apart, FS, ones, MODES["live"]).spikes == 2


def test_unusable_samples_are_not_counted():
    x = counts(add_bursts(noise(TEN_MIN, 3.0, seed=8), [10_000, 200_000]))
    usable = np.ones(x.size, bool)
    usable[:100_000] = False
    r = analyze_channel("x", x, FS, usable, MODES["live"])
    assert r.spikes == 1
    assert r.usable_s == pytest.approx((x.size - 100_000) / FS)


def test_bad_logger_samples_make_no_spike():
    x = counts(noise(TEN_MIN, 3.0, seed=9))
    bad = np.zeros(x.size, bool)
    bad[50_000:50_020] = True
    x[bad] = -32753  # the logger's bad-buffer marker
    r = analyze_channel("x", x, FS, ~bad, MODES["live"], bad=bad)
    assert r.spikes == 0
    assert r.bad_buffer_samples == 20


def test_fixed_threshold_and_floor():
    x = counts(add_bursts(noise(TEN_MIN, 3.0, seed=10), [50_000], size=30))
    ones = np.ones(x.size, bool)
    fixed = SpikeSettings(threshold_sigma=8, min_threshold_counts=10, fixed_threshold_counts=20)
    assert analyze_channel("x", x, FS, ones, fixed).threshold == 20
    flat = analyze_channel("x", np.zeros(10_000), FS, np.ones(10_000, bool), MODES["live"])
    assert flat.noise == 0 and flat.threshold == 10 and flat.spikes == 0


def test_checklist_numbers():
    x = counts(noise(TEN_MIN, 3.0, offset=-25, seed=11))
    r = analyze_channel("T1_dT1", x, FS, np.ones(x.size, bool), MODES["bench"])
    assert r.mean == pytest.approx(-25, abs=0.1)
    # 99.9% of a normal distribution lies within 3.29 standard deviations.
    assert r.typical_range == pytest.approx(3.29 * 3.0, rel=0.05)
