"""Jumps, trends and unusual files, on series with known answers."""

import numpy as np
import pytest

from rockland_noise.changes import ChangeSettings, find_changes

N = 30
DAYS = np.arange(N) / 6.0  # one file every 4 hours
NOISE = np.random.default_rng(1).normal(0, 0.15, N)  # log10 scatter of about x1.4
STEP = np.arange(N)


def test_flat_finds_nothing():
    c = find_changes(DAYS, NOISE)
    assert not (c.jumps or c.trends or c.unusual)


@pytest.mark.parametrize("size", [0.4, 1.0, 3.0])
def test_one_jump(size):
    c = find_changes(DAYS, NOISE + np.where(STEP >= 15, size, 0))
    assert [j.index for j in c.jumps] == [15]
    assert c.jumps[0].factor == pytest.approx(10**size, rel=0.5)
    assert not c.trends


def test_two_jumps():
    y = NOISE + np.where(STEP >= 10, 1.0, 0) + np.where(STEP >= 20, -0.7, 0)
    c = find_changes(DAYS, y)
    assert [j.index for j in c.jumps] == [10, 20]
    assert c.jumps[0].factor > 1 > c.jumps[1].factor


def test_a_ramp_is_a_trend_not_a_jump():
    c = find_changes(DAYS, NOISE + np.linspace(0, 2, N))
    assert not c.jumps
    assert len(c.trends) == 1
    assert c.trends[0].factor == pytest.approx(100, rel=0.5)


def test_ramp_then_jump():
    c = find_changes(DAYS, NOISE + np.linspace(0, 0.8, N) + np.where(STEP >= 20, 1.0, 0))
    assert [j.index for j in c.jumps] == [20]
    assert [(t.first, t.last) for t in c.trends] == [(0, 19)]


def test_one_odd_file_is_unusual_not_a_jump():
    y = NOISE.copy()
    y[12] += 1.5
    c = find_changes(DAYS, y)
    assert not c.jumps
    assert [(u.index, u.side) for u in c.unusual] == [(12, "both")]


@pytest.mark.parametrize("i", [1, N - 2])
def test_second_and_second_to_last_files_are_looked_at_too(i):
    y = NOISE.copy()
    y[i] += 1.5
    c = find_changes(DAYS, y)
    assert [(u.index, u.side) for u in c.unusual] == [(i, "both")]


def test_newest_file_is_compared_with_the_ones_before():
    y = NOISE.copy()
    y[-1] += 1.5
    c = find_changes(DAYS, y)
    assert [(u.index, u.side) for u in c.unusual] == [(N - 1, "before")]


def test_jumps_across_long_gaps_and_no_trend_across_them():
    days = np.concatenate([np.arange(20) / 6, 700 + np.arange(20) / 6, 1300 + np.arange(20) / 6])
    y = np.random.default_rng(2).normal(0, 0.15, 60) + np.repeat([0.0, 0.5, 1.6], 20)
    c = find_changes(days, y)
    assert [j.index for j in c.jumps] == [20, 40]
    assert c.groups == [(0, 20), (20, 40), (40, 60)]
    assert not c.trends


def test_short_excursion_beside_a_split_is_not_a_jump():
    """A few odd files next to an otherwise small shift must not make a big jump."""
    y = NOISE.copy()
    y[10:14] -= 0.7  # four files dip, then everything returns
    c = find_changes(DAYS, y)
    assert not [j for j in c.jumps if abs(np.log10(j.factor)) < 0.6]


def test_too_few_files():
    c = find_changes(DAYS[:5], NOISE[:5])
    assert c.too_few and not c.jumps


def test_same_answer_every_time():
    y = NOISE + np.where(STEP >= 15, 0.5, 0)
    a, b = find_changes(DAYS, y), find_changes(DAYS, y)
    assert [(j.index, j.p_value) for j in a.jumps] == [(j.index, j.p_value) for j in b.jumps]


def test_rarely_fooled_by_random_files():
    false = 0
    for seed in range(100):
        y = np.random.default_rng(100 + seed).normal(0, 0.15, N)
        c = find_changes(DAYS, y)
        false += bool(c.jumps or c.trends)
    assert false <= 2


def test_rarely_fooled_by_slowly_wandering_files():
    """Neighbouring files alike (lag-1 correlation 0.7): the hardest honest case."""
    false = 0
    for seed in range(100):
        rng = np.random.default_rng(500 + seed)
        y = np.zeros(N)
        for i in range(1, N):
            y[i] = 0.7 * y[i - 1] + rng.normal(0, 0.15)
        false += bool(find_changes(DAYS, y).jumps)
    assert false <= 8


def test_min_factor_sets_what_is_worth_reporting():
    y = NOISE + np.where(STEP >= 15, 0.4, 0)  # about x2.5
    assert find_changes(DAYS, y, ChangeSettings(min_factor=2)).jumps
    assert not find_changes(DAYS, y, ChangeSettings(min_factor=5)).jumps


def test_the_callers_check_can_veto_an_unusual_file():
    y = NOISE.copy()
    y[12] += 1.5
    seen = []

    def check(i, around, higher):
        seen.append((i, sorted(around.tolist()), higher))
        return False

    c = find_changes(DAYS, y, check=check)
    assert not c.unusual
    assert seen == [(12, [9, 10, 11, 13, 14, 15], True)]


@pytest.mark.parametrize("every", [3, 10])
def test_switching_back_and_forth_is_found(every):
    """Two levels 10x apart taking turns every few files: no jump, but SWITCHING."""
    n = 60
    days = np.arange(n) / 6.0
    y = np.random.default_rng(4).normal(0, 0.15, n) + np.where(np.arange(n) // every % 2, 1.0, 0)
    c = find_changes(days, y)
    assert not c.jumps
    assert len(c.switching) == 1
    sw = c.switching[0]
    assert (sw.first, sw.last) == (0, n - 1) and sw.switches >= 5
    assert sw.level(y, sw.high) - sw.level(y, sw.low) == pytest.approx(1.0, abs=0.15)


def test_no_switching_in_steady_or_wandering_files():
    assert not find_changes(DAYS, NOISE).switching
    for seed in range(50):
        rng = np.random.default_rng(900 + seed)
        y = np.zeros(N)
        for i in range(1, N):
            y[i] = 0.7 * y[i - 1] + rng.normal(0, 0.15)
        assert not find_changes(DAYS, y).switching
    one_jump = NOISE + np.where(STEP >= 15, 1.0, 0)
    assert not find_changes(DAYS, one_jump).switching
