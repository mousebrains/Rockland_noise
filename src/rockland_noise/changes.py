"""Find jumps, trends and unusual files in a series of ratios, oldest file first.

Everything here works on the base-10 logarithm of the ratio, so that "twice as
big" and "half as big" are changes of the same size, and one wild file cannot
dominate. Three things are looked for:

Jump
    The ratio sits at one level for a run of files, then moves to a clearly
    different level and stays there. Found with the standard test for a
    single shift in level: the largest two-sample t statistic over every place
    the files could be split, applied again on each side of every jump found
    ("binary segmentation"). A candidate is accepted only if

    * it is big enough and it lasts: the typical (median) value of the files
      just after it differs from that of the files just before it by at least
      ``min_factor``, both for the nearest 5 files on each side and for the
      nearest 15, so a short excursion next to it does not count;
    * it is sudden: around it, two flat levels fit the files better than one
      sloping line does, and after allowing for the slope on each side the
      step is still at least ``min_factor``. Otherwise it is part of a trend;
    * it is unlikely to be chance: the same test on the files in shuffled
      order gives a change this large less than ``alpha`` of the time.

    A single unusual file is replaced by its neighbours' typical value for
    this search, so one odd file can neither make nor hide a jump.

Trend
    Inside each run of files between jumps (and between long time gaps), a
    steady rise or fall over time. Kendall's tau (a rank correlation with time)
    must be significant at ``alpha``, and the Theil-Sen line (the median of the
    slopes between all pairs of files) must change by at least ``min_factor``
    from the first file of the run to its last.

Unusual file
    One file that differs from the files on both sides of it, in the same
    direction, by at least ``min_factor`` squared (and by much more than the
    usual file-to-file scatter). The first and the most recent file are
    compared with their one available side. The caller can add a test of its
    own (``check``): for spike ratios, that the file's spike counts are beyond
    what chance gives for its length, so a short file with a few spikes is
    not called unusual.

Every file counts the same in the jump and trend search, however long it is.
That is safe because the size of a jump is judged on the median of the files
on each side, which a few noisy short files cannot move; files with less
usable data than ``--min-minutes`` are left out altogether. The chances
reported are approximate: the shuffle test and Kendall's test assume files
are independent, and the allowance for similar neighbours is an estimate.

Neighbouring files are usually alike, which makes a run of files look like
stronger evidence than it is. So the chance of each jump and trend is worked
out twice: first as if the files were independent, to find candidates; then,
once every jump and trend is in place, again allowing for how alike
neighbouring files are in what is left over (see ``_inflation``). Jumps that no
longer pass are removed one at a time, weakest first. Every finding must also
pass a minimum size, so a tiny but steady change in hundreds of files is not
reported.
"""

import itertools
import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import kendalltau, norm, rankdata, theilslopes


@dataclass(frozen=True, slots=True)
class ChangeSettings:
    min_factor: float = 2.0  # smallest change worth reporting (a ratio, > 1)
    alpha: float = 0.01  # largest acceptable chance of a false finding
    min_segment: int = 3  # files needed on each side of a jump
    min_trend_files: int = 6  # files needed to look for a trend
    local_window: int = 5  # files each side that must show the jump...
    long_window: int = 15  # ...and over this many, so it is a lasting change
    gap_factor: float = 20.0  # a time gap this many times the usual spacing...
    min_gap_days: float = 2.0  # ...and at least this long separates groups of files
    n_shuffles: int = 1999  # shuffles for the jump test
    seed: int = 20261021  # fixed, so the same files always give the same answer
    unusual_alpha: float = 0.001  # for the caller's check of one unusual file (many are tested)


@dataclass(slots=True)
class Jump:
    index: int  # position of the first file after the jump
    before: float  # log10 level of the files just before
    after: float  # log10 level of the files just after
    p_value: float

    @property
    def factor(self) -> float:
        return 10 ** (self.after - self.before)


@dataclass(slots=True)
class Trend:
    first: int  # positions of the first and last files of the run
    last: int
    start: float  # fitted log10 value at the first and last file
    end: float
    days: float  # time from first to last file
    p_value: float

    @property
    def factor(self) -> float:
        return 10 ** (self.end - self.start)


@dataclass(slots=True)
class Unusual:
    index: int
    difference: float  # log10: how far it is from its neighbours (signed)
    side: str  # "both", "before" (most recent file) or "after" (first file)

    @property
    def factor(self) -> float:
        return 10**self.difference


@dataclass(slots=True)
class Changes:
    n: int
    jumps: list[Jump] = field(default_factory=list)
    trends: list[Trend] = field(default_factory=list)
    unusual: list[Unusual] = field(default_factory=list)
    segments: list[tuple[int, int]] = field(default_factory=list)  # [first, last) runs
    groups: list[tuple[int, int]] = field(default_factory=list)  # [first, last) by time gaps
    inflation: float = 1.0  # see _inflation
    too_few: bool = False


# --- small helpers -----------------------------------------------------------


def _l1_flat(y: np.ndarray) -> float:
    return float(np.sum(np.abs(y - np.median(y)))) if y.size else 0.0


def _line_residuals(y: np.ndarray, x: np.ndarray | None = None) -> np.ndarray:
    if y.size < 2:
        return np.zeros_like(y)
    x = np.arange(y.size, dtype=float) if x is None else x
    if np.ptp(x) == 0:
        return y - np.median(y)
    slope, intercept = theilslopes(y, x)[:2]
    return y - (intercept + slope * x)


def _slope(y: np.ndarray) -> float:
    """Theil-Sen slope per file, 0 for fewer than 3 files."""
    if y.size < 3:
        return 0.0
    return float(theilslopes(y, np.arange(y.size, dtype=float))[0])


def _levels(y: np.ndarray, k: int, w: int) -> tuple[float, float]:
    """Typical (median) value of up to w files before and after position k."""
    return float(np.median(y[max(0, k - w):k])), float(np.median(y[k:k + w]))


def _inflation(pieces: list[np.ndarray]) -> float:
    """How much neighbouring files' similarity overstates the evidence.

    Files next to each other are usually alike, so n files carry less
    information than n independent ones. With r the lag-1 correlation of the
    ranks of the residuals (what is left after fitting every jump and a line
    through each run of files), the effective number of files is
    n (1 - r) / (1 + r); test statistics are divided by the square root of
    (1 + r) / (1 - r). This is the usual correction of the modified
    Mann-Kendall test (Yue and Wang 2004). Pairs are only formed within a
    piece, never across a jump.
    """
    pieces = [p for p in pieces if p.size]
    if sum(p.size for p in pieces) < 4:
        return 1.0
    ranks = rankdata(np.concatenate(pieces))
    ranks -= ranks.mean()
    denom = float(np.dot(ranks, ranks))
    if denom == 0:
        return 1.0
    num, start = 0.0, 0
    for p in pieces:
        r = ranks[start:start + p.size]
        num += float(np.dot(r[:-1], r[1:]))
        start += p.size
    # A lag-1 correlation measured on short pieces, each with its own line
    # removed, comes out too small by about (1 + 3 r) / L for pieces of L
    # files; undo that to first order.
    length = sum(p.size for p in pieces) / len(pieces)
    rho = num / denom
    if length > 4:
        rho = (length * rho + 1) / (length - 3)
    rho = min(max(rho, 0.0), 0.9)
    return (1 + rho) / (1 - rho)


# --- jumps -------------------------------------------------------------------


def _t_scan(x: np.ndarray) -> np.ndarray:
    """Two-sample t statistic for a split after each k = 1..n-1.

    |mean after - mean before| divided by its standard error, using the spread
    of the files about their own side's mean. ``x`` may be 2-D, one shuffled
    series per row. The largest value over all splits is the standard test
    for a single shift in level.
    """
    n = x.shape[-1]
    k = np.arange(1, n)
    s1 = np.cumsum(x, axis=-1)[..., :-1]
    q1 = np.cumsum(x * x, axis=-1)[..., :-1]
    s_all = np.sum(x, axis=-1, keepdims=True)
    q_all = np.sum(x * x, axis=-1, keepdims=True)
    s2, q2 = s_all - s1, q_all - q1
    within = (q1 - s1 * s1 / k) + (q2 - s2 * s2 / (n - k))
    var = np.maximum(within, 0) / max(n - 2, 1)
    # A floor so that a perfectly clean step gives a huge but finite value.
    var = np.maximum(var, 1e-12 * (1 + np.var(x, axis=-1, keepdims=True)))
    return np.abs(s2 / (n - k) - s1 / k) / np.sqrt(var * (1 / k + 1 / (n - k)))


def _allowed(n: int, s: ChangeSettings) -> np.ndarray:
    k = np.arange(1, n)
    return (k >= s.min_segment) & (k <= n - s.min_segment)


def _split_p(y: np.ndarray, k: int, s: ChangeSettings, inflation: float,
             rng: np.random.Generator) -> float:
    """Chance that shuffled files give a split statistic as large as the one at k."""
    n = y.size
    allowed = _allowed(n, s)
    if not allowed.any():
        return 1.0
    observed = float(_t_scan(y)[k - 1]) / math.sqrt(inflation)
    exceed, done = 0, 0
    chunk = max(1, 2_000_000 // n)
    while done < s.n_shuffles:
        m = min(chunk, s.n_shuffles - done)
        z = _t_scan(rng.permuted(np.tile(y, (m, 1)), axis=1))
        exceed += int(np.count_nonzero(z[:, allowed].max(axis=1) >= observed - 1e-9))
        done += m
    return (exceed + 1) / (s.n_shuffles + 1)


def _refine(y: np.ndarray, k: int, s: ChangeSettings) -> int:
    """Move a split by up to 2 files to where two flat levels fit best nearby."""
    lo, hi = max(0, k - s.local_window - 2), min(y.size, k + s.local_window + 2)
    options = range(max(s.min_segment, k - 2), min(y.size - s.min_segment, k + 2) + 1)
    return min(options, key=lambda j: (_l1_flat(y[lo:j]) + _l1_flat(y[j:hi]), abs(j - k)))


def _find_jump(y: np.ndarray, s: ChangeSettings, rng: np.random.Generator) -> Jump | None:
    """The best acceptable jump in y, treating files as independent, or None."""
    if y.size < 2 * s.min_segment or np.ptp(y) == 0:
        return None
    allowed = _allowed(y.size, s)
    if not allowed.any():
        return None
    z = _t_scan(y)
    # Candidates: local peaks of the statistic (index i = split before file
    # i + 1), biggest first. Testing a few keeps a real jump from being hidden
    # by a bigger but gradual change elsewhere in the same run of files.
    cand = [i for i in np.flatnonzero(allowed)
            if (i == 0 or z[i] >= z[i - 1]) and (i == z.size - 1 or z[i] >= z[i + 1])]
    cand.sort(key=lambda i: -z[i])
    min_log = math.log10(s.min_factor)
    for i in cand[:5]:
        k = _refine(y, i + 1, s)
        # Big enough both right at the split and over a longer stretch, so a
        # short excursion next to the split is not mistaken for a lasting step.
        near = _levels(y, k, s.local_window)
        far = _levels(y, k, s.long_window)
        if min(abs(near[1] - near[0]), abs(far[1] - far[0])) < min_log:
            continue
        if (near[1] - near[0]) * (far[1] - far[0]) <= 0:
            continue
        left, right = y[max(0, k - s.local_window):k], y[k:k + s.local_window]
        sudden = _l1_flat(left) + _l1_flat(right)
        if sudden >= float(np.sum(np.abs(_line_residuals(np.concatenate((left, right)))))):
            continue  # a sloping line explains it as well: a trend, not a jump
        # What is left of the step after allowing for the slope on each side
        # must still be big enough; on a steep ramp nothing is left.
        slope = 0.5 * (_slope(y[max(0, k - s.long_window):k])
                       + _slope(y[k:k + s.long_window]))
        excess = (near[1] - near[0]) - slope * 0.5 * (left.size + right.size)
        if excess * (near[1] - near[0]) <= 0 or abs(excess) < min_log:
            continue
        p = _split_p(y, k, s, 1.0, rng)
        if p < s.alpha:
            return Jump(index=k, before=far[0], after=far[1], p_value=p)
    return None


def _edges(jumps: list[Jump], n: int) -> list[int]:
    return [0, *(j.index for j in jumps), n]


def _model_residuals(y: np.ndarray, jumps: list[Jump]) -> list[np.ndarray]:
    e = _edges(jumps, y.size)
    return [_line_residuals(y[a:b]) for a, b in itertools.pairwise(e)]


def _validate(y: np.ndarray, jumps: list[Jump], s: ChangeSettings,
              rng: np.random.Generator) -> tuple[list[Jump], float]:
    """Re-test every jump allowing for similar neighbours; drop failures one by one."""
    jumps = sorted(jumps, key=lambda j: j.index)
    while True:
        inflation = _inflation(_model_residuals(y, jumps))
        e = _edges(jumps, y.size)
        for i, j in enumerate(jumps):
            a, b = e[i], e[i + 2]  # between the neighbouring jumps
            j.p_value = _split_p(y[a:b], j.index - a, s, inflation, rng)
        failed = [j for j in jumps if j.p_value >= s.alpha]
        if not failed:
            break
        jumps.remove(max(failed, key=lambda j: j.p_value))
    # Report each jump's levels from the files between it and its neighbours.
    e = _edges(jumps, y.size)
    for i, j in enumerate(jumps):
        a, b = e[i], e[i + 2]
        j.before, j.after = _levels(y[a:b], j.index - a, s.long_window)
    return jumps, inflation


# --- trends, unusual files, groups ------------------------------------------


def _find_trend(days: np.ndarray, y: np.ndarray, first: int, s: ChangeSettings,
                inflation: float) -> Trend | None:
    if y.size < s.min_trend_files or np.ptp(y) == 0 or np.ptp(days) == 0:
        return None
    _, p = kendalltau(days, y)
    if not np.isfinite(p):
        return None
    p = float(2 * norm.sf(norm.isf(p / 2) / math.sqrt(inflation)))
    if p >= s.alpha:
        return None
    slope, intercept = theilslopes(y, days)[:2]
    start = intercept + slope * days[0]
    end = intercept + slope * days[-1]
    if abs(end - start) < math.log10(s.min_factor):
        return None
    return Trend(first=first, last=first + y.size - 1, start=float(start), end=float(end),
                 days=float(days[-1] - days[0]), p_value=p)


# check(i, neighbours, higher): is file i's difference from those files (positions
# in the series) beyond chance, given that it is higher (True) or lower?
Check = Callable[[int, np.ndarray, bool], bool]


def _find_unusual(y: np.ndarray, s: ChangeSettings, check: Check | None = None) -> list[Unusual]:
    if y.size < 4:
        return []
    steps = np.diff(y)
    scatter = 1.4826 * float(np.median(np.abs(steps - np.median(steps)))) / math.sqrt(2)
    limit = max(2 * math.log10(s.min_factor), 4 * scatter)
    found = []
    for i in range(y.size):
        before, after = y[max(0, i - 3):i], y[i + 1:i + 4]
        if before.size >= 2 and after.size >= 2:
            d_before, d_after = y[i] - np.median(before), y[i] - np.median(after)
            if d_before * d_after > 0 and min(abs(d_before), abs(d_after)) >= limit:
                d = math.copysign(min(abs(d_before), abs(d_after)), d_before)
                found.append(Unusual(i, d, "both"))
        elif after.size == 0 and before.size >= 3:
            d = float(y[i] - np.median(before))
            if abs(d) >= limit:
                found.append(Unusual(i, d, "before"))
        elif before.size == 0 and after.size >= 3:
            d = float(y[i] - np.median(after))
            if abs(d) >= limit:
                found.append(Unusual(i, d, "after"))
    if check is not None:
        found = [u for u in found if check(u.index, neighbours(u.index, y.size), u.difference > 0)]
    return found


def neighbours(i: int, n: int) -> np.ndarray:
    """Positions of the files an unusual file is compared with: up to 3 on each side."""
    return np.r_[max(0, i - 3):i, i + 1:min(n, i + 4)]


def time_groups(days: np.ndarray, s: ChangeSettings) -> list[tuple[int, int]]:
    """Split the files where a time gap is far longer than their usual spacing."""
    if days.size < 2:
        return [(0, days.size)]
    gaps = np.diff(days)
    usual = float(np.median(gaps))
    cuts = np.flatnonzero(gaps > max(s.gap_factor * usual, s.min_gap_days)) + 1
    edges = [0, *cuts.tolist(), days.size]
    return list(itertools.pairwise(edges))


def find_changes(days: np.ndarray, y: np.ndarray, settings: ChangeSettings | None = None,
                 check: Check | None = None) -> Changes:
    """Look for jumps, trends and unusual files.

    ``days`` is each file's start time in days (any origin), oldest first;
    ``y`` is log10 of the ratio for each file. Both must be finite. ``check``,
    if given, must also agree before a file is called unusual.
    """
    s = settings or ChangeSettings()
    days, y = np.asarray(days, dtype=float), np.asarray(y, dtype=float)
    out = Changes(n=y.size)
    out.groups = time_groups(days, s)
    if y.size < s.min_trend_files:
        out.too_few = True
        out.segments = [(0, y.size)]
        return out

    out.unusual = _find_unusual(y, s, check)
    # A single odd file must not make or hide a jump: for the jump search it is
    # replaced by the typical value of its neighbours.
    y_all = y
    y = y.copy()
    for u in out.unusual:
        if u.side == "both":
            i = u.index
            y[i] = float(np.median(np.concatenate((y_all[max(0, i - 3):i], y_all[i + 1:i + 4]))))

    # 1. Candidate jumps, treating files as independent.
    rng = np.random.default_rng(s.seed)
    pending, jumps = [(0, y.size)], []
    while pending and len(jumps) < 50:
        a, b = pending.pop()
        jump = _find_jump(y[a:b], s, rng)
        if jump is None:
            continue
        jump.index += a
        jumps.append(jump)
        pending += [(a, jump.index), (jump.index, b)]

    # 2. Keep only those that survive once similar neighbours are allowed for.
    out.jumps, out.inflation = _validate(y, jumps, s, rng)
    e = _edges(out.jumps, y.size)
    out.segments = list(itertools.pairwise(e))

    # 3. Trends, only within a run of files with no jump and no long time gap:
    # a straight line drawn across a gap of months would claim a steady change
    # where there is no data at all.
    group_edges = {a for a, _ in out.groups}
    for a, b in out.segments:
        cuts = [a, *sorted(g for g in group_edges if a < g < b), b]
        for lo, hi in itertools.pairwise(cuts):
            trend = _find_trend(days[lo:hi], y_all[lo:hi], lo, s, out.inflation)
            if trend is not None:
                out.trends.append(trend)
    return out
