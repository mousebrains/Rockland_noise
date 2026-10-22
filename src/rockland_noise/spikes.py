"""Measure the noise and count the spikes in one channel.

The recipe, in plain words:

1. Smooth the channel with a running median 0.1 s long. The median ignores a
   short burst, so the smoothed curve follows the slow signal but not spikes.
2. Subtract the smoothed curve. What is left (the "fast part") is the small
   electronic wiggle plus any spikes.
3. The **noise** is the typical size of that wiggle: the root-mean-square of
   the fast part, leaving out anything more than 5 times the typical size so
   that spikes do not inflate it. It is in raw counts.
4. A **spike** starts wherever the fast part is bigger (up or down) than a
   threshold: a multiple of the noise (6 for bench tests, 8 for live data),
   but never less than a few counts. Starts closer together than 0.1 s are one
   spike, because one electrical event rings for a few tens of milliseconds.

This is the rule used for the MR433 T1 fault in the AEM1-G project, with one
change: there the noise came from the median absolute deviation, which on
whole-number counts can only take a few values (2 or 3 counts), so the
threshold jumped by half between files. The clipped root-mean-square used here
changes smoothly.

Two channels are compared spike by spike too: a spike is SHARED when the other
channel has one starting within 0.1 s. Shaking and bumps reach both probes at
once; an electrical fault shows on one channel. Measured: on bench tests of
MicroRider SN 433 and 435 (shear) 46-60% of the spikes were shared, most of
them within 10 ms, where chance would give under 1%; MR433's faulty T1
shared 1% with T2.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import median_filter

# A sample at either end of a channel's range means the signal was clipped.
# Most channels are signed 16-bit; a few are unsigned (JAC_T, Turbidity, ...)
# or 32-bit (JAC_C), and the reader says which.
INT16_MIN, INT16_MAX = -32768, 32767
MEDIAN_BLOCK = 2**20  # samples per block of the running median
SHARED_WINDOW_S = 0.1  # spikes on the two channels this close in time are shared
CHECKLIST_MINUTE_S = 60.0  # the checklist's "typically within" is judged per minute
TYPICAL_PERCENT = 99.9  # "typically" = this percentage of the samples (as rsi-tpw bench)


@dataclass(frozen=True, slots=True)
class SpikeSettings:
    """How spikes are found. Every number has a command-line option."""

    threshold_sigma: float  # spike threshold, in multiples of the noise
    min_threshold_counts: float  # ...but never below this many counts
    fixed_threshold_counts: float | None = None  # use this threshold instead
    window_s: float = 0.1  # running-median length
    merge_s: float = 0.1  # spike starts closer than this are one spike
    amplitude_s: float = 0.05  # a spike's size is its largest value in this time


# The two kinds of data, and the settings that suit each.
#
# bench: test probes (open-circuit shear, fixed-resistor temperature), so the
#   signal is mostly the electronics' own noise. In quiet stretches of 4-hour
#   bench tests of MR433/MR435/MR533 the largest excursion of a healthy channel
#   is about 5.4 times the noise (median of 600 channel-files), as expected for
#   well-behaved noise; at 6 times, pure noise gives about 1 false spike every
#   250 hours. Handling and vibration still make real mechanical spikes,
#   mostly on both shear channels at once (27% of healthy shear channel-files
#   had at least one, 5% of thermistor channel-files).
# live: real probes, which also see the water (turbulence, plankton,
#   vibration). Real signal has more big excursions, so the threshold is the
#   AEM1-G value of 8 times the noise, and at least 10 counts.
MODES: dict[str, SpikeSettings] = {
    "bench": SpikeSettings(threshold_sigma=6.0, min_threshold_counts=5.0),
    "live": SpikeSettings(threshold_sigma=8.0, min_threshold_counts=10.0),
}


@dataclass(slots=True)
class ChannelResult:
    """Spike and noise numbers for one channel in one file."""

    name: str
    usable_s: float  # seconds of usable data
    noise: float  # counts; NaN when nothing was usable
    threshold: float  # counts
    spikes: int
    positive_first: int  # spikes whose first excursion was upward
    median_amplitude: float  # counts; NaN when there were no spikes
    largest: float  # largest |fast part|, counts
    saturated: int  # usable samples at either end of the channel's range
    bad_buffer_samples: int  # samples the logger marked as bad (not used)
    kind: str = ""  # sensor type from the file's setup, e.g. "therm", "shear"
    fs: float = float("nan")  # samples per second
    bits: int = 16  # 32 for a channel made of two words (e.g. JAC_C)
    fast: bool = True  # sampled every scan (False: a slow channel)
    mean: float = float("nan")  # mean of the usable raw counts (the "offset")
    # The checklist's range, per minute: in a typical (median) minute 99.9% of
    # the usable samples lie within +- typical_range of that minute's mean.
    typical_range: float = float("nan")
    worst_range: float = float("nan")  # the same, in the worst minute
    minute_ranges: list[float] = field(default_factory=list)  # every minute's, in order
    shared: int = 0  # spikes with a spike on the other channel within 0.1 s

    @property
    def rate_per_hour(self) -> float:
        return self.spikes / (self.usable_s / 3600) if self.usable_s > 0 else float("nan")

    @property
    def only_per_hour(self) -> float:
        """Spikes per hour on this channel alone (not shared with the other)."""
        return ((self.spikes - self.shared) / (self.usable_s / 3600) if self.usable_s > 0
                else float("nan"))


def odd_window(seconds: float, fs: float) -> int:
    """Running-median length in samples: odd, and at least 3."""
    n = round(seconds * fs)
    return max(3, n if n % 2 else n + 1)


def running_median(x: np.ndarray, size: int) -> np.ndarray:
    """``median_filter(x, size, mode="nearest")``, a block at a time.

    Each block is filtered with half a window of its neighbours on either side,
    so the answer is exactly the same, but a long channel needs little memory
    (scipy works on a full-length copy of whole-number input otherwise).
    """
    half = size // 2
    out = np.empty_like(x)
    for a in range(0, x.size, MEDIAN_BLOCK):
        b = min(x.size, a + MEDIAN_BLOCK)
        lo, hi = max(0, a - half), min(x.size, b + half)
        out[a:b] = median_filter(x[lo:hi], size=size, mode="nearest")[a - lo:b - lo]
    return out


def fast_part(counts: np.ndarray, fs: float, window_s: float) -> np.ndarray:
    """The channel minus its running median: the noise plus any spikes.

    Whole-number counts give whole numbers (the median of whole numbers is one
    of them), kept as 32-bit integers to save memory; anything else, float64.
    """
    median = running_median(counts, odd_window(window_s, fs))
    if np.issubdtype(counts.dtype, np.integer) and counts.dtype.itemsize <= 4:
        hp = counts.astype(np.int64 if counts.dtype == np.uint32 else np.int32)
    else:
        hp = counts.astype(np.float64)
    hp -= median
    return hp


def _median_of_histogram(values: np.ndarray, counts: np.ndarray) -> float:
    """The median of a sample given as sorted distinct values and how often each occurs."""
    n = int(counts.sum())
    cum = np.cumsum(counts)
    lower = values[np.searchsorted(cum, (n - 1) // 2 + 1)]
    upper = values[np.searchsorted(cum, n // 2 + 1)]
    return (float(lower) + float(upper)) / 2


def noise_level(values: np.ndarray) -> float:
    """Typical size of the small fast wiggles, ignoring spikes.

    The root-mean-square of the values within 5 times the current estimate,
    repeated three times. It starts from the median absolute deviation, which
    spikes cannot move. Returns 0 for a flat channel and NaN for no data.
    Whole numbers are counted in a histogram first, which gives the same
    answer with far less memory than sorting them.
    """
    if values.size == 0:
        return float("nan")
    if np.issubdtype(values.dtype, np.integer):
        lo = int(values.min())
        weights = np.bincount((values - lo).astype(np.intp, copy=False))
        distinct = np.flatnonzero(weights)
        weights = weights[distinct].astype(np.float64)
        centred = distinct + lo - _median_of_histogram(distinct + lo, weights)
    else:
        weights = None
        centred = values - np.median(values)
    size = np.abs(centred)
    if weights is None:
        mad = float(np.median(size))
    else:
        order = np.argsort(size, kind="stable")
        mad = _median_of_histogram(size[order], weights[order])
    sigma = max(1.4826 * mad, 0.5)
    for _ in range(3):
        keep = size < 5 * sigma
        if weights is None:
            if not keep.any():
                break
            sigma = float(np.sqrt(np.mean(centred[keep] ** 2)))
        else:
            if not keep.any():
                break
            sigma = float(np.sqrt(np.sum(weights[keep] * centred[keep] ** 2)
                                  / np.sum(weights[keep])))
    return sigma


def spike_starts(hp: np.ndarray, usable: np.ndarray, threshold: float, fs: float,
                 merge_s: float) -> np.ndarray:
    """Sample numbers where spikes start, with starts closer than merge_s merged."""
    hit = ((hp > threshold) | (hp < -threshold)) & usable
    starts = np.flatnonzero(hit & ~np.concatenate(([False], hit[:-1])))
    if starts.size < 2:
        return starts
    min_gap = merge_s * fs
    keep = [int(starts[0])]
    for i in starts[1:]:
        if i - keep[-1] > min_gap:
            keep.append(int(i))
    return np.asarray(keep, dtype=np.int64)


def shared_spikes(times: np.ndarray, other: np.ndarray, window_s: float) -> int:
    """How many of the spike times have one of the other channel's within window_s."""
    if times.size == 0 or other.size == 0:
        return 0
    i = np.searchsorted(other, times)
    before = other[np.clip(i - 1, 0, other.size - 1)]
    after = other[np.clip(i, 0, other.size - 1)]
    near = (np.abs(times - before) <= window_s) | (np.abs(after - times) <= window_s)
    return int(np.count_nonzero(near))


def minute_ranges(counts: np.ndarray, usable: np.ndarray, fs: float,
                  minute_s: float = CHECKLIST_MINUTE_S) -> np.ndarray:
    """The checklist's "typically within +-N counts" for each minute of the file.

    Rockland's checklist is written for a file of about a minute, so a long
    file is judged a minute at a time: for each minute with at least half of
    its samples usable, the range that 99.9% of them stay within, about that
    minute's own mean. A file with no such minute is judged as a whole.
    """
    n = max(1, round(minute_s * fs))
    out = []
    for a in range(0, counts.size, n):
        x = counts[a:a + n][usable[a:a + n]]
        if x.size >= n / 2:
            x = x.astype(np.float64)
            x -= x.mean()
            out.append(float(np.percentile(np.abs(x, out=x), TYPICAL_PERCENT)))
    if not out:
        x = counts[usable].astype(np.float64)
        if x.size:
            x -= x.mean()
            out.append(float(np.percentile(np.abs(x, out=x), TYPICAL_PERCENT)))
    return np.asarray(out)


def analyze_channel(name: str, counts: np.ndarray, fs: float, usable: np.ndarray,
                    settings: SpikeSettings, bad: np.ndarray | None = None,
                    limits: tuple[int, int] | None = (INT16_MIN, INT16_MAX)) -> ChannelResult:
    """Noise and spikes for one channel.

    ``usable`` marks the samples to count. ``bad`` marks samples the logger
    flagged as bad; they are never counted. ``limits`` is the smallest and
    largest count the channel can record (None: do not look for clipping).
    """
    return analyze_channel_many(name, counts, fs, usable, [settings], bad, limits)[0]


def analyze_channel_many(name: str, counts: np.ndarray, fs: float, usable: np.ndarray,
                         settings: list[SpikeSettings], bad: np.ndarray | None = None,
                         limits: tuple[int, int] | None = (INT16_MIN, INT16_MAX),
                         ) -> list[ChannelResult]:
    """Like analyze_channel for several settings at once (e.g. bench and live)."""
    return [r for r, _ in analyze_channel_times(name, counts, fs, usable, settings, bad, limits)]


def analyze_channel_times(name: str, counts: np.ndarray, fs: float, usable: np.ndarray,
                          settings: list[SpikeSettings], bad: np.ndarray | None = None,
                          limits: tuple[int, int] | None = (INT16_MIN, INT16_MAX),
                          ) -> list[tuple[ChannelResult, np.ndarray]]:
    """analyze_channel_many, and also when each spike started (seconds into the file).

    The slow part, the running median, is done once per window length.
    """
    counts = np.asarray(counts)
    usable = np.asarray(usable, dtype=bool)
    bad_buffer_samples = 0 if bad is None else int(np.count_nonzero(bad))
    if bad_buffer_samples and bad_buffer_samples < counts.size:
        # Replace the logger's bad samples by straight lines between good
        # neighbours, so the running median is not disturbed by them. They are
        # unusable, so they are never counted themselves.
        idx = np.arange(counts.size)
        counts = counts.astype(np.float64)
        counts[bad] = np.interp(idx[bad], idx[~bad], counts[~bad])
        usable = usable & ~bad
    saturated = 0 if limits is None else int(np.count_nonzero(
        ((counts <= limits[0]) | (counts >= limits[1])) & usable))

    # The bench-test checklist's numbers (see checklist.py): the offset (mean),
    # and per minute the range that 99.9% of the samples stay within.
    n_usable = int(np.count_nonzero(usable))
    mean = (float(np.sum(counts, where=usable, dtype=np.float64) / n_usable) if n_usable
            else float("nan"))
    ranges = minute_ranges(counts, usable, fs)
    typical_range = float(np.median(ranges)) if ranges.size else float("nan")
    worst_range = float(np.max(ranges)) if ranges.size else float("nan")

    # Per running-median length: the fast part, its noise, its largest value
    # and how many samples were usable.
    prepared: dict[float, tuple[np.ndarray, float, float, int]] = {}
    results = []
    for setting in settings:
        if setting.window_s not in prepared:
            hp = fast_part(counts, fs, setting.window_s)
            used = hp[usable]
            largest = float(max(used.max(), -used.min())) if used.size else float("nan")
            prepared[setting.window_s] = (hp, noise_level(used), largest, used.size)
            del used
        hp, noise, largest, n_used = prepared[setting.window_s]
        if setting.fixed_threshold_counts is not None:
            threshold = float(setting.fixed_threshold_counts)
        elif np.isfinite(noise):
            threshold = max(setting.threshold_sigma * noise, setting.min_threshold_counts)
        else:
            threshold = float("nan")

        if n_used == 0 or not np.isfinite(threshold):
            results.append((ChannelResult(name, 0.0, float("nan"), threshold, 0, 0, float("nan"),
                                          float("nan"), saturated, bad_buffer_samples, fs=fs),
                            np.empty(0)))
            continue
        starts = spike_starts(hp, usable, threshold, fs, setting.merge_s)
        span = max(1, round(setting.amplitude_s * fs))
        amplitudes = [float(np.max(np.abs(hp[i:i + span]))) for i in starts]
        results.append((ChannelResult(
            name=name,
            usable_s=float(n_used / fs),
            noise=noise,
            threshold=threshold,
            spikes=int(starts.size),
            positive_first=int(np.count_nonzero(hp[starts] > 0)),
            median_amplitude=float(np.median(amplitudes)) if amplitudes else float("nan"),
            largest=largest,
            saturated=saturated,
            bad_buffer_samples=bad_buffer_samples,
            mean=mean,
            typical_range=typical_range,
            worst_range=worst_range,
            minute_ranges=[round(float(x), 1) for x in ranges],
            fs=fs,
        ), starts / fs))
    return results
