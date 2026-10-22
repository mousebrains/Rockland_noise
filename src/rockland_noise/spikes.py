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
"""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter

# A sample at either end of the 16-bit range means the signal was clipped.
INT16_MIN, INT16_MAX = -32768, 32767


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
    saturated: int  # usable samples at the end of the 16-bit range
    bad_buffer_samples: int  # samples the logger marked as bad (not used)
    kind: str = ""  # sensor type from the file's setup, e.g. "therm", "shear"
    mean: float = float("nan")  # mean of the usable raw counts (the "offset")
    typical_range: float = float("nan")  # 99.9% of usable samples lie within +- this of the mean

    @property
    def rate_per_hour(self) -> float:
        return self.spikes / (self.usable_s / 3600) if self.usable_s > 0 else float("nan")


def odd_window(seconds: float, fs: float) -> int:
    """Running-median length in samples: odd, and at least 3."""
    n = round(seconds * fs)
    return max(3, n if n % 2 else n + 1)


def fast_part(counts: np.ndarray, fs: float, window_s: float) -> np.ndarray:
    """The channel minus its running median: the noise plus any spikes."""
    return counts - median_filter(counts, size=odd_window(window_s, fs), mode="nearest")


def noise_level(values: np.ndarray) -> float:
    """Typical size of the small fast wiggles, ignoring spikes.

    The root-mean-square of the values within 5 times the current estimate,
    repeated three times. It starts from the median absolute deviation, which
    spikes cannot move. Returns 0 for a flat channel and NaN for no data.
    """
    if values.size == 0:
        return float("nan")
    centred = values - np.median(values)
    sigma = max(1.4826 * float(np.median(np.abs(centred))), 0.5)
    for _ in range(3):
        kept = centred[np.abs(centred) < 5 * sigma]
        if kept.size == 0:
            break
        sigma = float(np.sqrt(np.mean(kept**2)))
    return sigma


def spike_starts(hp: np.ndarray, usable: np.ndarray, threshold: float, fs: float,
                 merge_s: float) -> np.ndarray:
    """Sample numbers where spikes start, with starts closer than merge_s merged."""
    hit = (np.abs(hp) > threshold) & usable
    starts = np.flatnonzero(hit & ~np.concatenate(([False], hit[:-1])))
    if starts.size < 2:
        return starts
    min_gap = merge_s * fs
    keep = [int(starts[0])]
    for i in starts[1:]:
        if i - keep[-1] > min_gap:
            keep.append(int(i))
    return np.asarray(keep, dtype=np.int64)


def analyze_channel(name: str, counts: np.ndarray, fs: float, usable: np.ndarray,
                    settings: SpikeSettings, bad: np.ndarray | None = None) -> ChannelResult:
    """Noise and spikes for one channel.

    ``usable`` marks the samples to count. ``bad`` marks samples the logger
    flagged as bad; they are never counted.
    """
    return analyze_channel_many(name, counts, fs, usable, [settings], bad)[0]


def analyze_channel_many(name: str, counts: np.ndarray, fs: float, usable: np.ndarray,
                         settings: list[SpikeSettings],
                         bad: np.ndarray | None = None) -> list[ChannelResult]:
    """Like analyze_channel for several settings at once (e.g. bench and live).

    The slow part, the running median, is done once per window length.
    """
    counts = np.asarray(counts, dtype=np.float64)
    usable = np.asarray(usable, dtype=bool)
    bad_buffer_samples = 0 if bad is None else int(np.count_nonzero(bad))
    if bad_buffer_samples and bad_buffer_samples < counts.size:
        # Replace the logger's bad samples by straight lines between good
        # neighbours, so the running median is not disturbed by them. They are
        # unusable, so they are never counted themselves.
        idx = np.arange(counts.size)
        counts = counts.copy()
        counts[bad] = np.interp(idx[bad], idx[~bad], counts[~bad])
        usable = usable & ~bad
    saturated = int(np.count_nonzero(((counts <= INT16_MIN) | (counts >= INT16_MAX)) & usable))

    # The bench-test checklist's numbers: offset, and the range that 99.9% of
    # the samples stay within (see checklist.py).
    raw = counts[usable]
    mean = float(np.mean(raw)) if raw.size else float("nan")
    typical_range = (float(np.percentile(np.abs(raw - mean), 99.9)) if raw.size
                     else float("nan"))

    prepared: dict[float, tuple[np.ndarray, np.ndarray, float]] = {}
    results = []
    for setting in settings:
        if setting.window_s not in prepared:
            hp = fast_part(counts, fs, setting.window_s)
            used = hp[usable]
            prepared[setting.window_s] = (hp, used, noise_level(used))
        hp, used, noise = prepared[setting.window_s]
        if setting.fixed_threshold_counts is not None:
            threshold = float(setting.fixed_threshold_counts)
        elif np.isfinite(noise):
            threshold = max(setting.threshold_sigma * noise, setting.min_threshold_counts)
        else:
            threshold = float("nan")

        if used.size == 0 or not np.isfinite(threshold):
            results.append(ChannelResult(name, 0.0, float("nan"), threshold, 0, 0, float("nan"),
                                         float("nan"), saturated, bad_buffer_samples))
            continue
        starts = spike_starts(hp, usable, threshold, fs, setting.merge_s)
        span = max(1, round(setting.amplitude_s * fs))
        amplitudes = [float(np.max(np.abs(hp[i:i + span]))) for i in starts]
        results.append(ChannelResult(
            name=name,
            usable_s=float(used.size / fs),
            noise=noise,
            threshold=threshold,
            spikes=int(starts.size),
            positive_first=int(np.count_nonzero(hp[starts] > 0)),
            median_amplitude=float(np.median(amplitudes)) if amplitudes else float("nan"),
            largest=float(np.max(np.abs(used))),
            saturated=saturated,
            bad_buffer_samples=bad_buffer_samples,
            mean=mean,
            typical_range=typical_range,
        ))
    return results
