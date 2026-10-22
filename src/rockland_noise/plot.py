"""A one-page picture of a comparison: four panels, one row per question.

1. Spikes per hour on each channel.
2. The spike ratio A/B, with jumps, trends and unusual files marked.
3. The noise on each channel.
4. The noise ratio A/B, marked the same way.

The x-axis is the file's number in the report table (#), so the picture and the
table line up; the date of some files is written under the axis. Colours carry
only which channel is which (blue = A, orange = B; checked for colour-blind
readers); findings are drawn in black and grey.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to a file; no window, works without a display
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

from .changes import Changes
from .compare import EXTRA_SPIKES_PER_HOUR, Comparison
from .report import factor

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
CHANNEL_A = "#2a78d6"  # categorical slot 1
CHANNEL_B = "#eb6834"  # categorical slot 2
WIDE_NOISE_RANGE = 20.0  # noise from smallest to largest beyond this: a log axis


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.yaxis.label.set_color(INK_2)


def _series(ax, x, y, color, label) -> None:
    ax.plot(x, y, color=color, linewidth=1.0, alpha=0.45, zorder=2)
    ax.plot(x, y, "o", color=color, markersize=4.5, markeredgecolor=SURFACE,
            markeredgewidth=0.8, label=label, zorder=3)


def _plain(value: float, _pos=None) -> str:
    """Axis numbers as people write them: 0.5, 1, 2, 10, 1,000."""
    if value == 0:
        return "0"
    return f"{value:,.0f}" if value >= 1 else f"{value:g}"


def _plain_log_axis(ax) -> None:
    """Plain numbers on a log axis; 1-2-5 steps only when the range is narrow."""
    lo, hi = ax.get_ylim()
    subs = (1, 2, 5) if np.log10(hi / lo) <= 2 else (1.0,)
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=subs))
    ax.yaxis.set_major_formatter(FuncFormatter(_plain))
    ax.yaxis.set_minor_formatter(NullFormatter())


def _end_labels(ax, x, ya, yb, a: str, b: str) -> None:
    """Name both series at their last point, pushed apart if they would overlap."""
    if not len(x):
        return
    # Compare the two end points in display space (works for any y scale).
    pa = ax.transData.transform((x[-1], ya[-1]))[1]
    pb = ax.transData.transform((x[-1], yb[-1]))[1]
    shift = 0.0 if abs(pa - pb) > 12 else 7.0
    up_a = pa >= pb
    for text, y, dy in ((a, ya[-1], shift if up_a else -shift),
                        (b, yb[-1], -shift if up_a else shift)):
        ax.annotate(text, (x[-1], y), xytext=(6, dy), textcoords="offset points",
                    va="center", fontsize=8, color=INK_2)


def _legend(ax) -> None:
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=8, frameon=False, ncols=2,
              borderaxespad=0.1, handletextpad=0.3)


def _findings(ax, changes: Changes, x: np.ndarray, y: np.ndarray, days: np.ndarray) -> None:
    """Draw jumps (vertical rules), trends (fitted lines) and unusual files (rings)."""
    top = ax.get_ylim()[1]
    # A label is about 80 points wide; one closer than that to the previous
    # label goes a line lower, so close jumps do not print over each other.
    width = 80 / 72 * ax.figure.dpi
    last_x, line = -np.inf, 0
    for j in changes.jumps:
        xj = 0.5 * (x[j.index - 1] + x[j.index])
        ax.axvline(xj, color=INK_2, linewidth=1.2, zorder=1)
        word = "up" if j.factor >= 1 else "down"
        size = j.factor if j.factor >= 1 else 1 / j.factor
        px = ax.transData.transform((xj, 1))[0]
        line = line + 1 if px - last_x < width else 0
        last_x = px
        ax.annotate(f"jump {word} x{factor(size)}", (xj, top), xytext=(3, -2 - 10 * (line % 3)),
                    textcoords="offset points", va="top", fontsize=8, color=INK)
    for t in changes.trends:
        sl = slice(t.first, t.last + 1)
        d = days[sl]
        fit = t.start + (t.end - t.start) * (d - d[0]) / max(d[-1] - d[0], 1e-12)
        ax.plot(x[sl], 10**fit, color=INK, linewidth=1.5, alpha=0.7, zorder=4)
        word = "rising" if t.factor >= 1 else "falling"
        ax.annotate(f"trend {word} x{factor(max(t.factor, 1 / t.factor))}",
                    (x[t.last], 10 ** t.end), xytext=(4, 6), textcoords="offset points",
                    fontsize=8, color=INK)
    for u in changes.unusual:
        ax.plot(x[u.index], y[u.index], "o", markersize=11, markerfacecolor="none",
                markeredgecolor=INK, markeredgewidth=1.2, zorder=5)


def _ratio_panel(ax, cmp: Comparison, rows: list[int], values: np.ndarray, changes: Changes,
                 ylabel: str) -> None:
    x = np.array([i + 1 for i in rows], dtype=float)
    _series(ax, x, values, INK_2, None)
    ax.set_yscale("log")
    ax.axhline(1, color=AXIS, linewidth=1.0, zorder=1)
    ax.set_ylabel(ylabel, fontsize=8)
    t0 = cmp.files[rows[0]].start if rows else None
    days = np.array([(cmp.files[i].start - t0).total_seconds() / 86400 for i in rows])
    if len(rows):
        lo, hi = np.nanmin(values), np.nanmax(values)
        ax.set_ylim(min(lo, 1) / 1.6, max(hi, 1) * 2.2)
    _plain_log_axis(ax)
    _findings(ax, changes, x, values, days)


def plot(cmp: Comparison, path: Path) -> None:
    rows = cmp.used
    if not rows:
        raise ValueError("there are no analysed files to plot")
    files = [cmp.files[i] for i in rows]
    x = np.array([i + 1 for i in rows], dtype=float)
    a, b = cmp.channel_a, cmp.channel_b

    fig, axes = plt.subplots(4, 1, figsize=(11, 10.5), sharex=True,
                             gridspec_kw={"hspace": 0.28})
    fig.patch.set_facecolor(SURFACE)
    # Every readable file has a place, including those left out. Set first, so
    # labels placed in screen units below see the final axis.
    last = max(x[-1], sum(not f.error for f in cmp.files))
    span = last - x[0] + 1
    axes[-1].set_xlim(x[0] - 0.8, last + max(2.5, 0.08 * span))
    for ax in axes:
        _style(ax)

    # 1. spikes per hour on each channel
    ax = axes[0]
    ra = np.array([f.a.rate_per_hour for f in files])
    rb = np.array([f.b.rate_per_hour for f in files])
    _series(ax, x, ra, CHANNEL_A, a)
    _series(ax, x, rb, CHANNEL_B, b)
    ax.set_yscale("symlog", linthresh=1)  # 0 sits on the axis, big rates on a log scale
    ax.set_ylim(bottom=0, top=max(10.0, float(np.nanmax(np.r_[ra, rb]))) * 3)
    ax.yaxis.set_major_formatter(FuncFormatter(_plain))
    ax.set_ylabel("spikes per hour", fontsize=8)
    _end_labels(ax, x, ra, rb, a, b)
    _legend(ax)

    # 2. spike ratio
    spike_y = np.array([((f.a.rate_per_hour + EXTRA_SPIKES_PER_HOUR)
                         / (f.b.rate_per_hour + EXTRA_SPIKES_PER_HOUR)) for f in files])
    _ratio_panel(axes[1], cmp, rows, spike_y, cmp.spikes,
                 f"spike ratio {a}/{b}\n(1 spike/h added to each)")

    # 3. noise on each channel
    ax = axes[2]
    na = np.array([f.a.noise for f in files])
    nb = np.array([f.b.noise for f in files])
    _series(ax, x, na, CHANNEL_A, a)
    _series(ax, x, nb, CHANNEL_B, b)
    ax.set_ylabel("noise (counts)", fontsize=8)
    both = np.r_[na, nb]
    both = both[np.isfinite(both) & (both > 0)]
    if both.size and both.max() / both.min() > WIDE_NOISE_RANGE:
        # A few very noisy files would squash the rest against the axis.
        ax.set_yscale("log")
        ax.set_ylim(both.min() / 1.5, both.max() * 2)
        _plain_log_axis(ax)
    else:
        ax.set_ylim(bottom=0, top=max(np.nanmax(na), np.nanmax(nb)) * 1.25)
    _end_labels(ax, x, na, nb, a, b)
    _legend(ax)

    # 4. noise ratio
    nrows = cmp.noise_used
    noise_y = np.array([cmp.files[i].a.noise / cmp.files[i].b.noise for i in nrows])
    _ratio_panel(axes[3], cmp, nrows, noise_y, cmp.noise, f"noise ratio {a}/{b}")

    # Long gaps in time: a hairline and a label, so the even spacing of files
    # on the x-axis does not hide them. Gaps closer together than a label is
    # wide share one label ("3 gaps of 6 to 11 days").
    width = 60 / 72 * fig.dpi
    labels: list[tuple[float, float, list[float], bool]] = []  # x, in pixels, days, all empty
    for g0, _ in cmp.spikes.groups[1:]:
        xg = 0.5 * (x[g0 - 1] + x[g0])
        gap = (files[g0].start - files[g0 - 1].start).total_seconds() / 86400
        for ax in axes:
            ax.axvline(xg, color=MUTED, linewidth=0.8, zorder=1)
        empty = x[g0] - x[g0 - 1] == 1  # no file left out in between
        px = axes[0].transData.transform((xg, 0))[0]
        if labels and px - labels[-1][1] < width:
            labels[-1] = (*labels[-1][:2], [*labels[-1][2], gap], labels[-1][3] and empty)
        else:
            labels.append((xg, px, [gap], empty))
    for xg, _, gaps, empty in labels:
        between = "files" if empty else "files compared"
        text = (f"{gaps[0]:.0f} days\nwithout {between}" if len(gaps) == 1
                else f"{len(gaps)} gaps of\n{min(gaps):.0f} to {max(gaps):.0f} days")
        axes[0].annotate(text, (xg, 0), xytext=(3, 3), textcoords="offset points", fontsize=7,
                         color=MUTED)

    # Files left out because a channel could not be measured: a pale band
    # over each run of them, so the lines drawn across are not mistaken for data.
    left_out = [i + 1 for i, f in enumerate(cmp.files) if f.problems]
    runs: list[list[int]] = []
    for n in left_out:
        if runs and n == runs[-1][-1] + 1:
            runs[-1].append(n)
        else:
            runs.append([n])
    for run in runs:
        for ax in axes:
            ax.axvspan(run[0] - 0.5, run[-1] + 0.5, color=GRID, alpha=0.6, linewidth=0,
                       zorder=0)

    # x-axis: report row number, with dates for about 10 files, evenly spread
    # (the rows are not evenly spaced when files were left out)
    ticks: list[int] = []
    for i, xi in enumerate(x):
        if not ticks or xi - x[ticks[-1]] >= span / 10:
            ticks.append(i)
    axes[-1].set_xticks(x[ticks])
    axes[-1].set_xticklabels([f"#{int(x[i])}\n{files[i].start:%Y-%m-%d}\n{files[i].start:%H:%M}"
                              for i in ticks], fontsize=7, color=INK_2)
    axes[-1].set_xlabel("file (# as in the report table), start time UTC", fontsize=8,
                        color=INK_2)

    inst = ", ".join(cmp.instruments) or "unknown instrument"
    fig.suptitle(f"{a} vs {b}  |  {inst}  |  {cmp.mode} mode  |  {len(rows)} files",
                 fontsize=11, color=INK, x=0.06, ha="left", y=0.955)
    fig.text(0.06, 0.925, "Vertical rule = jump, black line = trend, ring = unusual file."
             + (" Pale band = files left out (a channel could not be measured)." if runs else "")
             + " Ratios above 1 mean channel A is higher.", fontsize=8, color=INK_2)
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
