"""Turn a Comparison into words and tables a person can read.

Plain ASCII only, so the report prints correctly in any terminal, including
the old Windows command prompt.
"""

import csv
import math
from datetime import datetime
from pathlib import Path

import numpy as np

from . import checklist, health
from .changes import Changes, Jump, Switching, neighbours
from .compare import EXTRA_SPIKES_PER_HOUR, Comparison, FileResult
from .spikes import MODES, SHARED_WINDOW_S

TIME = "%Y-%m-%d %H:%M"
BENCH_BUSY_PER_HOUR = 10.0  # a typical rate above this on the bench gets a warning
COARSE_NOISE = 0.5  # counts: below this a channel is mostly whole-count steps
BOTH_CHANGED = 2.0  # both channels' noise changing this much means the signal changed
ONE_CHANNEL_FACTOR = 3.0  # a channel's own spikes must outnumber its partner's this much


# --- formatting helpers ----------------------------------------------------------


def factor(x: float) -> str:
    """A ratio as a person would say it: 790, 45, 3.2, 0.25, 0.091."""
    if not math.isfinite(x):
        return "?"
    if x >= 100:
        return f"{x:,.0f}"
    if x >= 10:
        return f"{x:.0f}"
    if x >= 1:
        return f"{x:.1f}"
    return f"{x:.2g}"


def rate(x: float) -> str:
    if not math.isfinite(x):
        return "-"
    if x == 0:
        return "0"
    if x < 10:
        return f"{x:.1f}"
    return f"{x:,.0f}"


def duration(days: float) -> str:
    hours = days * 24
    if hours < 1:
        return f"{hours * 60:.0f} minutes"
    if hours < 48:
        return f"{hours:.1f} hours"
    if days < 365:
        return f"{days:.1f} days"
    return f"{days / 365.25:.1f} years"


def how_sure(p: float, shuffles: int | None = None) -> str:
    """The chance that a finding is luck, in words. ``shuffles``: for a jump."""
    words = "very sure" if p <= 0.001 else "fairly sure"
    if shuffles is not None:
        if p * (shuffles + 1) <= 1.0001:  # no shuffle reached it: the smallest possible
            return f"{words}: none of {shuffles:,} shuffles of these files gave a change this big"
        return (f"{words}: a change this big turned up by chance in about 1 in {1 / p:,.0f}"
                " shuffles of these files")
    if p < 1e-4:
        return f"{words}: the chance of a steady change this big by luck is below 1 in 10,000"
    return f"{words}: the chance of a steady change this big by luck is about 1 in {1 / p:,.0f}"


def times_word(f: float) -> str:
    """'about 20x higher' / 'about 4x lower' for a ratio of ratios."""
    if f >= 1:
        return f"about {factor(f)}x higher"
    return f"about {factor(1 / f)}x lower"


def spike_ratio_text(r: FileResult) -> str:
    """The spike ratio for the table: a number, or who had spikes."""
    a, b = r.a, r.b
    if a is None or b is None or r.problems or r.usable_s <= 0:
        return "-"
    if a.spikes == 0 and b.spikes == 0:
        return "none"
    if b.spikes == 0:
        return "A only"
    if a.spikes == 0:
        return "B only"
    return factor(a.rate_per_hour / b.rate_per_hour)


def both_per_hour(r: FileResult) -> float:
    """Spikes per hour that happened on both channels at once (within 0.1 s)."""
    if r.a is None or r.b is None or r.usable_s <= 0:
        return float("nan")
    return min(r.a.shared, r.b.shared) / (r.usable_s / 3600)


def _setup_label(r: FileResult) -> str:
    s = r.setup
    if s is None or s.probes == "unknown":
        return "probes unknown"
    probes = {"test": "test probes", "real": "real probes"}
    moving = {True: "", False: ", pressure changing", None: ""}[s.steady]
    return probes[s.probes] + moving


def notes_for(r: FileResult, cmp: Comparison) -> list[str]:
    out = []
    if r.error:
        return [r.error]
    if r.usable_s <= 0:
        deeper = cmp.options.min_pressure_dbar
        out.append("no usable data, left out" + (f" (never deeper than {deeper:g} dbar?)"
                                                 if deeper is not None else ""))
    elif r.usable_s < cmp.options.min_usable_s:
        out.append("too little usable data, left out")
    for label, slot, ch in (("A", "a", r.a), ("B", "b", r.b)):
        if ch is None:
            continue
        problem = r.problems.get(slot)
        if problem is not None:
            out.append(f"left out: {label} {problem.kind} ({problem.detail})")
        elif ch.noise == 0:
            out.append(f"{label} is flat (no signal)")
        if ch.saturated and (problem is None or problem.kind != "railed"):
            out.append(f"{label} hit its limit {ch.saturated}x")
        if ch.bad_buffer_samples:
            out.append(f"{label} had {ch.bad_buffer_samples} bad samples (skipped)")
    labels = {_setup_label(f) for f in cmp.files if not f.error}
    if len(labels) > 1:  # say which files differ from the rest
        out.append(_setup_label(r))
    if cmp.mode == "bench":
        for label, ch in (("A", r.a), ("B", r.b)):
            limit = checklist.limit_for(ch.name) if ch else None
            if limit and checklist.problems(ch, limit):
                out.append(f"{label} outside bench checklist")
        if 0 < r.usable_s < checklist.MIN_FILE_SECONDS:
            out.append("shorter than the 60 s the checklist asks for")
    if r.setup is not None:
        out += [w.split(":")[0].replace(" is ", " ") for w in r.setup.warnings]
    if r.notes:
        out.append("reader warning (see CSV)")
    return out


# --- the table -------------------------------------------------------------------


def _jump_rows(changes: Changes, used: list[int]) -> dict[int, Jump]:
    """Position in the full file list -> the jump that happens just before it."""
    return {used[j.index]: j for j in changes.jumps}


def table(cmp: Comparison) -> list[str]:
    a, b = cmp.channel_a, cmp.channel_b
    w = max(8, len(a), len(b))
    name_w = max([len("P-file")] + [len(f.path.name) for f in cmp.files])
    group = 2 * w + 12  # two channel columns and a ratio column, with spaces
    head1 = (f"{'':>4}  {'':<{name_w}}  {'':<16}  {'':>6}  "
             f"{'spikes per hour':-^{group + 8}}  {'noise (counts)':-^{group}}")
    head2 = (f"{'#':>4}  {'P-file':<{name_w}}  {'Start (UTC)':<16}  {'Hours':>6}  "
             f"{a:>{w}}  {b:>{w}}  {'both':>6}  {'A/B':>8}  {a:>{w}}  {b:>{w}}  {'A/B':>8}  Notes")
    lines = [head1, head2, "-" * len(head2)]

    spike_jumps = _jump_rows(cmp.spikes, cmp.used)
    noise_jumps = _jump_rows(cmp.noise, cmp.noise_used)
    unusual_spike = {cmp.used[u.index] for u in cmp.spikes.unusual}
    unusual_noise = {cmp.noise_used[u.index] for u in cmp.noise.unusual}
    group_starts = {cmp.used[g[0]] for g in cmp.spikes.groups[1:]}

    previous: datetime | None = None
    for i, r in enumerate(cmp.files):
        if r.error:
            continue
        if i in group_starts and previous is not None:
            gap = (r.start - previous).total_seconds() / 86400
            lines.append(f"{'':>4}  ... {duration(gap)} with no files ...")
        if i in spike_jumps:
            lines.append(f"{'':>4}  ==== JUMP in spike ratio: {times_word(spike_jumps[i].factor)}"
                         " from here ====")
        if i in noise_jumps:
            lines.append(f"{'':>4}  ==== JUMP in noise ratio: {times_word(noise_jumps[i].factor)}"
                         " from here ====")
        notes = notes_for(r, cmp)
        if i in unusual_spike:
            notes.insert(0, "unusual spike ratio")
        if i in unusual_noise:
            notes.insert(0, "unusual noise ratio")
        noise_ratio = f"{r.a.noise / r.b.noise:.2f}" if r.b.noise > 0 and not r.problems else "-"
        noise_a, noise_b = (f"{x:.2f}" if math.isfinite(x) else "-" for x in (r.a.noise, r.b.noise))
        lines.append((
            f"{i + 1:>4}  {r.path.name:<{name_w}}  {r.start.strftime(TIME):<16}  "
            f"{r.usable_s / 3600:>6.2f}  {rate(r.a.rate_per_hour):>{w}}  "
            f"{rate(r.b.rate_per_hour):>{w}}  {rate(both_per_hour(r)):>6}  "
            f"{spike_ratio_text(r):>8}  "
            f"{noise_a:>{w}}  {noise_b:>{w}}  {noise_ratio:>8}  {'; '.join(notes)}"
        ).rstrip())
        previous = r.start
    bad = [r for r in cmp.files if r.error]
    if bad:
        lines.append("")
        lines.append("Files that could not be used:")
        for r in bad:
            lines.append(f"  {r.path}: {r.error}")
    return lines


# --- the findings ----------------------------------------------------------------


def _median_rates(cmp: Comparison, rows: list[int]) -> tuple[float, float]:
    ra = [cmp.files[i].a.rate_per_hour for i in rows]
    rb = [cmp.files[i].b.rate_per_hour for i in rows]
    return float(np.median(ra)), float(np.median(rb))


def _median_noise(cmp: Comparison, rows: list[int]) -> tuple[float, float]:
    return (float(np.median([cmp.files[i].a.noise for i in rows])),
            float(np.median([cmp.files[i].b.noise for i in rows])))


def _left_out_between(i: int, j: int) -> str:
    """' compared (5 left out)' when files lie between rows i and j but were not compared."""
    k = j - i - 1
    return f" compared ({k} left out)" if k > 0 else ""


def _describe_file(cmp: Comparison, i: int) -> str:
    r = cmp.files[i]
    return f"file {i + 1} ({r.path.name}, {r.start.strftime(TIME)} UTC)"


def _windows(changes: Changes, j: Jump, w: int) -> tuple[range, range]:
    """Positions (in the analysed series) of the files just before and after a jump."""
    edges = [0, *(x.index for x in changes.jumps), changes.n]
    k = edges.index(j.index)
    lo, hi = edges[k - 1], edges[k + 1]
    return range(max(lo, j.index - w), j.index), range(j.index, min(hi, j.index + w))


def _pairs_of_jumps(changes: Changes, min_factor: float) -> set[int]:
    """Jumps that go out and come back: the first of each such pair."""
    out = set()
    for first, second in zip(changes.jumps, changes.jumps[1:], strict=False):
        opposite = (first.factor > 1) != (second.factor > 1)
        returns = abs(math.log10(first.factor * second.factor)) < math.log10(min_factor)
        if opposite and returns:
            out.add(first.index)
    return out


def _both_noises_changed(cmp: Comparison, before: list[int], after: list[int],
                         spikes: bool) -> str:
    """If both channels' noise changed together, say so: the signal changed, not a channel."""
    (a0, b0), (a1, b1) = _median_noise(cmp, before), _median_noise(cmp, after)
    if not (a0 > 0 and b0 > 0 and a1 > 0 and b1 > 0):
        return ""
    fa, fb = a1 / a0, b1 / b0
    up = fa >= BOTH_CHANGED and fb >= BOTH_CHANGED
    down = fa <= 1 / BOTH_CHANGED and fb <= 1 / BOTH_CHANGED
    if not (up or down):
        return ""
    sizes = f"{factor(fa)}x and {factor(fb)}x" if up else f"{factor(1 / fa)}x and {factor(1 / fb)}x"
    text = f"Both channels' noise {'rose' if up else 'fell'} together ({sizes}): "
    if spikes:
        return text + ("the spike threshold follows the noise, so these spike counts cannot be"
                       " compared with the other files'.")
    if cmp.mode == "live":
        return text + ("the water or the vehicle changed (turbulence, speed, vibration), not one"
                       " channel. A noise-ratio change like this is not by itself a fault.")
    return text + ("something reached both (handling, vibration, electrical interference),"
                   " not one channel's electronics.")


def _channel_words(cmp: Comparison, before: list[int], after: list[int], spikes: bool) -> str:
    a, b = cmp.channel_a, cmp.channel_b
    if spikes:
        (a0, b0), (a1, b1) = _median_rates(cmp, before), _median_rates(cmp, after)
        unit, fmt = "spikes per hour", rate
    else:
        (a0, b0), (a1, b1) = _median_noise(cmp, before), _median_noise(cmp, after)
        unit, fmt = "counts", lambda x: f"{x:.2f}"
    parts = []
    for name, x0, x1 in ((a, a0, a1), (b, b0, b1)):
        same = (x0 == x1) or (x0 > 0 and x1 > 0 and abs(math.log10(x1 / x0)) < 0.1)
        if same:
            parts.append(f"{name} stayed at about {fmt(x0)}")
        else:
            parts.append(f"{name} went from about {fmt(x0)} to about {fmt(x1)}")
    return f"{parts[0]} {unit}; {parts[1]}."


def _not_alike(spikes: bool) -> str:
    """Each channel's median and the middle file disagree by more than min_factor squared."""
    busy = "busy" if spikes else "noisy"
    return (f"  These two do not agree because the files differ a lot from one another: the"
            f" channels were {busy} in different files, so no one file is typical. Look at the"
            " table, and compare files of one kind (bench tests, casts, deployments)"
            " separately.")


def findings(cmp: Comparison, changes: Changes, rows: list[int], what: str,
             min_factor: float) -> list[str]:
    """Plain-language description of the jumps, trends and unusual files in one ratio."""
    spikes = what == "spike"
    out: list[str] = []
    if not rows:
        if any(f.problems for f in cmp.files):
            return ["  No files could be compared: each had too little usable data or a"
                    " channel that could not be measured (see above)."]
        return ["  No files had enough usable data to compare."]
    y = np.array([cmp.files[i].spike_ratio_log if spikes else cmp.files[i].noise_ratio_log
                  for i in rows])
    # The middle file by ratio (the lower of the two middle ones): its numbers
    # belong together, unlike each channel's median, which may come from
    # different files.
    k = int(np.argsort(y, kind="stable")[(len(rows) - 1) // 2])
    middle = cmp.files[rows[k]]
    a, b = cmp.channel_a, cmp.channel_b
    where = f"In the middle file by {what} ratio, {_describe_file(cmp, rows[k])}"
    if spikes:
        ra, rb = _median_rates(cmp, rows)
        if ra == 0 and rb == 0:
            out.append(f"  In a typical file neither {a} nor {b} had any spikes.")
        else:
            out.append(f"  In a typical file {a} had about {rate(ra)} spikes per hour and "
                       f"{b} about {rate(rb)} (each channel's median over the {len(rows)}"
                       " files).")
            xa, xb = middle.a.rate_per_hour, middle.b.rate_per_hour
            if xa == 0 and xb == 0:
                out.append(f"  {where}, neither had spikes.")
            elif xb == 0 or xa == 0:
                only, x = (a, xa) if xb == 0 else (b, xb)
                out.append(f"  {where}, only {only} had spikes ({rate(x)} per hour).")
            else:
                more, less, f = (a, b, xa / xb) if xa >= xb else (b, a, xb / xa)
                out.append(f"  {where}, {a} had {rate(xa)} spikes per hour and {b} {rate(xb)}:"
                           f" {more} had {factor(f)} times as many as {less}.")
            unpaired = math.log10((ra + EXTRA_SPIKES_PER_HOUR) / (rb + EXTRA_SPIKES_PER_HOUR))
            if abs(unpaired - y[k]) > 2 * math.log10(min_factor):
                out.append(_not_alike(spikes))
        for name, ch in ((a, "a"), (b, "b")):
            if all(getattr(cmp.files[i], ch).spikes == 0 for i in rows):
                out.append(f"  {name} had no spikes at all in any of the {len(rows)} files.")
        out += shared_words(cmp, rows)
        if cmp.mode == "live" and (ra > 0 or rb > 0):
            out.append("  In live data most spikes on real probes are real signal (turbulence,"
                       " plankton, vibration), so the counts say more about the water than about"
                       " the electronics. Spikes on one channel only are the ones that point at a"
                       " channel.")
    else:
        na, nb = _median_noise(cmp, rows)
        out.append(f"  In a typical file the noise was {na:.2f} counts on {a} and {nb:.2f} on"
                   f" {b} (each channel's median over the {len(rows)} files).")
        xa, xb = middle.a.noise, middle.b.noise
        out.append(f"  {where}, it was {xa:.2f} counts on {a} and {xb:.2f} on {b}: a noise"
                   f" ratio of {xa / xb:.2f}.")
        if na > 0 and nb > 0 and abs(math.log10(na / nb) - y[k]) > 2 * math.log10(min_factor):
            out.append(_not_alike(spikes))
        if cmp.mode == "live":
            out.append("  In live data a real probe's 'noise' includes real signal, so it rises"
                       " and falls with the water. When that signal is far above the electronic"
                       " noise, the noise ratio shows how differently the two probes respond to"
                       " it (their sensitivity, where they sit, how vibration reaches them), so"
                       " it can change with no fault at all. It says most about the electronics"
                       " when both channels are quiet.")

    if changes.too_few:
        out.append(f"  Too few files ({len(rows)}) to look for jumps or trends; "
                   f"at least {cmp.options.spike_changes.min_trend_files} are needed.")
        return out

    if len(changes.groups) > 1:
        spans = []
        for g0, g1 in changes.groups:
            first, last = cmp.files[rows[g0]].start, cmp.files[rows[g1 - 1]].start
            spans.append(f"{first:%Y-%m-%d} to {last:%Y-%m-%d} ({g1 - g0} files)")
        compared = " compared" if len(rows) < sum(not f.error for f in cmp.files) else ""
        out.append(f"  The files{compared} fall in {len(changes.groups)} separate stretches of"
                   " time: " + "; ".join(spans) + ".")

    w = cmp.options.spike_changes.long_window
    shuffles = cmp.options.spike_changes.n_shuffles
    temporary = _pairs_of_jumps(changes, min_factor)
    skip_next = False
    for n, j in enumerate(changes.jumps):
        if skip_next:
            skip_next = False
            continue
        before, after = _windows(changes, j, w)
        before_rows = [rows[i] for i in before]
        after_rows = [rows[i] for i in after]
        here, prev = rows[j.index], rows[j.index - 1]
        if j.index in temporary:
            back = changes.jumps[n + 1]
            last = rows[back.index - 1]
            out.append("")
            out.append(f"  TEMPORARY CHANGE, files {here + 1} to {last + 1} "
                       f"({cmp.files[here].start:%Y-%m-%d %H:%M} to "
                       f"{cmp.files[last].start:%Y-%m-%d %H:%M} UTC):")
            out.append(f"    The {what} ratio was {times_word(j.factor)} than in the files "
                       "before and after, then went back.")
            out.append(f"    {_channel_words(cmp, before_rows, after_rows, spikes)}")
            inside = [rows[i] for i in range(j.index, back.index)]
            both = _both_noises_changed(cmp, before_rows, inside, spikes)
            if both:
                out.append(f"    {both}")
            out.append(f"    How sure: {how_sure(max(j.p_value, back.p_value), shuffles)}.")
            skip_next = True
            continue
        out.append("")
        out.append(f"  JUMP between file {prev + 1} and {_describe_file(cmp, here)}:")
        out.append(f"    The {what} ratio went from about {factor(10 ** j.before)} to about "
                   f"{factor(10 ** j.after)}: {times_word(j.factor)}, and stayed there.")
        out.append(f"    {_channel_words(cmp, before_rows, after_rows, spikes)}")
        both = _both_noises_changed(cmp, before_rows, after_rows, spikes)
        if both:
            out.append(f"    {both}")
        gap = (cmp.files[here].start - cmp.files[prev].start).total_seconds() / 86400
        if any(g[0] == j.index for g in changes.groups):
            out.append(f"    It happened across a gap of {duration(gap)} with no files"
                       f"{_left_out_between(prev, here)}.")
        out.append(f"    How sure: {how_sure(j.p_value, shuffles)}.")

    for t in changes.trends:
        first, last = rows[t.first], rows[t.last]
        direction = "rose" if t.factor > 1 else "fell"
        out.append("")
        out.append(f"  TREND, files {first + 1} to {last + 1} ({duration(t.days)}):")
        out.append(f"    The {what} ratio {direction} steadily from about {factor(10 ** t.start)}"
                   f" to about {factor(10 ** t.end)} ({times_word(t.factor)} by the end).")
        k = min(5, (t.last - t.first + 1) // 2)
        first_rows = [rows[i] for i in range(t.first, t.first + k)]
        last_rows = [rows[i] for i in range(t.last - k + 1, t.last + 1)]
        out.append("    " + _channel_words(cmp, first_rows, last_rows, spikes))
        both = _both_noises_changed(cmp, first_rows, last_rows, spikes)
        if both:
            out.append(f"    {both}")
        out.append(f"    How sure: {how_sure(t.p_value)}.")

    for u in changes.unusual:
        i = rows[u.index]
        where = {"both": "the files on either side of it",
                 "before": "the files before it (it is the last file compared)",
                 "after": "the files after it (it is the first file compared)"}[u.side]
        out.append("")
        out.append(f"  UNUSUAL: {_describe_file(cmp, i)} has a {what} ratio "
                   f"{times_word(u.factor)} than {where}.")
        both = _both_noises_changed(cmp, [rows[k] for k in neighbours(u.index, len(rows))], [i],
                                    spikes)
        if both:
            out.append(f"    {both}")

    for sw in changes.switching:
        out += _switching_words(cmp, sw, rows, y, what)

    if not (changes.jumps or changes.trends or changes.unusual or changes.switching):
        out.append(f"  No jumps, trends or unusual files in the {what} ratio.")
    return out


def _switching_words(cmp: Comparison, sw: Switching, rows: list[int], y: np.ndarray,
                     what: str) -> list[str]:
    spikes = what == "spike"
    first, last = rows[sw.first], rows[sw.last]
    high, low = [rows[i] for i in sw.high], [rows[i] for i in sw.low]
    if spikes:
        (ha, hb), (la, lb) = _median_rates(cmp, high), _median_rates(cmp, low)
        unit, fmt = "spikes per hour", rate
    else:
        (ha, hb), (la, lb) = _median_noise(cmp, high), _median_noise(cmp, low)
        unit, fmt = "counts", lambda x: f"{x:.2f}"
    out = ["",
           f"  SWITCHING, files {first + 1} to {last + 1} ({cmp.files[first].start:%Y-%m-%d} to"
           f" {cmp.files[last].start:%Y-%m-%d}):",
           f"    The {what} ratio went back and forth {sw.switches} times between about"
           f" {factor(10 ** sw.level(y, sw.high))} ({len(high)} files) and about"
           f" {factor(10 ** sw.level(y, sw.low))} ({len(low)} files). That is too often for the"
           " jump search, which needs each level to last about 15 to 20 files.",
           f"    {cmp.channel_a}: about {fmt(ha)} {unit} in the high files and {fmt(la)} in the low"
           f" ones; {cmp.channel_b}: about {fmt(hb)} and {fmt(lb)}."]
    both = _both_noises_changed(cmp, low, high, spikes)
    if both:
        out.append(f"    {both}")
    out.append("    A fault that comes and goes (a connector or probe), or two kinds of files mixed"
               " together (bench tests and casts, say), looks like this.")
    if cmp.mode == "live":
        out.append("    In live data the water can do it too: real probes follow it, and one may"
                   " respond more than the other.")
    return out


def mode_line(cmp: Comparison) -> str:
    a, b = cmp.spike_a, cmp.spike_b
    if a.fixed_threshold_counts is not None:
        rule = (f"a spike is a jump bigger than {a.fixed_threshold_counts:g} counts"
                + (f" on A and {b.fixed_threshold_counts:g} on B"
                   if b.fixed_threshold_counts != a.fixed_threshold_counts else ""))
    else:
        rule = (f"a spike is a jump bigger than {a.threshold_sigma:g} times the"
                f" channel's noise (and at least {a.min_threshold_counts:g} counts)")
    kind = {"bench": "bench test", "live": "live data"}[cmp.mode]
    custom = "" if MODES[cmp.mode] in (a, b) else ", with your own settings"
    return f"{cmp.mode} ({kind}{custom}): {rule}"


def _ranges(numbers: list[int], most: int = 12) -> str:
    """'4-6, 10, 12-14' for [4, 5, 6, 10, 12, 13, 14]."""
    runs: list[list[int]] = []
    for n in numbers:
        if runs and n == runs[-1][-1] + 1:
            runs[-1].append(n)
        else:
            runs.append([n])
    text = [f"{r[0]}-{r[-1]}" if len(r) > 2 else ", ".join(map(str, r)) for r in runs]
    return ", ".join(text[:most]) + (", ..." if len(text) > most else "")


def unmeasured_section(cmp: Comparison) -> list[str]:
    """Channels whose spikes and noise could not be measured, by channel and reason."""
    out = []
    for slot, name in (("a", cmp.channel_a), ("b", cmp.channel_b)):
        for kind in ("no probe", "railed", "flat"):
            hit = [(i, f) for i, f in enumerate(cmp.files)
                   if not f.error and slot in f.problems and f.problems[slot].kind == kind]
            if not hit:
                continue
            where = f"in {len(hit)} file(s) (#{_ranges([i + 1 for i, _ in hit])})"
            chans = [getattr(f, slot) for _, f in hit]
            if kind == "no probe":
                slow = health.slow_thermistor(chans[0].name)
                temps = sorted({round(f.setup.no_probe[slow], 2) for _, f in hit})
                read = (f"{temps[0]:.2f}" if len(temps) == 1
                        else f"{temps[0]:.2f} to {temps[-1]:.2f}")
                out.append(f"  {name}: no probe {where}. {slow} read {read} C, beyond any water"
                           " temperature, so nothing was connected and these spikes and noise"
                           " are not a probe's.")
            elif kind == "railed":
                parts = []
                share = max(ch.saturated / (ch.usable_s * ch.fs) for ch in chans)
                if share >= health.RAILED_SHARE:
                    parts.append(f"at the end of its range in up to {100 * share:.0f}% of the"
                                 " samples")
                over = [ch.threshold for ch in chans if ch.bits == 16 and ch.threshold > 2**15]
                if over:
                    parts.append(f"swinging so widely that its spike threshold reached"
                                 f" {max(over):,.0f} counts, more than half its range")
                out.append(f"  {name}: railed {where}: {'; '.join(parts)}. A spike could not"
                           " stand out.")
            else:
                noises = [ch.noise for ch in chans]
                how = ("it did not change at all" if max(noises) == 0
                       else f"its noise was {min(noises):.2f} to {max(noises):.2f} counts, less"
                            " than a working channel of its kind")
                out.append(f"  {name}: flat {where}: {how} (dead or disconnected?).")
    if not out:
        return []
    out.append("  These files are left out of the comparison. A channel that cannot show a"
               " spike would look like the clean one, and its partner like the faulty one.")
    railed = any(p.kind == "railed" for f in cmp.files for p in f.problems.values())
    if railed and cmp.mode == "live" and cmp.options.min_pressure_dbar is None:
        out.append("  If the instrument was out of the water for part of these files,"
                   " --min-pressure (for example --min-pressure 5) leaves that time out.")
    return ["", "Channels that could not be measured", *out]


def channel_warnings(cmp: Comparison) -> list[str]:
    """Warnings about channels this kind of analysis does not suit."""
    good = [f for f in cmp.files if not f.error and f.a is not None]
    out = []
    for slot, name in (("a", cmp.channel_a), ("b", cmp.channel_b)):
        results = [getattr(f, slot) for f in good]
        if not results:
            continue
        ch = results[0]
        if ch.bits == 32:
            out.append(f"WARNING: {name} is a 32-bit channel (a counter or a frequency, such as"
                       " JAC_C), not a voltage. Its spikes and noise in counts mean little.")
        elif not ch.fast:
            out.append(f"WARNING: {name} is a slow channel ({ch.fs:.0f} samples per second)."
                       " This program is made for the fast channels (T1_dT1, T2_dT2, sh1, sh2);"
                       " on a slow channel its 'spikes' are often ordinary steps in the signal.")
        coarse = sum(r.noise < COARSE_NOISE for r in results if math.isfinite(r.noise))
        if coarse * 2 >= len(results):
            out.append(f"WARNING: {name} moves in steps of whole counts with almost no noise in"
                       f" {coarse} of {len(results)} files, so its 'spikes' are steps in the"
                       " signal, not electrical spikes.")
    return out


def temperature_warnings(cmp: Comparison) -> list[str]:
    """Temperature channels that do not behave like their partner test probe."""
    found: dict[str, list[int]] = {}
    for i, f in enumerate(cmp.files):
        if not f.error and f.setup is not None:
            for w in f.setup.warnings:
                found.setdefault(w, []).append(i + 1)
    out = []
    for warning, rows in found.items():
        shown = ", ".join(str(r) for r in rows[:15]) + (" ..." if len(rows) > 15 else "")
        out.append(f"  {warning} ({len(rows)} file(s): #{shown})")
    return ["", "Temperature channels", *out] if out else []


def shared_words(cmp: Comparison, rows: list[int]) -> list[str]:
    """How many spikes the two channels had at the same moment, and what that means."""
    out = []
    parts = []
    hours = sum(cmp.files[i].usable_s for i in rows) / 3600
    for name, other, slot, other_slot in ((cmp.channel_a, cmp.channel_b, "a", "b"),
                                          (cmp.channel_b, cmp.channel_a, "b", "a")):
        total = sum(getattr(cmp.files[i], slot).spikes for i in rows)
        if total == 0:
            continue
        shared = sum(getattr(cmp.files[i], slot).shared for i in rows)
        # Chance: the other channel's spikes cover 2 x 0.1 s each.
        others = sum(getattr(cmp.files[i], other_slot).spikes for i in rows)
        chance = min(1.0, 2 * SHARED_WINDOW_S * others / (hours * 3600)) if hours > 0 else 0.0
        share = 100 * shared / total
        share_text = "under 1%" if 0 < share < 1 else f"{share:.0f}%"
        text = f"{share_text} of {name}'s spikes came within 0.1 s of one on {other}"
        if chance >= 0.05:
            text += f" (about {100 * chance:.0f}% would by chance at these rates)"
        parts.append(text)
    if parts:
        out.append("  " + "; ".join(parts) + ".")
        out.append("  Spikes on both channels at once come from something that reaches both"
                   " (a bump, shaking, real signal); spikes on one channel only point at that"
                   " channel.")
    return out


def bench_hints(cmp: Comparison) -> list[str]:
    """Extra plain-language warnings that only make sense for a bench test."""
    rows = cmp.used
    if cmp.mode != "bench" or not rows:
        return []
    out = []
    # Healthy thermistor channels in 163 bench tests averaged 0.5 spikes per
    # hour, shear 3.6, mostly shared between the two channels (up to about 90
    # per hour on both shear channels at once in a shaky setting); a faulty T1
    # showed 600 to 3000, on that channel alone.
    only = {name: float(np.median([getattr(cmp.files[i], slot).only_per_hour for i in rows]))
            for name, slot in ((cmp.channel_a, "a"), (cmp.channel_b, "b"))}
    both = float(np.median([both_per_hour(cmp.files[i]) for i in rows]))
    # Shaking also gives spikes on one channel only (a bump need not cross the
    # threshold on both): on the OOI SN 433 bench tests 19 and 44 per hour of
    # each shear channel's own, besides 48 per hour shared. So a channel is
    # singled out only when its own spikes far outnumber its partner's.
    (a, ra), (b, rb) = only.items()
    for name, r, other in ((a, ra, rb), (b, rb, ra)):
        if r >= BENCH_BUSY_PER_HOUR and r >= ONE_CHANNEL_FACTOR * (other + 1):
            out.append(f"  {name} had about {rate(r)} spikes per hour that the other channel did"
                       " not have, in a typical file. With test probes on a quiet bench a healthy"
                       " channel has few or none, so the channel, its connector or the test probe"
                       " may need attention.")
    if min(ra, rb) >= BENCH_BUSY_PER_HOUR and max(ra, rb) < ONE_CHANNEL_FACTOR * (min(ra, rb) + 1):
        out.append(f"  Both channels had many spikes of their own ({a} about {rate(ra)}, {b} about"
                   f" {rate(rb)} per hour in a typical file). That points at the surroundings"
                   " rather than one channel; repeat the test somewhere quieter to be sure.")
    if both >= BENCH_BUSY_PER_HOUR:
        out.append(f"  Both channels spiked at the same moment about {rate(both)} times per hour"
                   " in a typical file: the instrument was probably bumped or shaken (people"
                   " nearby, a ship, machinery). That does not point at either channel.")
    same_kind = all(cmp.files[i].a.kind and cmp.files[i].a.kind == cmp.files[i].b.kind
                    for i in rows)
    if same_kind and cmp.noise_used:
        y = np.array([cmp.files[i].noise_ratio_log for i in cmp.noise_used])
        typical = 10 ** float(np.median(y))
        if not 0.8 <= typical <= 1.25:
            out.append(f"  The two channels look like the same kind of sensor, but their noise"
                       f" differs by a factor of {typical:.2f}. On the MicroRiders checked so"
                       " far, healthy matching channels were within about 10% of each other.")
    return out


def checklist_section(cmp: Comparison) -> list[str]:
    """The Rockland bench checklist items about the two channels, file by file."""
    if cmp.mode != "bench":
        return []
    rows = [i for i, f in enumerate(cmp.files) if not f.error and f.usable_s > 0]
    if not rows:
        return []
    out = ["", "Rockland Bench Test Review Checklist (V3): items about these two channels",
           "  ('typically' = 99.9% of the samples, the same as rsi-tpw bench; the checklist is"
           " for a 1-minute file, so longer files are judged a minute at a time and the"
           " typical minute is compared with the limit)"]
    seen = set()
    for slot, name in (("a", cmp.channel_a), ("b", cmp.channel_b)):
        limit = checklist.limit_for(name)
        if limit is None:
            out.append(f"  The checklist has no limits for {name}.")
            continue
        if limit.wording not in seen:
            out.append(f'  "{limit.wording}."')
            seen.add(limit.wording)
        results = [(i, getattr(cmp.files[i], slot)) for i in rows]
        ranges = [ch.typical_range for _, ch in results]
        means = [ch.mean for _, ch in results]
        bad = [i + 1 for i, ch in results if checklist.problems(ch, limit)]
        text = (f"    {name}: in a typical minute within +-{min(ranges):,.0f} to"
                f" +-{max(ranges):,.0f} counts")
        if limit.offset is not None:
            text += f", offset {min(means):,.0f} to {max(means):,.0f} counts"
        out.append(text + ".")
        minutes = [x for _, ch in results for x in ch.minute_ranges]
        over = sum(x > limit.typical for x in minutes)
        if over:
            worst_i, worst = max(results, key=lambda r: r[1].worst_range)
            share = 100 * over / len(minutes)
            share_text = "under 1%" if share < 1 else f"{share:.0f}%"
            out.append(f"      {over:,} of {len(minutes):,} minutes ({share_text}) were over"
                       f" +-{limit.typical:g}; the worst was +-{worst.worst_range:,.0f} counts"
                       f" (file #{worst_i + 1}).")
        if bad:
            shown = ", ".join(str(b) for b in bad[:20]) + (" ..." if len(bad) > 20 else "")
            out.append(f"      OUTSIDE the checklist in {len(bad)} of {len(results)} files"
                       f" (#{shown}).")
        else:
            out.append(f"      Within the checklist in all {len(results)} files.")
    short = [i + 1 for i in rows if cmp.files[i].usable_s < checklist.MIN_FILE_SECONDS]
    if short:
        out.append(f"  The checklist asks for files of at least {checklist.MIN_FILE_SECONDS} s;"
                   f" {len(short)} file(s) are shorter.")
    out.append("  Not checked here: the spectra, accelerometer, pressure, inclinometer and EM"
               " items. Use rsi-tpw bench, Zissou Essentials or the ODAS quick_bench figures"
               " for those.")
    return out


def text_report(cmp: Comparison) -> str:
    good = [f for f in cmp.files if not f.error]
    lines = ["Rockland spike and noise comparison", ""]
    lines.append(f"  Channels:   A = {cmp.channel_a}   B = {cmp.channel_b}")
    lines.append(f"  Mode:       {mode_line(cmp)}")
    lines.append(f"              Why: {cmp.mode_reason}.")
    if cmp.options.min_pressure_dbar is not None:
        lines.append(f"  Using only data deeper than {cmp.options.min_pressure_dbar:g} dbar.")
    if cmp.options.skip_start_s:
        lines.append(f"  Skipping the first {cmp.options.skip_start_s:g} s of every file.")
    inst = cmp.instruments
    lines.append(f"  Instrument: {', '.join(inst) if inst else 'unknown'}")
    if len(inst) > 1:
        lines.append("  WARNING: these files come from more than one instrument. Jumps and"
                     " trends may only reflect switching between them.")
    lines += [f"  {w}" for w in channel_warnings(cmp)]
    span = (f"{good[0].start.strftime(TIME)} to {good[-1].start.strftime(TIME)} UTC"
            if good else "no readable files")
    unread = len(cmp.files) - len(good)
    unmeasured = sum(bool(f.problems) for f in good)
    lines.append(f"  Files:      {len(cmp.files)} P-files, {span}; {len(cmp.used)} analysed"
                 + (f", {unmeasured} left out (a channel could not be measured)"
                    if unmeasured else "")
                 + (f", {unread} could not be read" if unread else ""))
    lines.append("")
    if good:
        lines += table(cmp)
        lines.append("")
        lines.append("  A only / B only = only that channel had spikes; none = neither did."
                     "  Hours = usable data.")
    else:
        lines += table(cmp)
    lines += unmeasured_section(cmp)
    lines += ["", "What the numbers say", "====================", "", "Spikes"]
    lines += findings(cmp, cmp.spikes, cmp.used, "spike", cmp.options.spike_changes.min_factor)
    lines += ["", "Noise"]
    lines += findings(cmp, cmp.noise, cmp.noise_used, "noise",
                      cmp.options.noise_changes.min_factor)
    lines += temperature_warnings(cmp)
    hints = bench_hints(cmp)
    if hints:
        lines += ["", "Bench test", *hints]
    lines += checklist_section(cmp)
    lines += ["", "For a plain-language guide to every number here, run:  rockland-noise explain"]
    return "\n".join(lines)


# --- CSV -------------------------------------------------------------------------


def _round(x: float, digits: int):
    return round(x, digits) if math.isfinite(x) else ""


def _checklist_word(ch) -> str:
    limit = checklist.limit_for(ch.name)
    if limit is None or not math.isfinite(ch.typical_range):
        return ""
    found = checklist.problems(ch, limit)
    return "; ".join(found) if found else "ok"


def write_csv(cmp: Comparison, path: Path) -> None:
    """One row per file with every number, for a spreadsheet."""
    spike_jump = {cmp.used[j.index]: j for j in cmp.spikes.jumps}
    noise_jump = {cmp.noise_used[j.index]: j for j in cmp.noise.jumps}
    unusual_s = {cmp.used[u.index] for u in cmp.spikes.unusual}
    unusual_n = {cmp.noise_used[u.index] for u in cmp.noise.unusual}
    trend_s = {cmp.used[i] for t in cmp.spikes.trends for i in range(t.first, t.last + 1)}
    trend_n = {cmp.noise_used[i] for t in cmp.noise.trends for i in range(t.first, t.last + 1)}

    def switch_levels(changes: Changes, rows: list[int]) -> dict[int, str]:
        return {rows[i]: level for sw in changes.switching
                for level, where in (("high", sw.high), ("low", sw.low)) for i in where}

    switch_s = switch_levels(cmp.spikes, cmp.used)
    switch_n = switch_levels(cmp.noise, cmp.noise_used)
    used = set(cmp.used)

    def channel_cols(prefix: str, ch) -> dict:
        if ch is None:
            return {}
        return {
            f"{prefix}_spikes": ch.spikes,
            f"{prefix}_spikes_per_hour": round(ch.rate_per_hour, 3),
            f"{prefix}_spikes_shared": ch.shared,
            f"{prefix}_spikes_only_per_hour": (round(ch.only_per_hour, 3)
                                               if math.isfinite(ch.only_per_hour) else ""),
            f"{prefix}_noise_counts": round(ch.noise, 4),
            f"{prefix}_threshold_counts": round(ch.threshold, 3),
            f"{prefix}_spikes_starting_upward_pct": (
                round(100 * ch.positive_first / ch.spikes, 1) if ch.spikes else ""),
            f"{prefix}_median_spike_size_counts": (
                round(ch.median_amplitude, 1) if math.isfinite(ch.median_amplitude) else ""),
            f"{prefix}_largest_counts": round(ch.largest, 1) if math.isfinite(ch.largest) else "",
            f"{prefix}_saturated_samples": ch.saturated,
            f"{prefix}_bad_samples": ch.bad_buffer_samples,
            f"{prefix}_offset_counts": round(ch.mean, 2) if math.isfinite(ch.mean) else "",
            f"{prefix}_typical_range_counts": (
                round(ch.typical_range, 2) if math.isfinite(ch.typical_range) else ""),
            f"{prefix}_worst_minute_range_counts": (
                round(ch.worst_range, 2) if math.isfinite(ch.worst_range) else ""),
            f"{prefix}_bench_checklist": _checklist_word(ch),
        }

    rows = []
    for i, r in enumerate(cmp.files):
        both = both_per_hour(r)
        row = {
            "row": i + 1,
            "file": r.path.name,
            "path": str(r.path),
            "start_utc": r.start.isoformat() if r.start else "",
            "instrument": r.instrument,
            "file_number": r.file_number if r.file_number >= 0 else "",
            "usable_hours": round(r.usable_s / 3600, 4),
            "channel_A": cmp.channel_a,
            "channel_B": cmp.channel_b,
            "mode": cmp.mode,
        }
        row |= channel_cols("A", r.a)
        row |= channel_cols("B", r.b)
        for label, slot in (("A", "a"), ("B", "b")):
            if r.a is not None:
                p = r.problems.get(slot)
                row[f"{label}_not_measurable"] = f"{p.kind}: {p.detail}" if p else ""
        row["spikes_on_both_per_hour"] = round(both, 3) if math.isfinite(both) else ""
        ev, st = r.evidence, r.setup
        if ev is not None:
            row |= {
                "probes": st.probes,
                "temperature_warnings": " | ".join(st.warnings),
                "T1_follows_JAC_T": _round(ev.t1_follows, 3),
                "T2_follows_JAC_T": _round(ev.t2_follows, 3),
                "JAC_T_spread_C": _round(ev.water_spread, 3),
                "pressure_steady": {True: "yes", False: "no", None: ""}[st.steady],
                "setup_evidence": st.detail,
                "T1_C": _round(ev.t1, 3), "T2_C": _round(ev.t2, 3),
                "T1_reference_C": _round(ev.t1_ref, 3), "T2_reference_C": _round(ev.t2_ref, 3),
                "T1_spread_C": _round(ev.t1_spread, 4), "T2_spread_C": _round(ev.t2_spread, 4),
                "tilt_spread_deg": _round(ev.tilt_spread, 3), "shake_counts": _round(ev.shake, 1),
                "pressure_dbar": _round(ev.pressure, 2),
                "pressure_spread_dbar": _round(ev.pressure_spread, 2),
            }
        sr, nr = r.spike_ratio_log, r.noise_ratio_log
        plain = (r.a.rate_per_hour / r.b.rate_per_hour
                 if r.a and r.b and r.b.spikes > 0 and r.b.usable_s > 0 else math.nan)
        row["spike_ratio_A_over_B"] = round(plain, 4) if math.isfinite(plain) else ""
        row["spike_ratio_for_jumps_and_trends"] = round(10**sr, 4) if math.isfinite(sr) else ""
        row["noise_ratio_A_over_B"] = round(10**nr, 4) if math.isfinite(nr) else ""
        row["analysed"] = "yes" if i in used else "no"
        flags = []
        if i in spike_jump:
            flags.append(f"spike ratio jump x{factor(spike_jump[i].factor)} before this file")
        if i in noise_jump:
            flags.append(f"noise ratio jump x{factor(noise_jump[i].factor)} before this file")
        if i in trend_s:
            flags.append("in a spike ratio trend")
        if i in trend_n:
            flags.append("in a noise ratio trend")
        if i in unusual_s:
            flags.append("unusual spike ratio")
        if i in unusual_n:
            flags.append("unusual noise ratio")
        if i in switch_s:
            flags.append(f"{switch_s[i]} in a switching spike ratio")
        if i in switch_n:
            flags.append(f"{switch_n[i]} in a switching noise ratio")
        row["findings"] = "; ".join(flags)
        row["notes"] = " | ".join([r.error] if r.error else r.notes)
        rows.append(row)

    columns: list[str] = []
    for row in rows:
        columns += [c for c in row if c not in columns]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
