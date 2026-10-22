"""Turn a Comparison into words and tables a person can read.

Plain ASCII only, so the report prints correctly in any terminal, including
the old Windows command prompt.
"""

import csv
import math
from datetime import datetime
from pathlib import Path

import numpy as np

from . import checklist
from .changes import Changes, Jump
from .compare import Comparison, FileResult
from .spikes import MODES

TIME = "%Y-%m-%d %H:%M"
BENCH_BUSY_PER_HOUR = 10.0  # a typical rate above this on the bench gets a warning


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


def how_sure(p: float) -> str:
    tries = 1 / p
    words = "very sure" if p <= 0.001 else "fairly sure"
    return (f"{words}: a change this big turned up by chance in about 1 in "
            f"{tries:,.0f} shuffles of these files")


def times_word(f: float) -> str:
    """'about 20x higher' / 'about 4x lower' for a ratio of ratios."""
    if f >= 1:
        return f"about {factor(f)}x higher"
    return f"about {factor(1 / f)}x lower"


def spike_ratio_text(r: FileResult) -> str:
    """The spike ratio for the table: a number, or who had spikes."""
    a, b = r.a, r.b
    if a is None or b is None:
        return "-"
    if a.spikes == 0 and b.spikes == 0:
        return "none"
    if b.spikes == 0:
        return "A only"
    if a.spikes == 0:
        return "B only"
    return factor(a.rate_per_hour / b.rate_per_hour)


def notes_for(r: FileResult, cmp: Comparison) -> list[str]:
    out = []
    if r.error:
        return [r.error]
    if r.usable_s < cmp.options.min_usable_s:
        out.append("too little usable data, left out")
    for label, ch in (("A", r.a), ("B", r.b)):
        if ch is None:
            continue
        if ch.noise == 0:
            out.append(f"{label} is flat (no signal)")
        if ch.saturated:
            out.append(f"{label} hit its limit {ch.saturated}x")
        if ch.bad_buffer_samples:
            out.append(f"{label} had {ch.bad_buffer_samples} bad samples (skipped)")
    if cmp.options.mode == "bench":
        for label, ch in (("A", r.a), ("B", r.b)):
            limit = checklist.limit_for(ch.name) if ch else None
            if limit and checklist.problems(ch, limit):
                out.append(f"{label} outside bench checklist")
        if r.usable_s < checklist.MIN_FILE_SECONDS:
            out.append("shorter than the 60 s the checklist asks for")
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
             f"{'spikes per hour':-^{group}}  {'noise (counts)':-^{group}}")
    head2 = (f"{'#':>4}  {'P-file':<{name_w}}  {'Start (UTC)':<16}  {'Hours':>6}  "
             f"{a:>{w}}  {b:>{w}}  {'A/B':>8}  {a:>{w}}  {b:>{w}}  {'A/B':>8}  Notes")
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
        noise_ratio = f"{r.a.noise / r.b.noise:.2f}" if r.b.noise > 0 else "-"
        lines.append((
            f"{i + 1:>4}  {r.path.name:<{name_w}}  {r.start.strftime(TIME):<16}  "
            f"{r.usable_s / 3600:>6.2f}  {rate(r.a.rate_per_hour):>{w}}  "
            f"{rate(r.b.rate_per_hour):>{w}}  {spike_ratio_text(r):>8}  "
            f"{r.a.noise:>{w}.2f}  {r.b.noise:>{w}.2f}  {noise_ratio:>8}  {'; '.join(notes)}"
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


def findings(cmp: Comparison, changes: Changes, rows: list[int], what: str,
             min_factor: float) -> list[str]:
    """Plain-language description of the jumps, trends and unusual files in one ratio."""
    spikes = what == "spike"
    out: list[str] = []
    if not rows:
        return ["  No files had enough usable data to compare."]
    y = np.array([cmp.files[i].spike_ratio_log if spikes else cmp.files[i].noise_ratio_log
                  for i in rows])
    typical = 10 ** float(np.median(y))
    a, b = cmp.channel_a, cmp.channel_b
    if spikes:
        ra, rb = _median_rates(cmp, rows)
        if ra == 0 and rb == 0:
            out.append(f"  In a typical file neither {a} nor {b} had any spikes.")
        else:
            out.append(f"  In a typical file {a} had about {rate(ra)} spikes per hour and "
                       f"{b} about {rate(rb)}.")
            if rb > 0 and ra > 0:
                more, less, f = (a, b, typical) if typical >= 1 else (b, a, 1 / typical)
                out.append(f"  So {more} had about {factor(f)} times as many spikes as {less}"
                           f" (spike ratio {factor(typical)}).")
        for name, ch in ((a, "a"), (b, "b")):
            if all(getattr(cmp.files[i], ch).spikes == 0 for i in rows):
                out.append(f"  {name} had no spikes at all in any of the {len(rows)} files.")
    else:
        na, nb = _median_noise(cmp, rows)
        out.append(f"  In a typical file the noise was {na:.2f} counts on {a} and {nb:.2f} on"
                   f" {b}: a noise ratio of {typical:.2f}.")

    if changes.too_few:
        out.append(f"  Too few files ({len(rows)}) to look for jumps or trends; "
                   f"at least {cmp.options.spike_changes.min_trend_files} are needed.")
        return out

    if len(changes.groups) > 1:
        spans = []
        for g0, g1 in changes.groups:
            first, last = cmp.files[rows[g0]].start, cmp.files[rows[g1 - 1]].start
            spans.append(f"{first:%Y-%m-%d} to {last:%Y-%m-%d} ({g1 - g0} files)")
        out.append(f"  The files fall in {len(changes.groups)} separate stretches of time: "
                   + "; ".join(spans) + ".")

    w = cmp.options.spike_changes.long_window
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
            out.append(f"    How sure: {how_sure(max(j.p_value, back.p_value))}.")
            skip_next = True
            continue
        out.append("")
        out.append(f"  JUMP between file {prev + 1} and {_describe_file(cmp, here)}:")
        out.append(f"    The {what} ratio went from about {factor(10 ** j.before)} to about "
                   f"{factor(10 ** j.after)}: {times_word(j.factor)}, and stayed there.")
        out.append(f"    {_channel_words(cmp, before_rows, after_rows, spikes)}")
        gap = (cmp.files[here].start - cmp.files[prev].start).total_seconds() / 86400
        if any(g[0] == j.index for g in changes.groups):
            out.append(f"    It happened across a gap of {duration(gap)} with no files.")
        out.append(f"    How sure: {how_sure(j.p_value)}.")

    for t in changes.trends:
        first, last = rows[t.first], rows[t.last]
        direction = "rose" if t.factor > 1 else "fell"
        out.append("")
        out.append(f"  TREND, files {first + 1} to {last + 1} ({duration(t.days)}):")
        out.append(f"    The {what} ratio {direction} steadily from about {factor(10 ** t.start)}"
                   f" to about {factor(10 ** t.end)} ({times_word(t.factor)} by the end).")
        k = min(5, (t.last - t.first + 1) // 2)
        out.append("    " + _channel_words(cmp, [rows[i] for i in range(t.first, t.first + k)],
                                           [rows[i] for i in range(t.last - k + 1, t.last + 1)],
                                           spikes))
        out.append(f"    How sure: {how_sure(t.p_value)}.")

    for u in changes.unusual:
        i = rows[u.index]
        where = {"both": "the files on either side of it",
                 "before": "the files before it (it is the most recent file)",
                 "after": "the files after it (it is the first file)"}[u.side]
        out.append("")
        out.append(f"  UNUSUAL: {_describe_file(cmp, i)} has a {what} ratio "
                   f"{times_word(u.factor)} than {where}.")

    if not (changes.jumps or changes.trends or changes.unusual):
        out.append(f"  No jumps, trends or unusual files in the {what} ratio.")
    return out


def mode_line(cmp: Comparison) -> str:
    o = cmp.options
    if o.spike_a.fixed_threshold_counts is not None:
        rule = (f"a spike is a jump bigger than {o.spike_a.fixed_threshold_counts:g} counts"
                + (f" on A and {o.spike_b.fixed_threshold_counts:g} on B"
                   if o.spike_b.fixed_threshold_counts != o.spike_a.fixed_threshold_counts
                   else ""))
    else:
        rule = (f"a spike is a jump bigger than {o.spike_a.threshold_sigma:g} times the"
                f" channel's noise (and at least {o.spike_a.min_threshold_counts:g} counts)")
    kind = {"bench": "bench test with test probes", "live": "real probes"}[o.mode]
    custom = "" if MODES[o.mode] in (o.spike_a, o.spike_b) else ", with your own settings"
    return f"{o.mode} ({kind}{custom}): {rule}"


def bench_hints(cmp: Comparison) -> list[str]:
    """Extra plain-language warnings that only make sense for a bench test."""
    rows = cmp.used
    if cmp.options.mode != "bench" or not rows:
        return []
    out = []
    ra, rb = _median_rates(cmp, rows)
    for name, r in ((cmp.channel_a, ra), (cmp.channel_b, rb)):
        # Healthy channels on MR433/MR435/MR533 showed 0 to 3 spikes per hour
        # (the most on a patio, in air); a faulty T1 showed 600 to 3000.
        if r >= BENCH_BUSY_PER_HOUR:
            out.append(f"  {name} had about {rate(r)} spikes per hour in a typical file. With test"
                       " probes on a quiet bench a healthy channel has none, or a few from"
                       " bumps. First rule out bumps and handling during the test; if it is"
                       " not that, the channel, its connector or the test probe may need"
                       " attention.")
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
    if cmp.options.mode != "bench":
        return []
    rows = [i for i, f in enumerate(cmp.files) if not f.error and f.usable_s > 0]
    if not rows:
        return []
    out = ["", "Rockland Bench Test Review Checklist (V3): items about these two channels",
           "  ('typically' = 99.9% of the samples, the same as rsi-tpw bench)"]
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
        text = (f"    {name}: within +-{min(ranges):,.0f} to +-{max(ranges):,.0f} counts")
        if limit.offset is not None:
            text += f", offset {min(means):,.0f} to {max(means):,.0f} counts"
        out.append(text + ".")
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
    if cmp.options.min_pressure_dbar is not None:
        lines.append(f"  Using only data deeper than {cmp.options.min_pressure_dbar:g} dbar.")
    if cmp.options.skip_start_s:
        lines.append(f"  Skipping the first {cmp.options.skip_start_s:g} s of every file.")
    inst = cmp.instruments
    lines.append(f"  Instrument: {', '.join(inst) if inst else 'unknown'}")
    if len(inst) > 1:
        lines.append("  WARNING: these files come from more than one instrument. Jumps and"
                     " trends may only reflect switching between them.")
    span = (f"{good[0].start.strftime(TIME)} to {good[-1].start.strftime(TIME)} UTC"
            if good else "no readable files")
    unread = len(cmp.files) - len(good)
    lines.append(f"  Files:      {len(cmp.files)} P-files, {span}; {len(cmp.used)} analysed"
                 + (f", {unread} could not be read" if unread else ""))
    lines.append("")
    if good:
        lines += table(cmp)
        lines.append("")
        lines.append("  A only / B only = only that channel had spikes; none = neither did."
                     "  Hours = usable data.")
    else:
        lines += table(cmp)
    lines += ["", "What the numbers say", "====================", "", "Spikes"]
    lines += findings(cmp, cmp.spikes, cmp.used, "spike", cmp.options.spike_changes.min_factor)
    lines += ["", "Noise"]
    lines += findings(cmp, cmp.noise, cmp.noise_used, "noise",
                      cmp.options.noise_changes.min_factor)
    hints = bench_hints(cmp)
    if hints:
        lines += ["", "Bench test", *hints]
    lines += checklist_section(cmp)
    lines += ["", "For a plain-language guide to every number here, run:  rockland-noise explain"]
    return "\n".join(lines)


# --- CSV -------------------------------------------------------------------------


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
    used = set(cmp.used)

    def channel_cols(prefix: str, ch) -> dict:
        if ch is None:
            return {}
        return {
            f"{prefix}_spikes": ch.spikes,
            f"{prefix}_spikes_per_hour": round(ch.rate_per_hour, 3),
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
            f"{prefix}_bench_checklist": _checklist_word(ch),
        }

    rows = []
    for i, r in enumerate(cmp.files):
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
        }
        row |= channel_cols("A", r.a)
        row |= channel_cols("B", r.b)
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
