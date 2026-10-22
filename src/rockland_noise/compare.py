"""Run the two-channel comparison over a set of P-files.

For each file: read the two channels, measure noise and count spikes in each,
and remember the answer in a small cache so a second run (or a run that was
interrupted) does not have to read the file again. Then put the files in time
order and look for jumps, trends and unusual files in the two ratios.
"""

import functools
import glob
import hashlib
import json
import math
import os
import re
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
from scipy.stats import poisson

from .changes import Changes, ChangeSettings, find_changes
from .health import Problem, problem
from .reader import ChannelNotFoundError, Evidence, NoPressureError, PFileReadError, read_channels
from .setup import Setup, classify, run_mode
from .spikes import (
    SHARED_WINDOW_S,
    ChannelResult,
    SpikeSettings,
    analyze_channel_times,
    shared_spikes,
)

# For the jump and trend search, one spike per hour is added to both channels'
# rates, so the ratio stays finite when a channel has none (both having none
# gives a ratio of 1). Adding a rate, not a count, keeps a short file and a
# long file with the same spike rates at the same ratio. It only matters when a
# channel has a few spikes per hour or fewer. (AEM1-G clipped rates at 1/h.)
EXTRA_SPIKES_PER_HOUR = 1.0
WILDCARDS = set("*?[")
CACHE_ENTRY = re.compile(r"[0-9a-f]{64}\.(json|tmp)")  # the only files the cache writes


def _version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "unknown"


@functools.cache
def code_fingerprint() -> str:
    """A hash of this program's own source files.

    Remembered answers are keyed on it, so any change to the code, released
    or not, means files are analysed afresh rather than answered from an older
    version's results.
    """
    h = hashlib.sha256()
    for f in sorted(Path(__file__).parent.glob("*.py")):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()


@dataclass(slots=True)
class Options:
    """Everything that decides the answer for one file."""

    mode: str  # "auto" (decide from the files), "bench" or "live"
    # The spike settings (channel A, channel B) for each mode that may be used:
    # both bench and live for "auto", so the files need reading only once.
    spike: dict[str, tuple[SpikeSettings, SpikeSettings]]
    skip_start_s: float = 0.0
    min_pressure_dbar: float | None = None
    min_usable_s: float = 60.0  # files with less usable data are listed but not analysed
    spike_changes: ChangeSettings = field(default_factory=ChangeSettings)
    noise_changes: ChangeSettings = field(
        default_factory=lambda: ChangeSettings(min_factor=1.25))


@dataclass(slots=True)
class FileResult:
    path: Path
    start: datetime | None = None
    instrument: str = ""
    file_number: int = -1
    evidence: Evidence | None = None
    by_mode: dict[str, tuple[ChannelResult, ChannelResult]] = field(default_factory=dict)
    a: ChannelResult | None = None  # the results for the mode the run uses
    b: ChannelResult | None = None
    error: str = ""
    notes: list[str] = field(default_factory=list)
    from_cache: bool = False
    # Channels that could not be measured in this file ("a", "b" -> why); see health.py
    problems: dict[str, Problem] = field(default_factory=dict)

    @property
    def setup(self) -> Setup | None:
        return classify(self.evidence) if self.evidence is not None else None

    @property
    def usable_s(self) -> float:
        pair = (self.a, self.b) if self.a is not None else next(iter(self.by_mode.values()),
                                                                 (None, None))
        if pair[0] is None or pair[1] is None:
            return 0.0
        return min(pair[0].usable_s, pair[1].usable_s)

    @property
    def spike_ratio_log(self) -> float:
        """log10 of the spike-rate ratio A/B used for jumps and trends."""
        a, b = self.a, self.b
        if a is None or b is None or a.usable_s <= 0 or b.usable_s <= 0:
            return math.nan
        return math.log10((a.rate_per_hour + EXTRA_SPIKES_PER_HOUR)
                          / (b.rate_per_hour + EXTRA_SPIKES_PER_HOUR))

    @property
    def noise_ratio_log(self) -> float:
        a, b = self.a, self.b
        if a is None or b is None or not (a.noise > 0 and b.noise > 0):
            return math.nan
        return math.log10(a.noise / b.noise)


@dataclass(slots=True)
class Comparison:
    channel_a: str
    channel_b: str
    options: Options
    mode: str  # the mode used: "bench" or "live"
    mode_reason: str  # why, in plain words
    detected: str  # what the files themselves look like: "bench" or "live"
    files: list[FileResult]  # readable files in time order, then unreadable ones
    used: list[int]  # positions in ``files`` that went into the jump/trend search
    spikes: Changes
    noise: Changes
    noise_used: list[int]  # positions with a usable noise ratio (subset of used)

    @property
    def spike_a(self) -> SpikeSettings:
        return self.options.spike[self.mode][0]

    @property
    def spike_b(self) -> SpikeSettings:
        return self.options.spike[self.mode][1]

    @property
    def instruments(self) -> list[str]:
        return sorted({f.instrument for f in self.files if f.instrument})


# --- finding files -------------------------------------------------------------


def find_pfiles(inputs: list[str], recursive: bool = False) -> tuple[list[Path], list[str]]:
    """Turn files, folders and wildcard patterns into a list of P-files.

    Wildcards are expanded here as well as by the shell, so ``data/*.p`` also
    works in the Windows command prompt. Returns the files (each once, in the
    order found) and messages about inputs that matched nothing.
    """
    found: list[Path] = []
    problems: list[str] = []

    def add_folder(folder: Path) -> int:
        pattern = "**/*" if recursive else "*"
        hits = sorted(p for p in folder.glob(pattern) if p.is_file() and p.suffix.lower() == ".p")
        found.extend(hits)
        return len(hits)

    for item in inputs:
        path = Path(item).expanduser()
        if path.is_dir():
            if add_folder(path) == 0:
                deeper = "" if recursive else " (add --recursive to look in sub-folders)"
                problems.append(f"No .p files in folder {item}{deeper}")
        elif path.is_file():
            found.append(path)
        elif WILDCARDS & set(item):
            hits = sorted(Path(h) for h in glob.glob(os.path.expanduser(item), recursive=True))
            files = [h for h in hits if h.is_file() and h.suffix.lower() == ".p"]
            for h in hits:
                if h.is_dir():
                    add_folder(h)
            if not files and not any(h.is_dir() for h in hits):
                problems.append(f"Nothing matches {item}")
            found.extend(files)
        else:
            problems.append(f"Not found: {item}")

    unique: dict[Path, Path] = {}
    for p in found:
        unique.setdefault(p.resolve(), p)
    return list(unique.values()), problems


# --- the cache -----------------------------------------------------------------


def default_cache_dir() -> Path:
    """Where results are remembered between runs, by the usual rules of each system."""
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "rockland-noise" / "Cache"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "rockland-noise"
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "rockland-noise"


class Cache:
    """One small JSON file per (P-file, channel, settings).

    The key includes the P-file's full path, size and modification time, so an
    edited or replaced file is read again; every setting that changes the
    answer, so changing an option never reuses an old answer; and a hash of
    this program's source code plus the numpy and scipy versions, so a new
    version of any of them never does either.
    """

    def __init__(self, directory: Path | None):
        self.directory = directory
        self.warning = ""
        if directory is not None:
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except OSError as ex:
                self.warning = (f"Cannot use the cache folder {directory} ({ex});"
                                " continuing without it.")
                self.directory = None

    @staticmethod
    def key(path: Path, channel: str, settings: dict) -> str:
        st = path.stat()
        ident = {
            "path": str(path.resolve()),
            "size": st.st_size,
            "mtime_ns": st.st_mtime_ns,
            "channel": channel.casefold(),
            "settings": settings,
            "code": code_fingerprint(),
            "numpy": _version("numpy"),
            "scipy": _version("scipy"),
        }
        return hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()

    def clear(self) -> int:
        """Forget every remembered answer; returns how many were removed.

        Only the cache's own files (64 hex digits + .json) are touched, so a
        --cache-dir pointed at a folder with other files in it is safe.
        """
        if self.directory is None:
            return 0
        removed = 0
        for f in self.directory.iterdir():
            if CACHE_ENTRY.fullmatch(f.name):
                try:
                    f.unlink()
                    removed += 1
                except OSError:
                    pass
        return removed

    def get(self, key: str) -> dict | None:
        if self.directory is None:
            return None
        try:
            return json.loads((self.directory / f"{key}.json").read_text())
        except (OSError, ValueError):
            return None

    def put(self, key: str, value: dict) -> None:
        if self.directory is None:
            return
        target = self.directory / f"{key}.json"
        tmp = target.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(value))
            os.replace(tmp, target)  # never leaves a half-written entry behind
        except OSError:
            pass


# --- one file ------------------------------------------------------------------


def _channel_settings(spike: SpikeSettings, partner: str, partner_spike: SpikeSettings,
                      options: Options) -> dict:
    # The partner matters too: how many spikes the two channels share.
    return {**asdict(spike), "skip_start_s": options.skip_start_s,
            "min_pressure_dbar": options.min_pressure_dbar,
            "partner": partner.casefold(), "partner_settings": asdict(partner_spike)}


def analyze_file(path: Path, channel_a: str, channel_b: str, options: Options,
                 cache: Cache) -> FileResult:
    """Spike and noise numbers for both channels of one file. Never raises for a bad file."""
    result = FileResult(path=path)
    modes = list(options.spike)
    names = (channel_a, channel_b)
    try:
        keys = {(mode, slot): Cache.key(path, names[slot], _channel_settings(
                    options.spike[mode][slot], names[1 - slot], options.spike[mode][1 - slot],
                    options))
                for mode in modes for slot in (0, 1)}
    except OSError as ex:
        result.error = f"Cannot open the file: {ex.strerror or ex}"
        return result

    cached = {k: cache.get(key) for k, key in keys.items()}
    if all(c is not None for c in cached.values()):
        try:
            meta = cached[(modes[0], 0)]["file"]
            result.start = datetime.fromisoformat(meta["start"])
            result.instrument = meta["instrument"]
            result.file_number = meta["file_number"]
            result.notes = list(meta["notes"])
            result.evidence = Evidence(**meta["evidence"])
            result.by_mode = {m: (ChannelResult(**cached[(m, 0)]["channel"]),
                                  ChannelResult(**cached[(m, 1)]["channel"])) for m in modes}
            result.from_cache = True
            return result
        except (KeyError, TypeError, ValueError):
            result = FileResult(path=path)  # a damaged cache entry: read the file instead

    try:
        data = read_channels(path, [channel_a, channel_b], skip_start_s=options.skip_start_s,
                             min_pressure_dbar=options.min_pressure_dbar)
    except ChannelNotFoundError as ex:
        result.error = str(ex).splitlines()[0]
        result.notes = [str(ex)]
        return result
    except NoPressureError as ex:
        result.error = str(ex)
        return result
    except PFileReadError as ex:
        result.error = (f"Empty file: {ex}" if ex.empty
                        else f"Could not read this file as a P-file ({ex})")
        return result
    except (OSError, MemoryError) as ex:
        result.error = f"Could not read the file: {ex}"
        return result

    result.start = data.start
    result.instrument = data.instrument
    result.file_number = data.file_number
    result.notes = data.notes
    result.evidence = data.evidence
    meta = {"start": data.start.isoformat(), "instrument": data.instrument,
            "file_number": data.file_number, "notes": data.notes,
            "evidence": asdict(data.evidence)}
    per_slot = []
    for slot, requested in enumerate(names):
        ch = data.channels[data.resolved[requested]]
        results = analyze_channel_times(ch.name, ch.counts, ch.fs, ch.usable,
                                        [options.spike[m][slot] for m in modes], bad=ch.bad,
                                        limits=ch.limits)
        for res, _ in results:
            res.kind, res.bits, res.fast = ch.kind, ch.bits, ch.fast
        per_slot.append(results)
    for i, mode in enumerate(modes):
        (res_a, times_a), (res_b, times_b) = per_slot[0][i], per_slot[1][i]
        res_a.shared = shared_spikes(times_a, times_b, SHARED_WINDOW_S)
        res_b.shared = shared_spikes(times_b, times_a, SHARED_WINDOW_S)
        for slot, res in enumerate((res_a, res_b)):
            cache.put(keys[(mode, slot)], {"file": meta, "channel": asdict(res)})
    result.by_mode = {m: (per_slot[0][i][0], per_slot[1][i][0]) for i, m in enumerate(modes)}
    return result


# --- the whole set -------------------------------------------------------------


def _days(files: list[FileResult], idx: list[int]) -> np.ndarray:
    t0 = files[idx[0]].start if idx else None
    return np.array([(files[i].start - t0).total_seconds() / 86400 for i in idx])


def counting_check(files: list[FileResult], rows: list[int], alpha: float):
    """For the spike ratio: is a file's difference from its neighbours beyond counting chance?

    Spikes are counted, so a short file with a handful of them can have a
    wild ratio by luck. A file is only called unusual if one of its two
    counts is unlikely (below ``alpha``, by the Poisson distribution) at the
    neighbouring files' typical rate for that channel, in the direction of the
    change: A higher or B lower for a higher ratio, and the other way round.
    The rates include the same extra spike per hour as the ratio, so a quiet
    neighbourhood does not make one spike look impossible.
    """
    n = {s: np.array([getattr(files[i], s).spikes for i in rows], dtype=float) for s in "ab"}
    h = {s: np.array([getattr(files[i], s).usable_s / 3600 for i in rows]) for s in "ab"}

    def check(i: int, around: np.ndarray, higher: bool) -> bool:
        lam = {s: (float(np.median(n[s][around] / h[s][around])) + EXTRA_SPIKES_PER_HOUR)
               * h[s][i] for s in "ab"}
        more_a = poisson.sf(n["a"][i] - 1, lam["a"])  # chance of at least this many on A
        fewer_a = poisson.cdf(n["a"][i], lam["a"])  # ...of at most this many
        more_b = poisson.sf(n["b"][i] - 1, lam["b"])
        fewer_b = poisson.cdf(n["b"][i], lam["b"])
        p = min(more_a, fewer_b) if higher else min(fewer_a, more_b)
        return bool(p < alpha)

    return check


def compare(paths: list[Path], channel_a: str, channel_b: str, options: Options,
            cache: Cache, progress: Callable[[int, int, FileResult], None] | None = None,
            ) -> Comparison:
    """Analyse every file, decide bench or live, then look for changes in time order."""
    results = []
    for i, path in enumerate(paths, 1):
        res = analyze_file(path, channel_a, channel_b, options, cache)
        results.append(res)
        if progress is not None:
            progress(i, len(paths), res)

    good = sorted((r for r in results if not r.error),
                  key=lambda r: (r.start, r.path.name))
    bad = [r for r in results if r.error]
    files = good + bad

    detected, why = run_mode([r.setup for r in good])
    if options.mode == "auto":
        mode, reason = detected, why
    else:
        mode = options.mode
        reason = "chosen with --mode"
        if detected != mode:
            reason += f"; but the files look like {detected}: {why}"
    for r in good:
        r.a, r.b = r.by_mode[mode]
        if r.usable_s >= options.min_usable_s:  # a few samples are too few to judge
            setup = r.setup
            r.problems = {slot: p for slot, ch in (("a", r.a), ("b", r.b))
                          if (p := problem(ch, setup)) is not None}

    # A file where either channel could not be measured is left out: its
    # spike and noise ratios would compare a channel with a broken one.
    used = [i for i, r in enumerate(files)
            if not r.error and r.usable_s >= options.min_usable_s and not r.problems
            and math.isfinite(r.spike_ratio_log)]
    noise_used = [i for i in used if math.isfinite(files[i].noise_ratio_log)]
    spikes = find_changes(_days(files, used),
                          np.array([files[i].spike_ratio_log for i in used]),
                          options.spike_changes,
                          counting_check(files, used, options.spike_changes.unusual_alpha))
    noise = find_changes(_days(files, noise_used),
                         np.array([files[i].noise_ratio_log for i in noise_used]),
                         options.noise_changes)

    name_a = next((r.a.name for r in good if r.a), channel_a)
    name_b = next((r.b.name for r in good if r.b), channel_b)
    return Comparison(name_a, name_b, options, mode, reason, detected, files, used,
                      spikes, noise, noise_used)
