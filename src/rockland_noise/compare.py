"""Run the two-channel comparison over a set of P-files.

For each file: read the two channels, measure noise and count spikes in each,
and remember the answer in a small cache so a second run (or a run that was
interrupted) does not have to read the file again. Then put the files in time
order and look for jumps, trends and unusual files in the two ratios.
"""

import glob
import hashlib
import json
import math
import os
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from .changes import Changes, ChangeSettings, find_changes
from .reader import ChannelNotFoundError, PFileReadError, read_channels
from .spikes import ChannelResult, SpikeSettings, analyze_channel

# For the jump and trend search, one spike per hour is added to both channels'
# rates, so the ratio stays finite when a channel has none (both having none
# gives a ratio of 1). Adding a rate, not a count, keeps a short file and a
# long file with the same spike rates at the same ratio. It only matters when a
# channel has a few spikes per hour or fewer. (AEM1-G clipped rates at 1/h.)
EXTRA_SPIKES_PER_HOUR = 1.0
WILDCARDS = set("*?[")
CACHE_FORMAT = 2  # bump when ChannelResult or its meaning changes, so old answers are not reused


def _version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "unknown"


@dataclass(slots=True)
class Options:
    """Everything that decides the answer for one file."""

    mode: str
    spike_a: SpikeSettings
    spike_b: SpikeSettings
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
    a: ChannelResult | None = None
    b: ChannelResult | None = None
    error: str = ""
    notes: list[str] = field(default_factory=list)
    from_cache: bool = False

    @property
    def usable_s(self) -> float:
        if self.a is None or self.b is None:
            return 0.0
        return min(self.a.usable_s, self.b.usable_s)

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
    files: list[FileResult]  # readable files in time order, then unreadable ones
    used: list[int]  # positions in ``files`` that went into the jump/trend search
    spikes: Changes
    noise: Changes
    noise_used: list[int]  # positions with a usable noise ratio (subset of used)

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
    edited or replaced file is read again; and every setting that changes the
    answer, plus the program versions, so changing an option never reuses an
    old answer.
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
            "format": CACHE_FORMAT,
            "rockland-noise": _version("rockland-noise"),
            "microstructure-tpw": _version("microstructure-tpw"),
        }
        return hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()

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


def _channel_settings(spike: SpikeSettings, options: Options) -> dict:
    return {**asdict(spike), "skip_start_s": options.skip_start_s,
            "min_pressure_dbar": options.min_pressure_dbar}


def analyze_file(path: Path, channel_a: str, channel_b: str, options: Options,
                 cache: Cache) -> FileResult:
    """Spike and noise numbers for both channels of one file. Never raises for a bad file."""
    result = FileResult(path=path)
    try:
        keys = [Cache.key(path, channel_a, _channel_settings(options.spike_a, options)),
                Cache.key(path, channel_b, _channel_settings(options.spike_b, options))]
    except OSError as ex:
        result.error = f"Cannot open the file: {ex.strerror or ex}"
        return result

    cached = [cache.get(k) for k in keys]
    if all(c is not None for c in cached):
        try:
            meta = cached[0]["file"]
            result.start = datetime.fromisoformat(meta["start"])
            result.instrument = meta["instrument"]
            result.file_number = meta["file_number"]
            result.notes = list(meta["notes"])
            result.a = ChannelResult(**cached[0]["channel"])
            result.b = ChannelResult(**cached[1]["channel"])
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
    except PFileReadError as ex:
        result.error = f"Could not read this file as a P-file ({ex})"
        return result
    except (OSError, MemoryError) as ex:
        result.error = f"Could not read the file: {ex}"
        return result

    result.start = data.start
    result.instrument = data.instrument
    result.file_number = data.file_number
    result.notes = data.notes
    meta = {"start": data.start.isoformat(), "instrument": data.instrument,
            "file_number": data.file_number, "notes": data.notes}
    ch_a = data.channels[data.resolved[channel_a]]
    ch_b = data.channels[data.resolved[channel_b]]
    for slot, ch, spike, key in (("a", ch_a, options.spike_a, keys[0]),
                                 ("b", ch_b, options.spike_b, keys[1])):
        res = analyze_channel(ch.name, ch.counts, ch.fs, ch.usable, spike, bad=ch.bad)
        res.kind = ch.kind
        setattr(result, slot, res)
        cache.put(key, {"file": meta, "channel": asdict(res)})
    return result


# --- the whole set -------------------------------------------------------------


def _days(files: list[FileResult], idx: list[int]) -> np.ndarray:
    t0 = files[idx[0]].start if idx else None
    return np.array([(files[i].start - t0).total_seconds() / 86400 for i in idx])


def compare(paths: list[Path], channel_a: str, channel_b: str, options: Options,
            cache: Cache, progress: Callable[[int, int, FileResult], None] | None = None,
            ) -> Comparison:
    """Analyse every file, then look for changes in time order."""
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

    used = [i for i, r in enumerate(files)
            if not r.error and r.usable_s >= options.min_usable_s
            and math.isfinite(r.spike_ratio_log)]
    noise_used = [i for i in used if math.isfinite(files[i].noise_ratio_log)]
    spikes = find_changes(_days(files, used),
                          np.array([files[i].spike_ratio_log for i in used]),
                          options.spike_changes)
    noise = find_changes(_days(files, noise_used),
                         np.array([files[i].noise_ratio_log for i in noise_used]),
                         options.noise_changes)

    name_a = next((r.a.name for r in good if r.a), channel_a)
    name_b = next((r.b.name for r in good if r.b), channel_b)
    return Comparison(name_a, name_b, options, files, used, spikes, noise, noise_used)
