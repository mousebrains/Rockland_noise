"""Read the two channels being compared out of one Rockland P-file.

The P-file itself is decoded by ``pfile.py``. This module picks out the raw
counts of the requested channels, decides which samples may be used, and
gathers the few numbers that tell a bench test from data taken with real
probes (see ``setup.py``).

Why raw counts? A spike is an electrical event. Raw counts are what the
instrument's analog-to-digital converter actually recorded, before any
calibration or deconvolution, so nothing in the processing can create or hide
a spike. Rockland's own bench-test checklist is written in raw counts too.
"""

import contextlib
import difflib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from .pfile import EmptyPFileError, PFile, PFileError, reference_temperature


class PFileReadError(Exception):
    """The file could not be read as a P-file (damaged, empty, or not a P-file)."""

    def __init__(self, message: str, empty: bool = False):
        super().__init__(message)
        self.empty = empty  # the file holds no data at all


class NoPressureError(ValueError):
    """--min-pressure was given, but the file has no usable pressure channel."""


class ChannelNotFoundError(ValueError):
    """A requested channel name is not in the file."""

    def __init__(self, requested: str, available: list[str], path: Path):
        self.requested = requested
        self.available = sorted(available, key=str.casefold)
        self.suggestions = difflib.get_close_matches(requested, self.available, n=3, cutoff=0.5)
        hint = f" Did you mean {' or '.join(self.suggestions)}?" if self.suggestions else ""
        super().__init__(
            f"There is no channel called '{requested}' in {path.name}.{hint}\n"
            f"Channels in this file: {', '.join(self.available)}"
        )


@dataclass(slots=True)
class Channel:
    """One channel's samples and which of them may be used."""

    name: str  # exactly as spelled in the file
    counts: np.ndarray  # raw counts (integers, or float64 for a 32-bit channel)
    fs: float  # samples per second
    usable: np.ndarray  # True where the sample may be used
    kind: str  # Rockland sensor type from the file's setup, e.g. "therm", "shear"
    bad: np.ndarray  # True where the logger itself marked the sample as bad
    bits: int = 16  # 32 for a channel made of two words (e.g. JAC_C)
    limits: tuple[int, int] | None = None  # smallest and largest possible count (16-bit)
    fast: bool = True  # sampled every scan (False: once per matrix cycle)


@dataclass(slots=True)
class Evidence:
    """Numbers that show whether the instrument had test probes and was deployed.

    Spreads are interquartile ranges (the middle half of the samples), so
    handling at the start and end of a file does not count. NaN = no such
    channel in the file.
    """

    t1: float = float("nan")  # median thermistor temperature, deg C
    t2: float = float("nan")
    t1_ref: float = float("nan")  # the calibration's reference temperature T_0, deg C
    t2_ref: float = float("nan")
    t1_spread: float = float("nan")  # interquartile range, deg C
    t2_spread: float = float("nan")
    tilt_spread: float = float("nan")  # larger of the Incl_X and Incl_Y ranges, degrees
    shake: float = float("nan")  # larger of Ax and Ay: median over 1-s blocks of the sd, counts
    pressure: float = float("nan")  # median, dbar
    pressure_spread: float = float("nan")  # interquartile range, dbar
    pressure_high: float = float("nan")  # 95th percentile, dbar
    # VMPs only: the CT sensor's temperature (JAC_T), the water temperature.
    water_spread: float = float("nan")  # 5th to 95th percentile, deg C
    t1_follows: float = float("nan")  # rank correlation of T1 with JAC_T (1-s medians)
    t2_follows: float = float("nan")


@dataclass(slots=True)
class PFileData:
    """What this program needs from one P-file."""

    path: Path
    start: datetime  # start of the data, UTC
    duration_s: float
    instrument: str  # e.g. "MR1000RDL-EM SN 433"
    file_number: int
    channels: dict[str, Channel]  # keyed by the file's spelling of the name
    resolved: dict[str, str]  # requested name -> the file's spelling
    evidence: Evidence
    notes: list[str] = field(default_factory=list)  # oddities found while reading


def resolve_channel(requested: str, available: list[str], path: Path) -> str:
    """Return the file's spelling of ``requested``, ignoring upper/lower case.

    Raises ChannelNotFoundError, listing close matches, when there is none.
    """
    if requested in available:
        return requested
    folded = {name.casefold(): name for name in available}
    if requested.casefold() in folded:
        return folded[requested.casefold()]
    raise ChannelNotFoundError(requested, available, path)


def _open(path: Path) -> PFile:
    try:
        return PFile(path)
    except EmptyPFileError as ex:
        raise PFileReadError(str(ex), empty=True) from ex
    except PFileError as ex:
        raise PFileReadError(str(ex)) from ex


def list_channels(path: str | Path) -> list[dict]:
    """Describe every channel in a file: name, sensor type, rate, fast or slow."""
    pf = _open(Path(path))
    return [{"name": n, "type": pf.setup(n).type, "rate": "fast" if pf.is_fast(n) else "slow",
             "fs": pf.rate(n)} for n in sorted(pf.channel_names, key=str.casefold)]


def _usable(n: int, fs: float, skip_start_s: float, pressure: np.ndarray | None,
            t_pressure: np.ndarray | None, min_pressure_dbar: float | None) -> np.ndarray:
    """True for samples at least skip_start_s into the file and, if asked, deeper than
    min_pressure_dbar. Worked out a block at a time, so a long file needs little memory."""
    usable = np.ones(n, dtype=bool)
    # The first sample whose time i / fs is at least skip_start_s.
    first = max(0, int(skip_start_s * fs) - 2)
    while first < n and first / fs < skip_start_s:
        first += 1
    usable[:first] = False
    if pressure is not None:
        block = 2**20
        for a in range(0, n, block):
            t = np.arange(a, min(n, a + block)) / fs
            usable[a:a + t.size] &= np.interp(t, t_pressure, pressure) > min_pressure_dbar
    return usable


def _iqr(x: np.ndarray) -> float:
    q25, q75 = np.percentile(x, [25, 75])
    return float(q75 - q25)


def _seconds(x: np.ndarray, fs: float) -> np.ndarray:
    """Median of each whole second."""
    n = max(1, int(fs))
    return np.median(x[: x.size // n * n].reshape(-1, n), axis=1)


# Channels the bench/live evidence uses, if the file has them.
EVIDENCE_CHANNELS = ("T1", "T2", "Incl_X", "Incl_Y", "Ax", "Ay", "JAC_T", "P")


def _shake(x: np.ndarray, fs: float) -> float:
    """Median over 1-s blocks of the standard deviation, a block of seconds at a time."""
    n = max(1, int(fs))
    blocks = x.size // n
    if blocks == 0:
        return float("nan")
    step = 600  # seconds per pass, so a long file needs little memory
    sds = [x[i * n:min(blocks, i + step) * n].reshape(-1, n).std(axis=1)
           for i in range(0, blocks, step)]
    return float(np.median(np.concatenate(sds)))


def _evidence(pf: PFile) -> tuple[Evidence, list[str]]:
    """The numbers setup.py uses to recognise a bench test, and any problems.

    A missing channel gives NaN. So does one that cannot be converted (for
    example a calibration coefficient written as "0,0687"): that is noted, and
    the file is still analysed.
    """
    ev = Evidence()
    problems: list[str] = []
    names = {n.casefold(): n for n in pf.channel_names}

    def physical(name: str, kinds: tuple[str, ...]) -> np.ndarray | None:
        real = names.get(name.casefold())
        if real is None or pf.setup(real).type not in kinds:
            return None
        try:
            return pf.physical(real)[0]
        except PFileError as ex:
            problems.append(f"{real} could not be converted ({ex}), so it was not used to"
                            " tell a bench test from live data")
            return None

    temps = {}
    for key in ("t1", "t2"):
        t = physical(key, ("therm",))
        if t is not None:
            temps[key] = t
            setattr(ev, key, float(np.median(t)))
            setattr(ev, f"{key}_spread", _iqr(t))
            with contextlib.suppress(PFileError):  # setup.py then assumes 16.15 deg C
                setattr(ev, f"{key}_ref", reference_temperature(pf.setup(names[key]).params))
    tilts = [_iqr(v) for v in (physical("Incl_X", ("inclxy",)), physical("Incl_Y", ("inclxy",)))
             if v is not None]
    if tilts:
        ev.tilt_spread = max(tilts)
    shakes = [_shake(pf.raw(names[k]), pf.rate(names[k])) for k in ("ax", "ay") if k in names]
    shakes = [x for x in shakes if np.isfinite(x)]
    if shakes:
        ev.shake = max(shakes)
    water = physical("JAC_T", ("jac_t",))
    if water is not None:
        ev.water_spread = float(np.percentile(water, 95) - np.percentile(water, 5))
        w = _seconds(water, pf.rate(names["jac_t"]))
        for key, t in temps.items():
            ts = _seconds(t, pf.rate(names[key]))
            m = min(ts.size, w.size)
            if m >= 10 and np.ptp(ts[:m]) > 0 and np.ptp(w[:m]) > 0:
                setattr(ev, f"{key}_follows", float(spearmanr(ts[:m], w[:m]).statistic))
    p = physical("P", ("poly",))
    if p is not None:
        ev.pressure = float(np.median(p))
        ev.pressure_spread = _iqr(p)
        ev.pressure_high = float(np.percentile(p, 95))
    return ev, problems


def read_channels(
    path: str | Path,
    names: list[str],
    *,
    skip_start_s: float = 0.0,
    min_pressure_dbar: float | None = None,
) -> PFileData:
    """Read the named channels from one P-file.

    Parameters
    ----------
    path
        The P-file.
    names
        Channel names to read, matched without regard to upper/lower case.
    skip_start_s
        Ignore this many seconds at the start of the file (for example while
        the instrument settles after it is switched on or put in the water).
    min_pressure_dbar
        If given, only use samples taken while the pressure was above this
        value (dbar, about metres of water). Use it for deployed data so time
        at the surface or out of the water is not counted.

    Raises
    ------
    PFileReadError
        The file could not be decoded.
    ChannelNotFoundError
        A requested channel is missing.
    NoPressureError
        ``min_pressure_dbar`` was given but the pressure cannot be read.
    """
    path = Path(path)
    pf = _open(path)
    available = pf.channel_names
    requested_names = list(names)
    names = [resolve_channel(r, available, path) for r in requested_names]
    present = {n.casefold(): n for n in available}
    p_name = None
    if min_pressure_dbar is not None:
        p_name = present.get("p")
        if p_name is None:
            raise NoPressureError(f"--min-pressure needs a pressure channel (P), and"
                                  f" {path.name} has none. Leave out --min-pressure for this"
                                  " file (a bench test needs no pressure limit).")
    try:
        # One pass through the file for everything this program needs.
        pf.load(list(dict.fromkeys(names + [present[k.casefold()] for k in EVIDENCE_CHANNELS
                                            if k.casefold() in present])))
        pressure = t_pressure = None
        if p_name is not None:
            try:
                pressure = pf.physical(p_name)[0]
            except PFileError as ex:
                raise NoPressureError(f"--min-pressure needs the pressure, but {ex}.") from None
            t_pressure = np.arange(pressure.size) / pf.rate(p_name)

        channels: dict[str, Channel] = {}
        resolved: dict[str, str] = {}
        for requested, name in zip(requested_names, names, strict=True):
            resolved[requested] = name
            if name in channels:
                continue
            raw = pf.raw(name)
            fs = pf.rate(name)
            usable = _usable(raw.size, fs, skip_start_s, pressure, t_pressure, min_pressure_dbar)
            # The logger writes a special value in place of a sample it could
            # not record ("bad buffer"). Those are not measurements; a run of
            # them would otherwise look like a spike.
            bad = pf.bad(name)
            usable &= ~bad
            channels[name] = Channel(name=name, counts=raw, fs=fs, usable=usable,
                                     kind=pf.setup(name).type, bad=bad, bits=pf.bits(name),
                                     limits=pf.limits(name), fast=pf.is_fast(name))
    except PFileError as ex:
        raise PFileReadError(str(ex)) from ex
    # The bench/live evidence is a separate matter: a problem there (say, a
    # damaged coefficient on a channel nobody asked about) must not stop the
    # two channels from being analysed.
    try:
        evidence, problems = _evidence(pf)
    except (PFileError, ValueError, ArithmeticError) as ex:
        evidence, problems = Evidence(), [f"could not tell whether this is a bench test ({ex})"]

    return PFileData(
        path=path,
        start=pf.start_time.astimezone(UTC),
        duration_s=pf.duration_s,
        instrument=pf.instrument,
        file_number=int(pf.header["file_number"]),
        channels=channels,
        resolved=resolved,
        evidence=evidence,
        notes=list(pf.notes) + problems,
    )
