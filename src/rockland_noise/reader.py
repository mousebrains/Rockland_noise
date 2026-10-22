"""Read the two channels being compared out of one Rockland P-file.

Decoding the binary P-file format is done by the ``microstructure-tpw`` package
(imported as ``odas_tpw``). This module asks it for the *raw counts* of the
requested channels, decides which samples may be used, and hides everything
else from the rest of the program.

Why raw counts? A spike is an electrical event. Raw counts are what the
instrument's analog-to-digital converter actually recorded, before any
calibration or deconvolution, so nothing in the processing can create or hide
a spike. Rockland's own bench-test checklist is written in raw counts too.
"""

import difflib
import warnings
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from odas_tpw.rsi.p_file import BAD_BUFFER_SENTINEL, PFile


class PFileReadError(Exception):
    """The file could not be read as a P-file (damaged, empty, or not a P-file)."""


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
    counts: np.ndarray  # raw counts as float64
    fs: float  # samples per second
    usable: np.ndarray  # True where the sample may be used
    kind: str  # Rockland sensor type from the file's setup, e.g. "therm", "shear"
    bad: np.ndarray  # True where the logger itself marked the sample as bad


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
    notes: list[str] = field(default_factory=list)  # reader warnings, for the report


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


def _open(path: Path) -> tuple[PFile, list[str]]:
    """Open a P-file in raw counts, collecting the reader's warnings as notes."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            pf = PFile(path, deconvolve=False)
        except Exception as ex:  # any failure to decode is reported, never fatal
            raise PFileReadError(f"{type(ex).__name__}: {ex}") from ex
    return pf, [str(w.message) for w in caught]


def _instrument(pf: PFile) -> str:
    info = pf.config.get("instrument_info", {}) if isinstance(pf.config, dict) else {}
    model = str(info.get("model", "")).strip() or "unknown model"
    sn = str(info.get("sn", "")).strip() or "?"
    return f"{model} SN {sn}"


def list_channels(path: str | Path) -> list[dict]:
    """Describe every channel in a file: name, sensor type, rate, fast or slow."""
    path = Path(path)
    pf, _ = _open(path)
    rows = []
    for name in sorted(pf.channels_raw, key=str.casefold):
        fast = pf.is_fast(name)
        rows.append(
            {
                "name": name,
                "type": pf.channel_info.get(name, {}).get("type", ""),
                "rate": "fast" if fast else "slow",
                "fs": pf.fs_fast if fast else pf.fs_slow,
            }
        )
    return rows


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
        A requested channel (or ``P`` for the pressure limit) is missing.
    """
    path = Path(path)
    pf, notes = _open(path)
    available = list(pf.channels_raw)

    pressure = None
    if min_pressure_dbar is not None:
        p_name = resolve_channel("P", available, path)
        # Pressure is wanted in dbar here, so use the calibrated channel.
        pressure = np.asarray(pf.channels[p_name], dtype=np.float64)
        t_pressure = np.arange(pressure.size) / pf.fs_slow

    bad_report = (pf.bad_buffer_report or {}).get("confirmed", {})
    channels: dict[str, Channel] = {}
    resolved: dict[str, str] = {}
    for requested in names:
        name = resolve_channel(requested, available, path)
        resolved[requested] = name
        if name in channels:
            continue
        raw = np.asarray(pf.channels_raw[name])
        fs = pf.fs_fast if pf.is_fast(name) else pf.fs_slow
        t = np.arange(raw.size) / fs
        usable = t >= skip_start_s
        if pressure is not None:
            usable &= np.interp(t, t_pressure, pressure) > min_pressure_dbar
        # The logger writes a special value in place of a sample it could not
        # record properly ("bad buffer"). Those are not measurements; a long
        # run of them would otherwise look like a spike.
        bad = raw == BAD_BUFFER_SENTINEL
        for start, length in bad_report.get(name, {}).get("spans", []):
            bad[start : start + length] = True
        usable &= ~bad
        channels[name] = Channel(
            name=name,
            counts=raw.astype(np.float64),
            fs=float(fs),
            usable=usable,
            kind=str(pf.channel_info.get(name, {}).get("type", "")),
            bad=bad,
        )

    start = pf.start_time.astimezone(UTC)
    duration = len(pf.t_fast) / pf.fs_fast if len(pf.t_fast) else len(pf.t_slow) / pf.fs_slow
    return PFileData(
        path=path,
        start=start,
        duration_s=float(duration),
        instrument=_instrument(pf),
        file_number=int(pf.header.get("file_number", -1)),
        channels=channels,
        resolved=resolved,
        notes=notes,
    )
