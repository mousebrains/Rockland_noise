"""Read Rockland Scientific P-files, with no other Rockland software.

A P-file (ODAS format version 6, Rockland Technical Note 051 "Rockland Data
File Anatomy") is:

* record 0: a 128-byte header of 64 16-bit words, then the instrument's
  configuration as text (INI style: ``[root]``, ``[matrix]``, one
  ``[channel]`` section per channel, ``[instrument_info]``, ...);
* data records: each a 128-byte header and a block of 16-bit samples.

The samples are multiplexed. The ``[matrix]`` section is a grid of channel ids,
``n_rows`` rows by ``n_cols`` columns; the instrument samples it row by row,
one row per "scan". A channel whose id fills a whole column is sampled every
scan (a fast channel, about 512 Hz); a channel whose id appears once is
sampled once per matrix cycle (a slow channel, about 64 Hz). An id may also
appear in a few cells (Gnd fills spare cells, 2 to 6 per cycle on the
instruments seen so far); as in read_odas.m, every cell is read, in the order
sampled, and the rate is the fast rate times the cells per cycle over
``n_rows``. Every record holds whole matrix cycles.

This module follows the ODAS MATLAB library (read_odas.m, convert_odas.m) the
way microstructure-tpw's ``odas_tpw.rsi.p_file`` does, and was checked against
it sample for sample (see tests/test_pfile.py); the one difference is an id in
a few cells, where microstructure-tpw keeps only the first. It reads raw
counts for any channel, and converts to physical units only the few channel
types this program needs: pressure ("poly"), thermistors ("therm"), the
inclinometer ("inclxy", "inclt") and the VMP's CT-sensor temperature
("jac_t").

Not supported: the legacy pre-2015 header-version-1 files, which keep their
configuration in a separate setup file.
"""

import re
import struct
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

HEADER_BYTES = 128
# Header word positions, counting from 0 (TN-051 Table 3 counts from 1).
WORD = {
    "file_number": 0, "record_number": 1, "year": 3, "month": 4, "day": 5, "hour": 6,
    "minute": 7, "second": 8, "millisecond": 9, "header_version": 10, "config_size": 11,
    "timezone_min": 14, "buffer_status": 15, "header_size": 17, "record_size": 18,
    "clock_hz": 20, "clock_frac": 21, "fast_cols": 28, "slow_cols": 29, "n_rows": 30,
    "endian": 63,
}
# From format 6.1 the logger writes this value in place of a sample it could
# not record. It can also occur by chance in real data, so only runs of at
# least 4 count (the same rule as microstructure-tpw, which measured that real
# dropouts are 63-64 samples long and chance hits 1-2).
BAD_BUFFER_VALUE = -32753
BAD_BUFFER_MIN_RUN = 4
# Header values outside these ranges are kept, with a note: a damaged header
# word can give a wrong sample rate or date without making the file unreadable.
USUAL_FAST_RATE = (100.0, 2100.0)  # Hz; Rockland instruments sample at 512 or 1024
USUAL_YEARS = (2000, 2100)
READ_CHUNK_BYTES = 8 * 2**20  # records are read this much at a time


class PFileError(Exception):
    """The file is not a P-file this program can read."""


class EmptyPFileError(PFileError):
    """The file holds no data: it is empty, or has only its header and setup."""


@dataclass(slots=True)
class ChannelSetup:
    name: str
    ids: list[int]
    type: str
    params: dict[str, str]  # every key of the channel's [channel] section, lower case


def parse_config(text: str) -> dict:
    """The configuration text as {'matrix': rows, 'channels': [...], section: {...}}.

    Comments start with ';'. Section names are lower-cased with spaces folded
    to '_' (old files write '[cruise info]'). Keys are lower-cased.
    """
    out: dict = {"matrix": [], "channels": [], "root": {}}
    section = "root"
    channel: dict | None = None
    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = line.split(";", 1)[0].strip()
        if not line:
            continue
        m = re.fullmatch(r"\[(.+)\]", line)
        if m:
            section = "_".join(m.group(1).strip().lower().split())
            channel = None
            if section == "channel":
                channel = {}
                out["channels"].append(channel)
            elif section not in out:
                out[section] = {}
            continue
        m = re.fullmatch(r"(.+?)\s*=\s*(.*)", line)
        if not m:
            continue
        key, value = m.group(1).strip().lower(), m.group(2).strip()
        if channel is not None:
            channel[key] = value
        elif section == "matrix":
            if key.startswith("row"):
                try:
                    out["matrix"].append([int(x) for x in value.split()])
                except ValueError as ex:
                    raise PFileError(f"malformed matrix row {key} = {value!r}") from ex
        elif isinstance(out.get(section), dict):
            out[section][key] = value
    return out


def _number(params: dict[str, str], key: str, default: float) -> float:
    """A calibration coefficient: the default if absent, an error if unreadable."""
    raw = params.get(key, "")
    if raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        raise PFileError(f"channel {params.get('name', '?')}: coefficient {key} = {raw!r}"
                         " is not a number") from None


def reference_temperature(params: dict[str, str]) -> float:
    """A thermistor channel's T_0 in deg C: where its resistance equals the reference.

    A Rockland temperature test probe is a fixed resistor at that reference, so
    it reads T_0 (usually 16.15 deg C, but e.g. 7.88 or 9.45 on some probes).
    """
    return _number(params, "t_0", 289.0) - 273.15


def _runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start and length of each run of True values."""
    if not mask.any():
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    starts = np.flatnonzero(edges == 1)
    return starts, np.flatnonzero(edges == -1) - starts


def _adis14(raw: np.ndarray) -> np.ndarray:
    """ADIS16209 inclinometer words: clear the two status bits, 14-bit two's complement."""
    v = raw.astype(np.float64)
    v[v < -(2**14)] += 2**15  # bit 15: new-data flag
    v[v >= 2**14] -= 2**14  # bit 14: error flag
    v[v >= 2**13] -= 2**14
    return v


@dataclass
class PFile:
    """One P-file. Samples are read when asked for, not when the file is opened."""

    path: Path
    header: dict[str, int] = field(init=False)
    config: dict = field(init=False)
    notes: list[str] = field(init=False, default_factory=list)  # oddities, in plain words

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        with open(self.path, "rb") as f:
            head = f.read(HEADER_BYTES)
            if not head:
                raise EmptyPFileError("the file is empty (0 bytes)")
            if len(head) < HEADER_BYTES:
                raise PFileError("the file is too small to be a P-file")
            self.endian = self._endian(head)
            words = struct.unpack(f"{self.endian}64H", head)
            self.header = {name: words[i] for name, i in WORD.items()}
            version = self.header["header_version"]
            if sum(w != 0 for w in words) <= 2:
                raise PFileError("the header is blank (nearly all zeros), so the file was"
                                 " probably never finished, e.g. the power went off")
            if version == 1:
                raise PFileError("this is an old (pre-2015) format-1 P-file, which keeps its"
                                 " setup in a separate file; it is not supported")
            if version >> 8 < 6:
                raise PFileError(f"unknown P-file format version {version >> 8}.{version & 255}")
            f.seek(self.header["header_size"])
            text = f.read(self.header["config_size"]).decode("ascii", errors="replace")
            f.seek(0, 2)
            file_size = f.tell()
        self.config = parse_config(text)
        self._geometry(file_size)
        self._setups = self._channel_setups()
        self._words: dict[int, np.ndarray] = {}  # matrix address -> its samples
        self._checked = False

    # --- header and layout -------------------------------------------------

    @staticmethod
    def _endian(head: bytes) -> str:
        """'>' or '<'. Word 64 says 1 (little) or 2 (big); else the header size decides."""
        if struct.unpack_from(">H", head, 126)[0] == 2:
            return ">"
        if struct.unpack_from("<H", head, 126)[0] == 1:
            return "<"
        if struct.unpack_from(">H", head, 34)[0] == HEADER_BYTES:
            return ">"
        return "<"

    def _geometry(self, file_size: int) -> None:
        h = self.header
        self.n_rows = h["n_rows"]
        self.n_cols = h["fast_cols"] + h["slow_cols"]
        f_clock = h["clock_hz"] + h["clock_frac"] / 1000
        if self.n_rows < 1 or self.n_cols < 1 or f_clock <= 0:
            raise PFileError(f"damaged header (rows {self.n_rows}, columns {self.n_cols},"
                             f" clock {f_clock} Hz)")
        self.fs_fast = f_clock / self.n_cols
        self.fs_slow = self.fs_fast / self.n_rows

        rows = self.config["matrix"]
        if not rows or len({len(r) for r in rows}) != 1:
            raise PFileError("the setup has no usable [matrix] section")
        self.matrix = np.array(rows, dtype=np.int64)
        if self.matrix.shape != (self.n_rows, self.n_cols):
            raise PFileError(f"the setup's [matrix] is {self.matrix.shape[0]}x"
                             f"{self.matrix.shape[1]} but the header says {self.n_rows}x"
                             f"{self.n_cols}; the file cannot be unpacked safely")

        self.first_record = h["header_size"] + h["config_size"]
        self.record_size = h["record_size"]
        self.data_words = (self.record_size - h["header_size"]) // 2
        scans = self.data_words // self.n_cols if self.n_cols else 0
        if self.data_words < self.n_cols or self.data_words % self.n_cols or scans % self.n_rows:
            raise PFileError("damaged header: the record size does not hold whole matrix cycles")
        self.scans_per_record = scans
        self.n_records = (file_size - self.first_record) // self.record_size
        if self.n_records < 1:
            raise EmptyPFileError("the file has its header and setup but no data (the"
                                  " instrument probably stopped right after it started)")
        if (file_size - self.first_record) % self.record_size:
            self.notes.append("the file ends with a partial record, which was ignored")
        lo, hi = USUAL_FAST_RATE
        if not lo <= self.fs_fast <= hi:
            self.notes.append(f"the header gives a sample rate of {self.fs_fast:,.1f} per second,"
                              f" outside the usual {lo:g} to {hi:g}: the header may be damaged")

        tz = h["timezone_min"] - 2**16 if h["timezone_min"] >= 2**15 else h["timezone_min"]
        try:
            stamp = datetime(h["year"], h["month"], h["day"], h["hour"], h["minute"],
                             h["second"], h["millisecond"] * 1000,
                             tzinfo=timezone(timedelta(minutes=tz)))
        except ValueError as ex:
            raise PFileError(f"the header has no valid date and time ({ex})") from None
        if not USUAL_YEARS[0] <= h["year"] <= USUAL_YEARS[1]:
            self.notes.append(f"the header's date ({h['year']}-{h['month']:02d}-{h['day']:02d})"
                              " looks wrong: the instrument's clock may not have been set, so"
                              " the time order of the files may be wrong")
        # The header time marks the END of the first record (ODAS odas_p2mat.m).
        root = self.config.get("root", {})
        recsize = root.get("recsize", root.get("recordduration", "1"))
        try:
            recsize = float(recsize)
        except ValueError:
            recsize = 1.0
        self.start_time = stamp - timedelta(seconds=recsize)

    def _channel_setups(self) -> dict[str, ChannelSetup]:
        out = {}
        present = set(self.matrix.flatten().tolist())
        for ch in self.config["channels"]:
            name, kind = ch.get("name", "").strip(), ch.get("type", "").strip().lower()
            if not name or not kind:
                continue
            if "id" in ch:
                id_text = ch["id"]
            elif "id_even" in ch and "id_odd" in ch:
                id_text = f"{ch['id_even']} {ch['id_odd']}"
            else:
                continue
            try:
                ids = [int(x) for x in id_text.replace(",", " ").split()]
            except ValueError as ex:
                raise PFileError(f"channel {name}: malformed id {id_text!r}") from ex
            if ids and len(ids) <= 2 and all(i in present for i in ids):
                out[name] = ChannelSetup(name, ids, kind, ch)
        return out

    # --- what is in the file -------------------------------------------------

    @property
    def channel_names(self) -> list[str]:
        return list(self._setups)

    def setup(self, name: str) -> ChannelSetup:
        return self._setups[name]

    def is_fast(self, name: str) -> bool:
        ids = self._setups[name].ids
        if len(ids) != 1:
            return False
        col = int(np.nonzero(self.matrix == ids[0])[1][0])
        return bool(np.all(self.matrix[:, col] == ids[0]))

    def rate(self, name: str) -> float:
        """Samples per second: the fast rate times the cells per cycle over n_rows (read_odas.m)."""
        cells = int(np.count_nonzero(self.matrix == self._setups[name].ids[0]))
        return self.fs_fast if cells == self.n_rows else self.fs_fast * cells / self.n_rows

    @property
    def duration_s(self) -> float:
        return self.n_records * self.scans_per_record / self.fs_fast

    @property
    def instrument(self) -> str:
        info = self.config.get("instrument_info", {})
        model = info.get("model", "").strip() or "unknown model"
        sn = (info.get("sn") or info.get("serial_num") or "").strip() or "?"
        return f"{model} SN {sn}"

    # --- samples -------------------------------------------------------------

    def load(self, names: list[str]) -> None:
        """Read these channels' samples in one pass through the file.

        Only the requested matrix addresses are kept (16-bit integers), and the
        file is read a few megabytes at a time, so a long file never has to fit
        in memory as a whole. Channels not loaded here are read when asked for.
        """
        wanted = sorted({i for n in names for i in self._setups[n].ids} - set(self._words))
        if not wanted and self._checked:
            return
        cycle = self.n_rows * self.n_cols
        # Address -> its cells within one matrix cycle, row by row (the order
        # they are sampled): a slice when they are evenly spaced (a whole
        # column, or one cell), else the list of cells (e.g. Gnd in spare cells).
        where: dict[int, slice | np.ndarray] = {}
        for ch_id in wanted:
            cells = np.flatnonzero(self.matrix.ravel() == ch_id)
            steps = np.diff(np.append(cells, cells[0] + cycle))
            where[ch_id] = (slice(int(cells[0]), None, int(steps[0]))
                            if np.all(steps == steps[0]) else cells)
        per_record = {i: (self.data_words // w.step if isinstance(w, slice)
                          else self.data_words // cycle * w.size) for i, w in where.items()}
        out = {i: np.empty(self.n_records * per, dtype=np.int16) for i, per in per_record.items()}
        numbers = np.empty(self.n_records, dtype=np.int64)
        status = np.empty(self.n_records, dtype=np.int64)
        dtype = np.dtype([("hdr", f"{self.endian}u2", (self.header["header_size"] // 2,)),
                          ("data", f"{self.endian}i2", (self.data_words,))])
        chunk = max(1, READ_CHUNK_BYTES // self.record_size)
        with open(self.path, "rb") as f:
            f.seek(self.first_record)
            done = 0
            while done < self.n_records:
                records = np.fromfile(f, dtype=dtype, count=min(chunk, self.n_records - done))
                if records.size == 0:
                    raise PFileError("the file became shorter while it was being read")
                k = records.size
                numbers[done:done + k] = records["hdr"][:, WORD["record_number"]]
                status[done:done + k] = records["hdr"][:, WORD["buffer_status"]]
                data = records["data"]
                for ch_id, w in where.items():
                    per = per_record[ch_id]
                    out[ch_id][done * per:(done + k) * per].reshape(k, per)[:] = (
                        data[:, w] if isinstance(w, slice)
                        else data.reshape(-1, cycle)[:, w].reshape(k, per))
                done += k
        self._words.update(out)
        if not self._checked:
            self._check_record_headers(numbers, status)
            self._checked = True

    def _check_record_headers(self, numbers: np.ndarray, status: np.ndarray) -> None:
        # The record number is one 16-bit word, so it wraps after 65,535 (a file
        # of over 18 hours at one record per second); a step back is not a gap.
        steps = np.diff(numbers) % 2**16
        gaps = (steps > 1) & (steps < 2**15)
        if gaps.any():
            self.notes.append(f"{int(steps[gaps].sum() - gaps.sum())} record(s) are missing from"
                              " the file; later times are shifted earlier")
        if self.header["header_version"] >> 8 == 6 and self.header["header_version"] & 255 == 0:
            flagged = int(np.count_nonzero(status))
            if flagged:
                self.notes.append(f"{flagged} record(s) are flagged as bad by the logger")

    def _word_series(self, ch_id: int) -> np.ndarray:
        """One matrix address's samples in time order, as 16-bit integers."""
        if ch_id not in self._words:
            self.load([n for n, s in self._setups.items() if ch_id in s.ids][:1])
        return self._words[ch_id]

    @staticmethod
    def _unsigned(s: ChannelSetup) -> bool:
        return s.params.get("sign", "").strip().lower() == "unsigned" or s.type == "jac_t"

    def bits(self, name: str) -> int:
        """16 for an ordinary channel, 32 for one made of two words (e.g. JAC_C)."""
        return 16 * len(self._setups[name].ids)

    def limits(self, name: str) -> tuple[int, int] | None:
        """The smallest and largest count a 16-bit channel can record; None for 32-bit."""
        s = self._setups[name]
        if len(s.ids) != 1:
            return None
        return (0, 2**16 - 1) if self._unsigned(s) else (-(2**15), 2**15 - 1)

    def raw(self, name: str) -> np.ndarray:
        """Raw counts, as the ODAS MATLAB library reads them (read_odas.m)."""
        s = self._setups[name]
        if len(s.ids) == 1:
            x = self._word_series(s.ids[0]).astype(np.int32)
            if self._unsigned(s):
                x[x < 0] += 2**16
            return x
        # A 32-bit value from two words: the first id listed is the low word.
        low = self._word_series(s.ids[0]).astype(np.float64)
        high = self._word_series(s.ids[1]).astype(np.float64)
        low[low < 0] += 2**16
        high[high < 0] += 2**16
        x = high * 2**16 + low
        unsigned = s.params.get("sign", "").strip().lower() == "unsigned"
        if not unsigned and s.type not in {"sbt", "sbc", "jac_c", "o2_43f"}:
            x[x >= 2**31] -= 2**32
        return x

    def bad(self, name: str) -> np.ndarray:
        """True where the logger marked the sample as not recorded (format 6.1 and later)."""
        s = self._setups[name]
        series = [self._word_series(i) for i in s.ids]
        mask = np.zeros(series[0].size, dtype=bool)
        if self.header["header_version"] < 0x0601:
            return mask
        for words in series:
            starts, lengths = _runs(words == BAD_BUFFER_VALUE)
            for a, n in zip(starts, lengths, strict=True):
                if n >= BAD_BUFFER_MIN_RUN:
                    mask[a:a + n] = True
        return mask

    def physical(self, name: str) -> tuple[np.ndarray, str]:
        """Values in physical units, for the channel types this program uses."""
        s = self._setups[name]
        p, x = s.params, self.raw(name)
        if s.type == "poly":
            coefs = []
            for i in range(10):
                if f"coef{i}" not in p:
                    break
                coefs.append(_number(p, f"coef{i}", 0.0))
            if not coefs:
                return x.astype(np.float64), "counts"
            return np.polyval(coefs[::-1], x), p.get("units", "").strip("[]") or "unknown"
        if s.type == "therm":
            return self._therm(x, p), "deg_C"
        if s.type == "jac_t":  # JFE Advantech CT sensor temperature: a 5th-order polynomial
            coefs = [_number(p, k, 0.0) for k in ("f", "e", "d", "c", "b", "a")]
            return np.polyval(coefs, x.astype(np.float64)), "deg_C"
        if s.type == "inclxy":
            return _number(p, "coef1", 0.025) * _adis14(x) + _number(p, "coef0", 0.0), "deg"
        if s.type == "inclt":
            return _number(p, "coef1", -0.47) * _adis14(x) + _number(p, "coef0", 624.0), "deg_C"
        raise PFileError(f"no conversion to physical units for channel type {s.type!r}")

    @staticmethod
    def _therm(x: np.ndarray, p: dict[str, str]) -> np.ndarray:
        """FP07 thermistor (or a test probe's fixed resistor) in a half bridge."""
        a, b = _number(p, "a", 0.0), _number(p, "b", 1.0)
        adc_fs, adc_bits = _number(p, "adc_fs", 4.096), _number(p, "adc_bits", 16.0)
        gain, e_b, t_0 = _number(p, "g", 6.0), _number(p, "e_b", 0.68), _number(p, "t_0", 289.0)
        beta_1 = _number(p, "beta", _number(p, "beta_1", 3000.0)) if "beta" in p else \
            _number(p, "beta_1", 3000.0)
        z = ((x - a) / b) * (adc_fs / 2**adc_bits) * 2 / (gain * e_b)
        log_r = np.log((1 - np.clip(z, -0.6, 0.6)) / (1 + np.clip(z, -0.6, 0.6)))
        inv_t = 1 / t_0 + log_r / beta_1
        if p.get("beta_2", "") != "":
            inv_t = inv_t + log_r**2 / _number(p, "beta_2", 1e30)
            if p.get("beta_3", "") != "":
                inv_t = inv_t + log_r**3 / _number(p, "beta_3", 1e30)
        return 1 / inv_t - 273.15
