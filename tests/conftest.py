"""Shared test helpers: make small P-files whose contents we control.

``tests/data/MR_SL435_template.p`` is a real 20-second MicroRider file
(MR1000RDL-EM SN 435, from the microstructure-tpw test data). New files keep
its header and configuration, so the reader sees a genuine instrument setup,
but their samples are replaced: chosen channels get synthetic counts, the rest
repeat the template's own records.
"""

import struct
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from odas_tpw.rsi.p_file import PFile

TEMPLATE = Path(__file__).parent / "data" / "MR_SL435_template.p"
FS_FAST = 512.0327777777778  # the template's fast sample rate
SCANS_PER_RECORD = 512  # one record is one second


class PFileMaker:
    def __init__(self):
        self.pf = PFile(TEMPLATE, deconvolve=False)
        self.raw = TEMPLATE.read_bytes()
        h = self.pf.header
        self.first_size = h["header_size"] + h["config_size"]
        self.record_size = h["record_size"]
        self.n_cols, self.n_rows = self.pf.n_cols, self.pf.n_rows
        self.ids = {c["name"]: int(c["id"]) for c in self.pf.config["channels"]
                    if "id" in c and str(c["id"]).strip().isdigit()}

    def fast_samples(self, seconds: int) -> int:
        return seconds * SCANS_PER_RECORD

    def write(self, path: Path, *, start: datetime, seconds: int,
              channels: dict[str, np.ndarray] | None = None) -> Path:
        """Write a P-file of whole seconds; ``channels`` maps name -> raw counts."""
        first = bytearray(self.raw[:self.first_size])
        words = list(struct.unpack(">64H", first[:128]))
        # The header time marks the end of the first record (ODAS convention).
        t = start + timedelta(seconds=1)
        words[3:10] = [t.year, t.month, t.day, t.hour, t.minute, t.second,
                       t.microsecond // 1000]
        first[:128] = struct.pack(">64H", *words)

        dtype = np.dtype([("hdr", ">u2", (64,)),
                          ("data", ">i2", ((self.record_size - 128) // 2,))])
        template = np.frombuffer(self.raw, dtype=dtype, offset=self.first_size)
        records = np.empty(seconds, dtype=dtype)
        records["hdr"] = template["hdr"][np.arange(seconds) % len(template)]
        records["hdr"][:, 1] = np.arange(1, seconds + 1)  # record numbers, no gaps
        # A separate, contiguous block: reshaping records["data"] would give a
        # copy (the record headers sit between the data blocks), not a view.
        data = template["data"][np.arange(seconds) % len(template)].reshape(-1, self.n_cols)

        for name, values in (channels or {}).items():
            cid = self.ids[name]
            rows, cols = np.nonzero(self.pf.matrix == cid)
            col = int(cols[0])
            values = np.asarray(np.round(values), dtype=np.int64).clip(-32768, 32767)
            if np.all(self.pf.matrix[:, col] == cid):  # fast: the whole column
                data[:, col] = values
            else:  # slow: one cell, every n_rows scans
                data[int(rows[0])::self.n_rows, col] = values
        records["data"] = data.reshape(seconds, -1)
        path.write_bytes(bytes(first) + records.tobytes())
        return path


@pytest.fixture(scope="session")
def maker() -> PFileMaker:
    return PFileMaker()


def noise(n: int, sigma: float, offset: float = 0.0, seed: int = 0) -> np.ndarray:
    return offset + np.random.default_rng(seed).normal(0, sigma, n)


def add_bursts(x: np.ndarray, where: list[int], size: float = 80.0) -> np.ndarray:
    """Add short bipolar bursts (up first, then down) like the MR433 T1 fault."""
    x = x.copy()
    shape = size * np.array([1.0, 0.8, -0.6, -0.4, 0.3, 0.2, -0.1])
    for i in where:
        x[i:i + shape.size] += shape[:max(0, min(shape.size, x.size - i))]
    return x


T0 = datetime(2026, 1, 1, tzinfo=UTC)
