"""Rockland's Bench Test Review Checklist (V3): the items about single channels.

The checklist asks, for the time series in raw counts, that each channel be
"typically within +-N counts" of its mean, and for some channels that the
offset (the mean itself) be below a limit. "Typically" is taken as 99.9% of the
samples, the same reading as ``rsi-tpw bench`` (the ODAS quick_bench port in
the separate microstructure-tpw package), so a lone glitch does not fail an
otherwise clean channel and the two programs agree.

Only items about one channel at a time are here. The checklist's spectra,
accelerometer, inclinometer and EM-sensor items need ``rsi-tpw bench``,
Zissou Essentials or the ODAS quick_bench figures.
"""

import re
from dataclasses import dataclass

from .spikes import ChannelResult

TYPICAL_PERCENT = 99.9
MIN_FILE_SECONDS = 60  # "Collect a minimum 60s data file"


@dataclass(frozen=True, slots=True)
class Limit:
    pattern: str  # channel names it applies to (regular expression)
    typical: float  # "typically within +-N counts"
    offset: float | None  # |mean| must be below this, if the checklist says so
    wording: str  # the checklist's own words


LIMITS = (
    Limit(r"T\d_dT\d", 40, 100, "T1_dT1 and T2_dT2 counts are typically within +-40 counts,"
          " and their offsets are less than 100 counts"),
    Limit(r"sh\d", 30, 10, "sh1 and sh2 counts are typically within +-30 counts, and have a mean"
          " of less than 10 counts"),
    Limit(r"P", 2, None, "P counts are typically within +-2 counts"),
    Limit(r"P_dP", 10, None, "P_dP counts are typically within +-10 counts and seemingly random"
          " (no spikes or patterns at regular intervals)"),
    Limit(r"C\d_dC\d", 50, 6000, "C1_dC1 counts are typically within +-50 counts, and its offset"
          " is less than 6000 counts"),
    Limit(r"A[xy]", 500, None, "Ax and Ay counts are typically within +-500 counts"),
    Limit(r"U_EM", 2000, None, "U_EM counts are typically within +-2000 counts"),
    Limit(r"JAC_T", 50, None, "JAC_T counts are typically within +-50 counts"),
    Limit(r"Turbidity", 50, None, "Turbidity counts are typically within +-50 counts"),
    Limit(r"Chlorophyll", 400, None, "Chlorophyll counts are typically within +-400 counts"),
)


def limit_for(channel: str) -> Limit | None:
    for limit in LIMITS:
        if re.fullmatch(limit.pattern, channel, flags=re.IGNORECASE):
            return limit
    return None


def problems(result: ChannelResult, limit: Limit) -> list[str]:
    """What, if anything, is over the checklist's limits in one file."""
    out = []
    if result.typical_range > limit.typical:
        out.append(f"range +-{result.typical_range:,.0f} > +-{limit.typical:g}")
    if limit.offset is not None and abs(result.mean) >= limit.offset:
        out.append(f"offset {result.mean:,.0f} beyond +-{limit.offset:g}")
    return out
