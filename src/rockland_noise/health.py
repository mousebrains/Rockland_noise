"""Tell when a channel's spikes and noise say nothing about it.

A spike count of zero is good news only if the channel could have shown a
spike. In these cases it could not, so the file is left out of the
comparison; otherwise the broken channel looks like the clean one and its
healthy partner gets the blame. The limits come from 12,824 channel-files
of T1_dT1, T2_dT2, sh1 and sh2 (3,206 whole P-files from MicroRiders SN
330, 410, 429, 433, 434, 435, 533 and VMPs SN 132, 142, 409, 412, 428, 465,
479: bench tests, gliders, moorings, a REMUS and VMP casts).

No probe
    The thermistor's slow channel (T1 for T1_dT1) reads beyond any water
    temperature, steadily: the bridge has nothing connected and the
    conversion stops at -16.05 or 54.75 deg C (setup.py calls this "none").
    The fast channel is then either pinned (Hydrographer MR330 T2_dT2 from
    file palau2023_0113: noise 0.26 to 0.67 counts, no spikes in most files
    and bursts of glitches in a few) or floating
    (MR330 T1_dT1 in the last 27 files: noise 14 to 290 counts and up to
    3,366 spikes per hour). Either way it says nothing about a probe. Of the
    6,412 thermistor channel-files, 159 had no probe by this test; none of
    the others looked dead.

Railed
    The channel sits at the end of its range in at least 2% of its usable
    samples, or its spike threshold is more than half the channel's whole
    range, so only a swing from one end to the other could count. In whole
    files (surface and deck time included), the MicroRiders on gliders and
    moorings reached at most 1.0%. Others sat there far longer: 4 and 14%
    (sh2, T1_dT1) in one file of a MicroRider on a REMUS in a river test,
    and 3 to 19% on VMP shear channels in some files (ARCTERX 2023 Wake, SN
    142, Palau 2024 SN 465). VMP SN 479's shear channels swung from end to
    end short of the converter's limit, where no sample counts as "at its
    limit" (noise about 20,000 counts, thresholds of 160,000).

Flat
    The channel did not change at all (noise 0), or a fast thermistor
    channel moved by less than a count (noise below 1 count). Thermistor
    channels with a probe never went below 2.2 counts; pinned ones without a
    probe were at most 0.7. The noise of a shear channel says too little to
    tell: a healthy, quiet shear channel of VMP SN 132 had a noise of 0.67
    counts and never moved more than 2 counts in a 4-minute bench file.
"""

import math
import re
from dataclasses import dataclass

from .setup import Setup
from .spikes import ChannelResult

RAILED_SHARE = 0.02  # share of the usable samples at the end of the range
DEAD_THERMISTOR_NOISE = 1.0  # counts; below this a fast thermistor channel is not working


@dataclass(frozen=True, slots=True)
class Problem:
    kind: str  # "no probe", "railed" or "flat"
    detail: str  # the evidence in a few words, with numbers


def slow_thermistor(name: str) -> str | None:
    """'T1' for T1_dT1 or T1, 'T2' for T2_dT2 or T2; None for anything else."""
    m = re.fullmatch(r"T([12])(_dT\1)?", name, flags=re.IGNORECASE)
    return f"T{m.group(1)}" if m else None


def problem(ch: ChannelResult | None, setup: Setup | None) -> Problem | None:
    """Why this channel cannot be measured in this file, or None if it can."""
    if ch is None or ch.usable_s <= 0:
        return None
    slow = slow_thermistor(ch.name)
    if slow is not None and setup is not None and slow in setup.no_probe:
        return Problem("no probe", f"{slow} reads {setup.no_probe[slow]:.2f} C")
    samples = ch.usable_s * ch.fs
    share = ch.saturated / samples if samples > 0 else 0.0
    if share >= RAILED_SHARE:
        return Problem("railed", f"{100 * share:.0f}% of its samples at the end of its range")
    if ch.bits == 16 and math.isfinite(ch.threshold) and ch.threshold > 2**15:
        return Problem("railed", f"threshold {ch.threshold:,.0f} counts, over half its range")
    if ch.noise == 0:
        return Problem("flat", "it did not change at all")
    if ch.kind == "therm" and ch.fast and ch.noise < DEAD_THERMISTOR_NOISE:
        return Problem("flat", f"noise {ch.noise:.2f} counts")
    return None
