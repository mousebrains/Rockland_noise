# rockland-noise

Compare **spikes** and **noise** between two channels of a Rockland Scientific
instrument (MicroRider, VMP, ...) across a set of P-files, list the files in time
order, and point out **jumps**, **trends** and **unusual files**.

It answers questions like:

* "Is T1 spikier than T2, and since when?"
* "Did the shear channels change after the probe swap?"
* "Is this bench test clean, and does it pass Rockland's bench checklist?"

```
rockland-noise compare T1_dT1 T2_dT2 /path/to/pfiles --mode bench
```

You do not need to know statistics or programming to use it. The report explains
itself in plain words, and `rockland-noise explain` describes every number.

---

## 1. Install (once)

You need **uv**, a small program that installs Python tools and the right
version of Python for them. You do not need to install Python yourself.

**Mac or Linux**: open Terminal and paste:

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows**: open PowerShell and paste:

```
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Close the window and open a new one. Then install this tool:

```
uv tool install git+https://github.com/mousebrains/Rockland_noise
```

Check that it works:

```
rockland-noise --help
```

To get a newer version later: `uv tool upgrade rockland-noise`.

## 2. Use it

### Step 1: find the channel names

```
rockland-noise channels /path/to/one_file.p
```

The usual pairs to compare are the fast thermistor channels **`T1_dT1` and
`T2_dT2`**, and the shear channels **`sh1` and `sh2`**. Upper/lower case does
not matter.

### Step 2: compare two channels over many files

Give the two channel names first, then the files. Files can be a folder, a list
of files, or a pattern such as `data/*.p`.

| What you are looking at | Command |
|---|---|
| A **bench test** with Rockland test probes | `rockland-noise compare T1_dT1 T2_dT2 data/ --mode bench` |
| Real probes in a **tank** | `rockland-noise compare T1_dT1 T2_dT2 data/` |
| **Deployed** data (ignore time near the surface) | `rockland-noise compare sh1 sh2 data/ --min-pressure 5` |
| Save a **spreadsheet and a picture** for your supervisor | add `--csv results.csv --plot results.png` |

**Bench or live?** Use `--mode bench` when the instrument has Rockland's **test
probes** on (open-circuit shear, fixed-resistor temperature). The signal is then
only electronic noise, so the tool can be more sensitive: a spike is anything
over 6 times the noise. Use `--mode live` (the default) for **real probes**,
which also sense real things in the water, so the tool asks for 8 times the
noise. If unsure, leave it out: live is the cautious choice.

In bench mode the report also fills in the numbers for the items of Rockland's
**Bench Test Review Checklist (V3)** that concern the two channels: "typically
within ±40 counts", "offset less than 100 counts", and so on.

### Step 3: read the report

A shortened example: a bench test of MicroRider SN 433, whose T1 channel had a
fault.

```
  Channels:   A = T1_dT1   B = T2_dT2
  Mode:       bench (bench test with test probes): a spike is a jump bigger than 6 times the channel's noise (and at least 5 counts)

                                                   ------spikes per hour-------  -------noise (counts)-------
   #  P-file             Start (UTC)        Hours    T1_dT1    T2_dT2       A/B    T1_dT1    T2_dT2       A/B  Notes
   2  RIOT_SL684_0531.p  2026-10-19 04:16    0.70     2,930         0    A only      3.96      3.24      1.22  A outside bench checklist
   3  RIOT_SL684_0532.p  2026-10-19 05:11    4.00       635         0    A only      6.24      3.23      1.93  A outside bench checklist
   ...
Spikes
  In a typical file T1_dT1 had about 1,504 spikes per hour and T2_dT2 about 0.
Bench test
  T1_dT1 had about 1,504 spikes per hour in a typical file. With test probes on a quiet
  bench a healthy channel has none, or a few from bumps. ...
Rockland Bench Test Review Checklist (V3): items about these two channels
  "T1_dT1 and T2_dT2 counts are typically within +-40 counts, and their offsets are less than 100 counts."
    T1_dT1: within +-38 to +-315 counts, offset -17 to 153 counts.
      OUTSIDE the checklist in 11 of 12 files (#1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12).
    T2_dT2: within +-11 to +-11 counts, offset -26 to -20 counts.
      Within the checklist in all 12 files.
```

* **Spikes per hour**: sudden short bursts, counted per hour of data.
* **Noise**: the size of the small, fast wiggle every channel has, in raw counts.
  Healthy MicroRider channels: about 3.2 counts (thermistors) and 1.1 counts (shear).
* **A/B**: the ratio of channel A to channel B. `A only`: only A had spikes.
* A line `==== JUMP in spike ratio: about 20x higher from here ====` in the table
  marks where something changed. The text under **What the numbers say**
  describes each jump, trend and unusual file, which channel changed, and how
  sure the tool is.

Run `rockland-noise explain` for the full plain-language guide.

## 3. If something goes wrong

| You see | What to do |
|---|---|
| `There is no channel called 'T1'` | Run `rockland-noise channels FILE` and copy a name from the list. The tool suggests close matches. |
| `looks like a file, not a channel` | Put the two channel names **before** the files. |
| `No .p files in folder` | Check the folder, or add `--recursive` if the files are in sub-folders. |
| `PROBLEM: Could not read this file` | That file is damaged or not a P-file. The other files are still analysed. |
| `too little usable data, left out` | The file has less than a minute of usable data (change with `--min-minutes`). With `--min-pressure`, the instrument may never have gone that deep. |
| A long run was stopped | Run the same command again. Finished files are remembered and not read again. |
| `rockland-noise: command not found` | Open a new terminal window after installing uv, or run `uv tool update-shell`. |

## 4. How it works (for the curious and for supervisors)

**Spikes and noise** (`spikes.py`), on raw counts, one channel and one file at a time:

1. Subtract a 0.1-s running median. What remains is the noise plus any spikes.
2. **Noise** = root-mean-square of that remainder, leaving out values beyond 5
   times the current estimate (iterated). Unlike the median absolute deviation,
   this does not jump in steps on whole-number counts.
3. A **spike** starts where the remainder exceeds the threshold: 6 × noise
   (bench, at least 5 counts) or 8 × noise (live, at least 10 counts). Starts
   within 0.1 s merge.
4. Usable samples: after `--skip-start`, deeper than `--min-pressure` if given,
   and never the samples the logger marked as bad buffers.

This is the rule used for the MR433 T1 fault in the AEM1-G project. With
`--threshold-counts 23.7 --skip-start 600` it reproduces the AEM1-G counts file
for file (see `tests/test_real_data.py`). The bench threshold rests on
measurement: on healthy MR433, MR435 and MR533 channels, in 4-hour bench and tank
files, the largest excursion was 5.0–5.9 × the noise, as expected for Gaussian
noise, and never 6.

**Jumps, trends, unusual files** (`changes.py`) work on log10 of each ratio,
file by file in time order:

* **Jump**: binary segmentation with the maximum two-sample t statistic,
  p-value by shuffling the files (1999 shuffles, fixed seed). The change must
  be at least `--min-factor` (2× for spikes, 1.25× for noise) over both the
  nearest 5 and the nearest 15 files on each side, and still that big after
  allowing for the local slope (otherwise it is a trend).
* **Trend**: Kendall's tau against time, Theil–Sen slope, within runs of files
  with no jump and no long time gap; the fitted change must also be at least
  `--min-factor`.
* **Neighbouring files are alike**, which overstates the evidence, so every
  test statistic is deflated by the lag-1 autocorrelation of the residuals of
  the full fitted model (effective sample size, as in the modified
  Mann–Kendall test, Yue & Wang 2004), and jumps that no longer pass are
  dropped one at a time.
* **Unusual file**: differs from the median of its 3 neighbours on *both*
  sides, in the same direction, by at least `--min-factor`² and 4× the usual
  file-to-file scatter.
* To compare spike rates when one channel has none, 1 spike per hour is added
  to both rates before the ratio is taken for jump and trend detection.

Measured on synthetic series of 30 files: no false jumps or trends in 200
series of independent noise; 4% false jumps on strongly autocorrelated noise
(lag-1 correlation 0.7); a 3× step in correlated noise (lag-1 correlation 0.5)
found 87% of the time. On the AEM1-G per-file counts it finds the MR433
2025 → 2026 jump (about 270×; T2 had no spikes, so the ratio follows T1's rate),
the onset of MR435's T2 fault at file A685_0158 (2023-06-12 17:17 UTC, T2 about
10× busier, T1 unchanged), and a temporary T2 change in MR433's 2023 deployment.

Results are remembered per file, channel and setting in a cache folder (shown by
`rockland-noise compare --help`), so a second run, or one that was interrupted, is fast.

## 5. For developers

```
git clone git@github.com:mousebrains/Rockland_noise.git
cd Rockland_noise
uv venv --python 3.13 && uv pip install -e ".[dev]"
.venv/bin/python -m pytest
.venv/bin/ruff check src tests
```

The P-file reader is [microstructure-tpw](https://github.com/mousebrains/turbulence)
(`odas_tpw.rsi.p_file.PFile`, read with `deconvolve=False` for raw counts). The
tests build small P-files from `tests/data/MR_SL435_template.p`, a 20-s
MicroRider file from that project's test data, with synthetic counts written
into chosen channels. `tests/test_real_data.py` checks agreement with AEM1-G
when the AEM1-G tank files are present (`ROCKLAND_NOISE_AEM1G=/path/to/Tank_Test`).

License: GPL-3.0-or-later.
