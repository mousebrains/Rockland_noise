# rockland-noise

Compare **spikes** and **noise** between two channels of a Rockland Scientific
instrument (MicroRider, VMP, ...) across a set of P-files, list the files in time
order, and point out **jumps**, **trends** and **unusual files**.

It answers questions like:

* "Is T1 spikier than T2, and since when?"
* "Did the shear channels change after the probe swap?"
* "Is this bench test clean, and does it pass Rockland's bench checklist?"

```
rockland-noise compare T1_dT1 T2_dT2 /path/to/pfiles
```

You do not need to know statistics or programming to use it. It works out by
itself whether the files are a bench test, the report explains itself in plain
words, and `rockland-noise explain` describes every number.

It is self-contained: it reads P-files with its own reader and needs only
numpy, scipy and matplotlib.

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
| A **bench test** (test probes; on a bench, a ship's deck or in a tank) | `rockland-noise compare T1_dT1 T2_dT2 data/` |
| **Deployed** data (ignore time near the surface) | `rockland-noise compare sh1 sh2 data/ --min-pressure 5` |
| Save a **spreadsheet and a picture** for your supervisor | add `--csv results.csv --plot results.png` |

**Bench or live?** The tool decides from the files, and the report's `Mode:`
and `Why:` lines say what it found:

* **bench**: Rockland **test probes** are on (open-circuit shear,
  fixed-resistor temperature) and the instrument stays at the surface (bench,
  ship's deck, shallow test tank). The signal
  is only electronic noise, so the tool is more sensitive: a spike is anything
  over 6 times the noise.
* **live**: real probes, or an instrument in the water (deployed, or hung at
  a few metres). Real probes sense real
  things in the water, so a spike must be over 8 times the noise.

How it tells: a temperature test probe is a fixed resistor, so T1 or T2 reads a
steady temperature equal to the reference temperature T₀ in the instrument's
setup, usually 16.15 °C ("about 16.3" is the rule of thumb). And the pressure
stays near zero and steady: 95% of the readings at most 1.2 dbar, the middle
half within 0.5 dbar (so handling at the start and end of a file does not
count). An instrument hung in the water, even at a few metres, is not a bench
test. Tilt and shaking are shown but do not decide: a bench test on a rolling
ship's deck is still a good test of the temperature and shear channels.
If one temperature channel reads its T₀ and the other wanders, the file is
still a bench test and the wandering channel is flagged (bad temperature
electronics?); on a VMP the report says whether it follows the water
temperature from the CT sensor (JAC_T), as a real probe would. A set mixing
test-probe and real-probe files is analysed as live, with a note to run the
two kinds separately. `--mode bench` or `--mode live` overrides the choice;
the report then says if the files disagree.

In bench mode the report also fills in the numbers for the items of Rockland's
**Bench Test Review Checklist (V3)** that concern the two channels: "typically
within ±40 counts", "offset less than 100 counts", and so on.

### Step 3: read the report

A shortened example: a bench test of MicroRider SN 433, whose T1 channel had a
fault.

```
  Channels:   A = T1_dT1   B = T2_dT2
  Mode:       bench (bench test): a spike is a jump bigger than 6 times the channel's noise (and at least 5 counts)
              Why: T1 and T2 read their steady reference temperature (test probes) and the instrument stayed at the surface in 12 of 12 files.

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
| `looks like a file or folder, not a channel` | Put the two channel names **before** the files. |
| `'T1_dT1' is not a number` | An option such as `--threshold-counts` came before the channel names. Put the channel names and files first, then the options. |
| `No .p files in folder` | Check the folder, or add `--recursive` if the files are in sub-folders. |
| `PROBLEM: Could not read this file` | That file is damaged or not a P-file. The other files are still analysed. |
| `PROBLEM: Empty file` | The instrument wrote the file's header but no data (it usually stopped right after it started). Nothing to analyse; the other files are still used. |
| `--min-pressure needs a pressure channel (P)` | That file has no pressure. Leave out `--min-pressure`; a bench test does not need it. |
| `WARNING: ... is a slow channel` or `a 32-bit channel` | Spikes and noise are meant for the fast channels: `T1_dT1`/`T2_dT2` or `sh1`/`sh2`. |
| `... could not be converted ..., so it was not used to tell a bench test from live data` (CSV notes) | A calibration coefficient in that file's setup is unreadable (for example `0,0687` with a comma). The two channels are still analysed; check the setup, or choose `--mode` yourself. |
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
for file (see `tests/test_real_data.py`). The bench threshold: in quiet
stretches the largest excursion of a healthy channel in a 4-hour file is about
5.4 × the noise (median over 600 healthy channel-files in 163 bench tests of
MR433, MR435 and MR533), as expected for Gaussian noise, so 6 × gives about one
false spike per 250 hours of pure electronic noise. Real bench sessions also
have mechanical spikes from handling and vibration: 27% of the healthy shear
channel-files and 5% of the thermistor channel-files had at least one, and in
39 of the 49 shear files with spikes both sh1 and sh2 had them. A ratio between
the two channels is not fooled by spikes they share; the absolute counts are.

**Bench or live** (`setup.py`), from 946 files of MicroRiders (SN 330, 429,
433, 435, 533) and VMPs (SN 132, 142, 409, 465, 479):

* Bench thermistors read within 0.03 °C of their T₀ (99% of 324
  channel-files; 0.2 °C at most, on MR433's faulty T1); none of 376
  real-probe channel-files came within 0.5 °C of T₀ while steady. T₀ is
  usually 16.15 °C but 7.88, 9.45, 14.27 and 16.96 °C also occur, so the test
  is against each channel's own T₀, not a fixed temperature.
* Bench tests read −0.9 to +0.7 dbar (sensor offsets, shallow test tank),
  with the middle half of the readings within 0.12 dbar, or 0.42 on the deck
  of the Ocean Hunter III (ARCTERX 2023 Wake). VMPs hung overnight at 2–5 m in
  that lagoon read 1.9 dbar or more; casts, including the Wake VMP casts made
  with test probes, span tens of dbar.
* Real probes follow JAC_T (rank correlation ≥ 0.52, median 1.00) whenever
  the water temperature varies by more than 2 °C. Incl_T, the housing
  temperature, lags the water too much to help (correlation about 0 on VMP
  casts).

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

The P-file reader (`pfile.py`) follows Rockland's Technical Note 051 and the
ODAS MATLAB library. It was checked sample for sample against
[microstructure-tpw](https://github.com/mousebrains/turbulence)'s reader on 23
files (MicroRiders and VMPs, formats 6.0–6.3, 512 and 1024 Hz, 32-bit channels,
real bad-buffer runs). `tests/test_pfile.py` pins reference values from it for
four small real files in `tests/data/`, a few records each: a MicroRider at
format 6.3 and 512 Hz; a MicroRider at format 6.0 and 1024 Hz; a VMP with a
32-bit channel (JAC_C), unsigned channels (JAC_T, Chlorophyll, Turbidity) and a
real dropout; and a VMP with test probes on a ship's deck. The reader converts
to physical units only pressure, thermistor, inclinometer and JAC_T channels;
everything else stays in counts. It reads only the channels it needs, a few
megabytes of the file at a time; analysing a 4-hour, 512 Hz file needs about
230 MB of working data.
The other tests build small P-files from `tests/data/MR_SL435_template.p` with
synthetic counts written into chosen channels. `tests/test_real_data.py` checks
agreement with AEM1-G when the AEM1-G tank files are present
(`ROCKLAND_NOISE_AEM1G=/path/to/Tank_Test`).

License: GPL-3.0-or-later.
