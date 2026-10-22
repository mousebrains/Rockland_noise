"""The text shown by ``rockland-noise explain``."""

GUIDE = """\
HOW TO READ A ROCKLAND-NOISE REPORT
===================================

What this program does
----------------------
It reads a set of P-files from a Rockland instrument (MicroRider, VMP, ...),
looks at two channels you choose (for example T1_dT1 and T2_dT2, or sh1 and
sh2), and for every file counts the SPIKES and measures the NOISE on each
channel. Then it compares the two channels file by file, in time order, and
points out JUMPS, TRENDS and UNUSUAL files.

Comparing two channels is the trick. Many things make both channels busier at
once (rough water, the instrument being bumped, a noisy power supply). A
problem with ONE probe, connector or circuit board channel shows up as that
channel getting worse while its partner does not.

All numbers are in raw COUNTS: the numbers the instrument's electronics
actually recorded, before any calibration. That way nothing in the processing
can create or hide a spike.

Noise
-----
Every channel wiggles a little all the time, even with nothing happening. That
is noise. The program measures the size of these small, fast wiggles (faster
than about 10 per second) and reports it in counts. Spikes are left out, so a
few spikes do not change the noise number.

  * On MicroRiders checked so far, a healthy temperature channel (T1_dT1,
    T2_dT2) has a noise of about 3.2 counts and a healthy shear channel (sh1,
    sh2) about 1.1 counts.
  * NOISE RATIO = noise of channel A divided by noise of channel B. For two
    channels of the same kind it is normally close to 1 (healthy pairs checked
    so far were within about 10%). A ratio of 1.5 means A wiggles 50% more
    than B.

Spikes
------
A spike is a sudden, short burst that sticks out far above the normal
wiggle. The program counts how many start in each file.

  * In bench mode a spike must be more than 6 times the channel's noise
    (and at least 5 counts). With test probes there is no real signal, so
    anything that big is an electrical event.
  * In live mode a spike must be more than 8 times the noise (and at least 10
    counts), because real probes in water also see real things.
  * Bursts that start within 0.1 seconds of each other count as one spike.
  * The table shows SPIKES PER HOUR, so short and long files can be compared.
  * SPIKE RATIO = spikes per hour on A divided by spikes per hour on B.
    "45" means A had 45 times as many spikes as B. "A only" means only A had
    any; "none" means neither did.
  * To look for jumps and trends the program adds 1 spike per hour to both
    channels first, so a file where one channel had no spikes still gets a
    number. That only matters when a channel has very few spikes.

Bench or live?  (worked out for you; --mode to choose yourself)
---------------------------------------------------------------
  bench  Rockland TEST PROBES are on (open-circuit shear probes and
         fixed-resistor temperature probes) and the instrument is not being
         deployed: on a bench, on a ship's deck, or in a test tank. The signal
         should be only the electronics' own small noise, so a healthy
         channel usually shows NO spikes at all, and the noise of matching
         channels should be nearly equal.
  live   Real probes, or an instrument that is being deployed. Real probes
         sense real things: turbulence, temperature changes, plankton hitting
         the probe tip, vibration. Some spikes are real and expected. What
         matters most is how the two channels compare, and whether that
         changes. For deployed data add --min-pressure (for example
         --min-pressure 5) so time at the surface or out of the water is not
         counted.

How the program tells them apart (the report says what it found and why):
  * Test probes: a temperature test probe is a fixed resistor, so T1 or T2
    reads a steady temperature equal to the reference temperature T_0 in the
    instrument's setup, usually 16.15 C ("about 16.3" is the rule of thumb).
    One such channel is enough: an unsteady partner channel next to it is
    flagged, because it usually means bad temperature electronics. On a VMP
    the report also says whether that channel follows the water temperature
    measured by the CT sensor (JAC_T), as a real probe would.
  * Not deployed: the pressure stays steady (within 2 dbar). Tilt and shaking
    are shown but do not decide, because a bench test on a rolling ship is
    still a good test of the temperature and shear channels.
  * If some files have test probes and others real probes, the program uses
    live and asks you to run the two kinds separately.

In bench mode the report also fills in the numbers for the items of Rockland's
Bench Test Review Checklist (V3) that are about the two channels, for example
"T1_dT1 and T2_dT2 counts are typically within +-40 counts" and "offset values
are less than 100 counts". "Typically" means 99.9% of the samples. The
checklist's spectra and accelerometer items need the bench figures from
rsi-tpw bench, Zissou Essentials or the ODAS quick_bench.

Rockland's bench test instructions, in short: test probes installed; the
instrument lying flat on something soft (open-cell foam), pressure port up;
at least 60 s of data; no vibration or bumps. Things that spoil a bench test:
a tall building, a ship at sea, noisy electrical equipment nearby, people
moving near the instrument.

Jumps, trends and unusual files
-------------------------------
These are looked for separately in the spike ratio and in the noise ratio.
The program works with the RATIOS, so "twice as big" and "half as big" count
as the same size of change.

  JUMP   The ratio stayed at one level for a run of files, then moved to a
         clearly different level and stayed there. The table marks it with a
         line like
             ==== JUMP in spike ratio: about 20x higher from here ====
         just before the first file after the jump. This often means
         something changed: a probe was swapped, a connector got wet, the
         instrument was moved somewhere new.

  TEMPORARY CHANGE
         A jump that came back to where it started a few files later.

  TREND  The ratio rose or fell steadily over many files, rather than all at
         once. A slowly failing probe or a connector slowly corroding can do
         this.

  UNUSUAL FILE
         One file that is very different from the files on both sides of it.
         Often something happened during just that file (the instrument was
         handled, or the file is very short). If it is the most recent file,
         keep an eye on the next files: it may be the start of a jump.

To be reported, a change must be at least 2x for the spike ratio (1.25x for
the noise ratio; both can be changed with --min-factor and
--min-noise-factor), and it must be unlikely to be chance.

"How sure" explained
--------------------
For a jump, the program shuffles the files into random order thousands of
times and asks how often a change at least this big appears just by luck.
"About 1 in 2,000 shuffles" means it would be very surprising by luck alone.
Nothing is reported unless that chance is below 1 in 100. Files next to each
other are usually alike, and the program allows for that, so it does not get
over-excited by a long, slowly wandering record.

Some things a report cannot tell you
------------------------------------
  * WHY something changed. A jump says WHEN to look in the log book.
  * Whether a spike was electrical or real, in live data. Compare with the
    partner channel: a real event usually hits both.
  * Anything about files from different instruments mixed together; the
    report warns you when that happens.
  * Jumps within the first or last 3 files: there is not enough data on one
    side to be sure. Run again when more files are in.

What to send your supervisor
----------------------------
Run with --csv results.csv --plot results.png and send both files, together
with the report text and a note of what you were testing. The CSV has every
number for every file, including some extra ones not shown in the table:
  *_threshold_counts            the spike threshold used for that file
  *_spikes_starting_upward_pct  % of spikes whose first swing was upward
                                (real signal is usually about half and half;
                                some electrical faults are almost all one way)
  *_median_spike_size_counts    how big a typical spike was
  *_saturated_samples           samples at the very end of the range (clipped)
  *_bad_samples                 samples the logger itself marked as bad
"""
