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
  * In LIVE data a real probe's "noise" includes real signal (turbulence,
    temperature changes, vibration), so it rises and falls with the water.
    When that signal is far bigger than the electronic noise, the noise ratio
    shows how differently the two probes respond to it, so it can change with
    no fault at all. The report says so when both channels' noise rose (or
    fell) together, at least 2x each, around a change.

Spikes
------
A spike is a sudden, short burst that sticks out far above the normal
wiggle. The program counts how many start in each file.

  * In bench mode a spike must be more than 6 times the channel's noise
    (and at least 5 counts). With test probes there is no real signal, so
    anything that big is either an electrical event or the instrument being
    bumped or shaken. Shaking usually shows on both shear channels at once;
    an electrical fault usually shows on one channel only.
  * In live mode a spike must be more than 8 times the noise (and at least 10
    counts), because real probes in water also see real things.
  * Bursts that start within 0.1 seconds of each other count as one spike.
  * The table shows SPIKES PER HOUR, so short and long files can be compared.
  * BOTH = spikes per hour that happened on both channels at the same moment
    (within 0.1 s of each other). A bump, shaking or real signal reaches both
    probes at once; an electrical fault shows on one channel only. On bench
    tests of shear probes about half of the spikes were shared; MR433's faulty
    T1 shared 1% of its spikes with T2. The report says what share of each
    channel's spikes were shared, and, when the channels are busy enough for
    it to matter, how many would be shared by chance alone.
  * SPIKE RATIO = spikes per hour on A divided by spikes per hour on B.
    "45" means A had 45 times as many spikes as B. "A only" means only A had
    any; "none" means neither did.
  * The summary gives each channel's typical (median) number over the files,
    and the numbers of the MIDDLE FILE: the one with as many files above it
    as below in that ratio. The middle file's numbers belong together; the
    two medians may come from different files. When they disagree a lot, the
    report says so: the files are not alike (bench tests mixed with casts,
    say), and no one file is typical.
  * To look for jumps and trends the program adds 1 spike per hour to both
    channels first, so a file where one channel had no spikes still gets a
    number. That only matters when a channel has very few spikes.
  * The spike threshold follows the noise. When both channels' noise rises
    many times over (rough water), fewer things count as spikes, so the
    counts in those files cannot be compared with the others; the report
    says so.

Channels that could not be measured
-----------------------------------
A channel with no spikes is only clean if it could have shown one. The report
lists, and leaves out of the comparison, the files where a channel could not:

  no probe  The thermistor's slow channel (T1 for T1_dT1) reads far beyond
            any water temperature (-16.05 or 54.75 C): nothing is connected.
            The fast channel then sits still, or swings about by itself;
            either way it says nothing about a probe.
  railed    The channel sat at the end of its range in at least 2% of its
            samples, or swung so widely that its spike threshold was more
            than half its range. Time near the surface or out of the
            water can do this; --min-pressure leaves that time out.
  flat      The channel did not change at all, or a fast thermistor channel
            moved by less than one count (a working one moves by 2 or more).

Without this, a dead channel would look like the clean one and its healthy
partner like the faulty one.

Bench or live?  (worked out for you; --mode to choose yourself)
---------------------------------------------------------------
  bench  Rockland TEST PROBES are on (open-circuit shear probes and
         fixed-resistor temperature probes) and the instrument stays at the
         surface: on a bench, on a ship's deck, or in a shallow test tank.
         The signal should be only the electronics' own small noise, so a
         healthy channel on a quiet bench shows few or no spikes (bumps and
         shaking add some, mostly on both shear channels at once), and the
         noise of matching channels should be nearly equal.
  live   Real probes, or an instrument in the water (deployed, or hung at
         a few metres). Real probes
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
  * At the surface: the pressure stays near zero (95% of the readings at
    most 1.2 dbar) and steady (the middle half within 0.5 dbar), so handling
    at the start and end of a file does not count. An instrument hung in the
    water, even at a few metres, is not a bench test. Tilt and shaking are
    shown but do not decide, because a bench test on a rolling ship's deck
    is still a good test of the temperature and shear channels.
  * If some files have test probes and others real probes, the program uses
    live and asks you to run the two kinds separately.

In bench mode the report also fills in the numbers for the items of Rockland's
Bench Test Review Checklist (V3) that are about the two channels, for example
"T1_dT1 and T2_dT2 counts are typically within +-40 counts" and "offset values
are less than 100 counts". "Typically" means 99.9% of the samples. The
checklist is written for a file of about one minute, so a longer file is
judged a minute at a time: the typical (median) minute must be within the
limit, and the report also says how many minutes were over it and how bad the
worst one was. The checklist's spectra and accelerometer items need the bench
figures from rsi-tpw bench, Zissou Essentials or the ODAS quick_bench.

In bench mode the report also warns when one channel has many spikes of its
own (the channel, its connector or the test probe may need attention), or
when both channels spiked together often (the instrument was bumped or
shaken, which says nothing against either channel).

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

  SWITCHING
         The ratio went back and forth between two clear levels, a few
         files at each, too often for the jump search (which needs each
         level to last about 15 to 20 files). A fault that comes and goes
         (a connector or probe) or two kinds of files mixed together (bench
         tests and casts) look like this; so can the water, in live data.
         The report gives each channel's typical numbers in the high and the
         low files, which shows which channel changes.

  UNUSUAL FILE
         One file that is very different from the files on both sides of it.
         Often something happened during just that file (the instrument was
         handled, or the water was rough). If it is the most recent file,
         keep an eye on the next files: it may be the start of a jump. For
         the spike ratio, a file is only called unusual if its spike counts
         are too many (or too few) to be luck for a file of its length, so
         one spike in a two-minute file is not enough.

To be reported, a change must be at least 2x for the spike ratio (1.25x for
the noise ratio; both can be changed with --min-factor and
--min-noise-factor), and it must be unlikely to be chance. That is the
smallest change worth reporting, not the smallest that is sure to be found:
in tests on 30 files that each differed from the next by about 1.4x, a
lasting 3x change was found nearly 9 times in 10, a 2x change only 1 time in
3.

"How sure" explained
--------------------
For a jump, the program shuffles the files into random order 1,999 times and
asks how often a change at least this big appears just by luck. "None of
1,999 shuffles" is the surest it can say: luck would give a change this big
less often than about 1 time in 2,000. For a trend it uses a standard test
of steady rising or falling (Kendall's). Nothing is reported unless the
chance is below 1 in 100. Files next to each other are usually alike, and
the program allows for that, so it does not get over-excited by a long,
slowly wandering record.

These chances are APPROXIMATE: the allowance for similar neighbouring files
is an estimate, and many places in the record are looked at. Treat "very
sure" as "worth looking into", not as proof.

Every file counts the same in the search for jumps and trends, however long
it is. The size of a jump is judged from the typical (median) file on each
side, which a few short, noisy files cannot move; files with less usable
data than --min-minutes (default 1 minute) are left out.

Some things a report cannot tell you
------------------------------------
  * WHY something changed. A jump says WHEN to look in the log book.
  * Whether a spike was electrical or real, in live data. Compare with the
    partner channel: a real event usually hits both (see BOTH above).
  * Anything about files from different instruments mixed together; the
    report warns you when that happens.
  * Jumps within the first or last 3 files: there is not enough data on one
    side to be sure. Run again when more files are in.
  * Changes that come and go every few files. The report calls them
    SWITCHING when the ratio flips between two clear levels (each at least
    2x from the middle for spikes); a ratio that just wanders, or flips by
    less, is not reported. Look at the table and the plot.

What to send your supervisor
----------------------------
Run with --csv results.csv --plot results.png and send both files, together
with the report text and a note of what you were testing. The CSV has every
number for every file, including some extra ones not shown in the table:
  *_threshold_counts            the spike threshold used for that file
  *_spikes_shared               spikes with one on the other channel within 0.1 s
  *_spikes_only_per_hour        spikes per hour on this channel alone
  *_worst_minute_range_counts   the checklist range in the worst minute
  *_not_measurable              why that channel could not be measured
  *_spikes_starting_upward_pct  % of spikes whose first swing was upward
                                (real signal is usually about half and half;
                                some electrical faults are almost all one way)
  *_median_spike_size_counts    how big a typical spike was
  *_saturated_samples           samples at the very end of the range (clipped)
  *_bad_samples                 samples the logger itself marked as bad
"""
