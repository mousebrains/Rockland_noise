"""The ``rockland-noise`` command.

    rockland-noise compare T1_dT1 T2_dT2 /path/to/pfiles --mode bench
    rockland-noise channels /path/to/one_file.p
    rockland-noise explain
"""

import argparse
import dataclasses
import re
import sys
from pathlib import Path

from . import __version__
from .changes import ChangeSettings
from .compare import Cache, FileResult, Options, compare, default_cache_dir, find_pfiles
from .explain import GUIDE
from .reader import ChannelNotFoundError, PFileReadError, list_channels
from .report import text_report, write_csv
from .spikes import MODES

CHANNEL_NAME = re.compile(r"[A-Za-z0-9_\-]+")
SHOW_PROBLEMS = 20  # when no file could be used, list this many reasons

EXAMPLES = """\
examples:
  rockland-noise channels data/RIOT_SL684_0520.p
      list the channel names in a file

  rockland-noise compare T1_dT1 T2_dT2 data/
      compare the two thermistor channels in every .p file in the folder data;
      it works out by itself whether this is a bench test

  rockland-noise compare sh1 sh2 data/*.p --min-pressure 5 --csv sh.csv --plot sh.png
      compare the shear channels in deployed data, below 5 dbar, and save a
      spreadsheet and a picture

  rockland-noise explain
      what every number in the report means
"""


class FriendlyParser(argparse.ArgumentParser):
    """Show the short usage, the problem and where to get help, not a wall of text."""

    def error(self, message: str):
        self.print_usage(sys.stderr)
        self.exit(2, f"\nProblem: {message}\nFor help, run:  {self.prog} --help\n")


def _number(text: str) -> float:
    """A number, or a message that says what probably went wrong."""
    try:
        return float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"'{text}' is not a number. Put the two channel names and the files first, then"
            " the options, e.g.  rockland-noise compare T1_dT1 T2_dT2 data/ --threshold-counts"
            " 20") from None


def _positive(text: str) -> float:
    value = _number(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be more than 0, not {text}")
    return value


def _not_negative(text: str) -> float:
    value = _number(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be 0 or more, not {text}")
    return value


def _factor(text: str) -> float:
    value = _number(text)
    if value <= 1:
        raise argparse.ArgumentTypeError(f"must be more than 1 (2 means 'twice'), not {text}")
    return value


def _chance(text: str) -> float:
    value = _number(text)
    if not 0 < value < 0.5:
        raise argparse.ArgumentTypeError(f"must be between 0 and 0.5, not {text}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = FriendlyParser(
        prog="rockland-noise",
        description="Compare spikes and noise between two channels across a set of Rockland"
                    " P-files, and point out jumps and trends.",
        epilog=EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND", parser_class=FriendlyParser)

    c = sub.add_parser(
        "compare", help="compare two channels across P-files (the main job)",
        description="Count spikes and measure noise on two channels in every P-file, list the"
                    " files in time order, and point out jumps, trends and unusual files.",
        epilog=EXAMPLES, formatter_class=argparse.RawDescriptionHelpFormatter)
    c.add_argument("channel_a", metavar="CHANNEL_A",
                   help="first channel, e.g. T1_dT1 or sh1 (see: rockland-noise channels FILE)")
    c.add_argument("channel_b", metavar="CHANNEL_B", help="second channel, e.g. T2_dT2 or sh2")
    c.add_argument("paths", metavar="PFILES", nargs="+",
                   help="P-files, folders of P-files, or patterns like data/*.p")
    c.add_argument("--mode", choices=["auto", *sorted(MODES)], default="auto",
                   help="auto (default) = decide from the files: test probes (T1 and T2 read"
                        " their steady reference temperature, usually 16.15 C) and steady"
                        " pressure mean a bench test; bench = bench test with Rockland test"
                        " probes (more sensitive); live = real probes, or deployed")
    c.add_argument("-r", "--recursive", action="store_true",
                   help="also look for P-files in sub-folders of any folder given")
    c.add_argument("--csv", metavar="FILE", type=Path,
                   help="also save every number for every file in this spreadsheet (CSV) file")
    c.add_argument("--plot", metavar="FILE", type=Path,
                   help="also save a picture of the results (e.g. results.png or results.pdf)")
    c.add_argument("--min-pressure", metavar="DBAR", type=_number, default=None,
                   help="only use data deeper than this pressure in dbar (about metres);"
                        " use for deployed data, e.g. 5")
    c.add_argument("--skip-start", metavar="SECONDS", type=_not_negative, default=0.0,
                   help="ignore this many seconds at the start of every file (default 0)")
    c.add_argument("--min-minutes", metavar="MINUTES", type=_not_negative, default=1.0,
                   help="leave out of the jump/trend search files with less usable data than"
                        " this (default 1)")
    c.add_argument("-q", "--quiet", action="store_true", help="do not show progress")

    t = c.add_argument_group(
        "fine-tuning (leave alone unless you have been asked to change them)")
    t.add_argument("--threshold-sigma", metavar="N", type=_positive,
                   help="a spike is bigger than N times the noise (bench 6, live 8)")
    t.add_argument("--min-threshold-counts", metavar="N", type=_not_negative,
                   help="...and bigger than N counts (bench 5, live 10)")
    t.add_argument("--threshold-counts", metavar="N", type=_positive, nargs="+",
                   help="use a fixed spike threshold in counts instead; one number for both"
                        " channels or two numbers, A then B")
    t.add_argument("--min-factor", metavar="F", type=_factor, default=2.0,
                   help="smallest change in the spike ratio worth reporting (default 2)")
    t.add_argument("--min-noise-factor", metavar="F", type=_factor, default=1.25,
                   help="smallest change in the noise ratio worth reporting (default 1.25)")
    t.add_argument("--alpha", metavar="P", type=_chance, default=0.01,
                   help="largest chance of a jump or trend being luck (default 0.01)")
    t.add_argument("--cache-dir", metavar="FOLDER", type=Path, default=None,
                   help="where results are remembered between runs"
                        f" (default {default_cache_dir()})")
    t.add_argument("--no-cache", action="store_true",
                   help="do not remember results; read every file every time")
    t.add_argument("--clear-cache", action="store_true",
                   help="forget every remembered result first, then run as usual")

    ch = sub.add_parser("channels", help="list the channel names in a P-file",
                        description="List the channels in a P-file, so you know what to compare.")
    ch.add_argument("file", metavar="PFILE", type=Path)

    sub.add_parser("explain", help="explain every number in the report, in plain language")
    return parser


def _options(args: argparse.Namespace) -> Options:
    changes = {}
    if args.threshold_sigma is not None:
        changes["threshold_sigma"] = args.threshold_sigma
    if args.min_threshold_counts is not None:
        changes["min_threshold_counts"] = args.min_threshold_counts
    if args.threshold_counts and len(args.threshold_counts) > 2:
        raise SystemExit("Problem: --threshold-counts takes one or two numbers.")
    spike = {}
    for mode in (MODES if args.mode == "auto" else [args.mode]):
        spike_a = spike_b = dataclasses.replace(MODES[mode], **changes)
        if args.threshold_counts:
            spike_a = dataclasses.replace(spike_a, fixed_threshold_counts=args.threshold_counts[0])
            spike_b = dataclasses.replace(spike_b, fixed_threshold_counts=args.threshold_counts[-1])
        spike[mode] = (spike_a, spike_b)
    return Options(
        mode=args.mode,
        spike=spike,
        skip_start_s=args.skip_start,
        min_pressure_dbar=args.min_pressure,
        min_usable_s=args.min_minutes * 60,
        spike_changes=ChangeSettings(min_factor=args.min_factor, alpha=args.alpha),
        noise_changes=ChangeSettings(min_factor=args.min_noise_factor, alpha=args.alpha),
    )


def _progress(quiet: bool):
    def show(i: int, n: int, r: FileResult) -> None:
        if quiet:
            return
        if r.error:
            status = f"PROBLEM: {r.error}"
        else:
            status = "remembered from an earlier run" if r.from_cache else "done"
            status += f" ({r.usable_s / 3600:.2f} h usable)"
        print(f"[{i:>{len(str(n))}}/{n}] {r.path.name}  {status}", file=sys.stderr, flush=True)
    return show


def run_compare(args: argparse.Namespace) -> int:
    if args.channel_a.casefold() == args.channel_b.casefold():
        print("Problem: the two channels are the same. Give two different channels, e.g."
              " T1_dT1 T2_dT2.", file=sys.stderr)
        return 1
    for name in (args.channel_a, args.channel_b):
        # Channel names are letters, digits, '_' and '-' (T1_dT1, sh1, P_dP). A
        # name ending in .p, or with a slash, dot or wildcard in it, is a path.
        if Path(name).suffix.lower() == ".p" or not CHANNEL_NAME.fullmatch(name):
            print(f"Problem: '{name}' looks like a file or folder, not a channel. Give the two"
                  " channel names first, then the files:\n  rockland-noise compare T1_dT1"
                  " T2_dT2 data/*.p", file=sys.stderr)
            return 1

    options = _options(args)
    paths, problems = find_pfiles(args.paths, recursive=args.recursive)
    for msg in problems:
        print(f"Note: {msg}", file=sys.stderr)
    if not paths:
        print("Problem: no P-files to look at.", file=sys.stderr)
        return 1

    folder = args.cache_dir or default_cache_dir()
    if args.clear_cache:
        removed = Cache(folder).clear() if folder.is_dir() else 0
        print(f"Forgot {removed} remembered result(s) in {folder}.", file=sys.stderr)
    cache = Cache(None if args.no_cache else folder)
    if cache.warning:
        print(f"Note: {cache.warning}", file=sys.stderr)
    if not args.quiet:
        print(f"Reading {len(paths)} P-file(s)...", file=sys.stderr)
    try:
        result = compare(paths, args.channel_a, args.channel_b, options, cache,
                         progress=_progress(args.quiet))
    except KeyboardInterrupt:
        print("\nStopped. Every file finished so far is remembered; run the same command"
              " again to carry on from there.", file=sys.stderr)
        return 130

    good = [f for f in result.files if not f.error]
    if not good:
        # Say why, whatever --quiet says: there is no report to read instead.
        missing = [f for f in result.files if f.notes and "There is no channel" in f.notes[0]]
        if missing:
            print(f"Problem: {missing[0].notes[0]}", file=sys.stderr)
            for name in (args.channel_a, args.channel_b):
                if Path(name).exists():
                    print(f"('{name}' is also the name of a file or folder here. Give the two"
                          " channel names first, then the files.)", file=sys.stderr)
            return 1
        print("Problem: none of the files could be used:", file=sys.stderr)
        for f in result.files[:SHOW_PROBLEMS]:
            print(f"  {f.path.name}: {f.error}", file=sys.stderr)
        if len(result.files) > SHOW_PROBLEMS:
            print(f"  ... and {len(result.files) - SHOW_PROBLEMS} more", file=sys.stderr)
        return 1

    print(text_report(result))
    if args.csv:
        try:
            write_csv(result, args.csv)
            print(f"\nSaved the spreadsheet: {args.csv}")
        except OSError as ex:
            print(f"Problem: could not save {args.csv}: {ex}", file=sys.stderr)
            return 1
    if args.plot:
        from .plot import plot  # matplotlib is slow to load; only when asked

        try:
            plot(result, args.plot)
            print(f"Saved the picture: {args.plot}")
        except (OSError, ValueError) as ex:
            print(f"Problem: could not save {args.plot}: {ex}", file=sys.stderr)
            return 1
    return 0


def run_channels(args: argparse.Namespace) -> int:
    try:
        rows = list_channels(args.file)
    except FileNotFoundError:
        print(f"Problem: there is no file {args.file}", file=sys.stderr)
        return 1
    except (PFileReadError, OSError) as ex:
        print(f"Problem: could not read {args.file} as a P-file ({ex})", file=sys.stderr)
        return 1
    w = max([len("name")] + [len(r["name"]) for r in rows])
    print(f"Channels in {args.file.name}:")
    print(f"  {'name':<{w}} {'sensor type':<12} {'samples/s':>9}")
    for r in rows:
        print(f"  {r['name']:<{w}} {r['type']:<12} {r['fs']:>9.1f}")
    print("\nFor spikes, the fast pre-emphasised thermistor channels (T1_dT1, T2_dT2) and the"
          "\nshear channels (sh1, sh2) are the usual pairs to compare.")
    return 0


def main(argv: list[str] | None = None) -> int:
    # Never crash on a character the terminal cannot show.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "explain":
        print(GUIDE)
        return 0
    if args.command == "channels":
        return run_channels(args)
    try:
        return run_compare(args)
    except ChannelNotFoundError as ex:  # pragma: no cover - handled per file
        print(f"Problem: {ex}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
