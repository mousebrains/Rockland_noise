"""The whole program, run as a technician would, on synthetic P-files."""

import csv
from datetime import timedelta

import numpy as np
import pytest

from rockland_noise.cli import main

from .conftest import DATA, T0, add_bursts, bench_channels, counts_for_temperature, noise

SECONDS = 120  # each test file is two minutes long


def write_set(maker, folder, bursts_per_file, *, t1_offset=-20, seconds=SECONDS):
    """One file per entry: that many T1 bursts; T2 always clean."""
    folder.mkdir(exist_ok=True)
    n = maker.fast_samples(seconds)
    paths = []
    for k, nb in enumerate(bursts_per_file):
        where = list(np.linspace(1000, n - 1000, nb).astype(int)) if nb else []
        t1 = add_bursts(noise(n, 3.0, t1_offset, seed=k), where, size=80)
        t2 = noise(n, 3.0, -20, seed=1000 + k)
        sh = noise(n, 1.1, 0, seed=2000 + k)
        paths.append(maker.write(folder / f"TEST_{k + 1:04d}.p", start=T0 + timedelta(hours=4 * k),
                                 seconds=seconds,
                                 channels={"T1_dT1": t1, "T2_dT2": t2, "sh1": sh, "sh2": sh}))
    return paths


def run(capsys, *args):
    code = main([str(a) for a in args])
    out = capsys.readouterr()
    return code, out.out, out.err


@pytest.fixture
def cache(tmp_path):
    return ["--cache-dir", tmp_path / "cache"]


def test_finds_the_jump_and_writes_csv_and_plot(maker, tmp_path, capsys, cache):
    write_set(maker, tmp_path / "data", [1] * 8 + [40] * 8)
    csv_path, png_path = tmp_path / "out.csv", tmp_path / "out.png"
    code, out, err = run(capsys, "compare", "T1_dT1", "T2_dT2", tmp_path / "data",
                         "--csv", csv_path, "--plot", png_path, *cache)
    assert code == 0, err
    assert "==== JUMP in spike ratio" in out
    assert "JUMP between file 8 and file 9" in out
    assert "none of 1,999 shuffles of these files gave a change this big" in out
    assert "T2_dT2 had no spikes at all" in out
    rows = list(csv.DictReader(csv_path.open()))
    assert len(rows) == 16
    assert rows[8]["findings"].startswith("spike ratio jump")
    assert int(rows[0]["A_spikes"]) == 1 and int(rows[8]["A_spikes"]) == 40
    assert png_path.stat().st_size > 10_000


def test_sorted_by_time_not_by_name_or_order_given(maker, tmp_path, capsys, cache):
    paths = write_set(maker, tmp_path / "data", [0, 5, 10])
    code, out, _ = run(capsys, "compare", "t1_dt1", "T2_DT2", *reversed(paths), *cache)
    assert code == 0
    lines = [ln for ln in out.splitlines() if "TEST_000" in ln]
    assert [ln.split()[1] for ln in lines] == ["TEST_0001.p", "TEST_0002.p", "TEST_0003.p"]


def test_bench_mode_fills_in_the_checklist(maker, tmp_path, capsys, cache):
    write_set(maker, tmp_path / "data", [0, 0, 0], t1_offset=150)  # offset over 100
    code, out, _ = run(capsys, "compare", "T1_dT1", "T2_dT2", tmp_path / "data",
                       "--mode", "bench", *cache)
    assert code == 0
    assert "Rockland Bench Test Review Checklist (V3)" in out
    assert "T1_dT1: in a typical minute within" in out
    assert "OUTSIDE the checklist in 3 of 3 files" in out
    assert "T2_dT2: in a typical minute within" in out
    assert "Within the checklist in all 3 files" in out
    assert "neither T1_dT1 nor T2_dT2 had any spikes" in out


def test_shear_pair(maker, tmp_path, capsys, cache):
    write_set(maker, tmp_path / "data", [0, 0, 0])
    code, out, _ = run(capsys, "compare", "sh1", "sh2", tmp_path / "data", *cache)
    assert code == 0
    assert "noise ratio of 1.00" in out


def test_a_bad_file_does_not_stop_the_run(maker, tmp_path, capsys, cache):
    write_set(maker, tmp_path / "data", [0, 3, 6])
    (tmp_path / "data" / "TEST_9999.p").write_bytes(b"not a P-file")
    code, out, err = run(capsys, "compare", "T1_dT1", "T2_dT2", tmp_path / "data", *cache)
    assert code == 0
    assert "Files that could not be used" in out and "TEST_9999.p" in out
    assert "PROBLEM" in err


def test_unknown_channel_suggests_the_right_one(maker, tmp_path, capsys, cache):
    write_set(maker, tmp_path / "data", [0, 0])
    code, _, err = run(capsys, "compare", "T1_dT", "T2_dT2", tmp_path / "data", *cache)
    assert code == 1
    assert "no channel called 'T1_dT'" in err and "Did you mean T1_dT1" in err


def test_files_given_where_channels_belong(tmp_path, capsys):
    code, _, err = run(capsys, "compare", "a.p", "b.p", tmp_path)
    assert code == 1 and "looks like a file" in err
    code, _, err = run(capsys, "compare", "data/", "T1_dT1", tmp_path)
    assert code == 1 and "looks like a file" in err


def test_same_channel_twice(tmp_path, capsys):
    code, _, err = run(capsys, "compare", "T1_dT1", "t1_dt1", tmp_path)
    assert code == 1 and "the same" in err


def test_nothing_found(tmp_path, capsys):
    code, _, err = run(capsys, "compare", "T1_dT1", "T2_dT2", tmp_path / "nowhere")
    assert code == 1 and "Not found" in err


def test_second_run_uses_the_cache_and_new_settings_do_not(maker, tmp_path, capsys, cache):
    write_set(maker, tmp_path / "data", [0, 2])
    args = ["compare", "T1_dT1", "T2_dT2", tmp_path / "data", *cache]
    run(capsys, *args)
    _, _, err = run(capsys, *args)
    assert err.count("remembered from an earlier run") == 2
    _, _, err = run(capsys, *args, "--mode", "bench")  # auto already worked out bench
    assert err.count("remembered from an earlier run") == 2
    _, _, err = run(capsys, *args, "--threshold-sigma", "7")
    assert "remembered" not in err


def test_new_code_never_reuses_old_answers(maker, tmp_path, capsys, cache, monkeypatch):
    from rockland_noise import compare

    write_set(maker, tmp_path / "data", [0, 2])
    args = ["compare", "T1_dT1", "T2_dT2", tmp_path / "data", *cache]
    run(capsys, *args)
    monkeypatch.setattr(compare, "code_fingerprint", lambda: "a different version")
    _, _, err = run(capsys, *args)
    assert "remembered" not in err


def test_clear_cache(maker, tmp_path, capsys, cache):
    write_set(maker, tmp_path / "data", [0, 2])
    args = ["compare", "T1_dT1", "T2_dT2", tmp_path / "data", *cache]
    run(capsys, *args)
    keep = tmp_path / "cache" / "my notes.txt"
    keep.write_text("not the cache's")
    _, _, err = run(capsys, *args, "--clear-cache")
    assert "Forgot 8 remembered result(s)" in err and "remembered from" not in err  # 2 x 2 x 2
    assert keep.exists()
    _, _, err = run(capsys, *args)
    assert err.count("remembered from an earlier run") == 2


def test_min_pressure_uses_only_deep_data(maker, tmp_path, capsys, cache):
    """P is above 20 dbar for the second half only; only those spikes count."""
    cal = maker.pf.setup("P").params
    deep = (20 - float(cal["coef0"])) / float(cal["coef1"])  # counts for 20 dbar (c2 is tiny)
    n, n_slow = maker.fast_samples(SECONDS), SECONDS * 64
    p = np.where(np.arange(n_slow) < n_slow // 2, 0.0, deep)
    t1 = add_bursts(noise(n, 3.0, seed=1), [5000, 10_000, 40_000, 50_000])
    maker.write(tmp_path / "deep.p", start=T0, seconds=SECONDS,
                channels={"T1_dT1": t1, "T2_dT2": noise(n, 3.0, seed=2), "P": p})
    csv_path = tmp_path / "out.csv"
    code, _, err = run(capsys, "compare", "T1_dT1", "T2_dT2", tmp_path / "deep.p",
                       "--min-pressure", 10, "--csv", csv_path, *cache)
    assert code == 0, err
    row = next(csv.DictReader(csv_path.open()))
    assert int(row["A_spikes"]) == 2
    assert float(row["usable_hours"]) == pytest.approx(SECONDS / 2 / 3600, rel=0.02)


def test_channels_and_explain(maker, tmp_path, capsys):
    path = write_set(maker, tmp_path / "data", [0])[0]
    code, out, _ = run(capsys, "channels", path)
    assert code == 0 and "T1_dT1" in out and "sh2" in out
    code, out, _ = run(capsys, "explain")
    assert code == 0 and "HOW TO READ" in out


def test_no_command_prints_help(capsys):
    code, out, _ = run(capsys)
    assert code == 0 and "compare" in out


def write_bench_set(maker, folder, bursts_per_file):
    """Files that look like a bench test: test probes, steady pressure."""
    folder.mkdir(exist_ok=True)
    n = maker.fast_samples(SECONDS)
    for k, nb in enumerate(bursts_per_file):
        where = list(np.linspace(1000, n - 1000, nb).astype(int)) if nb else []
        channels = bench_channels(maker, SECONDS, seed=k)
        channels |= {"T1_dT1": add_bursts(noise(n, 3.0, -20, seed=k), where),
                     "T2_dT2": noise(n, 3.0, -20, seed=100 + k)}
        maker.write(folder / f"BENCH_{k + 1:04d}.p", start=T0 + timedelta(hours=k),
                    seconds=SECONDS, channels=channels)


def test_auto_mode_recognises_a_bench_test(maker, tmp_path, capsys, cache):
    write_bench_set(maker, tmp_path / "bench", [0, 0, 0])
    code, out, _ = run(capsys, "compare", "T1_dT1", "T2_dT2", tmp_path / "bench", *cache)
    assert code == 0
    assert "Mode:       bench" in out
    assert "test probes" in out and "stayed at the surface in 3 of 3 files" in out
    assert "Rockland Bench Test Review Checklist" in out


def test_auto_mode_says_live_for_real_probes(maker, tmp_path, capsys, cache):
    write_set(maker, tmp_path / "data", [0, 0, 0])  # the template's own probes: 12 and 11 C
    code, out, _ = run(capsys, "compare", "T1_dT1", "T2_dT2", tmp_path / "data", *cache)
    assert code == 0
    assert "Mode:       live" in out and "real probes" in out
    assert "Checklist" not in out


def test_forcing_a_mode_that_disagrees_is_pointed_out(maker, tmp_path, capsys, cache):
    write_bench_set(maker, tmp_path / "bench", [0, 0])
    code, out, _ = run(capsys, "compare", "T1_dT1", "T2_dT2", tmp_path / "bench",
                       "--mode", "live", *cache)
    assert code == 0
    assert "Mode:       live" in out and "but the files look like bench" in out


def test_bad_temperature_electronics_on_one_channel(maker, tmp_path, capsys, cache):
    """T1 a test probe; T2 wanders (bad electronics): still a bench test, T2 flagged."""
    folder = tmp_path / "bench"
    folder.mkdir()
    n, n_slow = maker.fast_samples(SECONDS), SECONDS * 64
    for k in range(3):
        channels = bench_channels(maker, SECONDS, seed=k)
        wander = np.cumsum(np.random.default_rng(k).normal(0, 20, n_slow))
        channels["T2"] = channels["T2"] + wander
        channels |= {"T1_dT1": noise(n, 3.0, -20, seed=k), "T2_dT2": noise(n, 3.0, -20, seed=9)}
        maker.write(folder / f"B_{k}.p", start=T0 + timedelta(hours=k), seconds=SECONDS,
                    channels=channels)
    code, out, _ = run(capsys, "compare", "T1_dT1", "T2_dT2", folder, *cache)
    assert code == 0
    assert "Mode:       bench" in out
    assert "Temperature channels" in out and "T2 is unsteady" in out and "3 file(s)" in out


def test_a_damaged_coefficient_elsewhere_does_not_lose_the_file(maker, tmp_path, capsys, cache):
    """A decimal comma in P's calibration: sh1 vs sh2 is still analysed, with a note."""
    path = write_set(maker, tmp_path / "data", [0])[0]
    raw = path.read_bytes()
    assert raw.count(b"coef1       = 0.058512") == 1
    path.write_bytes(raw.replace(b"coef1       = 0.058512", b"coef1       = 0,058512"))
    csv_path = tmp_path / "out.csv"
    code, _, err = run(capsys, "compare", "sh1", "sh2", path, "--csv", csv_path, *cache)
    assert code == 0, err
    row = next(csv.DictReader(csv_path.open()))
    assert row["analysed"] == "yes" and "P could not be converted" in row["notes"]
    # ...but a pressure limit needs the pressure, so then the file cannot be used.
    code, _, err = run(capsys, "compare", "sh1", "sh2", path, "--min-pressure", 5, "-q", *cache)
    assert code == 1 and "--min-pressure needs the pressure" in err


def test_quiet_still_says_why_no_file_could_be_used(tmp_path, capsys, cache):
    (tmp_path / "a.p").write_bytes(b"not a P-file at all, but long enough to have a header" * 3)
    (tmp_path / "b.p").write_bytes(b"")
    code, out, err = run(capsys, "compare", "sh1", "sh2", tmp_path, "-q", *cache)
    assert code == 1 and not out
    assert "a.p: Could not read this file" in err and "b.p: Empty file" in err


def test_option_before_the_channels(capsys):
    with pytest.raises(SystemExit):
        main(["compare", "--threshold-counts", "23.7", "T1_dT1", "T2_dT2", "data/"])
    err = capsys.readouterr().err
    assert "'T1_dT1' is not a number" in err and "channel names and the files first" in err


def test_min_pressure_on_a_file_without_pressure(maker, tmp_path, capsys, cache):
    path = write_set(maker, tmp_path / "data", [0])[0]
    raw = path.read_bytes()
    path.write_bytes(raw.replace(b"name        = P\r\n", b"name        = Q\r\n"))
    code, _, err = run(capsys, "compare", "sh1", "sh2", path, "--min-pressure", 5, *cache)
    assert code == 1 and "needs a pressure channel (P)" in err


def test_a_channel_named_like_a_folder_here(maker, tmp_path, capsys, cache, monkeypatch):
    write_set(maker, tmp_path / "data", [0, 0])
    (tmp_path / "sh1").mkdir()
    monkeypatch.chdir(tmp_path)
    code, _, err = run(capsys, "compare", "sh1", "sh2", "data", *cache)
    assert code == 0, err
    code, _, err = run(capsys, "compare", "data", "sh2", "data", *cache)
    assert code == 1 and "also the name of a file or folder here" in err


def test_unsuitable_channels_are_pointed_out(maker, tmp_path, capsys, cache):
    write_set(maker, tmp_path / "data", [0, 0])
    code, out, _ = run(capsys, "compare", "Incl_X", "Incl_Y", tmp_path / "data", *cache)
    assert code == 0 and "Incl_X is a slow channel (64 samples per second)" in out
    code, out, _ = run(capsys, "compare", "JAC_C", "JAC_T", DATA / "VMP_SN479_dropout.p",
                       "--min-minutes", 0.1, *cache)
    assert code == 0 and "JAC_C is a 32-bit channel" in out
    assert "hit its limit" not in out  # JAC_T is unsigned: 0 to 65535, not -32768 to 32767


def write_bench_bursts(maker, folder, t1_bursts, t2_bursts, seconds=SECONDS):
    """Bench files; per file the burst positions (samples) on T1_dT1 and on T2_dT2."""
    folder.mkdir(exist_ok=True)
    n = maker.fast_samples(seconds)
    for k, (w1, w2) in enumerate(zip(t1_bursts, t2_bursts, strict=True)):
        channels = bench_channels(maker, seconds, seed=k)
        channels |= {"T1_dT1": add_bursts(noise(n, 3.0, -20, seed=k), w1),
                     "T2_dT2": add_bursts(noise(n, 3.0, -20, seed=100 + k), w2)}
        maker.write(folder / f"BENCH_{k + 1:04d}.p", start=T0 + timedelta(hours=k),
                    seconds=seconds, channels=channels)


def test_shaking_is_told_from_a_faulty_channel(maker, tmp_path, capsys, cache):
    n = maker.fast_samples(SECONDS)
    where = [list(np.linspace(1000, n - 1000, 8).astype(int))] * 3
    write_bench_bursts(maker, tmp_path / "shaken", where, where)  # the same moments: shaking
    csv_path = tmp_path / "shaken.csv"
    code, out, _ = run(capsys, "compare", "T1_dT1", "T2_dT2", tmp_path / "shaken", "--csv",
                       csv_path, *cache)
    assert code == 0
    assert "100% of T1_dT1's spikes came within 0.1 s of one on T2_dT2" in out
    assert "Both channels spiked at the same moment about 240 times per hour" in out
    assert "may need attention" not in out
    row = next(csv.DictReader(csv_path.open()))
    assert (row["A_spikes_shared"], row["B_spikes_shared"], row["A_spikes_only_per_hour"]) == \
        ("8", "8", "0.0")
    write_bench_bursts(maker, tmp_path / "faulty", where, [[]] * 3)  # T1 alone: a fault
    code, out, _ = run(capsys, "compare", "T1_dT1", "T2_dT2", tmp_path / "faulty", *cache)
    assert code == 0
    assert "0% of T1_dT1's spikes came within 0.1 s" in out
    assert "T1_dT1 had about 240 spikes per hour that the other channel did not have" in out
    assert "may need attention" in out and "spiked at the same moment" not in out


def test_one_spike_in_a_short_file_is_not_unusual(maker, tmp_path, capsys, cache):
    """1 spike in 2 minutes is 30 per hour, but one spike is easily luck."""
    n = maker.fast_samples(SECONDS)
    bursts = [[] for _ in range(12)]
    bursts[6] = [n // 2]
    write_bench_bursts(maker, tmp_path / "data", bursts, [[]] * 12)
    code, out, _ = run(capsys, "compare", "T1_dT1", "T2_dT2", tmp_path / "data", *cache)
    assert code == 0 and "UNUSUAL" not in out
    bursts[6] = list(np.linspace(1000, n - 1000, 10).astype(int))  # 10 in 2 minutes is not luck
    write_bench_bursts(maker, tmp_path / "data2", bursts, [[]] * 12)
    code, out, _ = run(capsys, "compare", "T1_dT1", "T2_dT2", tmp_path / "data2", *cache)
    assert code == 0 and "UNUSUAL: file 7" in out


def test_both_noises_rising_is_called_a_change_in_the_signal(maker, tmp_path, capsys, cache):
    """Live data: sh1 and sh2 both get far noisier (rough water); their ratio moves too."""
    folder = tmp_path / "live"
    folder.mkdir()
    n = maker.fast_samples(SECONDS)
    for k in range(16):
        rough = k >= 8
        sh1 = noise(n, 10.0 if rough else 1.0, seed=k)
        sh2 = noise(n, 7.0 if rough else 1.0, seed=50 + k)
        maker.write(folder / f"LIVE_{k + 1:04d}.p", start=T0 + timedelta(hours=k),
                    seconds=SECONDS, channels={"sh1": sh1, "sh2": sh2})
    code, out, _ = run(capsys, "compare", "sh1", "sh2", folder, *cache)
    assert code == 0 and "Mode:       live" in out
    assert "JUMP between file 8 and file 9" in out
    assert "Both channels' noise rose together" in out and "not by itself a fault" in out


def test_a_probe_that_dies_is_not_the_clean_channel(maker, tmp_path, capsys, cache):
    """As Hydrographer MR330's T2 from file 107: T2 at the end of its conversion, T2_dT2
    pinned. Its zero spikes must not make a jump that blames T1."""
    folder = tmp_path / "bench"
    folder.mkdir()
    n = maker.fast_samples(SECONDS)
    where = list(np.linspace(1000, n - 1000, 10).astype(int))
    for k in range(16):
        channels = bench_channels(maker, SECONDS, seed=k)
        channels["T1_dT1"] = add_bursts(noise(n, 3.0, -20, seed=k), where)
        if k < 8:
            channels["T2_dT2"] = add_bursts(noise(n, 3.0, -20, seed=100 + k),
                                            [w + 500 for w in where])
        else:  # no probe: four values near the end of the range
            channels["T2"] = np.full(SECONDS * 64, counts_for_temperature(maker, "T2", -16.05))
            channels["T2_dT2"] = -32347 + np.random.default_rng(k).integers(0, 4, n)
        maker.write(folder / f"B_{k + 1:04d}.p", start=T0 + timedelta(hours=k), seconds=SECONDS,
                    channels=channels)
    csv_path = tmp_path / "out.csv"
    code, out, err = run(capsys, "compare", "T1_dT1", "T2_dT2", folder, "--csv", csv_path, *cache)
    assert code == 0, err
    assert "JUMP" not in out
    assert "16 P-files" in out and "8 analysed, 8 left out (a channel could not be measured)" in out
    assert "Channels that could not be measured" in out
    assert "T2_dT2: no probe in 8 file(s) (#9-16). T2 read " in out
    assert "left out: B no probe (T2 reads " in out
    rows = list(csv.DictReader(csv_path.open()))
    assert rows[8]["analysed"] == "no" and rows[8]["B_not_measurable"].startswith("no probe: T2")
    assert rows[0]["analysed"] == "yes" and rows[0]["B_not_measurable"] == ""
    assert "T2 no probe (" in rows[8]["setup_evidence"]


def test_a_railed_channel_is_not_the_clean_one(maker, tmp_path, capsys, cache):
    """As VMP SN 465's sh1 in the overnight hangs: at +-32767 in a share of its samples."""
    folder = tmp_path / "live"
    folder.mkdir()
    n = maker.fast_samples(SECONDS)
    for k in range(12):
        sh1 = noise(n, 1.1, seed=k)
        if 4 <= k < 8:  # 5% of the samples at the end of the range, in 30 stretches
            for a in np.linspace(1000, n - 1000, 30).astype(int):
                sh1[a:a + n // 600] = 32767
        maker.write(folder / f"L_{k + 1:04d}.p", start=T0 + timedelta(hours=k), seconds=SECONDS,
                    channels={"sh1": sh1, "sh2": noise(n, 1.1, seed=50 + k)})
    code, out, _ = run(capsys, "compare", "sh1", "sh2", folder, *cache)
    assert code == 0 and "Mode:       live" in out
    assert "sh1: railed in 4 file(s) (#5-8): at the end of its range in up to 5% of the" in out
    assert "--min-pressure (for example --min-pressure 5) leaves that time out" in out
    assert "left out: A railed (5% of its samples at the end of its range)" in out
    assert "hit its limit" not in out
    assert "JUMP" not in out and "UNUSUAL" not in out


def test_a_gap_full_of_left_out_files_is_not_called_empty(maker, tmp_path, capsys, cache):
    """Files 5-8 have no T2 probe and lie 3 days apart: the compared files have a gap
    there, but the instrument did record."""
    folder = tmp_path / "bench"
    folder.mkdir()
    n = maker.fast_samples(SECONDS)
    start = T0
    for k in range(12):
        channels = bench_channels(maker, SECONDS, seed=k)
        channels |= {"T1_dT1": noise(n, 3.0, -20, seed=k),
                     "T2_dT2": noise(n, 3.0, -20, seed=50 + k)}
        if 4 <= k < 8:
            channels["T2"] = np.full(SECONDS * 64, counts_for_temperature(maker, "T2", -16.05))
            channels["T2_dT2"] = -32347 + np.random.default_rng(k).integers(0, 4, n)
        maker.write(folder / f"B_{k + 1:04d}.p", start=start, seconds=SECONDS, channels=channels)
        start += timedelta(days=3) if 3 <= k < 8 else timedelta(hours=1)
    code, out, _ = run(capsys, "compare", "T1_dT1", "T2_dT2", folder, *cache)
    assert code == 0
    assert "The files compared fall in 2 separate stretches of time" in out
    assert "The files fall in" not in out


def test_files_with_no_usable_data_say_so(maker, tmp_path, capsys, cache):
    """--min-pressure deeper than any file: no 'nan', no ratio, no checklist length complaint."""
    write_bench_set(maker, tmp_path / "bench", [0, 0])
    code, out, _ = run(capsys, "compare", "T1_dT1", "T2_dT2", tmp_path / "bench",
                       "--min-pressure", 100, *cache)
    assert code == 0
    rows = [ln for ln in out.splitlines() if "BENCH_000" in ln and ln.split()[0].isdigit()]
    assert len(rows) == 2
    for row in rows:
        assert "nan" not in row and "none" not in row and "shorter than" not in row
        assert row.split()[4:11] == ["0.00", "-", "-", "-", "-", "-", "-"]
        assert "no usable data, left out (never deeper than 100 dbar?)" in row


def test_a_fault_that_comes_and_goes_is_called_switching(maker, tmp_path, capsys, cache):
    """T1 busy for 3 files, quiet for 3, and so on: too quick for a jump."""
    n = maker.fast_samples(SECONDS)
    busy = list(np.linspace(1000, n - 1000, 40).astype(int))
    bursts = [busy if k // 3 % 2 else [busy[0]] for k in range(24)]
    write_bench_bursts(maker, tmp_path / "data", bursts, [[]] * 24)
    csv_path = tmp_path / "out.csv"
    code, out, _ = run(capsys, "compare", "T1_dT1", "T2_dT2", tmp_path / "data", "--csv",
                       csv_path, *cache)
    assert code == 0 and "JUMP" not in out
    assert "SWITCHING, files 1 to 24" in out
    assert "The spike ratio went back and forth 7 times between about 1,201 (12 files) and" \
        " about 31 (12 files)" in out
    assert "T1_dT1: about 1,200 spikes per hour in the high files and 30 in the low ones;" \
        " T2_dT2: about 0 and 0." in out
    rows = list(csv.DictReader(csv_path.open()))
    assert rows[0]["findings"] == "low in a switching spike ratio"
    assert rows[3]["findings"] == "high in a switching spike ratio"
