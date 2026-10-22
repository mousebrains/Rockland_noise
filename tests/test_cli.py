"""The whole program, run as a technician would, on synthetic P-files."""

import csv
from datetime import timedelta

import numpy as np
import pytest

from rockland_noise.cli import main

from .conftest import T0, add_bursts, noise

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
    assert "T1_dT1: within" in out and "OUTSIDE the checklist in 3 of 3 files" in out
    assert "T2_dT2: within" in out and "Within the checklist in all 3 files" in out
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
    _, _, err = run(capsys, *args, "--mode", "bench")
    assert "remembered" not in err


def test_min_pressure_uses_only_deep_data(maker, tmp_path, capsys, cache):
    """P is above 20 dbar for the second half only; only those spikes count."""
    cal = maker.pf.channel_info["P"]["cal"]
    deep = (20 - cal["coef0"]) / cal["coef1"]  # counts for 20 dbar (ignoring the tiny c2)
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
