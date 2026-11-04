"""The calibration and pilot code behind the locked E4 and E5 predictions: its rules agree with the repo's, its logic is pinned by
small cases, and the committed raw data still reproduces the numbers the predictions quoted. None of this is evidence."""
from __future__ import annotations

import json
import math
from pathlib import Path
import random
import statistics

import pytest

from phonebook import analyze_capacity as ac
from phonebook.calibration import e4_calibration as cal
from phonebook.calibration import e4_montecarlo as mc
from phonebook.calibration import e5_pilot_report as rep
from phonebook.calibration import e5_pilots as pil

ROOT = Path(__file__).resolve().parents[1]
CAL = ROOT / "phonebook" / "calibration"


def row(d, n, seed, epochs, first, final=0.0, control=0.0):
    return {"d": d, "n": n, "seed": seed, "epochs": epochs, "first_perfect": first, "final_train_seq": final,
            "best_train_seq": final, "max_control": control, "seconds": 1.0, "params": 3072, "curve": [[10, final]]}


def best_of(rows):
    return {(r["d"], r["n"], r["seed"]): r for r in rows}


# ------------------------------------------------------------------ the Monte Carlo prior

def test_censored_runs_count_as_the_assumed_completion_epochs_and_short_runs_say_nothing() -> None:
    best = best_of([row(8, 10, 100, 1000, None, final=0.9), row(8, 20, 100, 1000, None, final=0.1),
                    row(8, 40, 100, 300, None, final=0.0), row(8, 80, 100, 1000, 700), row(16, 10, 100, 1000, 100)])
    assert mc.observed(best, 8) == {10: mc.STALL, 20: 2 * mc.STALL, 80: 700.0}      # the 300-epoch run and the other width are ignored


def test_the_completion_curve_is_log_log_linear_never_decreasing_and_extrapolates_at_least_linearly() -> None:
    m = mc.curve(best_of([row(8, 10, 100, 1000, 100), row(8, 20, 100, 1000, 400), row(8, 40, 100, 1000, 300)]), 8)
    assert m(10) == pytest.approx(100) and m(20) == pytest.approx(400)
    assert m(10 * math.sqrt(2)) == pytest.approx(200)                              # halfway in log n is halfway in log epochs
    assert m(5) == pytest.approx(100)                                              # below the first rung: flat
    assert m(40) == pytest.approx(400)                                             # the dip to 300 is lifted to keep the curve non-decreasing
    assert m(80) == pytest.approx(800)                                             # beyond the last rung: at least linear in n


def test_the_monte_carlo_capacity_rule_is_the_rule_of_analyze_capacity() -> None:
    rng = random.Random(3)
    ladder = [10, 20, 40, 80]
    for _ in range(400):
        times = {n: sorted(rng.choice([rng.uniform(50, 1200), math.inf]) for _ in range(3)) for n in ladder}
        budget = rng.choice([100, 300, 500, 1000])
        cells = [{"n": n, "seed": s, "first_perfect": None if t == math.inf else t} for n in ladder for s, t in enumerate(times[n])]
        # `times` is sorted per rung, so seed identity is irrelevant to the rule: it only counts how many are in time
        expected = ac.capacity(cells, budget=budget, min_seeds=2)["capacity"] or 0
        assert mc.capacity_from_times(times, ladder, budget) == expected


def test_simulation_is_reproducible_and_every_trial_is_counted() -> None:
    best = mc.load_results(CAL / "e4_calibration_results.jsonl")
    first = mc.simulate(best, 16, 0.4, trials=500, seed=5)
    again = mc.simulate(best, 16, 0.4, trials=500, seed=5)
    assert first[0] == again[0] and first[1] == again[1]
    assert all(sum(counter.values()) == 500 for counter in first[0].values()) and sum(first[1].values()) == 500


def test_the_committed_calibration_still_reproduces_the_numbers_behind_the_e4_predictions() -> None:
    best = mc.load_results(CAL / "e4_calibration_results.jsonl")
    assert len(best) == 46 and len((CAL / "e4_calibration_results.jsonl").read_text(encoding="utf-8").splitlines()) == 57   # 300-epoch runs collapse into their 1000-epoch twins
    expected = {8: {1000: {10: 0.10, 14: 0.57, 20: 0.32}}, 64: {1000: {320: 0.21, 453: 0.74, 640: 0.05}}}
    for d, by_budget in expected.items():
        caps, boundary, _ = mc.simulate(best, d, 0.4, trials=20000, seed=7)
        for budget, shares in by_budget.items():
            for rung, share in shares.items():
                assert caps[budget][rung] / 20000 == pytest.approx(share, abs=0.006), (d, budget, rung)
        assert max(caps[1000], key=caps[1000].get) == {8: 14, 64: 453}[d]
    _, boundary32, _ = mc.simulate(best, 32, 0.4, trials=20000, seed=7)
    assert boundary32["stable"] / 20000 == pytest.approx(0.48, abs=0.006)         # the d=32 coin flip behind the "stable" prediction


# ------------------------------------------------------------------ the E4 calibration runner

def test_collapse_keeps_the_longest_run_of_each_cell() -> None:
    rows = [row(8, 10, 100, 300, None), row(8, 10, 100, 1000, 520), row(8, 10, 101, 300, 450), row(16, 10, 100, 300, 230)]
    best = cal.collapse(rows)
    assert len(best) == 3 and best[(8, 10, 100)]["epochs"] == 1000 and best[(8, 10, 100)]["first_perfect"] == 520


def test_the_summary_marks_short_runs_and_cells_that_never_finished() -> None:
    text = cal.summary([row(8, 10, 100, 1000, 520), row(8, 10, 101, 1000, 450), row(8, 20, 100, 1000, None, final=0.95), row(16, 10, 100, 300, 230)])
    lines = text.splitlines()
    assert "d=8   n=10: 520/450 | n=20: no(0.95)" in lines[1] and lines[2].startswith("d=16  n=10: 230*")


def test_cells_parse_and_a_malformed_cell_is_rejected() -> None:
    assert cal.parse_cells("8:10:1000,64:453:1000:101") == [(8, 10, 1000), (64, 453, 1000, 101)]
    for bad in ("8:10", "8:10:1000:1:2", "a:b:c"):
        with pytest.raises(ValueError):
            cal.parse_cells(bad)


def test_a_failing_cell_is_reported_and_never_recorded(tmp_path: Path, capsys) -> None:
    def runner(work, d, n, epochs, seed=100):
        if n == 20:
            return {"d": d, "n": n, "epochs": epochs, "error": "boom"}
        if n == 40:
            raise RuntimeError("crash")
        return row(d, n, seed, epochs, 100, final=1.0)

    results = tmp_path / "results.jsonl"
    failed = cal.run_batch(tmp_path, [(8, 10, 300), (8, 20, 300), (8, 40, 300), (8, 80, 300, 101)], workers=2, results=results, runner=runner)
    recorded = [json.loads(line) for line in results.read_text(encoding="utf-8").splitlines()]
    assert failed == 2 and sorted(r["n"] for r in recorded) == [10, 80] and recorded[0]["seed"] in (100, 101)
    out = capsys.readouterr().out
    assert "FAILED cell (8, 20, 300): boom" in out and "FAILED cell (8, 40, 300)" in out


def test_the_extra_rungs_are_prefixes_of_the_top_book_and_existing_files_are_never_overwritten(tmp_path: Path) -> None:
    rungs = cal.write_calibration_books(tmp_path, seed=100, n_max=160, extras=(14, 57, 113))
    assert rungs == [10, 14, 20, 40, 57, 80, 113, 160]
    top = cal.book_path(tmp_path, 160).read_text(encoding="ascii").splitlines()
    for n in (14, 57, 113):
        assert cal.book_path(tmp_path, n).read_text(encoding="ascii").splitlines() == top[:n]
    assert (tmp_path / "capacity_seed100_control.txt").exists()
    with pytest.raises(ValueError):
        cal.write_calibration_books(tmp_path, seed=100, n_max=160, extras=(14,))
    with pytest.raises(ValueError, match="already on the ladder or outside the top book"):
        cal.write_calibration_books(tmp_path / "other", seed=100, n_max=160, extras=(20,))            # 20 is already on the doubling ladder
    for outside in (0, 160, 999):                                                                    # at or beyond the top book, or empty
        with pytest.raises(ValueError, match="already on the ladder or outside the top book"):
            cal.write_calibration_books(tmp_path / f"outside{outside}", seed=100, n_max=160, extras=(outside,))
    with pytest.raises(ValueError, match="already exists"):
        cal.write_calibration_books(tmp_path / "dup", seed=100, n_max=160, extras=(14, 14))              # the same extra rung twice


def test_the_summary_command_reads_the_committed_results(capsys) -> None:
    assert cal.main(["summary", "--results", str(CAL / "e4_calibration_results.jsonl")]) == 0
    assert "d=64" in capsys.readouterr().out


def test_a_malformed_cell_list_stops_the_command_with_a_message(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        cal.main(["run", "--work", str(tmp_path), "--cells", "bad"])


# ------------------------------------------------------------------ the E5 pilots

def test_job_grids_have_the_documented_size_and_unique_keys() -> None:
    main_jobs, lr003 = pil.main_jobs([15, 17]), pil.lr003_jobs([15, 17])
    assert len(main_jobs) == 2 * (18 + 4 + 12) and len(lr003) == 2 * 12
    assert len({pil.job_key(j) for j in main_jobs}) == len(main_jobs) and len({pil.job_key(j) for j in lr003}) == len(lr003)
    assert pil.job_key((15, 42, 3, "0.001", 42, 120)) == "15|42|3|0.001|42|120"
    assert {j[3] for j in main_jobs} == {"0.001", "0.0003"} and {j[3] for j in lr003} == {"0.003"}
    extra = [j for j in main_jobs if j[4] != 42]
    assert sorted({(j[1], j[2], j[4]) for j in extra}) == [(42, 1, 1), (42, 1, 2), (42, 3, 1), (42, 3, 2)]


def test_the_job_grids_are_exactly_the_cells_of_the_recorded_result_files() -> None:
    """Epochs, seeds, ks and learning rates of each grid are pinned by the runs that actually exist."""
    recorded = {name: set(json.loads((CAL / "e5_pilot_results" / name).read_text(encoding="utf-8"))) for name in
                ("phonebooks_a.json", "phonebooks_b.json", "phonebooks_lr0.003.json")}
    assert recorded["phonebooks_a.json"] == {pil.job_key(j) for j in pil.main_jobs([15, 17, 18, 20])}
    assert recorded["phonebooks_b.json"] == {pil.job_key(j) for j in pil.main_jobs([26, 32, 34, 36, 38, 39])}
    assert recorded["phonebooks_lr0.003.json"] == {pil.job_key(j) for j in pil.lr003_jobs([15, 17, 18, 20, 26, 32, 34, 36, 38, 39])}


def test_a_checkpoint_that_differs_from_e3s_pin_is_refused(monkeypatch) -> None:
    monkeypatch.setattr(pil, "sha256", lambda path: "0" * 64)
    with pytest.raises(ValueError, match="differs from E3's pinned hash"):
        pil.verify_d0_pins()


def test_the_e3_checkpoints_on_disk_match_their_pins() -> None:
    if not all((ROOT / p).exists() for p in pil.D0_CHECKPOINTS.values()):
        pytest.skip("the E3 Phase-A checkpoints are local files and are absent from this checkout")
    pil.verify_d0_pins()


def test_summarize_metrics_reads_joint_recall_and_the_dip(tmp_path: Path) -> None:
    header = "Epoch,Train Loss,Val Loss,Train Token Acc,Val Token Acc,Train Seq Acc,Val Seq Acc,Training Time (sec),Param Count\n"
    rows = [(1, 8, 0), (2, 10, 5), (3, 10, 10), (4, 10, 10)]          # (epoch, entries of A recalled, entries of B recalled), k = 1
    body = "".join(f"{e},1.0,1.0,0.5,0.5,{(a + b) / 20:.4f},{a / 10:.4f},0.1,110528\n" for e, a, b in rows)
    path = tmp_path / "metrics.csv"
    path.write_text(header + body, encoding="utf-8")
    assert pil.summarize_metrics(path, 1) == {"stable_joint": 3, "first_joint": 3, "first_b": 3, "min_a": 8}


# ------------------------------------------------------------------ the E5 pilot report

def test_spreads_skip_single_seed_groups_and_reject_a_censored_group() -> None:
    assert rep.spreads({"a": [40, 40, 41], "b": [50], "c": [30, 30]}) == [1, 0]
    with pytest.raises(ValueError):
        rep.spreads({"a": [40, None]})


def test_the_pilot_claims_behind_the_e5_predictions_still_reproduce() -> None:
    files = rep.load_files(rep.RESULTS_DIR)
    noise = rep.noise_sets(files)
    assert sorted(len(v) for v in noise.values()) == [6, 12, 18, 20]                # 56 groups of several shuffle seeds ...
    assert all(max(v) == 0 for v in noise.values())                                 # ... and none moved the joint-recall epoch
    census = rep.censoring(files)
    assert census["10 other phonebooks, lr 0.001 and 0.0003"] == {"cells": 340, "censored": 0, "flicker": 2}
    assert census["10 other phonebooks, lr 0.003"] == {"cells": 120, "censored": 0, "flicker": 0}
    assert census["d0, all files"] == {"cells": 150, "censored": 0, "flicker": 0}


def test_the_phonebook_tables_reproduce_the_pooled_counts_behind_the_predictions() -> None:
    pbs, tab, dip, tab3 = rep.build_tables(rep.load_files(rep.RESULTS_DIR))
    assert len(pbs) == 11 and pbs[0] == "d0"
    assert [tab["d0"][k] for k in rep.KS] == [41, 40, 39, 38, 44, 54] and [tab["pb15"][k] for k in rep.KS] == [40, 41, 43, 50, 55, 67]
    faster = {k: sum(tab[p][1] > tab[p][k] for p in pbs) for k in rep.KS[1:]}
    assert faster == {2: 9, 3: 4, 4: 1, 5: 0, 9: 0}
    assert statistics.median(tab[p][1] for p in pbs) == 36 and statistics.median(dip[p][1] for p in pbs) == 9
    assert all(tab3[p][2] < tab3[p][1] for p in tab3) and len(tab3) == 11


def test_the_best_k_per_phonebook_lines_reproduce_the_ones_the_predictions_used() -> None:
    lines = rep.pooled_lines(*rep.build_tables(rep.load_files(rep.RESULTS_DIR)))
    assert lines[-2] == ("  best k per pb at 0.0003: {'d0': 5, 'pb15': 2, 'pb17': 2, 'pb18': 2, 'pb20': 3, 'pb26': 3, 'pb32': 3, "
                         "'pb34': 3, 'pb36': 2, 'pb38': 3, 'pb39': 3}")
    assert lines[-1] == ("  best k per pb at 0.001 : {'d0': 4, 'pb15': 1, 'pb17': 2, 'pb18': 1, 'pb20': 2, 'pb26': 2, 'pb32': 2, "
                         "'pb34': 2, 'pb36': 2, 'pb38': 2, 'pb39': 2}")


def test_the_tables_use_shuffle_seed_42_and_no_other() -> None:
    def cell(value: int) -> dict:
        return {"stable_joint": value, "first_joint": value, "first_b": value, "min_a": 9}

    d0, d0_3, pb_a = {}, {}, {}
    for init in (42, 123, 999):
        for k in rep.KS:
            d0[f"d0_seed{init}|k{k}|s42"] = cell(100 + k)
            pb_a[f"99|{init}|{k}|0.001|42|120"] = cell(10 * k)
        for k in rep.KS3:
            d0_3[f"d0_seed{init}|k{k}|s42"] = cell(200 + k)
            pb_a[f"99|{init}|{k}|0.0003|42|200"] = cell(20 * k)
        for shuffle in (1, 2):                                  # other shuffle seeds, written after the seed-42 cells, must not win
            for k in (1, 3):
                d0[f"d0_seed{init}|k{k}|s{shuffle}"] = cell(7777)
                d0_3[f"d0_seed{init}|k{k}|s{shuffle}"] = cell(8888)
                pb_a[f"99|{init}|{k}|0.001|{shuffle}|120"] = cell(9999)
                pb_a[f"99|{init}|{k}|0.0003|{shuffle}|200"] = cell(6666)
    for init in (42, 123, 999):                                 # pb98: all three inits at lr 0.001 but only two at lr 0.0003
        for k in rep.KS:
            pb_a[f"98|{init}|{k}|0.001|42|120"] = cell(30 * k)
        for k in rep.KS3:
            if init != 999:
                pb_a[f"98|{init}|{k}|0.0003|42|200"] = cell(40 * k)
    files = {"d0_1": d0, "d0_3": d0_3, "d0_mt": {}, "pb_a": pb_a, "pb_b": {}, "pb_003": {}}
    pbs, tab, dip, tab3 = rep.build_tables(files)
    assert pbs == ["d0", "pb99", "pb98"] and sorted(tab3) == ["d0", "pb99"]       # a median over two inits is not comparable, so pb98 is left out of the lr 0.0003 table
    assert tab["d0"] == {k: 100 + k for k in rep.KS} and tab["pb99"] == {k: 10 * k for k in rep.KS}
    assert tab3["d0"] == {k: 200 + k for k in rep.KS3} and tab3["pb99"] == {k: 20 * k for k in rep.KS3}
    assert dip["pb99"] == {k: 9 for k in rep.KS}


def test_the_bootstrap_is_reproducible_and_matches_the_distribution_the_predictions_used() -> None:
    pbs, tab, dip, tab3 = rep.build_tables(rep.load_files(rep.RESULTS_DIR))
    assert rep.bootstrap(pbs, tab, dip, tab3, trials=300, seed=2) == rep.bootstrap(pbs, tab, dip, tab3, trials=300, seed=2)
    lines = rep.bootstrap(pbs, tab, dip, tab3, trials=20000, seed=1)
    assert lines[1] == " fastest k: {1: 0.168, 2: 0.8, 3: 0.012, 4: 0.02}"
    assert lines[3].startswith(" P(k beats k=1 under rule, band 0): {2: 0.699, 3: 0.027}")
    assert lines[-2] == " plateau (band 0): {(2,): 0.742, (1, 2): 0.092, (1,): 0.058, (2, 3): 0.049, (4,): 0.02, (1, 2, 3): 0.017}"
    assert lines[-1] == " best k at 0.0003 vs 0.001: {'same': 0.458, 'larger': 0.521, 'smaller': 0.021}"


def test_the_report_runs_end_to_end() -> None:
    text = rep.report(trials=200, seed=1)
    for heading in ("== 1.", "== 2.", "== 3.", "== 4."):
        assert heading in text
    assert "0 with a nonzero spread" in text
