from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest

from phonebook import analyze_replay_k as ar

INF = math.inf


# ------------------------------------------------------------------ work

def test_work_is_updates_times_sequences_per_update() -> None:
    assert ar.work_sequences(41, 1) == 41 * 20      # 10 of A + 10 of B
    assert ar.work_sequences(39, 3) == 39 * 40      # 10 of A + 30 of B
    assert ar.work_sequences(10, 9, n_a=10, n_b=10) == 10 * 100


def test_a_run_that_never_finished_has_no_work() -> None:
    assert ar.work_sequences(None, 3) is None


# ------------------------------------------------------------------ cells and medians

def cell(pb, init, k, stable, *, lr=0.001, seed=42, min_a=8, plain=False):
    return {"phonebook": pb, "init": init, "seed": seed, "lr": lr, "k": None if plain else k, "plain": plain,
            "stable_joint": stable, "first_joint": stable, "min_a": min_a}


def test_phonebook_medians_take_the_median_over_inits_and_count_censored_runs_as_infinitely_slow() -> None:
    cells = [cell(2, 42, 1, 40), cell(2, 123, 1, 50), cell(2, 999, 1, 60),
             cell(2, 42, 3, None), cell(2, 123, 3, 30), cell(2, 999, 3, 35),
             cell(4, 42, 3, None), cell(4, 123, 3, None), cell(4, 999, 3, 30)]
    table = ar.phonebook_medians(cells, lr=0.001, seed=42)
    assert table[2][1] == 50 and table[2][3] == 35
    assert table[4][3] == INF


def test_phonebook_medians_use_only_the_requested_learning_rate_and_shuffle_seed() -> None:
    cells = [cell(2, 42, 1, 40), cell(2, 42, 1, 99, seed=7), cell(2, 42, 1, 77, lr=0.003)]
    assert ar.phonebook_medians(cells, lr=0.001, seed=42) == {2: {1: 40}}


def test_the_plain_fine_tuning_control_never_enters_the_k_table() -> None:
    cells = [cell(2, 42, 1, 40), cell(2, 42, None, 33, plain=True)]
    assert ar.phonebook_medians(cells, lr=0.001, seed=42) == {2: {1: 40}}


def test_a_repeated_run_for_one_unit_and_seed_is_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        ar.phonebook_medians([cell(2, 42, 1, 40), cell(2, 42, 1, 41)], lr=0.001, seed=42)


# ------------------------------------------------------------------ paired comparison

TABLE = {2: {1: 50, 3: 35}, 4: {1: 60, 3: 66}, 7: {1: INF, 3: 70}, 8: {1: 40, 3: INF}, 9: {1: INF, 3: INF}}


def test_paired_gain_is_updates_saved_against_k_equals_one_within_each_phonebook() -> None:
    assert ar.paired_gains(TABLE, 3) == {2: 15, 4: -6, 7: INF, 8: -INF, 9: 0}


def test_sign_counts_split_faster_tied_and_slower() -> None:
    assert ar.sign_counts([15, -6, INF, -INF, 0]) == {"faster": 2, "tied": 1, "slower": 2}


def test_a_phonebook_missing_the_baseline_or_the_k_is_rejected() -> None:
    with pytest.raises(ValueError, match="phonebook 5"):
        ar.paired_gains({5: {3: 10}}, 3)
    with pytest.raises(ValueError, match="phonebook 5"):
        ar.paired_gains({5: {1: 10}}, 3)


def test_k_beats_one_only_when_consistent_and_larger_than_the_noise() -> None:
    gains = [5, 6, 7, 8, 9, 10]
    assert ar.beats_baseline(gains, band=4, min_phonebooks=5)["beats"] is True
    assert ar.beats_baseline(gains, band=8, min_phonebooks=5)["beats"] is False          # median 7.5 is inside the noise
    assert ar.beats_baseline([50, 50, 50, 50, -1, -1], band=0, min_phonebooks=5)["beats"] is False  # not consistent


def test_the_boundary_cases_of_the_decision_rule_are_strict() -> None:
    five_of_six = [5, 6, 7, 8, 9, -1]                  # median 6.5
    assert ar.beats_baseline(five_of_six, band=6, min_phonebooks=5)["beats"] is True
    assert ar.beats_baseline(five_of_six, band=6.5, min_phonebooks=5)["beats"] is False   # median must exceed, not equal
    assert ar.beats_baseline([5, 6, 7, 8, 0, 0], band=0, min_phonebooks=5)["faster"] == 4  # ties are not faster


# ------------------------------------------------------------------ noise

def test_the_noise_band_is_the_largest_spread_among_identical_configurations() -> None:
    cells = [cell(2, 42, 1, 40, seed=1), cell(2, 42, 1, 43, seed=2), cell(2, 42, 1, 41, seed=3),
             cell(4, 42, 3, 50, seed=1), cell(4, 42, 3, 55, seed=2),
             cell(7, 42, 1, 99, seed=1)]               # a single seed has no spread
    noise = ar.noise_band(cells)
    assert noise["band"] == 5 and noise["groups"] == 2


def test_runs_at_different_learning_rates_or_k_are_different_configurations_not_noise() -> None:
    cells = [cell(2, 42, 1, 40, seed=1), cell(2, 42, 3, 80, seed=2), cell(2, 42, 1, 41, seed=3, lr=0.003)]
    with pytest.raises(ValueError, match="noise"):
        ar.noise_band(cells)


def test_a_censored_run_in_the_noise_arm_is_rejected_because_its_spread_is_unbounded() -> None:
    with pytest.raises(ValueError, match="censored"):
        ar.noise_band([cell(2, 42, 1, 40, seed=1), cell(2, 42, 1, None, seed=2)])


# ------------------------------------------------------------------ plateau, cheapest, dip

def test_plateau_is_every_k_within_the_noise_band_of_the_best() -> None:
    medians = {1: 41, 2: 40, 3: 39, 5: 44, 9: 54}
    assert ar.plateau(medians, band=2) == [1, 2, 3]
    assert ar.plateau(medians, band=0) == [3]
    assert ar.plateau({1: INF, 3: INF}, band=5) == []


def test_cheapest_k_in_work_breaks_ties_toward_the_smaller_k() -> None:
    assert ar.cheapest_k({1: 820, 2: 1200, 3: 1560}) == 1
    assert ar.cheapest_k({2: 5, 1: 5}) == 1


def test_the_dip_is_monotone_when_more_copies_never_keep_more_of_a() -> None:
    assert ar.dip_is_monotone({1: 8, 2: 5, 3: 3, 5: 1, 9: 1}) is True
    assert ar.dip_is_monotone({1: 8, 2: 3, 3: 5}) is False


# ------------------------------------------------------------------ whole analysis on synthetic cells

def synthetic(gain: int, *, spread: int = 2) -> list[dict]:
    cells = []
    for pb in (2, 4, 7, 8, 9, 12):
        for init in (42, 123, 999):
            base = 40 + pb + init % 3
            cells += [cell(pb, init, 1, base, min_a=8), cell(pb, init, 3, base - gain, min_a=3)]
    for seed in (1, 2, 3, 4, 5):                       # the noise arm: three units, k in {1, 3}
        for pb in (2, 4, 7):
            cells += [cell(pb, 42, 1, 40 + pb + 42 % 3 + (seed % 2) * spread, seed=seed),
                      cell(pb, 42, 3, 40 + pb + 42 % 3 - gain + (seed % 3 == 0) * spread, seed=seed)]
    return cells


def test_a_consistent_gain_above_the_noise_is_reported_as_beating_k_equals_one() -> None:
    result = ar.analyze(synthetic(gain=6), lrs=[0.001], primary_lr=0.001, ks=[1, 3], phonebooks=[2, 4, 7, 8, 9, 12])
    lr = result["lr"][0.001]
    assert result["noise"]["band"] == 2
    assert lr["paired"][3]["beats"] is True and lr["paired"][3]["faster"] == 6
    assert lr["median_stable_joint"][3] < lr["median_stable_joint"][1]
    assert lr["plateau"] == [3]
    assert lr["cheapest_k_in_work"] == 1                # 20 sequences per update beats 40 even when 3 needs fewer updates


def test_a_gain_inside_the_noise_is_not_reported_as_an_effect() -> None:
    lr = ar.analyze(synthetic(gain=2), lrs=[0.001], primary_lr=0.001, ks=[1, 3], phonebooks=[2, 4, 7, 8, 9, 12])["lr"][0.001]
    assert lr["paired"][3]["faster"] == 6 and lr["paired"][3]["beats"] is False
    assert lr["plateau"] == [1, 3]


def test_other_learning_rates_are_descriptive_and_get_no_verdict() -> None:
    cells = synthetic(gain=6) + [{**c, "lr": 0.003} for c in synthetic(gain=6) if c["seed"] == 42]
    result = ar.analyze(cells, lrs=[0.001, 0.003], primary_lr=0.001, ks=[1, 3], phonebooks=[2, 4, 7, 8, 9, 12])
    assert result["lr"][0.001]["paired"][3]["beats"] is True
    assert result["lr"][0.003]["paired"][3]["beats"] is None
    assert result["lr"][0.003]["paired"][3]["faster"] == 6


def test_the_analysis_refuses_the_wrong_set_of_phonebooks() -> None:
    with pytest.raises(ValueError, match="phonebooks"):
        ar.analyze(synthetic(gain=6), lrs=[0.001], primary_lr=0.001, ks=[1, 3], phonebooks=[2, 4, 7, 8, 9])


# ------------------------------------------------------------------ loading evidence

def fake_replay_evidence(root: Path, runs: list[dict], seed: int = 42) -> None:
    root.mkdir(parents=True)
    (root / "protocol.json").write_text(
        json.dumps({"spec": {"seed": seed}, "runs": [r["protocol"] for r in runs]}), encoding="utf-8")
    for r in runs:
        p = r["protocol"]
        folder = root / p["scenario"]
        (folder / p["condition"]).mkdir(parents=True, exist_ok=True)
        books = {"A": {"generation": {"total": 10, "correct": r["curve"][-1][1]}},
                 "B": {"generation": {"total": 10, "correct": r["curve"][-1][2]}}}
        (folder / f"{p['condition']}.evaluation.json").write_text(json.dumps({"books": books}), encoding="utf-8")
        a_copies, b_copies = p["train"]["a_copies"], p["train"]["b_copies"]
        with (folder / p["condition"] / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["Epoch", "Train Loss", "Val Loss", "Train Token Acc", "Val Token Acc",
                             "Train Seq Acc", "Val Seq Acc", "Training Time (sec)", "Param Count"])
            for epoch, a, b in r["curve"]:
                train = (a_copies * a + b_copies * b) / (10 * (a_copies + b_copies))
                writer.writerow([epoch, "1.0", "1.0", "0.5", "0.5", f"{train:.4f}", f"{a / 10:.4f}", "0.1", 110_528])


def replay_protocol_run(scenario: str, a_copies: int, b_copies: int, condition: str, lr: float = 0.001) -> dict:
    return {"id": f"{scenario}/{condition}", "scenario": scenario, "condition": condition,
            "mode": f"a{a_copies}b{b_copies}_evalA", "train": {"a_copies": a_copies, "b_copies": b_copies},
            "eval_book": "A", "lr": lr, "epochs": 4, "batch_size": 128, "accuracy_interval": 1,
            "l2sp_lambda": 0.0, "adapt": True, "n_train": 10 * (a_copies + b_copies), "updates_per_epoch": 1, "seq_len": 29}


def test_loading_reads_phonebook_init_shuffle_seed_k_and_the_stable_joint_epoch(tmp_path: Path) -> None:
    fake_replay_evidence(tmp_path / "root", [
        {"protocol": replay_protocol_run("pb2_s123", 1, 3, "k3_lr0.001"),
         "curve": [(1, 8, 2), (2, 9, 9), (3, 10, 10), (4, 10, 10)]},
        {"protocol": replay_protocol_run("pb2_s123", 0, 1, "plain_lr0.001"),
         "curve": [(1, 4, 6), (2, 0, 10), (3, 0, 10), (4, 0, 10)]},
    ])
    cells = {(c["plain"], c["k"]): c for c in ar.load_replay_cells(tmp_path / "root")}
    replay, plain = cells[(False, 3)], cells[(True, None)]
    assert (replay["phonebook"], replay["init"], replay["seed"], replay["lr"]) == (2, 123, 42, 0.001)
    assert replay["stable_joint"] == 3 and replay["first_joint"] == 3 and replay["min_a"] == 8
    assert plain["stable_joint"] is None and plain["min_a"] == 0


def test_a_replay_run_that_loses_a_book_again_is_stable_only_from_its_last_perfect_stretch(tmp_path: Path) -> None:
    fake_replay_evidence(tmp_path / "root", [
        {"protocol": replay_protocol_run("pb2_s42", 1, 1, "k1_lr0.001"),
         "curve": [(1, 10, 10), (2, 9, 10), (3, 10, 10), (4, 10, 10)]}])
    cell_ = ar.load_replay_cells(tmp_path / "root")[0]
    assert cell_["first_joint"] == 1 and cell_["stable_joint"] == 3


def test_more_than_one_copy_of_a_is_outside_the_experiment_and_rejected(tmp_path: Path) -> None:
    fake_replay_evidence(tmp_path / "root", [
        {"protocol": replay_protocol_run("pb2_s42", 2, 1, "x"), "curve": [(1, 10, 10), (2, 10, 10), (3, 10, 10), (4, 10, 10)]}])
    with pytest.raises(ValueError, match="copies of A"):
        ar.load_replay_cells(tmp_path / "root")


def test_a_scenario_id_outside_the_naming_scheme_is_rejected(tmp_path: Path) -> None:
    fake_replay_evidence(tmp_path / "root", [
        {"protocol": replay_protocol_run("weird", 1, 1, "x"), "curve": [(1, 10, 10), (2, 10, 10), (3, 10, 10), (4, 10, 10)]}])
    with pytest.raises(ValueError, match="scenario id"):
        ar.load_replay_cells(tmp_path / "root")


# ------------------------------------------------------------------ edge cases and report files

def test_an_undefined_median_gain_never_counts_as_an_effect() -> None:
    gains = [-INF, -INF, -INF, INF, INF, INF]          # the middle two average to nan
    assert ar.beats_baseline(gains, band=0, min_phonebooks=3)["beats"] is False


def test_report_files_are_strict_json_and_csv_even_with_censored_runs(tmp_path: Path) -> None:
    def refuse(constant):
        raise ValueError(f"non-strict JSON constant {constant}")

    cells = synthetic(gain=6) + [cell(12, 42, 5, None)]
    ar.write_report(cells, tmp_path, prefix="e5", lrs=[0.001], primary_lr=0.001, ks=[1, 3],
                    phonebooks=[2, 4, 7, 8, 9, 12])
    analysis = json.loads((tmp_path / "e5_analysis.json").read_text(encoding="utf-8"), parse_constant=refuse)
    assert analysis["lr"]["0.001"]["paired"]["3"]["beats"] is True
    with (tmp_path / "e5_cells.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == len(cells)
    censored = [row for row in rows if row["phonebook"] == "12" and row["k"] == "5"]
    assert len(censored) == 1 and censored[0]["stable_joint"] == ""


# ------------------------------------------------------------------ review fixes

def test_censored_runs_are_counted_per_k_so_the_median_cannot_hide_them() -> None:
    cells = synthetic(gain=6)
    victim = next(c for c in cells if (c["phonebook"], c["init"], c["k"], c["seed"]) == (8, 123, 1, 42))
    victim["stable_joint"] = None
    lr = ar.analyze(cells, lrs=[0.001], primary_lr=0.001, ks=[1, 3], phonebooks=[2, 4, 7, 8, 9, 12])["lr"][0.001]
    assert lr["censored"] == {1: 1, 3: 0}
    assert lr["phonebook_medians"][8][1] != INF            # two of three inits finished, so the median looks fine


def test_the_command_line_writes_strict_files(tmp_path: Path) -> None:
    def refuse(constant):
        raise ValueError(f"non-strict JSON constant {constant}")

    def runs(k1_curve):
        return [{"protocol": replay_protocol_run("pb2_s42", 1, 1, "k1_lr0.001"), "curve": k1_curve},
                {"protocol": replay_protocol_run("pb2_s42", 1, 3, "k3_lr0.001"),
                 "curve": [(1, 8, 2), (2, 10, 10), (3, 10, 10), (4, 10, 10)]}]

    fake_replay_evidence(tmp_path / "main", runs([(1, 8, 2), (2, 8, 5), (3, 10, 10), (4, 10, 10)]), seed=42)
    fake_replay_evidence(tmp_path / "noise", runs([(1, 8, 2), (2, 8, 5), (3, 8, 5), (4, 10, 10)]), seed=1)
    ar.main(["--root", str(tmp_path / "main"), "--root", str(tmp_path / "noise"), "--lrs", "0.001", "--primary-lr", "0.001",
             "--ks", "1", "3", "--phonebooks", "2", "--out", str(tmp_path / "out"), "--prefix", "e5"])
    analysis = json.loads((tmp_path / "out" / "e5_analysis.json").read_text(encoding="utf-8"), parse_constant=refuse)
    assert analysis["noise"]["band"] == 1
    with (tmp_path / "out" / "e5_cells.csv").open(newline="", encoding="utf-8") as file:
        assert len(list(csv.DictReader(file))) == 4
