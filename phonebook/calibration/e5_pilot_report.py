"""Report on the E5 pilots (scratch, NOT evidence): noise between shuffle seeds, censored and flickering runs, and a bootstrap
over phonebooks that gave the locked E5 predictions their ranges.

    python phonebook/calibration/e5_pilot_report.py [--results-dir DIR] [--trials 20000] [--seed 1]

Reads the six raw result files in `e5_pilot_results/` (see the README of this folder for the command that made each).
Sections: 1 timing noise between shuffle seeds, 2 censored and flickering runs, 3 phonebook-level tables and pooled counts,
4 a bootstrap that draws 6 phonebooks with replacement and applies the E5 rule with a noise band of 0.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import random
import statistics as st

RESULTS_DIR = Path(__file__).with_name("e5_pilot_results")
KS = (1, 2, 3, 4, 5, 9)
KS3 = (1, 2, 3, 5)
FILES = {"d0_1": "d0_lr0.001.json", "d0_3": "d0_lr0.0003.json", "d0_mt": "d0_lr0.001_default_threads.json",
         "pb_a": "phonebooks_a.json", "pb_b": "phonebooks_b.json", "pb_003": "phonebooks_lr0.003.json"}


def load_files(results_dir: Path) -> dict:
    return {name: json.loads((Path(results_dir) / file).read_text(encoding="utf-8")) for name, file in FILES.items()}


def d0_cells(table: dict):
    """(scenario, k, shuffle seed, result) for a d0 file."""
    for key, value in table.items():
        scenario, k, seed = key.split("|")
        yield scenario, int(k[1:]), int(seed[1:]), value


def pb_cells(table: dict):
    """(phonebook, init, k, lr, shuffle seed, result) for a phonebook file."""
    for key, value in table.items():
        pb, init, k, lr, seed, _ = key.split("|")
        yield int(pb), int(init), int(k), lr, int(seed), value


# ------------------------------------------------------------------ 1. noise

def spreads(groups: dict) -> list[int]:
    """Largest minus smallest stable_joint over the shuffle seeds of each group that has at least two; a censored run is an error."""
    out = []
    for key, values in groups.items():
        if len(values) < 2:
            continue
        if None in values:
            raise ValueError(f"a censored run in noise group {key}")
        out.append(max(values) - min(values))
    return out


def noise_sets(files: dict) -> dict:
    sets = {}
    for label, name in (("d0, lr 0.001, one thread, 5 shuffle seeds", "d0_1"), ("d0, lr 0.001, default threads, 4 shuffle seeds", "d0_mt"),
                        ("d0, lr 0.0003, 3 shuffle seeds", "d0_3")):
        groups = defaultdict(list)
        for scenario, k, seed, value in d0_cells(files[name]):
            groups[(scenario, k)].append(value["stable_joint"])
        sets[label] = spreads(groups)
    groups = defaultdict(list)
    for name in ("pb_a", "pb_b"):
        for pb, init, k, lr, seed, value in pb_cells(files[name]):
            if lr == "0.001" and init == 42 and k in (1, 3):
                groups[(pb, k)].append(value["stable_joint"])
    sets["10 other phonebooks, lr 0.001, init 42, k in {1, 3}, 3 shuffle seeds"] = spreads(groups)
    return sets


# ------------------------------------------------------------------ 2. censoring and flicker

def censoring(files: dict) -> dict:
    def tally(cells):
        cells = list(cells)
        return {"cells": len(cells), "censored": sum(v["stable_joint"] is None for v in cells),
                "flicker": sum(v["stable_joint"] is not None and v["first_joint"] != v["stable_joint"] for v in cells)}
    return {"10 other phonebooks, lr 0.001 and 0.0003": tally(v for name in ("pb_a", "pb_b") for *_, v in pb_cells(files[name])),
            "10 other phonebooks, lr 0.003": tally(v for *_, v in pb_cells(files["pb_003"])),
            "d0, all files": tally(v for name in ("d0_1", "d0_3", "d0_mt") for *_, v in d0_cells(files[name]))}


# ------------------------------------------------------------------ 3. tables

def build_tables(files: dict):
    """data[phonebook][lr][init][k] for shuffle seed 42; d0 is the E3 phonebook."""
    data: dict = {}

    def add(pb, lr, init, k, value):
        data.setdefault(pb, {}).setdefault(lr, {}).setdefault(init, {})[k] = value
    for name, lr in (("d0_1", "0.001"), ("d0_3", "0.0003")):
        for scenario, k, seed, value in d0_cells(files[name]):
            if seed == 42:
                add("d0", lr, int(scenario.split("seed")[1]), k, value)
    for name in ("pb_a", "pb_b"):
        for pb, init, k, lr, seed, value in pb_cells(files[name]):
            if seed == 42:
                add(f"pb{pb}", lr, init, k, value)

    def med(pb, lr, k, field="stable_joint"):
        return st.median(data[pb][lr][i][k][field] for i in data[pb][lr])
    pbs = [p for p in data if len(data[p].get("0.001", {})) == 3 and all(k in data[p]["0.001"][42] for k in KS)]
    tab = {p: {k: med(p, "0.001", k) for k in KS} for p in pbs}
    dip = {p: {k: med(p, "0.001", k, "min_a") for k in KS} for p in pbs}
    tab3 = {p: {k: med(p, "0.0003", k) for k in KS3} for p in pbs if len(data[p].get("0.0003", {})) == 3}
    return pbs, tab, dip, tab3


def pooled_lines(pbs, tab, dip, tab3) -> list[str]:
    lines = ["phonebooks: " + str(pbs), "", "phonebook-level medians at lr 0.001 (k=1,2,3,4,5,9) | dip medians"]
    for p in pbs:
        lines.append(f"  {p:6s} {[tab[p][k] for k in KS]} | dip {[dip[p][k] for k in KS]} | lr0.0003 {[tab3[p][k] for k in KS3] if p in tab3 else None}")
    lines += ["", "observed, all phonebooks pooled:"]
    for k in KS[1:]:
        g = [tab[p][1] - tab[p][k] for p in pbs]
        lines.append(f"  k={k}: faster {sum(x > 0 for x in g)} tied {sum(x == 0 for x in g)} slower {sum(x < 0 for x in g)} of {len(pbs)}; gains {g}; median gain {st.median(g)}")
    lines.append(f"  median over phonebooks of k=1: {st.median(tab[p][1] for p in pbs)}")
    for k in KS:
        lines.append(f"  k={k}: median time {st.median(tab[p][k] for p in pbs)}  median dip {st.median(dip[p][k] for p in pbs)}  mean time {st.mean(tab[p][k] for p in pbs):.1f}")
    lines += ["", "lr 0.0003 (ks 1,2,3,5), faster than k=1 counts:"]
    for k in KS3[1:]:
        g = [tab3[p][1] - tab3[p][k] for p in tab3]
        lines.append(f"  k={k}: faster {sum(x > 0 for x in g)} of {len(g)}; gains {g}")
    lines.append("  best k per pb at 0.0003: " + str({p: min(KS3, key=lambda k: (tab3[p][k], k)) for p in tab3}))
    lines.append("  best k per pb at 0.001 : " + str({p: min(KS, key=lambda k: (tab[p][k], k)) for p in pbs}))
    return lines


# ------------------------------------------------------------------ 4. bootstrap

def bootstrap(pbs, tab, dip, tab3, trials: int = 20000, seed: int = 1) -> list[str]:
    """Draw 6 phonebooks with replacement (E5 has six), apply the E5 rule with a noise band of 0."""
    random.seed(seed)
    res = {"fast": Counter(), "cheap": Counter(), "beats": Counter(), "any": 0, "m1": [], "dip1": [], "dip9": [],
           "cnt": {k: Counter() for k in KS[1:]}, "best3_vs_best": Counter(), "plateau0": Counter()}
    pool3 = list(tab3)
    for _ in range(trials):
        s = [random.choice(pbs) for _ in range(6)]
        med_t = {k: st.median(tab[p][k] for p in s) for k in KS}
        res["fast"][min(KS, key=lambda k: (med_t[k], k))] += 1
        work = {k: med_t[k] * (10 + 10 * k) for k in KS}
        res["cheap"][min(KS, key=lambda k: (work[k], k))] += 1
        res["m1"].append(med_t[1])
        res["dip1"].append(st.median(dip[p][1] for p in s))
        res["dip9"].append(st.median(dip[p][9] for p in s))
        anyb = False
        for k in KS[1:]:
            g = [tab[p][1] - tab[p][k] for p in s]
            c = sum(x > 0 for x in g)
            res["cnt"][k][c] += 1
            if c >= 5 and st.median(g) > 0:
                res["beats"][k] += 1
                anyb = True
        res["any"] += anyb
        res["plateau0"][tuple(k for k in KS if med_t[k] <= min(med_t.values()))] += 1
        s3 = [random.choice(pool3) for _ in range(6)]
        m3 = {k: st.median(tab3[p][k] for p in s3) for k in KS3}
        best3 = min(KS3, key=lambda k: (m3[k], k))
        m1 = {k: st.median(tab[p][k] for p in s3) for k in KS}
        best1 = min(KS, key=lambda k: (m1[k], k))
        res["best3_vs_best"]["larger" if best3 > best1 else "same" if best3 == best1 else "smaller"] += 1
    N = trials
    lines = [f"BOOTSTRAP of 6 phonebooks (pool of {len(pbs)}) , noise band 0:",
             " fastest k: " + str({k: round(v / N, 3) for k, v in sorted(res["fast"].items())}),
             " cheapest k in work: " + str({k: round(v / N, 3) for k, v in sorted(res["cheap"].items())}),
             " P(k beats k=1 under rule, band 0): " + str({k: round(v / N, 3) for k, v in sorted(res["beats"].items())}) + f"  any: {round(res['any'] / N, 3)}"]
    for k in KS[1:]:
        tot = res["cnt"][k]
        lines.append(f" count faster for k={k}: " + str({c: round(tot[c] / N, 2) for c in range(7)}) + f"  mode {tot.most_common(1)[0][0]}")
    a = sorted(res["m1"])
    lines.append(f" median k=1 time: mean {round(st.mean(a), 1)}  10-90%: {a[N // 10]} {a[9 * N // 10]}")
    for name in ("dip1", "dip9"):
        lines.append(f" {name}:  {Counter(res[name]).most_common(5)}")
    lines.append(" plateau (band 0): " + str({k: round(v / N, 3) for k, v in res["plateau0"].most_common(6)}))
    lines.append(" best k at 0.0003 vs 0.001: " + str({k: round(v / N, 3) for k, v in res["best3_vs_best"].items()}))
    return lines


def report(results_dir: Path = RESULTS_DIR, trials: int = 20000, seed: int = 1) -> str:
    files = load_files(results_dir)
    out = ["== 1. timing noise between shuffle seeds (spread = largest minus smallest stable_joint over the seeds of one cell)"]
    for label, sp in noise_sets(files).items():
        out.append(f"  {label}: {len(sp)} groups, {sum(x != 0 for x in sp)} with a nonzero spread, largest spread {max(sp)}")
    out += ["", "== 2. runs that never reach a stable joint recall (censored) and runs whose first joint recall was lost again (flicker)"]
    for label, t in censoring(files).items():
        out.append(f"  {label}: {t['cells']} cells, {t['censored']} censored, {t['flicker']} flicker")
    pbs, tab, dip, tab3 = build_tables(files)
    out += ["", "== 3. tables"] + pooled_lines(pbs, tab, dip, tab3)
    out += ["", "== 4."] + bootstrap(pbs, tab, dip, tab3, trials, seed)
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--trials", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(argv)
    print(report(args.results_dir, args.trials, args.seed))


if __name__ == "__main__":
    main()
