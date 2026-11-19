"""Tests for report/make_numbers.py (R1): every number the report prints comes from report/numbers.tex through \\num{key}.

Three layers check each value. (1) PINNED: headline literals typed from the verified recompute outputs of 2026-11-18. (2) RAW: an independent
recompute from the per-run metrics.csv files, written here without importing anything from phonebook/ or from the generator. (3) the committed
report/numbers.tex must equal what the generator writes now. The generator itself reads only tracked files: the derived files in phonebook/results,
the E4/E6/E7 protocol.json files, the three control books and the three E7 arm W width-64 metrics files (for the dips).
"""
from __future__ import annotations

import csv
import json
import math
import re
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

import pytest

from report import build_report as br
from report import gate
from report import make_numbers as mn

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "phonebook" / "experiments"
WIDTHS = (8, 16, 32, 64)

# ---- what the generator may read (a copy of exactly these files must be enough to build) ------------------------------------------------------
INPUT_FILES = [
    "phonebook/results/e4_capacity.json", "phonebook/results/e6_capacity.json", "phonebook/results/e7w_capacity.json", "phonebook/results/e7s_capacity.json",
    "phonebook/results/e6_extension.json", "phonebook/results/e7w_extension.json", "phonebook/results/e7s_extension.json",
    "phonebook/results/e4_prediction_scores.json", "phonebook/results/e6_prediction_scores.json", "phonebook/results/e7_scores.json",
    "phonebook/results/fig_e6_curves.csv",
]
INPUT_GLOBS = [
    "phonebook/experiments/week_2026-10-31/e[46]_*/protocol.json",
    "phonebook/experiments/week_2026-11-13/e7_*/protocol.json",
    "phonebook/experiments/week_2026-11-13/e7_w_d64_seed*/s*_n640/d64/metrics.csv",
    "phonebook/inputs/capacity_seed*_control.txt",
]


def copy_inputs(dst: Path) -> None:
    paths = [ROOT / rel for rel in INPUT_FILES]
    for pattern in INPUT_GLOBS:
        paths += [p for p in ROOT.glob(pattern) if "__interrupted" not in str(p)]
    assert len(paths) == len(INPUT_FILES) + 24 + 12 + 18 + 3 + 3, len(paths)  # E4 24, E6 12 and E7 18 protocols; 3 metrics files; 3 control books
    for path in paths:
        target = dst / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(path, target)


# ---- layer 1: headline literals, typed from evidence\recompute_*_output.txt (every line there is [ok]) ------------------------------------------
PINNED = {
    # E4 (CL-1, CL-2, CL-6)
    "e4.cap.w8.b1000": "14", "e4.cap.w16.b1000": "113", "e4.cap.w32.b1000": "226", "e4.cap.w64.b1000": "453",
    "e4.cap.w8.b500": "none", "e4.cap.w16.b500": "40", "e4.cap.w32.b500": "226", "e4.cap.w64.b500": "320",
    "e4.cap.w8.b300": "none", "e4.cap.w16.b300": "20", "e4.cap.w32.b300": "160", "e4.cap.w64.b300": "320",
    "e4.cap.w8.b100": "none", "e4.cap.w16.b100": "none", "e4.cap.w32.b100": "none", "e4.cap.w64.b100": "20",
    "e4.params.w8": "3,064", "e4.params.w16": "9,216", "e4.params.w32": "30,720", "e4.params.w64": "110,656",
    "e4.bits.w8": "0.149", "e4.bits.w16": "0.399", "e4.bits.w32": "0.240", "e4.bits.w64": "0.133",
    "e4.boundary.w8": "moving", "e4.boundary.w16": "moving", "e4.boundary.w32": "stable", "e4.boundary.w64": "moving",
    "e4.upe.w8": "1", "e4.upe.w16": "2", "e4.upe.w32": "4", "e4.upe.w64": "8",
    "e4.updates.w8": "1,000", "e4.updates.w16": "2,000", "e4.updates.w32": "4,000", "e4.updates.w64": "8,000",
    "e4.pred.scored": "23", "e4.pred.hits": "23", "e4.pred.exact": "13",
    # E6 (CL-3, CL-4, CL-6)
    "e6.n.w8": "20", "e6.n.w16": "160", "e6.n.w32": "320", "e6.n.w64": "640",
    "e6.passes.w8": "yes", "e6.passes.w16": "yes", "e6.passes.w32": "yes", "e6.passes.w64": "no",
    "e6.firstbudget.w8": "2000", "e6.firstbudget.w16": "4000", "e6.firstbudget.w32": "3000", "e6.firstbudget.w64": "none",
    "e6.perfectby.w8": "3", "e6.perfectby.w16": "3", "e6.perfectby.w32": "2", "e6.perfectby.w64": "0",
    "e6.reached.w8": "3", "e6.reached.w16": "3", "e6.reached.w32": "2", "e6.reached.w64": "0",
    "e6.atend.w8": "3", "e6.atend.w16": "0", "e6.atend.w32": "0", "e6.atend.w64": "0",
    "e6.first.w16.s1": "3270", "e6.first.w16.s2": "1210", "e6.first.w16.s3": "3200",
    "e6.first.w32.s1": "940", "e6.first.w32.s2": "2960", "e6.first.w32.s3": "never",
    "e6.first.w64.s1": "never", "e6.first.w64.s2": "never", "e6.first.w64.s3": "never",
    "e6.logged100.w8.s1": "283", "e6.logged100.w8.s2": "290", "e6.logged100.w8.s3": "219",
    "e6.logged100.w16.s1": "3", "e6.logged100.w16.s2": "21", "e6.logged100.w16.s3": "2",
    "e6.logged100.w32.s1": "6", "e6.logged100.w32.s2": "1", "e6.logged100.w32.s3": "0",
    "e6.logged100.w64.s1": "0", "e6.logged100.w64.s2": "0", "e6.logged100.w64.s3": "0",
    "e6.late100.w8.s1": "95", "e6.late100.w8.s2": "96", "e6.late100.w8.s3": "94",
    "e6.late100.w16.s1": "3", "e6.late100.w16.s2": "2", "e6.late100.w16.s3": "2",
    "e6.late100.w32.s1": "2", "e6.late100.w32.s2": "0", "e6.late100.w32.s3": "0",
    "e6.recall4000.w16.s1": "147", "e6.recall4000.w16.s2": "158", "e6.recall4000.w16.s3": "154",
    "e6.recall4000.w32.s1": "316", "e6.recall4000.w32.s2": "311", "e6.recall4000.w32.s3": "316",
    "e6.recall4000.w64.s1": "610", "e6.recall4000.w64.s2": "633", "e6.recall4000.w64.s3": "623",
    "e6.mean.w16.early": "0.9351", "e6.mean.w16.late": "0.9526", "e6.mean.w32.early": "0.9636", "e6.mean.w32.late": "0.9733",
    "e6.mean.w64.early": "0.9684", "e6.mean.w64.late": "0.9754",
    "e6.bits.w8": "0.213", "e6.bits.w16": "0.566", "e6.bits.w32": "0.339", "e6.bits.w64": "none",
    "e6.holdingtotal": "0",
    "e6.pred.scored": "17", "e6.pred.hits": "17", "e6.pred.exact": "8",
    # E7 arm W (CL-7)
    "e7w.first.w16.s1": "2020", "e7w.first.w16.s2": "1310", "e7w.first.w16.s3": "1830",
    "e7w.first.w32.s1": "820", "e7w.first.w32.s2": "1660", "e7w.first.w32.s3": "1770",
    "e7w.first.w64.s1": "1310", "e7w.first.w64.s2": "1270", "e7w.first.w64.s3": "1500",
    "e7w.late100.w16.s1": "13", "e7w.late100.w16.s2": "100", "e7w.late100.w16.s3": "100",
    "e7w.late100.w32.s1": "100", "e7w.late100.w32.s2": "100", "e7w.late100.w32.s3": "100",
    "e7w.late100.w64.s1": "95", "e7w.late100.w64.s2": "90", "e7w.late100.w64.s3": "100",
    "e7w.passes.w16": "yes", "e7w.passes.w32": "yes", "e7w.passes.w64": "yes",
    "e7w.firstbudget.w16": "2000", "e7w.firstbudget.w32": "2000", "e7w.firstbudget.w64": "2000",
    "e7w.atend.w16": "3", "e7w.atend.w32": "3", "e7w.atend.w64": "3",
    "e7w.holding.w16": "2", "e7w.holding.w32": "3", "e7w.holding.w64": "3", "e7w.holdingtotal": "8",
    "e7w.mean.w16.early": "0.9666", "e7w.mean.w16.late": "0.9937", "e7w.mean.w32.early": "0.9802", "e7w.mean.w32.late": "1.0000",
    "e7w.mean.w64.early": "0.9928", "e7w.mean.w64.late": "0.9928",
    "e7w.bits.w16": "0.566", "e7w.bits.w32": "0.339", "e7w.bits.w64": "0.188",
    "e7w.dipmin.w64.s1": "390", "e7w.dipmin.w64.s2": "25", "e7w.dipmin.w64.s3": "none",
    # E7 arm S
    "e7s.first.w16.s1": "2010", "e7s.first.w16.s2": "1210", "e7s.first.w16.s3": "2510",
    "e7s.first.w32.s1": "940", "e7s.first.w32.s2": "2010", "e7s.first.w32.s3": "2010",
    "e7s.first.w64.s1": "2010", "e7s.first.w64.s2": "2010", "e7s.first.w64.s3": "2010",
    "e7s.late100.w16.s1": "100", "e7s.late100.w32.s2": "100", "e7s.late100.w64.s3": "100",
    "e7s.passes.w16": "yes", "e7s.passes.w32": "yes", "e7s.passes.w64": "yes",
    "e7s.firstbudget.w16": "3000", "e7s.firstbudget.w32": "3000", "e7s.firstbudget.w64": "3000",
    "e7s.atend.w16": "3", "e7s.atend.w32": "3", "e7s.atend.w64": "3",
    "e7s.holding.w16": "3", "e7s.holding.w32": "3", "e7s.holding.w64": "3", "e7s.holdingtotal": "9",
    "e7s.mean.w16.early": "0.9351", "e7s.mean.w16.late": "1.0000", "e7s.mean.w32.early": "0.9636", "e7s.mean.w32.late": "1.0000",
    "e7s.mean.w64.early": "0.9684", "e7s.mean.w64.late": "1.0000",
    "e7s.bits.w16": "0.566", "e7s.bits.w32": "0.339", "e7s.bits.w64": "0.188",
    # E7 scoring
    "e7.pred.scored": "44", "e7.pred.hits": "41", "e7.pred.exact": "25", "e7.pred.inrange": "16", "e7.pred.misses": "3",
    "e7.tripwire.fired": "2", "e7.tripwire.total": "10",
    # design constants (protocol.json of E4, E6, E7; the control books)
    "design.batch": "64", "design.lr": "0.001", "design.layers": "2", "design.heads": "2", "design.clip": "1.0",
    "design.wd.default": "0.1", "design.wd.w": "0.0", "design.epochs.e4": "1000", "design.epochs.e6": "4000", "design.loginterval": "10",
    "design.s.step": "2000", "design.s.factor": "0.1", "design.seeds": "3", "design.control": "20", "design.bitsentry": "32.575",
    "design.upe.n10": "1", "design.upe.n14": "1", "design.upe.n20": "1", "design.upe.n28": "1", "design.upe.n40": "1", "design.upe.n57": "1",
    "design.upe.n80": "2", "design.upe.n113": "2", "design.upe.n160": "3", "design.upe.n226": "4", "design.upe.n320": "5",
    "design.upe.n453": "8", "design.upe.n640": "10", "design.upe.n905": "15",
}


# ---- layer 2: independent recompute from the raw per-run files -----------------------------------------------------------------------------
BITS = math.log2(800 * 800 * 10_000)


def raw_runs(week: str, prefix: str) -> dict[tuple[int, int, int], tuple[list[tuple[int, int]], int]]:
    """{(width, rung size, seed): (sorted [(epoch, entries recalled)], parameter count)} for the run folders whose name starts with prefix."""
    out = {}
    for path in sorted((EXPERIMENTS / week).glob(f"{prefix}*/s*_n*/d*/metrics.csv")):
        if "__interrupted" in path.parts[-4]:
            continue
        seed, n = map(int, re.fullmatch(r"s(\d+)_n(\d+)", path.parts[-3]).groups())
        width = int(path.parts[-2][1:])
        rows, params = [], set()
        with open(path, newline="") as handle:
            for record in csv.DictReader(handle):
                recalled = float(record["Train Seq Acc"]) * n
                assert abs(recalled - round(recalled)) < 0.05
                rows.append((int(record["Epoch"]), int(round(recalled))))
                params.add(int(record["Param Count"]))
        assert len(params) == 1
        out[(width, n, seed)] = (sorted(rows), params.pop())
    return out


def first_at(rows, n):
    return next((e for e, k in rows if k == n), None)


def word(value) -> str:
    return "none" if value is None else str(value)


def mean4(runs, lo, hi) -> str:
    values = [k / n for (rows, n) in runs for e, k in rows if lo <= e <= hi]
    return f"{sum(values) / len(values):.4f}"


def dip_min(rows, n):
    run = 0
    for i, (e, k) in enumerate(rows):
        run = run + 1 if k == n else 0
        if run == 50:
            settled = rows[i - 49][0]
            later = [k2 for e2, k2 in rows if e2 > settled and k2 < n]
            return min(later) if later else None
    return None


def raw_expected() -> dict[str, str]:
    out: dict[str, str] = {}
    # E4
    e4 = raw_runs("week_2026-10-31", "e4_")
    assert len(e4) == 96
    for d in WIDTHS:
        rungs = sorted({n for (w, n, s) in e4 if w == d})
        caps = {}
        for budget in (100, 300, 500, 1000):
            cap = None
            for n in rungs:
                if sum(1 for s in (1, 2, 3) if (first_at(e4[(d, n, s)][0], n) or 10**9) <= budget) < 2:
                    break
                cap = n
            caps[budget] = cap
            out[f"e4.cap.w{d}.b{budget}"] = word(cap)
        cap = caps[1000]
        params = statistics.median(e4[(d, cap, s)][1] for s in (1, 2, 3))
        out[f"e4.params.w{d}"] = f"{int(params):,}"
        out[f"e4.bits.w{d}"] = f"{cap * BITS / params:.3f}"
        out[f"e4.boundary.w{d}"] = "stable" if caps[1000] == caps[500] else "moving"
        out[f"e4.upe.w{d}"] = str(math.ceil(cap / 64))
        out[f"e4.updates.w{d}"] = f"{math.ceil(cap / 64) * 1000:,}"
    # E6 and E7
    arms = {"e6": raw_runs("week_2026-10-31", "e6_"), "e7w": raw_runs("week_2026-11-13", "e7_w_"), "e7s": raw_runs("week_2026-11-13", "e7_s_")}
    assert [len(v) for v in arms.values()] == [12, 9, 9]
    for arm, runs in arms.items():
        total_holding = 0
        for d in sorted({w for (w, n, s) in runs}):
            n = next(n for (w, n, s) in runs if w == d)
            per_seed = [runs[(d, n, s)][0] for s in (1, 2, 3)]
            firsts = [first_at(rows, n) for rows in per_seed]
            passes = lambda budget: sum(1 for f in firsts if f is not None and f <= budget) >= 2
            out[f"{arm}.n.w{d}"] = str(n)
            out[f"{arm}.passes.w{d}"] = "yes" if passes(4000) else "no"
            out[f"{arm}.firstbudget.w{d}"] = word(next((b for b in (1000, 2000, 3000, 4000) if passes(b)), None))
            out[f"{arm}.perfectby.w{d}"] = str(sum(1 for f in firsts if f is not None and f <= 4000))
            out[f"{arm}.reached.w{d}"] = str(sum(1 for f in firsts if f is not None))
            out[f"{arm}.atend.w{d}"] = str(sum(1 for rows in per_seed if rows[-1] == (4000, n)))
            late = [sum(1 for e, k in rows if 3001 <= e <= 4000 and k == n) for rows in per_seed]
            holding = sum(1 for c in late if c >= 50)
            out[f"{arm}.holding.w{d}"] = str(holding)
            if d != 8:
                total_holding += holding
            out[f"{arm}.mean.w{d}.early"] = mean4([(rows, n) for rows in per_seed], 1001, 2000)
            out[f"{arm}.mean.w{d}.late"] = mean4([(rows, n) for rows in per_seed], 3001, 4000)
            params = statistics.median(runs[(d, n, s)][1] for s in (1, 2, 3))
            out[f"{arm}.bits.w{d}"] = f"{n * BITS / params:.3f}" if passes(4000) else "none"
            for s in (1, 2, 3):
                out[f"{arm}.first.w{d}.s{s}"] = "never" if firsts[s - 1] is None else str(firsts[s - 1])
                out[f"{arm}.late100.w{d}.s{s}"] = str(late[s - 1])
                if arm == "e6":
                    out[f"e6.logged100.w{d}.s{s}"] = str(sum(1 for e, k in per_seed[s - 1] if k == n))
                    out[f"e6.recall4000.w{d}.s{s}"] = str(dict(per_seed[s - 1])[4000])
                if arm == "e7w" and d == 64:
                    out[f"e7w.dipmin.w64.s{s}"] = word(dip_min(per_seed[s - 1], n))
        out[f"{arm}.holdingtotal"] = str(total_holding)
    # prediction scores
    results = lambda name: [row["result"] for row in json.loads((ROOT / "phonebook" / "results" / name).read_text(encoding="utf-8"))]
    for prefix, name in (("e4", "e4_prediction_scores.json"), ("e6", "e6_prediction_scores.json"), ("e7", "e7_scores.json")):
        r = results(name)
        hits, exact, misses = sum(x.startswith("HIT") for x in r), sum(x == "HIT (exact)" for x in r), sum(x == "MISS" for x in r)
        out[f"{prefix}.pred.scored"], out[f"{prefix}.pred.hits"], out[f"{prefix}.pred.exact"] = str(hits + misses), str(hits), str(exact)
        if prefix == "e7":
            out["e7.pred.inrange"], out["e7.pred.misses"] = str(hits - exact), str(misses)
            fired = sum(x == "TRIPWIRE FIRED" for x in r)
            out["e7.tripwire.fired"], out["e7.tripwire.total"] = str(fired), str(fired + sum(x == "TRIPWIRE NOT FIRED" for x in r))
    # design constants: read from the protocols here too, by separate code, and required to be uniform
    def protocols(week: str, prefix: str) -> list[dict]:
        found = [p for p in sorted((EXPERIMENTS / week).glob(f"{prefix}*/protocol.json")) if "__interrupted" not in p.parts[-2]]
        return [json.loads(p.read_text(encoding="utf-8")) for p in found]

    def unique(values) -> str:
        values = {str(v) for v in values}
        assert len(values) == 1, values
        return values.pop()

    p4, p6 = protocols("week_2026-10-31", "e4_"), protocols("week_2026-10-31", "e6_")
    pw, ps = protocols("week_2026-11-13", "e7_w_"), protocols("week_2026-11-13", "e7_s_")
    allp = p4 + p6 + pw + ps
    assert [len(p4), len(p6), len(pw), len(ps)] == [24, 12, 9, 9]
    runs = lambda plist: [r for p in plist for r in p["runs"]]
    out["design.batch"] = unique(r["batch_size"] for r in runs(allp))
    out["design.lr"] = unique(r["lr"] for r in runs(allp))
    out["design.layers"] = unique(r["model"]["n_layers"] for r in runs(allp))
    out["design.heads"] = unique(r["model"]["n_heads"] for r in runs(allp))
    out["design.loginterval"] = unique(r["accuracy_interval"] for r in runs(allp))
    out["design.clip"] = unique(p["optimizer"]["grad_clip"] for p in allp)
    out["design.wd.default"] = unique(p["optimizer"]["weight_decay"] for p in allp)
    out["design.wd.w"] = unique(r["weight_decay"] for r in runs(pw))
    out["design.epochs.e4"] = unique(r["epochs"] for r in runs(p4))
    out["design.epochs.e6"] = unique(r["epochs"] for r in runs(p6 + pw + ps))
    kind, step, factor = unique(r["lr_schedule"] for r in runs(ps)).split(":")
    assert kind == "step"
    out["design.s.step"], out["design.s.factor"] = step, factor
    out["design.bitsentry"] = f"{BITS:.3f}"
    out["design.seeds"] = "3"
    for n in (10, 14, 20, 28, 40, 57, 80, 113, 160, 226, 320, 453, 640, 905):
        out[f"design.upe.n{n}"] = str(math.ceil(n / 64))
    books = [ROOT / "phonebook" / "inputs" / f"capacity_seed{s}_control.txt" for s in (1, 2, 3)]
    sizes = {len([line for line in b.read_text(encoding="utf-8").split("\n") if line.strip()]) for b in books}
    assert len(sizes) == 1
    out["design.control"] = str(sizes.pop())
    return out


@pytest.fixture(scope="module")
def built():
    return mn.build(ROOT)


@pytest.fixture(scope="module")
def raw():
    return raw_expected()


# ---- the TeX file: rendering ----------------------------------------------------------------------------------------------------------------
def test_render_defines_every_key_with_numdef_in_sorted_order():
    text = mn.render({"b.y": "2", "a.x": "1"})
    assert text.index("\\numdef{a.x}{1}") < text.index("\\numdef{b.y}{2}")


def test_render_is_deterministic_and_has_no_carriage_return_and_no_timestamp():
    values = {"a.x": "1", "b.y": "0.5"}
    assert mn.render(values) == mn.render(dict(reversed(list(values.items()))))
    assert "\r" not in mn.render(values)
    assert not re.search(r"20\d\d-\d\d-\d\d|\d\d:\d\d", mn.render(values))


def test_render_defines_the_num_macro_that_fails_loudly_on_an_unknown_key():
    text = mn.render({"a.x": "1"})
    assert "\\newcommand{\\num}[1]" in text and "\\ifcsname num@#1\\endcsname" in text
    assert "\\PackageError{numbers}{Undefined number key `#1'}" in text


@pytest.mark.parametrize("bad_value", ["", " ", "1}", "{1", "1\\x", "5%", "a b", "1\n2"])
def test_render_refuses_a_value_that_could_break_the_tex_file(bad_value):
    with pytest.raises(mn.NumbersError):
        mn.render({"a.x": bad_value})


@pytest.mark.parametrize("bad_key", ["", "a b", "a}", "a\\b", "a%", "A.x", "a..", "9.x"])
def test_render_refuses_a_key_outside_the_documented_alphabet(bad_key):
    with pytest.raises(mn.NumbersError):
        mn.render({bad_key: "1"})


# ---- using the keys in the report -----------------------------------------------------------------------------------------------------------
def test_num_uses_lists_the_keys_in_order_ignoring_comments():
    tex = "A \\num{a.x} and \\num{b.y}.\n% \\num{c.z}\nAnd \\num{a.x} \\% \\num{d.w}"
    assert mn.num_uses(tex) == ["a.x", "b.y", "a.x", "d.w"]


def test_undefined_and_unused_keys():
    tex = "\\num{a.x} \\num{nope}"
    assert mn.undefined_keys(tex, {"a.x": "1", "b.y": "2"}) == ["nope"]
    assert mn.unused_keys(tex, {"a.x": "1", "b.y": "2"}) == ["b.y"]


VALUES = {"e6.bits.w16": "0.566", "e4.params.w64": "110,656", "design.epochs.e4": "1000", "design.layers": "2", "e6.n.w8": "20", "e6.n.w64": "640"}


@pytest.mark.parametrize("tex, flagged", [
    ("We found 0.566 bits.", ["0.566"]),
    ("We found \\num{e6.bits.w16} bits.", []),
    ("% a comment says 0.566\nNothing.", []),
    ("There are 110,656 parameters.", ["110,656"]),
    ("There are 110656 parameters.", ["110656"]),
    ("Trained for 1000 epochs.", ["1000"]),
    ("A model with 2 layers and width 8 has 20 entries.", []),   # small integers are structure, not results
    ("See \\ref{fig:0.566} and \\cite{x1000} and \\label{t:1000}.", []),
    ("Two numbers, 0.566 and 1000.", ["0.566", "1000"]),
    ("It is 10.5660 and 0.5661.", []),                            # only an exact literal counts
    ("Values 0.566,0.566 and (1000,1000).", ["0.566", "0.566", "1000", "1000"]),   # a comma without a space does not hide the next number
    ("It has 110,656 parameters, 20,000 steps.", ["110,656"]),    # a thousands group is one number
    ("See \\autoref{fig:0.566}, \\Cref{fig:0.566}, \\Citet{a,1000} and \\citep[p.~1000]{b,1000}.", []),
    ("It took 1000x longer and 0.566th.", []),                    # a number glued to letters is not a literal
    ("The 640 entries.", ["640"]),                                # three digits are a result
])
def test_typed_numbers_matching_flags_a_result_typed_by_hand(tex, flagged):
    assert mn.typed_numbers_matching(tex, VALUES) == flagged


# ---- reading the inputs: loud failures ------------------------------------------------------------------------------------------------------
@pytest.fixture
def inputs(tmp_path):
    copy_inputs(tmp_path)
    return tmp_path


def edit_json(path: Path, change) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    change(data)
    path.write_text(json.dumps(data), encoding="utf-8")   # json.dumps writes NaN as a bare NaN, which json.loads reads back


def test_build_works_from_a_copy_of_exactly_the_files_it_is_allowed_to_read(inputs):
    assert mn.build(inputs) == mn.build(ROOT)


def test_a_missing_input_file_is_an_error_not_a_silent_zero(inputs):
    (inputs / "phonebook/results/e6_extension.json").unlink()
    with pytest.raises(mn.NumbersError, match="e6_extension.json"):
        mn.build(inputs)


@pytest.mark.parametrize("bad", [True, -1, 2.5, "3", float("nan"), None])
def test_a_count_that_is_a_boolean_negative_fraction_string_nan_or_missing_is_an_error(inputs, bad):
    edit_json(inputs / "phonebook/results/e6_extension.json", lambda d: d["16"].__setitem__("reached_perfect", bad))
    with pytest.raises(mn.NumbersError, match="reached_perfect"):
        mn.build(inputs)


@pytest.mark.parametrize("bad", [True, "0.5", float("nan"), float("inf"), -0.1, 1.5])
def test_a_mean_accuracy_outside_zero_to_one_or_not_a_number_is_an_error(inputs, bad):
    edit_json(inputs / "phonebook/results/e7w_extension.json", lambda d: d["16"]["mean_accuracy"].__setitem__("1001_2000", bad))
    with pytest.raises(mn.NumbersError, match="mean_accuracy"):
        mn.build(inputs)


def test_protocols_that_disagree_about_the_batch_size_are_an_error(inputs):
    path = next((inputs / "phonebook/experiments/week_2026-11-13").glob("e7_s_d16_seed1/protocol.json"))
    edit_json(path, lambda d: d["runs"][0].__setitem__("batch_size", 32))
    with pytest.raises(mn.NumbersError, match="batch_size"):
        mn.build(inputs)


def test_protocols_that_disagree_about_updates_per_epoch_are_an_error(inputs):
    path = next((inputs / "phonebook/experiments/week_2026-11-13").glob("e7_w_d64_seed1/protocol.json"))
    edit_json(path, lambda d: d["runs"][0].__setitem__("updates_per_epoch", 9))
    with pytest.raises(mn.NumbersError, match="updates_per_epoch"):
        mn.build(inputs)


def test_an_arm_w_run_that_does_not_record_weight_decay_zero_is_an_error(inputs):
    path = next((inputs / "phonebook/experiments/week_2026-11-13").glob("e7_w_d32_seed2/protocol.json"))
    edit_json(path, lambda d: d["runs"][0].pop("weight_decay"))
    with pytest.raises(mn.NumbersError, match="weight_decay"):
        mn.build(inputs)


def test_a_metrics_file_with_a_fractional_recall_count_is_an_error(inputs):
    path = next((inputs / "phonebook/experiments/week_2026-11-13").glob("e7_w_d64_seed1/s1_n640/d64/metrics.csv"))
    lines = path.read_text(encoding="utf-8").splitlines()
    cells = lines[5].split(",")
    cells[5] = "0.5004"           # Train Seq Acc: 640 * 0.5004 = 320.256 is not a whole number of entries (four-decimal rounding moves it by at most 0.032)
    lines[5] = ",".join(cells)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(mn.NumbersError, match="Train Seq Acc"):
        mn.build(inputs)


def test_a_setting_that_no_protocol_records_is_an_error_not_the_text_none(inputs):
    for path in inputs.glob("phonebook/experiments/*/e[467]_*/protocol.json"):
        edit_json(path, lambda d: [run.pop("batch_size") for run in d["runs"]])
    with pytest.raises(mn.NumbersError, match="batch_size"):
        mn.build(inputs)


@pytest.mark.parametrize("damage", ["no runs", "no n_train"])
def test_a_malformed_protocol_is_a_numbers_error_not_a_key_error(inputs, damage):
    path = next((inputs / "phonebook/experiments/week_2026-10-31").glob("e6_extend_d8_seed1/protocol.json"))
    edit_json(path, (lambda d: d.pop("runs")) if damage == "no runs" else (lambda d: d["runs"][0].pop("n_train")))
    with pytest.raises(mn.NumbersError, match="protocol"):
        mn.build(inputs)


@pytest.mark.parametrize("damage", ["repeated", "missing"])
def test_a_curve_with_a_repeated_or_missing_logged_epoch_is_an_error(inputs, damage):
    path = inputs / "phonebook/results/fig_e6_curves.csv"
    lines = path.read_text(encoding="utf-8").splitlines()
    lines = lines + [lines[5]] if damage == "repeated" else lines[:5] + lines[6:]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(mn.NumbersError, match="fig_e6_curves.csv"):
        mn.build(inputs)


def test_a_metrics_file_with_a_missing_epoch_row_is_an_error(inputs):
    path = next((inputs / "phonebook/experiments/week_2026-11-13").glob("e7_w_d64_seed2/s2_n640/d64/metrics.csv"))
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:100] + lines[101:]) + "\n", encoding="utf-8")
    with pytest.raises(mn.NumbersError, match="metrics.csv"):
        mn.build(inputs)


def test_control_books_of_different_sizes_are_an_error(inputs):
    path = inputs / "phonebook/inputs/capacity_seed2_control.txt"
    path.write_text(path.read_text(encoding="utf-8") + "extra line\n", encoding="utf-8")
    with pytest.raises(mn.NumbersError, match="control"):
        mn.build(inputs)


@pytest.mark.parametrize("bad", ["nan", "inf", "-inf", "", "abc"])
def test_a_non_finite_or_empty_recall_value_in_a_metrics_file_is_a_numbers_error(inputs, bad):
    path = next((inputs / "phonebook/experiments/week_2026-11-13").glob("e7_w_d64_seed1/s1_n640/d64/metrics.csv"))
    lines = path.read_text(encoding="utf-8").splitlines()
    cells = lines[5].split(",")
    cells[5] = bad
    lines[5] = ",".join(cells)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(mn.NumbersError, match="Train Seq Acc"):
        mn.build(inputs)


@pytest.mark.parametrize("name, change, match", [
    ("e7w_extension", lambda d: d["16"]["epochs_at_100"]["by_seed"]["3"].__setitem__("3001_4000", 101), "3001_4000"),
    ("e6_extension", lambda d: d["16"].__setitem__("reached_perfect", 4), "reached_perfect"),
    ("e6_extension", lambda d: d["16"]["perfect_by"].__setitem__("4000", 2), "perfect_by"),
    ("e6_extension", lambda d: d["32"]["passes"].__setitem__("4000", False), "passes"),
    ("e6_extension", lambda d: d["64"].__setitem__("first_pass_budget", 4000), "first_pass_budget"),
    ("e6_extension", lambda d: d["16"].__setitem__("perfect_at_end", 4), "perfect_at_end"),
    ("e6_extension", lambda d: d["16"]["first_perfect"].__setitem__("2", None), "first_perfect"),
])
def test_counts_that_are_impossible_or_disagree_with_each_other_are_an_error(inputs, name, change, match):
    edit_json(inputs / f"phonebook/results/{name}.json", change)
    with pytest.raises(mn.NumbersError, match=match):
        mn.build(inputs)


def test_a_seed_count_that_the_keys_do_not_cover_is_an_error(inputs):
    for name in ("e6_extension", "e7w_extension", "e7s_extension"):   # all files agree on 4 seeds, but the keys cover seeds 1 to 3
        edit_json(inputs / f"phonebook/results/{name}.json", lambda d: [d[w].__setitem__("seeds", 4) for w in d])
    with pytest.raises(mn.NumbersError, match="seeds"):
        mn.build(inputs)


def test_main_reports_an_unwritable_output_as_a_numbers_error(tmp_path, capsys):
    assert mn.main(["--out", str(tmp_path)]) == 2          # a folder, not a file
    assert "numbers" in capsys.readouterr().err


def test_a_failed_write_leaves_the_old_file_untouched_and_no_temporary_file(tmp_path, monkeypatch, capsys):
    out = tmp_path / "numbers.tex"
    out.write_bytes(b"old")
    monkeypatch.setattr(mn.os, "replace", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    assert mn.main(["--out", str(out)]) == 2
    assert out.read_bytes() == b"old" and list(tmp_path.iterdir()) == [out]
    assert "disk full" in capsys.readouterr().err


def test_a_successful_write_leaves_no_temporary_file(tmp_path):
    assert mn.main(["--out", str(tmp_path / "numbers.tex")]) == 0
    assert [p.name for p in tmp_path.iterdir()] == ["numbers.tex"]


def set_recall(path: Path, changes: dict[int, int], n: int = 640) -> None:
    """Overwrite the logged Train Seq Acc at the given epochs with changes[epoch] entries recalled."""
    lines = path.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines[1:], start=1):
        cells = line.split(",")
        if int(cells[0]) in changes:
            cells[5] = f"{changes[int(cells[0])] / n:.4f}"
            lines[i] = ",".join(cells)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


SEED3 = "phonebook/experiments/week_2026-11-13/e7_w_d64_seed3/s3_n640/d64/metrics.csv"


def test_seed_three_has_no_dip_and_is_full_from_epoch_1500_which_the_next_two_tests_rely_on(inputs):
    rows = [line.split(",") for line in (inputs / SEED3).read_text(encoding="utf-8").splitlines()[1:]]
    assert all(c[5] == "1.0000" for c in rows if 1500 <= int(c[0]) <= 4000)
    assert mn.build(inputs)["e7w.dipmin.w64.s3"] == "none"


def test_a_dip_right_after_exactly_fifty_full_epochs_counts(inputs):
    set_recall(inputs / SEED3, {2000: 100})                       # 1500 to 1990 are 50 full logged epochs, then a dip
    assert mn.build(inputs)["e7w.dipmin.w64.s3"] == "100"


def test_a_dip_after_only_forty_nine_full_epochs_is_not_a_dip_yet(inputs):
    set_recall(inputs / SEED3, {1990: 100})                       # 1500 to 1980 are 49 full logged epochs: not settled
    assert mn.build(inputs)["e7w.dipmin.w64.s3"] == "none"


def test_the_deepest_of_several_dips_is_reported(inputs):
    set_recall(inputs / SEED3, {2000: 300, 2500: 7, 3000: 100})
    assert mn.build(inputs)["e7w.dipmin.w64.s3"] == "7"


def test_holding_needs_at_least_fifty_of_the_last_hundred(inputs):
    def change(d):
        for seed, count in (("1", 50), ("2", 49), ("3", 100)):
            d["16"]["epochs_at_100"]["by_seed"][seed]["3001_4000"] = count
    edit_json(inputs / "phonebook/results/e7w_extension.json", change)
    built = mn.build(inputs)
    assert (built["e7w.holding.w16"], built["e7w.holdingtotal"]) == ("2", "8")


@pytest.mark.parametrize("name, change, match", [
    ("e6_extension", lambda d: d["32"]["passes"].__setitem__("4000", "true"), "passes"),
    ("e4_capacity", lambda d: d["16"].__setitem__("boundary", "weird"), "boundary"),
    ("e4_capacity", lambda d: d["64"]["capacity_by_budget"].__setitem__("1000", 454), "updates per epoch"),
    ("e7_scores", lambda d: None, "e7_scores.json"),
])
def test_other_malformed_values_are_numbers_errors(inputs, name, change, match):
    path = inputs / f"phonebook/results/{name}.json"
    if name == "e7_scores":
        path.write_text("{}", encoding="utf-8")                     # a mapping where a list of scored rows belongs
    else:
        edit_json(path, change)
    with pytest.raises(mn.NumbersError, match=match):
        mn.build(inputs)


def test_a_file_that_is_not_json_is_a_numbers_error(inputs):
    (inputs / "phonebook/results/e6_capacity.json").write_text("{", encoding="utf-8")
    with pytest.raises(mn.NumbersError, match="not valid JSON"):
        mn.build(inputs)


E6_CSV = "phonebook/results/fig_e6_curves.csv"


def edit_curves(inputs: Path, change) -> None:
    path = inputs / E6_CSV
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(change(lines)) + "\n", encoding="utf-8")


def test_the_late_window_of_e6_is_epochs_3010_to_4000(inputs):
    def full(lines):          # width 64 seed 1 at epochs 3000 (outside) and 3010 (inside) recalled in full
        return [("64,640,1,%s,640" % l.split(",")[3]) if l.startswith("64,640,1,") and l.split(",")[3] in ("3000", "3010") else l for l in lines]
    edit_curves(inputs, full)
    built = mn.build(inputs)
    assert (built["e6.late100.w64.s1"], built["e6.logged100.w64.s1"]) == ("1", "2")


def test_a_missing_curve_or_one_with_a_different_entry_count_is_an_error(inputs):
    edit_curves(inputs, lambda lines: [l for l in lines if not l.startswith("16,160,1,")])
    with pytest.raises(mn.NumbersError, match="fig_e6_curves.csv"):
        mn.build(inputs)
    shutil.copy(ROOT / E6_CSV, inputs / E6_CSV)   # restore the curve, then change one entry count
    edit_curves(inputs, lambda lines: [l.replace("8,20,2,", "8,21,2,", 1) if l.startswith("8,20,2,") else l for l in lines])
    with pytest.raises(mn.NumbersError, match="fig_e6_curves.csv"):
        mn.build(inputs)


def test_a_mean_accuracy_of_exactly_zero_or_one_is_accepted(inputs):
    edit_json(inputs / "phonebook/results/e7w_extension.json", lambda d: d["16"]["mean_accuracy"].__setitem__("1001_2000", 0))
    assert mn.build(inputs)["e7w.mean.w16.early"] == "0.0000"


def test_check_mode_accepts_a_file_with_windows_line_endings(tmp_path):
    out = tmp_path / "numbers.tex"
    out.write_bytes(mn.render(mn.build(ROOT)).encode("utf-8").replace(b"\n", b"\r\n"))
    assert mn.main(["--check", "--out", str(out)]) == 0


def test_a_protocol_that_is_not_json_or_has_no_usable_runs_is_a_numbers_error(inputs):
    path = next((inputs / "phonebook/experiments/week_2026-10-31").glob("e6_extend_d8_seed1/protocol.json"))
    path.write_text("{", encoding="utf-8")
    with pytest.raises(mn.NumbersError, match="protocol.json"):
        mn.build(inputs)
    for runs in ([], [1], "x", [{}, 1]):
        path.write_text(json.dumps({"runs": runs}), encoding="utf-8")
        with pytest.raises(mn.NumbersError, match="protocol.json"):
            mn.build(inputs)


def test_an_interrupted_run_folder_is_never_read(inputs):
    source = next((inputs / "phonebook/experiments/week_2026-10-31").glob("e4_*/protocol.json")).parent
    ghost = source.parent / (source.name + "__interrupted")
    shutil.copytree(source, ghost)
    edit_json(ghost / "protocol.json", lambda d: [run.__setitem__("batch_size", 32) for run in d["runs"]])
    assert mn.build(inputs) == mn.build(ROOT)


def test_a_score_file_with_a_row_that_has_no_result_is_a_numbers_error(inputs):
    (inputs / "phonebook/results/e7_scores.json").write_text(json.dumps([{"result": "HIT"}, {"q": "Q2"}]), encoding="utf-8")
    with pytest.raises(mn.NumbersError, match="e7_scores.json"):
        mn.build(inputs)


def test_a_file_that_is_not_text_is_a_numbers_error(inputs):
    (inputs / "phonebook/results/e4_prediction_scores.json").write_bytes(b"\xff\xfe\x00")
    with pytest.raises(mn.NumbersError, match="cannot read"):
        mn.build(inputs)


def test_the_order_of_the_rows_in_a_metrics_file_does_not_matter(inputs):
    path = inputs / SEED3
    set_recall(path, {1990: 100})                                  # 49 full logged epochs, a dip, then full: not a dip after settling
    lines = path.read_text(encoding="utf-8").splitlines()
    row = next(i for i, line in enumerate(lines) if line.startswith("1990,"))
    lines.append(lines.pop(row))                                   # the dip row now comes last in the file
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert mn.build(inputs)["e7w.dipmin.w64.s3"] == "none"


def test_a_run_already_full_at_the_first_logged_epochs_needs_fifty_of_them_to_be_settled(inputs):
    set_recall(inputs / SEED3, {**{e: 640 for e in range(10, 500, 10)}, 500: 100})   # 49 full logged epochs, then a dip
    assert mn.build(inputs)["e7w.dipmin.w64.s3"] == "none"


@pytest.mark.parametrize("cell", ["x", "-1", "1.5", "1000000"])
def test_a_bad_cell_in_the_curves_file_is_a_numbers_error(inputs, cell):
    path = inputs / "phonebook/results/fig_e6_curves.csv"
    lines = path.read_text(encoding="utf-8").splitlines()
    cells = lines[3].split(",")
    cells[4] = cell                                              # recalled
    lines[3] = ",".join(cells)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(mn.NumbersError, match="fig_e6_curves.csv"):
        mn.build(inputs)


def test_no_protocol_files_for_an_experiment_is_an_error(inputs):
    for path in inputs.glob("phonebook/experiments/week_2026-11-13/e7_s_*/protocol.json"):
        path.unlink()
    with pytest.raises(mn.NumbersError, match="no protocol.json"):
        mn.build(inputs)


def test_an_arm_s_schedule_that_is_not_a_step_schedule_is_an_error(inputs):
    for path in inputs.glob("phonebook/experiments/week_2026-11-13/e7_s_*/protocol.json"):
        edit_json(path, lambda d: [run.__setitem__("lr_schedule", "linear:2000:0.1") for run in d["runs"]])
    with pytest.raises(mn.NumbersError, match="step"):
        mn.build(inputs)


def test_build_accepts_the_root_as_text():
    assert mn.build(str(ROOT)) == mn.build(ROOT)


def test_main_without_out_uses_report_numbers_tex_under_the_root_and_says_what_it_did(inputs, capsys):
    assert mn.main(["--root", str(inputs), "--check"]) == 1          # no report/numbers.tex there yet
    assert mn.main(["--root", str(inputs)]) == 0
    assert "wrote" in capsys.readouterr().out and (inputs / "report" / "numbers.tex").is_file()
    assert mn.main(["--root", str(inputs), "--check"]) == 0


def test_main_creates_missing_parent_folders_of_the_output(tmp_path):
    assert mn.main(["--out", str(tmp_path / "a" / "b" / "numbers.tex")]) == 0
    assert (tmp_path / "a" / "b" / "numbers.tex").is_file()


def test_the_module_runs_as_a_command_and_the_committed_file_is_up_to_date():
    done = subprocess.run([sys.executable, "-m", "report.make_numbers", "--check"], cwd=ROOT, capture_output=True, text=True)
    assert (done.returncode, "up to date" in done.stdout) == (0, True), done.stdout + done.stderr


def test_report_readme_documents_the_numbers_generator():
    text = (ROOT / "report" / "README.md").read_text(encoding="utf-8")
    for needle in ("make_numbers.py", "numbers.tex", "python -m report.make_numbers", "--check", "\\num{", "test_report_numbers.py"):
        assert needle in text, needle


# ---- the numbers themselves -----------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("key", sorted(PINNED))
def test_every_pinned_key_equals_its_typed_literal(built, key):
    assert built[key] == PINNED[key]


def test_every_pinned_key_is_a_key_the_generator_writes_and_a_key_the_raw_recompute_makes(built, raw):
    assert set(PINNED) <= set(built) and set(PINNED) <= set(raw)


def test_the_generator_and_the_raw_recompute_have_exactly_the_same_keys(built, raw):
    assert sorted(set(built) ^ set(raw)) == []


def test_every_value_equals_the_independent_recompute_from_the_raw_per_run_files(built, raw):
    wrong = {k: (built[k], raw[k]) for k in raw if built.get(k) != raw[k]}
    assert wrong == {}


def test_the_key_count_is_the_one_the_plan_documents(built):
    assert len(built) == 268   # e4 39, e6 92, e7w 52, e7s 49, e7 7, design 29


def test_build_is_deterministic():
    assert mn.build(ROOT) == mn.build(ROOT)


# ---- the committed file and the command line ------------------------------------------------------------------------------------------------
def test_the_committed_numbers_tex_is_what_the_generator_writes_now(built):
    committed = (ROOT / "report" / "numbers.tex").read_bytes().replace(b"\r\n", b"\n")   # git for Windows may check out CRLF
    assert committed == mn.render(built).encode("utf-8")


def test_main_writes_the_file_and_check_mode_accepts_it_then_rejects_a_changed_copy(tmp_path, capsys):
    out = tmp_path / "numbers.tex"
    assert mn.main(["--out", str(out)]) == 0
    assert out.read_bytes() == mn.render(mn.build(ROOT)).encode("utf-8")
    assert mn.main(["--check", "--out", str(out)]) == 0
    out.write_bytes(out.read_bytes().replace(b"{14}", b"{15}", 1))
    capsys.readouterr()
    assert mn.main(["--check", "--out", str(out)]) == 1
    assert "differs" in capsys.readouterr().out


def test_main_check_mode_never_writes(tmp_path):
    out = tmp_path / "numbers.tex"
    assert mn.main(["--check", "--out", str(out)]) == 1
    assert not out.exists()


def test_main_returns_two_when_an_input_is_missing(tmp_path, capsys):
    assert mn.main(["--root", str(tmp_path), "--out", str(tmp_path / "n.tex")]) == 2
    assert "numbers" in capsys.readouterr().err


# ---- a real TeX build ------------------------------------------------------------------------------------------------------------------------
# Needs pdflatex on PATH. There is no skip: a machine without TeX must fail these (see report/README.md).
def build_document(tmp_path: Path, body: str):
    src = tmp_path / "src"
    src.mkdir()
    (src / "numbers.tex").write_bytes(mn.render(mn.build(ROOT)).encode("utf-8"))
    (src / "main.tex").write_text("\\documentclass{article}\n\\input{numbers}\n\\begin{document}\n" + body + "\n\\end{document}\n", encoding="utf-8")
    return gate.build_pdf(src / "main.tex", src / "build")


@pytest.mark.integration
def test_keys_print_their_values_on_a_real_build(tmp_path):
    result = build_document(tmp_path, "Batch \\num{design.batch}, bits \\num{e7w.bits.w64}, parameters \\num{e4.params.w64}, none \\num{e4.cap.w8.b100}.")
    assert br.parse_log(result.log_text)["errors"] == []
    text = " ".join(" ".join(gate.pdf_pages(result.pdf)).split())
    assert "Batch 64, bits 0.188, parameters 110,656, none none." in text


@pytest.mark.integration
def test_an_undefined_key_is_a_latex_error_on_a_real_build_so_the_gate_cannot_pass_it(tmp_path):
    result = build_document(tmp_path, "Missing: [\\num{no.such.key}].")
    errors = br.parse_log(result.log_text)["errors"]
    assert any(e.startswith("Package numbers Error: Undefined number key `no.such.key'") for e in errors)
