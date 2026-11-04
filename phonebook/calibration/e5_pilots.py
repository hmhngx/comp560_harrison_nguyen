"""Scratch pilots behind the locked E5 predictions: shuffle-seed timing noise and the k effect. NOT evidence.

Same recipe as E5 (lr as given, full batch, adapt from a Phase-A checkpoint, accuracy logged every update), run with the
isolated loader stream. Two kinds of pilot:

  d0          the E3 phonebook and its three pinned Phase-A checkpoints (hashes checked against E3's spec); several shuffle seeds
              per cell, so the spread between seeds is pure shuffle noise
  phonebooks  phonebooks that E5 does not use (any qualifying generator seed): train three Phase-A models per phonebook
              (init seeds 42, 123, 999), then replay A once + B x k

    python phonebook/calibration/e5_pilots.py d0 --work WORK --out OUT.json [--lr 0.001] [--ks 1,2,3,4,5,9] [--seeds 42,1,2,3,4]
        [--epochs 120] [--workers 10] [--threads 1|default] [--tag ""]
    python phonebook/calibration/e5_pilots.py phonebooks --work WORK --out OUT.json --pbs 15,17,18,20 [--stage main|lr003] [--workers 8]

Result files map a cell key to {stable_joint, first_joint, min_a, first_b}. d0 keys read `d0_seed42|k3|s42`; phonebook keys read
`PB|INIT|K|LR|SHUFFLE_SEED|EPOCHS`. A cell that fails to train is reported and the command exits 1.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from completion_core.vocabulary import Vocabulary
from phonebook.analyze_experiment import curve, load_metrics, summarize_curve
from phonebook.evaluate_checkpoint import sha256
from phonebook.make_heldout_dataset import generate_phonebook_lines, split_books
from phonebook.make_replay_input import build_mixture_lines
from phonebook.prepare_phonebook import load_phonebook_lines

DEFAULT_WORK = Path(tempfile.gettempdir()) / "phonebook_calibration"
D0_CHECKPOINTS = {"d0_seed42": "phonebook/out_phaseA/model.pth", "d0_seed123": "phonebook/out_phaseA_seed123/model.pth",
                  "d0_seed999": "phonebook/out_phaseA_seed999/model.pth"}
INITS = (42, 123, 999)
KS = (1, 2, 3, 4, 5, 9)
ROBUST_KS = (1, 2, 3, 5)                             # the k values of the lr 0.0003 and lr 0.003 arms


def train_env(threads: str = "1") -> dict:
    env = dict(os.environ, MPLBACKEND="Agg")
    if threads == "1":
        env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    return env


def train(cmd: list[str], env: dict) -> int:
    done = subprocess.run(cmd, cwd=ROOT_DIR, capture_output=True, text=True, env=env)
    if done.returncode != 0:
        print("FAILED", " ".join(cmd[-10:]), done.stderr[-400:], flush=True)
    return done.returncode


def prepare_dir(work_inputs: Path, out: Path, train_lines: list[str], eval_lines: list[str]) -> Path:
    work_inputs.mkdir(parents=True, exist_ok=True)
    (work_inputs / "train.txt").write_text("".join(train_lines), encoding="utf-8", newline="\n")
    (work_inputs / "eval.txt").write_text("".join(eval_lines), encoding="utf-8", newline="\n")
    if not (out / "train.bin").exists():
        subprocess.run([sys.executable, "phonebook/prepare_phonebook.py", "--train-input", str(work_inputs / "train.txt"),
                        "--eval-input", str(work_inputs / "eval.txt"), "--out-dir", str(out)],
                       cwd=ROOT_DIR, check=True, capture_output=True, text=True)
    return out


def summarize_metrics(metrics: Path, k: int) -> dict:
    s = summarize_curve(curve(load_metrics(metrics), a_copies=1, b_copies=k, eval_book="A", n_a=10, n_b=10), n_a=10, n_b=10, budgets=(100,))
    return {"stable_joint": s["stable_joint"], "first_joint": s["first_joint"], "first_b": s["first_b_complete"], "min_a": s["min_a"]}


# ------------------------------------------------------------------ d0: the E3 phonebook

def verify_d0_pins() -> None:
    """The three Phase-A checkpoints must be the ones E3 pinned."""
    spec = json.loads((ROOT_DIR / "phonebook" / "specs" / "e3_replay_ratio.json").read_text(encoding="utf-8"))
    pins = {scenario["id"]: scenario["init"] for scenario in spec["scenarios"]}
    for name, relative in D0_CHECKPOINTS.items():
        pin = pins[name]
        if pin["checkpoint"] != relative:
            raise ValueError(f"{name}: E3 pinned {pin['checkpoint']}, not {relative}")
        if sha256(ROOT_DIR / relative) != pin["checkpoint_sha256"]:
            raise ValueError(f"{name}: the checkpoint on disk differs from E3's pinned hash")


def d0_data(work: Path, k: int) -> Path:
    a = load_phonebook_lines(ROOT_DIR / "phonebook" / "inputs" / "phonebookA.txt")
    b = load_phonebook_lines(ROOT_DIR / "phonebook" / "inputs" / "phonebookB.txt")
    return prepare_dir(work / "inputs" / f"k{k}", work / "data" / f"k{k}", build_mixture_lines(a, b, a_copies=1, b_copies=k), a)


def d0_cell_dir(work: Path, lr: str, tag: str, scenario: str, k: int, seed: int) -> Path:
    return work / f"lr{lr}{tag}" / scenario / f"k{k}_s{seed}"


def d0_run(work: Path, lr: str, tag: str, epochs: int, threads: str, job: tuple) -> int:
    scenario, k, seed = job
    out_dir = d0_cell_dir(work, lr, tag, scenario, k, seed)
    if (out_dir / "metrics.csv").exists() and (out_dir / "model.pth").exists():
        return 0
    cmd = [sys.executable, "train.py", "phonebook/config/phonebook.py", "--device", "cpu", "--data-dir", str(d0_data(work, k)),
           "--out-dir", str(out_dir), "--seed", str(seed), "--lr", lr, "--epochs", str(epochs), "--batch-size", "128",
           "--accuracy-interval", "1", "--l2sp-lambda", "0.0", "--adapt-from", str(ROOT_DIR / D0_CHECKPOINTS[scenario])]
    return train(cmd, train_env(threads))


def run_d0(args) -> int:
    verify_d0_pins()
    ks = [int(x) for x in args.ks.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    for k in ks:
        d0_data(args.work, k)
    jobs = [(s, k, seed) for s in D0_CHECKPOINTS for k in ks for seed in seeds]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        codes = list(pool.map(lambda job: d0_run(args.work, args.lr, args.tag, args.epochs, args.threads, job), jobs))
    if any(codes):
        print(f"{sum(1 for c in codes if c)} of {len(jobs)} cells failed; no results written")
        return 1
    results = {f"{s}|k{k}|s{seed}": summarize_metrics(d0_cell_dir(args.work, args.lr, args.tag, s, k, seed) / "metrics.csv", k)
               for s, k, seed in jobs}
    args.out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"wrote {args.out}: {len(results)} cells")
    return 0


# ------------------------------------------------------------------ phonebooks that E5 does not use

def main_jobs(pbs: list[int], inits: tuple = INITS) -> list[tuple]:
    """(phonebook, init, k, lr, shuffle seed, epochs): the k grid at lr 0.001, extra shuffle seeds for k=1,3 on init 42, a small lr 0.0003 arm."""
    jobs = [(pb, i, k, "0.001", 42, 120) for pb in pbs for i in inits for k in KS]
    jobs += [(pb, 42, k, "0.001", sh, 120) for pb in pbs for k in (1, 3) for sh in (1, 2)]
    jobs += [(pb, i, k, "0.0003", 42, 220) for pb in pbs for i in inits for k in ROBUST_KS]
    return jobs


def lr003_jobs(pbs: list[int], inits: tuple = INITS) -> list[tuple]:
    return [(pb, i, k, "0.003", 42, 500) for pb in pbs for i in inits for k in ROBUST_KS]


def job_key(job: tuple) -> str:
    return "|".join(map(str, job))


def pb_setup(work: Path, pb: int) -> Path:
    lines = [line + "\n" for line in generate_phonebook_lines(pb)]
    a, b = split_books(lines)
    root = work / f"pb{pb}"
    phase_a = prepare_dir(root / "inputs_a1b0_evalB", root / "data_a1b0_evalB", a, b)
    for k in KS:
        data = prepare_dir(root / f"inputs_a1b{k}_evalA", root / f"data_a1b{k}_evalA", build_mixture_lines(a, b, a_copies=1, b_copies=k), a)
        if Vocabulary.from_pickle(data / "meta.pkl").stoi != Vocabulary.from_pickle(phase_a / "meta.pkl").stoi:
            raise ValueError(f"phonebook {pb}: the replay vocabulary differs from Phase A's")
    return phase_a


def pb_phase_a(work: Path, pb: int, init: int, data: Path) -> int:
    out = work / f"pb{pb}" / f"phaseA_s{init}"
    if (out / "model.pth").exists():
        return 0
    return train([sys.executable, "train.py", "phonebook/config/phonebook.py", "--device", "cpu", "--data-dir", str(data), "--out-dir", str(out),
                  "--seed", str(init), "--lr", "0.001", "--epochs", "100", "--batch-size", "64", "--accuracy-interval", "10",
                  "--l2sp-lambda", "0.0"], train_env())


def pb_cell_dir(work: Path, job: tuple) -> Path:
    pb, init, k, lr, seed, _ = job
    return work / f"pb{pb}" / f"lr{lr}" / f"s{init}_k{k}_sh{seed}"


def pb_replay(work: Path, job: tuple) -> int:
    pb, init, k, lr, seed, epochs = job
    out = pb_cell_dir(work, job)
    if (out / "metrics.csv").exists() and (out / "model.pth").exists():
        return 0
    return train([sys.executable, "train.py", "phonebook/config/phonebook.py", "--device", "cpu", "--data-dir", str(work / f"pb{pb}" / f"data_a1b{k}_evalA"),
                  "--out-dir", str(out), "--seed", str(seed), "--lr", str(lr), "--epochs", str(epochs), "--batch-size", "128",
                  "--accuracy-interval", "1", "--l2sp-lambda", "0.0", "--adapt-from", str(work / f"pb{pb}" / f"phaseA_s{init}" / "model.pth")], train_env())


def run_phonebooks(args) -> int:
    pbs = [int(x) for x in args.pbs.split(",")]
    datas = {pb: pb_setup(args.work, pb) for pb in pbs}
    with ThreadPoolExecutor(args.workers) as pool:
        codes = list(pool.map(lambda x: pb_phase_a(args.work, *x), [(pb, i, datas[pb]) for pb in pbs for i in INITS]))
    if any(codes):
        print("a Phase-A model failed to train; no results written")
        return 1
    for pb in pbs:                                       # every Phase-A model must recall its own book in full
        for i in INITS:
            last = load_metrics(args.work / f"pb{pb}" / f"phaseA_s{i}" / "metrics.csv")[-1]
            if last["Train Seq Acc"] != 1.0:
                raise ValueError(f"phonebook {pb}, init {i}: Phase A reached {last['Train Seq Acc']} at epoch {last['epoch']}, not 1.0")
    jobs = main_jobs(pbs) if args.stage == "main" else lr003_jobs(pbs)
    with ThreadPoolExecutor(args.workers) as pool:
        codes = list(pool.map(lambda job: pb_replay(args.work, job), jobs))
    if any(codes):
        print(f"{sum(1 for c in codes if c)} of {len(jobs)} cells failed; no results written")
        return 1
    results = {job_key(job): summarize_metrics(pb_cell_dir(args.work, job) / "metrics.csv", job[2]) for job in jobs}
    args.out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"wrote {args.out}: {len(results)} cells")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    d0 = sub.add_parser("d0")
    d0.add_argument("--work", type=Path, default=DEFAULT_WORK)
    d0.add_argument("--out", type=Path, required=True)
    d0.add_argument("--lr", default="0.001")
    d0.add_argument("--ks", default="1,2,3,4,5,9")
    d0.add_argument("--seeds", default="42,1,2,3,4")
    d0.add_argument("--epochs", type=int, default=120)
    d0.add_argument("--workers", type=int, default=10)
    d0.add_argument("--threads", choices=("1", "default"), default="1")
    d0.add_argument("--tag", default="")
    pbs = sub.add_parser("phonebooks")
    pbs.add_argument("--work", type=Path, default=DEFAULT_WORK)
    pbs.add_argument("--out", type=Path, required=True)
    pbs.add_argument("--pbs", required=True)
    pbs.add_argument("--stage", choices=("main", "lr003"), default="main")
    pbs.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    try:
        return run_d0(args) if args.command == "d0" else run_phonebooks(args)
    except ValueError as error:
        sys.exit(f"error: {error}")


if __name__ == "__main__":
    sys.exit(main())
