"""Pinned replay membership and prepared text; no training is performed here.

python phonebook/replay_protocol.py --out-root NEW_DIRECTORY
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from completion_core.prep import build_char_vocab
from completion_core.validation import is_torch_seed
from phonebook.prepare_phonebook import REQUIRED_ALNUM_CHARS

SCHEMA = "phonebook-replay-membership-v1"
ALGORITHM = "sha256-rank-v1"


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_digest(obj) -> str:
    return digest_bytes(json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"))


def read_json(path: Path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    def constant(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")
    obj = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=constant)
    canonical_digest(obj)  # Reject valid JSON numeric syntax that overflows floats.
    return obj


def _keys(obj, names, where):
    if not isinstance(obj, dict) or set(obj) != set(names):
        raise ValueError(f"{where} must have exactly these fields: {sorted(names)}")


def _resolve(root, path):
    path = Path(path)
    return path if path.is_absolute() else Path(root) / path


def _normalized(data):
    return data.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")


def _book(path, root):
    text = _normalized(_resolve(root, path).read_bytes())
    if not text:
        raise ValueError("Books must be nonempty")
    lines = text.split("\n")
    if lines[-1] == "":
        lines.pop()
    records, prompts = [], set()
    for line in lines:
        if line.count("=") != 1:
            raise ValueError("Each record must have exactly one '=' delimiter")
        prompt, target = line.split("=")
        if not prompt or not target or prompt != prompt.strip() or target != target.strip() or "_" in line:
            raise ValueError("Records need nonempty, unpadded prompt/target with no surrounding whitespace")
        if any(ord(c) < 32 or ord(c) == 127 for c in line):
            raise ValueError("Control characters are not allowed inside records")
        if prompt in prompts:
            raise ValueError("Duplicate or conflicting prompts are not allowed")
        prompts.add(prompt)
        records.append({"id": digest_bytes((line + "\n").encode("utf-8")), "prompt": prompt, "target": target})
    canonical = "".join(f"{r['prompt']}={r['target']}\n" for r in records)
    return {"source": str(path).replace("\\", "/"), "normalized_sha256": digest_bytes(canonical.encode()),
            "records": records}


def _rank(ids, seed):
    return sorted(ids, key=lambda entry: (canonical_digest([ALGORITHM, seed, entry]), entry))


def _unique_ints(values, name, *, seeds=False):
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"{name} must be a nonempty list")
    for value in values:
        valid = is_torch_seed(value) if seeds else (type(value) is int and value >= 1)
        if not valid:
            raise ValueError(f"Invalid {name} value: {value!r}")
    if len(set(values)) != len(values):
        raise ValueError(f"Duplicate {name} values are not allowed")


def build_protocol(book_a, book_b, *, probe_size=3, probe_seed=1733,
                   subset_seeds=(1701,), memory_sizes=(1, 2, 5), repo_root=ROOT):
    _unique_ints(subset_seeds, "subset_seeds", seeds=True)
    _unique_ints(memory_sizes, "memory_sizes")
    if not is_torch_seed(probe_seed) or isinstance(probe_size, bool) or not isinstance(probe_size, int) or probe_size < 1:
        raise ValueError("probe_seed and probe_size must be valid integers; probe_size must be positive")
    books = {"A": _book(book_a, repo_root), "B": _book(book_b, repo_root)}
    a, b = books["A"]["records"], books["B"]["records"]
    if {r["prompt"] for r in a} & {r["prompt"] for r in b}:
        raise ValueError("A/B prompt overlap is not allowed, including conflicting targets")
    a_ids, b_ids = [r["id"] for r in a], [r["id"] for r in b]
    if len(set(a_ids + b_ids)) != len(a_ids + b_ids):
        raise ValueError("Record IDs must be unique across books")
    if max(len(r["prompt"]) + len(r["target"]) + 2 for r in a) != \
            max(len(r["prompt"]) + len(r["target"]) + 2 for r in b):
        raise ValueError("A/B must have equal longest record length under the current inferred-context trainer")
    if probe_size >= len(a):
        raise ValueError("Probe must leave a nonempty replay pool")
    probe = _rank(a_ids, probe_seed)[:probe_size]
    pool = [entry for entry in a_ids if entry not in probe]
    if max(memory_sizes) > len(pool):
        raise ValueError("Memory size exceeds the replay pool after excluding the probe")
    draws = [{"seed": seed, "permutation": _rank(pool, seed)} for seed in subset_seeds]
    lookup = {r["id"]: f"{r['prompt']}={r['target']}\n" for r in a + b}
    conditions = []
    def add(name, kind, replay):
        training = list(replay) + b_ids
        conditions.append({"id": name, "kind": kind, "initialization": "fresh" if kind == "fresh_joint" else "phase_a",
                           "replay_ids": list(replay), "training_ids": training,
                           "training_sha256": digest_bytes("".join(lookup[i] for i in training).encode())})
    add("plain_b", "b_only", [])
    for draw in draws:
        for size in memory_sizes:
            add(f"partial_m{size}_seed{draw['seed']}", "partial", draw["permutation"][:size])
    add("full_replay", "full_replay", a_ids)
    add("fresh_joint", "fresh_joint", a_ids)
    text = "".join(lookup.values())
    stoi, _ = build_char_vocab(text + REQUIRED_ALNUM_CHARS, "_")
    return {"schema": SCHEMA, "selection": {"algorithm": ALGORITHM, "probe_size": probe_size,
            "probe_seed": probe_seed, "subset_seeds": list(subset_seeds), "memory_sizes": list(memory_sizes)},
            "books": books, "probe_ids": probe, "replay_pool_ids": pool, "draws": draws,
            "context_length": max(len(line) for line in lookup.values()), "vocabulary": stoi,
            "conditions": conditions}


@dataclass(frozen=True)
class Record:
    id: str
    prompt: str
    target: str

    @property
    def line(self):
        return f"{self.prompt}={self.target}\n"


@dataclass(frozen=True)
class Membership:
    protocol_sha256: str
    condition_id: str
    kind: str
    a: tuple[Record, ...]
    b: tuple[Record, ...]
    probe_ids: tuple[str, ...]
    replay_ids: tuple[str, ...]
    training_ids: tuple[str, ...]
    training_sha256: str
    context_length: int
    vocabulary: tuple[tuple[str, int], ...]

    def partitions(self):
        replay = set(self.replay_ids)
        return {"A_all": tuple(r.id for r in self.a), "A_replayed": self.replay_ids,
                "A_unreplayed": tuple(r.id for r in self.a if r.id not in replay),
                "A_probe": self.probe_ids, "B": tuple(r.id for r in self.b)}

    def training_text(self):
        lookup = {r.id: r.line for r in self.a + self.b}
        return "".join(lookup[entry] for entry in self.training_ids)


def validate_protocol(protocol, expected_digest, *, repo_root=ROOT):
    if not isinstance(expected_digest, str) or len(expected_digest) != 64 \
            or any(c not in "0123456789abcdef" for c in expected_digest):
        raise ValueError("A pinned 64-character lowercase protocol digest is required")
    if canonical_digest(protocol) != expected_digest:
        raise ValueError("Protocol differs from the external digest pin")
    _keys(protocol, {"schema", "selection", "books", "probe_ids", "replay_pool_ids", "draws",
                     "context_length", "vocabulary", "conditions"}, "protocol")
    if protocol["schema"] != SCHEMA:
        raise ValueError("Unsupported protocol schema")
    selection = protocol["selection"]
    _keys(selection, {"algorithm", "probe_size", "probe_seed", "subset_seeds", "memory_sizes"}, "selection")
    if selection["algorithm"] != ALGORITHM:
        raise ValueError("Unsupported selection algorithm")
    _keys(protocol["books"], {"A", "B"}, "books")
    for book in protocol["books"].values():
        _keys(book, {"source", "normalized_sha256", "records"}, "book")
        if not isinstance(book["source"], str) or not book["source"]:
            raise ValueError("Book source must be a nonempty path")
    expected = build_protocol(protocol["books"]["A"]["source"], protocol["books"]["B"]["source"],
                              repo_root=repo_root, **{k: v for k, v in selection.items() if k != "algorithm"})
    if canonical_digest(expected) != canonical_digest(protocol):
        raise ValueError("Protocol membership, records, vocabulary or context do not match the source books and selection")
    return protocol


def membership(protocol, condition_id, expected_digest, *, repo_root=ROOT):
    validate_protocol(protocol, expected_digest, repo_root=repo_root)
    condition = next((c for c in protocol["conditions"] if c["id"] == condition_id), None)
    if condition is None:
        raise ValueError(f"Unknown condition: {condition_id}")
    records = {name: tuple(Record(**r) for r in protocol["books"][name]["records"]) for name in "AB"}
    return Membership(expected_digest, condition_id, condition["kind"], records["A"], records["B"],
                      tuple(protocol["probe_ids"]), tuple(condition["replay_ids"]), tuple(condition["training_ids"]),
                      condition["training_sha256"], protocol["context_length"], tuple(protocol["vocabulary"].items()))


def validate_training_input(selected: Membership, path: Path):
    data = Path(path).read_bytes()
    text = _normalized(data)
    if text != selected.training_text() or digest_bytes(text.encode()) != selected.training_sha256:
        raise ValueError("Prepared training text differs from the pinned membership, order or targets")
    return {"sha256": digest_bytes(data), "normalized_sha256": selected.training_sha256,
            "entry_ids": list(selected.training_ids)}


def write_protocol(protocol, out_root, *, repo_root=ROOT):
    pin = canonical_digest(protocol)
    validate_protocol(protocol, pin, repo_root=repo_root)
    payload = json.dumps(protocol, indent=2, allow_nan=False) + "\n"
    lookup = {r["id"]: f"{r['prompt']}={r['target']}\n" for book in protocol["books"].values() for r in book["records"]}
    texts = {c["id"]: "".join(lookup[i] for i in c["training_ids"]) for c in protocol["conditions"]}
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=False)
    (out_root / "protocol.json").write_text(payload, encoding="utf-8", newline="\n")
    for name, text in texts.items():
        folder = out_root / name
        folder.mkdir()
        (folder / "train.txt").write_text(text, encoding="utf-8", newline="\n")
    return pin


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phonebook-a", type=Path, default=Path("phonebook/inputs/phonebookA.txt"))
    parser.add_argument("--phonebook-b", type=Path, default=Path("phonebook/inputs/phonebookB.txt"))
    parser.add_argument("--probe-size", type=int, default=3)
    parser.add_argument("--probe-seed", type=int, default=1733)
    parser.add_argument("--subset-seeds", type=int, nargs="+", default=[1701])
    parser.add_argument("--memory-sizes", type=int, nargs="+", default=[1, 2, 5])
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()
    if args.out_root.exists():
        parser.error("Output root already exists; choose an unused directory")
    protocol = build_protocol(args.phonebook_a, args.phonebook_b, probe_size=args.probe_size,
                              probe_seed=args.probe_seed, subset_seeds=args.subset_seeds, memory_sizes=args.memory_sizes)
    print(write_protocol(protocol, args.out_root))


if __name__ == "__main__":
    main()
