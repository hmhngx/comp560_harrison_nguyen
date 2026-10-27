from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Vocabulary:
    vocab_size: int
    stoi: dict[str, int]
    itos: dict[int, str]

    pad_id: int = field(init=False)
    equal_id: int = field(init=False)
    eot_id: int = field(init=False)
    is_2char: bool = field(init=False)

    def __post_init__(self) -> None:
        self.pad_id = self.stoi["_"]
        self.equal_id = self.stoi["="]
        self.eot_id = self.stoi["\n"]
        self.is_2char = any(len(k) == 2 for k in self.stoi.keys())

    @classmethod
    def from_pickle(cls, path: Path) -> "Vocabulary":
        if not path.exists():
            raise FileNotFoundError(
                f"Vocabulary file not found: {path}. Run prepare.py first."
            )
        with path.open("rb") as fh:
            meta = pickle.load(fh)
        return cls(
            vocab_size=meta["vocab_size"],
            stoi=meta["stoi"],
            itos=meta["itos"],
        )

    def tokenize_string(self, text: str) -> list[str]:
        if not self.is_2char:
            return list(text)
        if len(text) % 2 != 0:
            text += "_"
        return [text[i:i + 2] for i in range(0, len(text), 2)]
