from __future__ import annotations

from pathlib import Path
import random
import string
from collections.abc import Callable


def bounded_charset(charset_size: int) -> str:
    safe_size = min(max(1, charset_size), 26)
    return string.ascii_lowercase[:safe_size]


def resolve_charset(raw_charset: str) -> str:
    if raw_charset.isdigit():
        return bounded_charset(int(raw_charset))
    return raw_charset


def print_generation_header(
    n: int,
    charset: str,
    num_lines: int,
    output_filename: str | Path,
) -> None:
    print("Generating dataset...")
    print(f"Max string length (n) : {n}")
    print(f"Using charset [{len(charset)}]: {charset}")
    print(f"Writing {num_lines} lines to {output_filename}...\n")


def default_inputs_path(filename: str) -> Path:
    return Path("inputs") / filename


def write_lines(output_filename: str | Path, num_lines: int, builder: Callable[[], str]) -> None:
    output_path = Path(output_filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for _ in range(num_lines):
            fh.write(builder())


def random_string(charset: str, length: int) -> str:
    return "".join(random.choice(charset) for _ in range(length))
