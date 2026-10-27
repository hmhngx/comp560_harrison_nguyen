from __future__ import annotations


def print_default_args_message(n: int, charset_size: int, num_lines: int) -> None:
    print(
        "No parameters passed. Using defaults: "
        f"n={n}, charset={charset_size}, num_lines={num_lines}"
    )
