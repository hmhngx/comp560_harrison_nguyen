r"""Mechanical checks on the prose of the report (main.tex): its numbers and its claims.

    python -m report.check_text report/main.tex [--final] [--root REPO]

Numbers (IC-1, G7). Every result in the text comes from numbers.tex through \num{key}. A problem is an undefined key, and a number typed in the prose that
equals a generated value (of three digits or more, or with a decimal point; small whole numbers are structure). Listed for your own read, never a problem:
other numbers typed in the prose, and the keys that nothing uses.

Claims (G7). The report's claims are the accepted ledger's CL-1 to CL-7. Put a marker comment on the line above the paragraph that states a claim,
with a blank line before it:

    % claim: CL-4
    The claim that a wide model keeps ... was Not true in E6 ...

The paragraph, up to the next blank line, must contain the claim's status words (copied from its row of phonebook/RESULTS.md; case, markup, line breaks and
hyphens do not matter) and none of the claim's "do not say" phrases or of the words prove, impossible, optimal and guarantee. A marker may name several
claims (% claim: CL-2, CL-3) and markers may be stacked. Text outside marked paragraphs is not checked. With --final every claim must be marked at least once.
"No stronger than the ledger" is still a human read; this check removes the mechanical ways to get it wrong.

Exit status: 0 no problems; 1 problems (printed, one per line); 2 the file cannot be read or the numbers cannot be built.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from report import build_report as br
from report import make_numbers as mn


@dataclass(frozen=True)
class Claim:
    required: tuple    # ((label, regex), ...): every pattern must match the prose of the paragraph
    forbidden: tuple   # ((label, regex), ...): no pattern may match


NOT = r"(?<!not )(?<!never )(?<!not yet )(?<!not been )(?<!not yet been )(?<!n t )"   # "not established" is the opposite of "established"; so are "never", "not yet", "isn't"; the prose has single spaces and lower case
ESTABLISHED = NOT + r"\bestablished\b"
SUPPORTED = NOT + r"\bsupported\b"

# The accepted ledger (v2, 2026-11-13), section 1 (status words) and each claim's "Do not say" line.
CLAIMS = {
    "CL-1": Claim((("Established", ESTABLISHED),), (("can store", r"\bcan store\b"), ("has a capacity of", r"\bhas a capacity of\b"))),
    "CL-2": Claim((("Supported", SUPPORTED), ("not shown for width 64", r"\bnot shown for width 64\b")), (("established", ESTABLISHED), ("shown", NOT + r"\bshown\b"))),
    "CL-3": Claim((("Established", ESTABLISHED),), (("holds", r"\bholds\b"), ("keeps", r"\bkeeps\b"))),
    "CL-4": Claim((("Not true in E6", r"\bnot true in e6\b"),), (("cannot hold", r"\bcannot hold\b"),)),
    "CL-5": Claim((("Not established", r"\bnot established\b"),), ()),
    "CL-6": Claim((("lower bound", r"\blower bounds?\b"),), ()),
    "CL-7": Claim((("Supported", SUPPORTED),), (("established", ESTABLISHED),)),
}
GLOBAL_FORBIDDEN = (("prove", r"\bprove[sd]?\b|\bproven\b|\bproof\b"), ("impossible", r"\bimpossible\b"), ("optimal", r"\boptimal(?:ly)?\b"),
                    ("guarantee", r"\bguarantee[sd]?\b"))
FINAL_CLAIMS = tuple(CLAIMS)

_MARKER = re.compile(r"^\s*%\s*claim\s*:(.*)$")
_MARKER_IDS = re.compile(r"\s*(CL-\d+)(?:\s*,\s*(CL-\d+))*\s*")
_ID = re.compile(r"CL-\d+")
MARKER_FORM = "a claim marker must look like '% claim: CL-4'"


@dataclass(frozen=True)
class ClaimParagraph:
    ids: tuple
    line: int
    text: str


@dataclass(frozen=True)
class Report:
    problems: list
    notes: list


def prose(text: str) -> str:
    """The words of a piece of TeX source, lower case, single spaces: comments, the arguments of cite, ref, label, input and num commands, every other
    command name, and all punctuation and markup are gone (so "\\textit{Not true in E6}" and "not\ntrue in  E6." are the same words)."""
    body = mn.COMMAND_ARGUMENTS.sub(" ", br.strip_comments(text))
    body = re.sub(r"\\[a-zA-Z]+\*?", " ", body)
    return " ".join(re.sub(r"[^a-z0-9]+", " ", body.lower()).split())


def _lf(tex: str) -> str:
    """The source with every kind of line ending (CRLF, a lone CR) as a newline, so the paragraphs are the same whatever the file's line endings."""
    return tex.replace("\r\n", "\n").replace("\r", "\n")


def _scan(tex: str) -> tuple[list[ClaimParagraph], list[tuple[int, str]]]:
    """(the claim paragraphs, the marker problems as (line, text)) of a source; a paragraph runs from its marker block to the next blank line."""
    lines = _lf(tex).split("\n")
    paragraphs: list[ClaimParagraph] = []
    problems: list[tuple[int, str]] = []
    i, after_blank = 0, True
    while i < len(lines):
        if not _MARKER.match(lines[i]):
            after_blank = not lines[i].strip()
            i += 1
            continue
        first, ids = i, []
        if not after_blank:
            problems.append((first + 1, "a claim marker must start a paragraph (put a blank line before it)"))
        while i < len(lines) and _MARKER.match(lines[i]):
            rest = _MARKER.match(lines[i]).group(1)
            if _MARKER_IDS.fullmatch(rest) is None:
                problems.append((i + 1, MARKER_FORM))
            else:
                for claim in _ID.findall(rest):
                    if claim not in CLAIMS:
                        problems.append((i + 1, f"unknown claim '{claim}' (use CL-1 to CL-{len(CLAIMS)})"))
                    ids.append(claim)
            i += 1
        body_start = i
        while i < len(lines) and lines[i].strip() and not _MARKER.match(lines[i]):
            i += 1
        if i == body_start:
            problems.append((first + 1, "claim marker with no paragraph after it"))
        elif ids:
            paragraphs.append(ClaimParagraph(tuple(ids), first + 1, "\n".join(lines[body_start:i])))
        after_blank = False
    return paragraphs, problems


def claim_paragraphs(tex: str) -> list[ClaimParagraph]:
    return _scan(tex)[0]


def claim_problems(tex: str, *, final: bool = False) -> list[str]:
    paragraphs, scan = _scan(tex)
    found: list[tuple[int, str]] = list(scan)
    marked = set()
    for paragraph in paragraphs:
        words = prose(paragraph.text)
        for claim in paragraph.ids:
            if claim not in CLAIMS:
                continue
            marked.add(claim)
            spec = CLAIMS[claim]
            for label, pattern in spec.required:
                if re.search(pattern, words) is None:
                    found.append((paragraph.line, f"{claim} must contain '{label}'"))
            for label, pattern in spec.forbidden + GLOBAL_FORBIDDEN:
                if re.search(pattern, words):
                    found.append((paragraph.line, f"{claim} must not contain '{label}'"))
    found.sort(key=lambda item: item[0])      # stable: file order, then the order the rules are listed in
    out = [f"line {line}: {message}" for line, message in found]
    if final:
        out += [f"{claim}: no paragraph is marked '% claim: {claim}'" for claim in FINAL_CLAIMS if claim not in marked]
    return out


def number_problems(tex: str, values: dict[str, str]) -> list[str]:
    out = [f"undefined number key '{key}'" for key in mn.undefined_keys(tex, values)]
    carriers: dict[str, list[str]] = defaultdict(list)
    for key, value in values.items():
        carriers[value.replace(",", "")].append(key)
    for literal in mn.typed_numbers_matching(tex, values):
        keys = sorted(carriers[literal.replace(",", "")])
        out.append(f"the number {literal} typed in the text equals a generated value ({', '.join(keys)}): write \\num{{{keys[0]}}}")
    return out


def number_notes(tex: str, values: dict[str, str]) -> list[str]:
    known = {value.replace(",", "") for value in values.values()}
    others = list(dict.fromkeys(lit for lit in mn.typed_numbers(tex) if lit.replace(",", "") not in known))
    notes = []
    if others:
        notes.append("typed numbers equal to no generated value, to read yourself: " + ", ".join(others))
    unused = mn.unused_keys(tex, values)
    if unused:
        notes.append(f"keys never used: {len(unused)} of {len(values)}")
    return notes


def check_text(tex: str, values: dict[str, str], *, final: bool = False) -> Report:
    return Report(claim_problems(tex, final=final) + number_problems(tex, values), number_notes(tex, values))


# ------------------------------------------------------------------ the command line


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tex", type=Path, help="the report's main .tex file")
    parser.add_argument("--final", action="store_true", help="every claim CL-1 to CL-7 must be marked at least once")
    parser.add_argument("--root", type=Path, default=ROOT_DIR, help="the repository the numbers are built from (default: this repository)")
    args = parser.parse_args(argv)
    try:
        raw = args.tex.read_bytes()
    except OSError as error:
        print(f"cannot read {args.tex}: {error}", file=sys.stderr)
        return 2
    try:
        tex = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        print(f"{args.tex} is not UTF-8", file=sys.stderr)
        return 2
    try:
        values = mn.build(args.root)
    except mn.NumbersError as error:
        print(f"numbers: {error}", file=sys.stderr)
        return 2
    report = check_text(tex, values, final=args.final)
    if report.problems:
        print(f"{args.tex}: {len(report.problems)} problem{'s' if len(report.problems) != 1 else ''}")
        for problem in report.problems:
            print(f"  {problem}")
    else:
        print(f"{args.tex}: no problems")
    if report.notes:
        print("for your own read (not problems):")
        for note in report.notes:
            print(f"  {note}")
    return 1 if report.problems else 0


if __name__ == "__main__":
    sys.exit(main())
