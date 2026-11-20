"""A strict reader and checker for the report's BibTeX file.

    python -m report.check_bib report/references.bib [--tex report/main.tex]

The file is read by a small strict parser (a malformed entry is an error with its line number, never silently skipped), then every entry is held to the
rules below. The point is that a bibliography entry in this report was copied from a page that someone opened, and that a later edit cannot change a
verified field or leave an entry half-written without a failing check.

Rules, per entry (each problem is one line "key: what is wrong"):
- type is article, inproceedings or misc; the key is lower-case letters, a four-digit year, then letters or digits, and its year equals the year field;
- required fields: author, title, year and verified, plus journal (article), booktitle (inproceedings) or howpublished (misc); no empty field; only known
  field names (a typo such as "auther" is a problem); ASCII only; no "%";
- author: "Last, First" joined by "and", every author listed (no "et al", no "others"); a name in braces is one author;
- title: one brace group around the whole title, so that BibTeX keeps its capitals; no period at the end; no space inside the braces;
- year: four digits from 1900 to 2026;
- an entry with an eprint is an arXiv entry: eprint like 2404.05405, archiveprefix arXiv, a primaryclass such as cs.CL, url https://arxiv.org/abs/<eprint>,
  and for a misc entry howpublished "arXiv preprint arXiv:<eprint>";
- verified: "YYYY-MM-DD, https://..." with a real calendar date, naming the page that was opened (for an arXiv entry, its abstract page).
Across entries: no duplicate key, title or eprint. With --tex, an entry that the .tex file never cites, and a citation key without an entry, are problems.

Exit status: 0 no problems; 1 problems (printed, one per line); 2 the files could not be read.
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from report import build_report as br

ALLOWED_TYPES = ("article", "inproceedings", "misc")
REQUIRED = {"article": ("author", "title", "journal", "year"), "inproceedings": ("author", "title", "booktitle", "year"),
            "misc": ("author", "title", "year", "howpublished")}
ALWAYS_REQUIRED = ("verified",)
ALLOWED_FIELDS = frozenset({"author", "title", "year", "journal", "booktitle", "howpublished", "eprint", "archiveprefix", "primaryclass", "url", "verified",
                            "volume", "number", "pages", "publisher", "doi", "note"})
FIRST_YEAR, LAST_YEAR = 1900, 2026

_KEY = re.compile(r"[a-z]+([0-9]{4})[a-z0-9]*")
_EPRINT = re.compile(r"[0-9]{4}\.[0-9]+")
_CATEGORY = re.compile(r"[a-z]+(-[a-z]+)*(\.([A-Z]{2}|[a-z]+(-[a-z]+)*))?")
_VERIFIED = re.compile(r"([0-9]{4}-[0-9]{2}-[0-9]{2}), (https://\S+)")
_FIELD_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")
_KEY_TEXT = re.compile(r"[^\s,{}\"=@#%()]*")


def is_arxiv_id(text: str) -> bool:
    """A new-style arXiv identifier YYMM.NNNNN: the month is 01 to 12, and the number has 5 digits from 1501 on and 4 digits before."""
    match = _EPRINT.fullmatch(text)
    if match is None:
        return False
    yymm, number = text[:4], text[5:]
    return 1 <= int(yymm[2:]) <= 12 and len(number) == (5 if int(yymm) >= 1501 else 4)


class BibError(ValueError):
    """The text is not a BibTeX file this checker can read; the message starts with the line number."""


@dataclass(frozen=True)
class Entry:
    type: str
    key: str
    fields: dict


# ------------------------------------------------------------------ the parser


def _line(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


def _skip_space(text: str, pos: int) -> int:
    while pos < len(text) and text[pos].isspace():
        pos += 1
    return pos


def _read_braced(text: str, pos: int):
    """(inner text, position after the closing brace) for the brace group that opens at pos; (None, len(text)) when it never closes."""
    depth = 0
    for i in range(pos, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[pos + 1:i], i + 1
    return None, len(text)


def _read_quoted(text: str, pos: int):
    """(inner text, position after the closing quote) for the quoted value that opens at pos; a quote inside braces does not end it."""
    depth = 0
    for i in range(pos + 1, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        elif c == '"' and depth <= 0:
            return text[pos + 1:i], i + 1
    raise BibError(f"line {_line(text, pos)}: unterminated quoted value")


def _parse_entry(text: str, start: int):
    n = len(text)
    here = _line(text, start)

    def unclosed() -> BibError:
        return BibError(f"line {here}: the entry is not closed")

    kind_match = re.compile(r"[A-Za-z]+").match(text, start + 1)
    if kind_match is None:
        raise BibError(f"line {here}: an entry type is missing")
    kind = kind_match.group().lower()
    pos = _skip_space(text, kind_match.end())
    if kind in ("string", "preamble"):
        raise BibError(f"line {here}: @{kind} is not supported")
    if pos >= n or text[pos] != "{":
        raise BibError(f"line {here if pos >= n else _line(text, pos)}: an entry must open with a brace")
    if kind == "comment":
        inner, end = _read_braced(text, pos)
        if inner is None:
            raise unclosed()
        return None, end
    pos = _skip_space(text, pos + 1)
    key = _KEY_TEXT.match(text, pos).group()
    if key == "":
        raise BibError(f"line {here if pos >= n else _line(text, pos)}: missing citation key")
    pos = _skip_space(text, pos + len(key))
    if pos >= n:
        raise unclosed()
    fields: dict = {}
    if text[pos] == "}":
        return Entry(kind, key, fields), pos + 1
    if text[pos] != ",":
        raise BibError(f"line {_line(text, pos)}: expected a comma after the citation key")
    pos += 1
    while True:
        pos = _skip_space(text, pos)
        if pos >= n:
            raise unclosed()
        if text[pos] == "}":
            return Entry(kind, key, fields), pos + 1
        name_match = _FIELD_NAME.match(text, pos)
        if name_match is None:
            raise BibError(f"line {_line(text, pos)}: a field name is missing")
        name = name_match.group().lower()
        pos = _skip_space(text, name_match.end())
        if pos >= n:
            raise unclosed()
        if text[pos] != "=":
            raise BibError(f"line {_line(text, pos)}: expected '=' after the field name")
        pos = _skip_space(text, pos + 1)
        if pos >= n:
            raise unclosed()
        c = text[pos]
        if c == "{":
            value, pos = _read_braced(text, pos)          # (None, end of text) when it never closes: the end-of-text check below reports it
        elif c == '"':
            value, pos = _read_quoted(text, pos)
        elif c.isdigit():
            number = re.compile(r"[0-9]+").match(text, pos)
            value, pos = number.group(), number.end()
        elif c in ",}":
            raise BibError(f"line {_line(text, pos)}: a value is missing")
        else:
            raise BibError(f"line {_line(text, pos)}: a bare word is not allowed as a value")
        if name in fields:
            raise BibError(f"line {_line(text, pos)}: duplicate field '{name}'")
        fields[name] = value
        pos = _skip_space(text, pos)
        if pos >= n:
            raise unclosed()
        if text[pos] == ",":
            pos += 1
        elif text[pos] == "#":
            raise BibError(f"line {_line(text, pos)}: '#' concatenation is not supported")
        elif text[pos] != "}":
            raise BibError(f"line {_line(text, pos)}: expected a comma or the end of the entry")


def parse_bib(text: str) -> list[Entry]:
    entries: list[Entry] = []
    pos = 0
    while pos < len(text):
        c = text[pos]
        if c.isspace():
            pos += 1
        elif c == "%":
            end = text.find("\n", pos)
            pos = len(text) if end < 0 else end
        elif c == "@":
            entry, pos = _parse_entry(text, pos)
            if entry is not None:
                entries.append(entry)
        else:
            raise BibError(f"line {_line(text, pos)}: text outside an entry")
    return entries


# ------------------------------------------------------------------ the rules


def _mask(s: str) -> str:
    """s with every character inside braces (and the braces) replaced by '_', so that splitting and counting see only the top level."""
    depth, out = 0, []
    for c in s:
        if c == "{":
            depth += 1
            out.append("_")
        elif c == "}":
            out.append("_")
            depth = max(depth - 1, 0)
        else:
            out.append("_" if depth else c)
    return "".join(out)


def is_single_group(s: str) -> bool:
    if not (s.startswith("{") and s.endswith("}")):
        return False
    depth = 0
    for i, c in enumerate(s):
        depth += (c == "{") - (c == "}")
        if depth == 0 and i < len(s) - 1:
            return False
    return depth == 0


def split_authors(value: str) -> list[str]:
    """The authors of a BibTeX author field: split at every "and" (any case) that stands alone outside braces, each part stripped."""
    mask = _mask(value)
    cuts = [m.span() for m in re.finditer(r"(?i)(?<!\S)and(?!\S)", mask)]
    bounds = [0] + [edge for span in cuts for edge in span] + [len(value)]
    return [value[bounds[i]:bounds[i + 1]].strip() for i in range(0, len(bounds), 2)]


def _author_problems(value: str) -> list[str]:
    people = split_authors(value)
    problems = []
    if re.search(r"(?i)\bet\.?\s+al\b", _mask(value)):
        problems.append("author list contains 'et al'; list every author")
    if people[-1].lower() == "others":
        problems.append("author list ends in 'others'; list every author")
        people = people[:-1]
    if any(person == "" for person in people):
        problems.append("author list has an empty author")
    for person in people:
        if person == "" or is_single_group(person):
            continue
        commas = _mask(person).count(",")
        last, _, first = person.partition(",")
        if commas != 1 or not last.strip() or not first.strip():
            problems.append(f"author '{person}' is not in the form 'Last, First'")
    return problems


def entry_problems(entry: Entry) -> list[str]:
    key, fields = entry.key, entry.fields
    out: list[str] = []

    def add(message: str) -> None:
        out.append(f"{key}: {message}")

    if entry.type not in ALLOWED_TYPES:
        add(f"entry type '{entry.type}' is not allowed")
    key_match = _KEY.fullmatch(key)
    if key_match is None:
        add("citation key must be lower-case letters, then a four-digit year, then optional lower-case letters or digits")
    year = fields.get("year", "")
    year_ok = re.fullmatch(r"[0-9]{4}", year) is not None and FIRST_YEAR <= int(year) <= LAST_YEAR
    if key_match is not None and year_ok and key_match.group(1) != year:
        add(f"the year in the key ({key_match.group(1)}) does not match the year field ({year})")
    for name in REQUIRED.get(entry.type, ()) + ALWAYS_REQUIRED:
        if name not in fields:
            add(f"missing field '{name}'")
    for name, value in fields.items():
        if name not in ALLOWED_FIELDS:
            add(f"unknown field '{name}'")
        if value.strip() == "":
            add(f"field '{name}' is empty")
        if not value.isascii():
            add(f"field '{name}' has a non-ASCII character")
        if "%" in value:
            add(f"field '{name}' contains a '%'")
    if fields.get("author", "").strip():
        for problem in _author_problems(fields["author"]):
            add(problem)
    title = fields.get("title", "")
    if title.strip():
        if not is_single_group(title):
            add("title must be one brace group around the whole title, so BibTeX keeps its capitals")
        else:
            inner = title[1:-1]
            if inner.endswith("."):
                add("title ends in a period")
            if inner != inner.strip():
                add("title has a space after its opening brace or before its closing brace")
    if "year" in fields and fields["year"].strip() and not year_ok:
        add(f"year '{year}' is not a four-digit year from 1900 to 2026")
    eprint = fields.get("eprint")
    if eprint is not None:
        if not is_arxiv_id(eprint):
            add(f"eprint '{eprint}' is not an arXiv identifier")
        if "archiveprefix" not in fields:
            add("missing field 'archiveprefix'")
        elif fields["archiveprefix"] != "arXiv":
            add("archiveprefix must be 'arXiv'")
        if "primaryclass" not in fields:
            add("missing field 'primaryclass'")
        elif _CATEGORY.fullmatch(fields["primaryclass"]) is None:
            add(f"primaryclass '{fields['primaryclass']}' is not an arXiv category")
        page = f"https://arxiv.org/abs/{eprint}"
        if "url" not in fields:
            add("missing field 'url'")
        elif fields["url"] != page:
            add(f"url must be {page}")
        if entry.type == "misc" and "howpublished" in fields and fields["howpublished"] != f"arXiv preprint arXiv:{eprint}":
            add(f"howpublished must be 'arXiv preprint arXiv:{eprint}'")
    verified = fields.get("verified")
    if verified is not None and verified.strip():
        match = _VERIFIED.fullmatch(verified)
        if match is None:
            add("verified must look like 'YYYY-MM-DD, https://...'")
        else:
            try:
                date.fromisoformat(match.group(1))
            except ValueError:
                add(f"verified has the date '{match.group(1)}', which is not a calendar date")
            if eprint is not None and match.group(2) != f"https://arxiv.org/abs/{eprint}":
                add(f"verified must name the page https://arxiv.org/abs/{eprint}")
    return out


def _title_form(title: str) -> str:
    return " ".join(re.sub(r"[{}]", "", title).lower().split())


def bib_problems(entries: list[Entry]) -> list[str]:
    out: list[str] = []
    for entry in entries:
        out.extend(entry_problems(entry))
    seen_key: dict = {}
    seen_title: dict = {}
    seen_eprint: dict = {}
    for entry in entries:
        if entry.key in seen_key:
            out.append(f"duplicate citation key '{entry.key}'")
        seen_key.setdefault(entry.key, entry.key)
        title = _title_form(entry.fields.get("title", ""))
        if title:
            if title in seen_title and seen_title[title] != entry.key:
                out.append(f"duplicate title (keys '{seen_title[title]}' and '{entry.key}')")
            seen_title.setdefault(title, entry.key)
        eprint = entry.fields.get("eprint")
        if eprint:
            if eprint in seen_eprint and seen_eprint[eprint] != entry.key:
                out.append(f"duplicate eprint {eprint} (keys '{seen_eprint[eprint]}' and '{entry.key}')")
            seen_eprint.setdefault(eprint, entry.key)
    return out


def _check(text: str) -> tuple[list[Entry], list[str]]:
    try:
        entries = parse_bib(text)
    except BibError as error:
        return [], [str(error)]
    return entries, (bib_problems(entries) if entries else ["no entries"])


def check_text(text: str) -> list[str]:
    return _check(text)[1]


# ------------------------------------------------------------------ the command line


def _read(path: Path) -> str:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise OSError(f"cannot read {path}: {error}") from None
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise OSError(f"{path} is not UTF-8") from None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bib", type=Path, help="the .bib file to check")
    parser.add_argument("--tex", type=Path, help="also check that the .tex file cites every entry and cites nothing that has no entry")
    args = parser.parse_args(argv)
    try:
        text = _read(args.bib)
        tex = _read(args.tex) if args.tex is not None else None
    except OSError as error:
        print(error, file=sys.stderr)
        return 2
    entries, problems = _check(text)
    if tex is not None:
        problems += [f"{key}: not cited in {args.tex}" for key in br.uncited_bib_entries(tex, text)]
        known = set(br.bib_keys(text))
        problems += [f"citation key '{key}' has no entry in {args.bib}" for key in sorted(br.cited_keys(tex) - known)]
    if problems:
        for problem in problems:
            print(problem)
        return 1
    print(f"{args.bib}: {len(entries)} entries, no problems")
    return 0


if __name__ == "__main__":
    sys.exit(main())
