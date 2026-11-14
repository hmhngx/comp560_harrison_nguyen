"""Gate helpers for the DNU-style report. Pure Python: nothing here needs a TeX engine.

Each function checks one requirement of the DNU Lab report guidelines (dnulab-web, public/internal/report-guidelines.md):
the nine-content-page limit, the required sections, a caption and an in-text reference for every figure and table, and that
every bibliography entry is cited. Failures come back as readable strings or as exceptions; nothing is swallowed.
"""
from __future__ import annotations

import re

MAX_CONTENT_PAGES = 9
REQUIRED_SECTIONS = (
    "Abstract",
    "Introduction",
    "Related Work",
    "Conclusion",
    "References",
    "Use of Artificial Intelligence",
)


def strip_comments(tex: str) -> str:
    """Remove LaTeX comments: from an unescaped % to the end of its line (an odd number of backslashes before a % escapes it)."""
    return "\n".join(re.sub(r"(?<!\\)((?:\\\\)*)%.*$", r"\1", line) for line in tex.splitlines())


def content_last_page(aux_text: str, label: str = "endcontent") -> int:
    r"""Page recorded for \label{label} in a .aux file; ValueError if the label is absent or has no plain page number."""
    pattern = r"\\newlabel\{" + re.escape(label) + r"\}\{\{(?:[^{}]|\{[^{}]*\})*\}\{(\d+)\}"
    match = re.search(pattern, aux_text)
    if match is None:
        raise ValueError(
            f"label {label!r} not found in the .aux text (put \\label{{{label}}} as the last token of the last content paragraph)"
        )
    return int(match.group(1))


def _unwrap(log_text: str) -> str:
    """TeX breaks log lines at 79 characters; join each 79-character line to the next one."""
    lines: list[str] = []
    joined = ""
    for line in log_text.splitlines():
        if len(line) == 79:
            joined += line
        else:
            lines.append(joined + line)
            joined = ""
    if joined:
        lines.append(joined)
    return "\n".join(lines)


_ERROR = re.compile(r"^! (.+)$", re.M)
_UNDEFINED_REFERENCE = re.compile(r"Reference `([^']+)' on page \d+ undefined")
_UNDEFINED_CITATION = re.compile(r"Citation `([^']+)' on page \d+ undefined")
_MULTIPLY_DEFINED = re.compile(r"Label `([^']+)' multiply defined")


def parse_log(log_text: str) -> dict[str, list[str]]:
    text = _unwrap(log_text)
    return {
        "errors": _ERROR.findall(text),
        "undefined_refs": _UNDEFINED_REFERENCE.findall(text),
        "undefined_cites": _UNDEFINED_CITATION.findall(text),
        "multiply_defined": _MULTIPLY_DEFINED.findall(text),
    }


def missing_sections(tex: str, required: tuple[str, ...] = REQUIRED_SECTIONS) -> list[str]:
    body = strip_comments(tex)
    present = {m.group(1).strip().lower() for m in re.finditer(r"\\(?:sub)*section\*?\{([^}]*)\}", body)}
    if re.search(r"\\begin\{abstract\}", body):
        present.add("abstract")
    if re.search(r"\\bibliography\{", body):
        present.add("references")
    return [name for name in required if name.lower() not in present]


_CITE = re.compile(r"\\[cC]ite[a-zA-Z]*\*?(?:\[[^\]]*\])*\{([^}]*)\}")


def keys_in(pattern: re.Pattern[str], tex: str) -> list[str]:
    """The comma-separated keys inside the braces that group 1 of the pattern captures, in order, with comments stripped."""
    return [key.strip() for match in pattern.finditer(strip_comments(tex)) for key in match.group(1).split(",") if key.strip()]


def cited_keys(tex: str) -> set[str]:
    return set(keys_in(_CITE, tex))


def bib_keys(bib_text: str) -> list[str]:
    return re.findall(r"^@(?!comment|string|preamble)\w+\s*\{\s*([^,\s]+)\s*,", bib_text, flags=re.M | re.I)


def uncited_bib_entries(tex: str, bib_text: str) -> list[str]:
    cited = cited_keys(tex)
    return [key for key in bib_keys(bib_text) if key not in cited]


_FLOAT = re.compile(r"\\begin\{(figure|table)\*?\}(.*?)\\end\{\1\*?\}", re.S)
_REFERENCE = re.compile(r"\\(?:auto|c|C)?ref\{([^}]*)\}")


def referenced_keys(tex: str) -> set[str]:
    return set(keys_in(_REFERENCE, tex))


def float_problems(tex: str) -> list[str]:
    body = strip_comments(tex)
    referenced = referenced_keys(body)
    problems: list[str] = []
    for match in _FLOAT.finditer(body):
        kind, inner = match.group(1), match.group(2)
        if "\\caption" not in inner:
            problems.append(f"{kind} without a caption")
        labels = re.findall(r"\\label\{([^}]*)\}", inner)
        if not labels:
            problems.append(f"{kind} without a label")
        for label in labels:
            if label not in referenced:
                problems.append(f"label {label} is never referenced")
    return problems


def first_references_page(pages: list[str]) -> int:
    """1-based number of the first page with a line that is exactly 'References' (text extracted page by page)."""
    for number, text in enumerate(pages, start=1):
        if any(line.strip() == "References" for line in text.splitlines()):
            return number
    raise ValueError("no page has a line that is exactly 'References'")


def text_above_references(pages: list[str], references_page: int) -> list[str]:
    """The non-blank lines above the 'References' heading on its page (a 1-based page number)."""
    lines = pages[references_page - 1].splitlines()
    heading = next(index for index, line in enumerate(lines) if line.strip() == "References")
    return [line.strip() for line in lines[:heading] if line.strip()]


def check_page_budget(label_page: int, references_page: int) -> list[str]:
    """Two independent measurements of where the content ends must agree, and the content must fit the limit."""
    problems: list[str] = []
    if references_page - label_page not in (0, 1):
        problems.append(
            f"page measurements disagree: end-of-content label on page {label_page}, "
            f"'References' heading on page {references_page}"
        )
    if label_page > MAX_CONTENT_PAGES:
        problems.append(f"content ends on page {label_page}; the limit is {MAX_CONTENT_PAGES} content pages")
    return problems


def check_report(tex: str, bib_text: str, aux_text: str, log_text: str, pages: list[str]) -> list[str]:
    """Every defect found, as readable strings; an empty list means every gate passed.

    A page measurement that cannot be made (no end label in the .aux text, no 'References' heading line) is itself a problem, listed with the others.
    """
    problems: list[str] = []
    log = parse_log(log_text)
    problems += [f"LaTeX error: {item}" for item in log["errors"]]
    problems += [f"undefined reference: {item}" for item in log["undefined_refs"]]
    problems += [f"undefined citation: {item}" for item in log["undefined_cites"]]
    problems += [f"label defined twice: {item}" for item in log["multiply_defined"]]
    problems += [f"missing required section: {item}" for item in missing_sections(tex)]
    problems += [f"bibliography entry never cited: {item}" for item in uncited_bib_entries(tex, bib_text)]
    problems += float_problems(tex)
    measured: list[int] = []
    for measure, source in ((content_last_page, aux_text), (first_references_page, pages)):
        try:
            measured.append(measure(source))
        except ValueError as error:
            problems.append(f"page budget not measured: {error}")
    if len(measured) == 2:
        label_page, references_page = measured
        problems += check_page_budget(label_page, references_page)
        if references_page == label_page + 1:  # the heading starts the next page, so no content may sit above it
            above = text_above_references(pages, references_page)
            if above:
                problems.append(
                    f"content continues on page {references_page} above the 'References' heading (first line: {above[0]!r}), "
                    f"so the end-of-content label on page {label_page} is too early"
                )
    return problems
