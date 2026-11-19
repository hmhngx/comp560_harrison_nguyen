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


def _references_headings(line: str) -> int:
    """How many 'References' headings a text line holds: 1 for the heading, 2 when pdftotext joins two headings printed one under the other, else 0."""
    words = line.split()
    return len(words) if words and all(word == "References" for word in words) else 0


def _is_references_line(line: str) -> bool:
    return _references_headings(line) > 0


def first_references_page(pages: list[str]) -> int:
    """1-based number of the first page with a line that is exactly 'References' (text extracted page by page)."""
    for number, text in enumerate(pages, start=1):
        if any(_is_references_line(line) for line in text.splitlines()):
            return number
    raise ValueError("no page has a line that is exactly 'References'")


def text_above_references(pages: list[str], references_page: int) -> list[str]:
    """The non-blank lines above the 'References' heading on its page (a 1-based page number)."""
    lines = pages[references_page - 1].splitlines()
    heading = next(index for index, line in enumerate(lines) if _is_references_line(line))
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


# ---- polish: what a defect-free LaTeX run can still get wrong -------------------------------------------------------------
# Layout warnings in the log, template text left in the PDF, the title and abstract word limits, text kept outside main.tex, and a
# bibliography heading printed twice. check_build runs these after check_report; check_report itself is unchanged.

MAX_OVERFULL_PT = 1.0
TITLE_WORDS = (8, 15)        # RP-08
ABSTRACT_WORDS = (150, 200)  # RP-09
TEMPLATE_PLACEHOLDERS = (
    "DNU Research Report Title",
    "Author Name",
    "youremail@dickinson.edu",
    "Summarize the research question, approach",
    "Introduce the problem, explain its importance",
    "See the DNU Lab web pages",
    "Explain the data, materials, methods",
    "Present the principal findings, using figures",
    "Interpret the findings, describe limitations",
    "Summarize the report's main conclusions",
    "Optionally, acknowledge",
    "Add the report's references here",
    "Add optional supplementary methods",
)
_LAYOUT = re.compile(
    r"^(?:Overfull \\[hv]box \((?P<amount>\d+(?:\.\d+)?)pt (?P<way>too wide|too high)\)(?P<where>.*)"
    r"|Missing character: (?P<missing>.+)"
    r"|LaTeX Font Warning: (?P<font>.+)"
    r"|LaTeX Warning: Float too large for page by (?P<float>\d+(?:\.\d+)?pt))",
    re.M,
)


# a heading as pdftotext shows it: an optional section number or appendix letter, then a capitalised title that ends in "(optional)"
_OPTIONAL_HEADING = re.compile(r"(?:(?:\d+(?:\.\d+)*|[A-Z])\s+)?[A-Z][A-Za-z ,-]*\(optional\)")


def layout_problems(log_text: str, max_overfull_pt: float = MAX_OVERFULL_PT) -> list[str]:
    """Visible layout defects LaTeX only warns about: overfull boxes wider (or taller) than the limit, missing characters, font substitutions, floats too large."""
    problems: list[str] = []
    for match in _LAYOUT.finditer(_unwrap(log_text)):
        if match["amount"] is not None:
            if float(match["amount"]) > max_overfull_pt:
                where = match["where"].strip()
                problems.append(f"overfull box: {match['amount']}pt {match['way']}" + (f" {where}" if where else ""))
        elif match["missing"] is not None:
            problems.append(f"missing character: {match['missing'].strip()}")
        elif match["font"] is not None:
            problems.append(f"font warning: {match['font'].strip()}")
        else:
            problems.append(f"float too large for the page by {match['float']}")
    return problems


def leftover_template_text(pages: list[str]) -> list[str]:
    """Placeholder sentences of the DNU template, and headings that still end in "(optional)", found in the text of the PDF pages."""
    problems: list[str] = []
    for number, text in enumerate(pages, start=1):
        flat = " ".join(text.replace("\u2019", "'").split())
        problems += [f"template text left in the PDF (page {number}): '{phrase}'" for phrase in TEMPLATE_PLACEHOLDERS if phrase in flat]
        for line in text.splitlines():
            if _OPTIONAL_HEADING.fullmatch(line.strip()) and len(line.split()) <= 7:
                problems.append(f"heading still says (optional) (page {number}): '{line.strip()}'")
    return problems


def count_words(text: str) -> int:
    r"""Words of a piece of LaTeX source: \num{...} and a citation count as one word each, inline math as one, braces and commands vanish,
    ~ is a space, and a token needs a letter or digit to count ("--" does not)."""
    text = _drop_command_groups(text, ("thanks", "footnote"))   # a note is not part of the line it hangs from
    text = re.sub(r"\\label\s*\{[^{}]*\}", " ", text)           # prints nothing
    text = re.sub(r"\\href\s*\{[^{}]*\}", " ", text)            # the address is not printed, the text after it is
    text = re.sub(r"\\url\s*\{[^{}]*\}|\\[a-zA-Z]*ref\s*\{[^{}]*\}", " R ", text)   # an address or a reference prints as one word
    text = re.sub(r"\\num\{[^{}]*\}", " N ", text)
    text = re.sub(r"\\[cC]ite[a-zA-Z]*\*?(?:\[[^\]]*\])*\{[^{}]*\}", " C ", text)
    text = re.sub(r"\\([%&_#])", r"\1", text).replace("\\$", "D")
    text = re.sub(r"\$[^$]*\$", " M ", text)
    text = re.sub(r"\\\\|\\[a-zA-Z]+\*?|\\.", " ", text)
    text = text.replace("~", " ").replace("{", " ").replace("}", " ")
    return sum(1 for token in text.split() if re.search(r"[A-Za-z0-9]", token))


def _group_end(text: str, open_at: int) -> int:
    """Index just after the brace group that opens at text[open_at] == '{'; ValueError if the braces never balance."""
    depth = 0
    for index in range(open_at, len(text)):
        depth += {"{": 1, "}": -1}.get(text[index], 0)
        if depth == 0:
            return index + 1
    raise ValueError("the braces do not balance")


def _drop_command_groups(text: str, names: tuple[str, ...]) -> str:
    r"""Remove \name{...} with its whole (possibly nested) group, for each name."""
    pattern = re.compile(r"\\(?:" + "|".join(names) + r")\s*\{")
    while (found := pattern.search(text)) is not None:
        text = text[: found.start()] + " " + text[_group_end(text, found.end() - 1):]
    return text


def title_words(tex: str) -> int | None:
    r"""Words in \title{...}; None if there is no title; ValueError if its braces do not balance."""
    body = strip_comments(tex)
    found = re.search(r"\\title\s*\{", body)
    if found is None:
        return None
    return count_words(body[found.end(): _group_end(body, found.end() - 1) - 1])


def abstract_words(tex: str) -> int | None:
    found = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", strip_comments(tex), re.S)
    return None if found is None else count_words(found.group(1))


def word_count_problems(tex: str) -> list[str]:
    """RP-08 (title 8 to 15 words) and RP-09 (abstract 150 to 200 words). A missing abstract is left to missing_sections."""
    problems: list[str] = []
    for name, count_in, (low, high), unbalanced in (
        ("title", title_words, TITLE_WORDS, "the braces of \\title{...} do not balance"),
        ("abstract", abstract_words, ABSTRACT_WORDS, "the braces in the abstract do not balance"),
    ):
        try:
            count = count_in(tex)
        except ValueError:
            problems.append(unbalanced)
            continue
        if count is None and name == "title":
            problems.append("no \\title{...} found in the source")
        elif count is not None and not low <= count <= high:
            problems.append(f"{name} has {count} word{'' if count == 1 else 's'}; the guideline asks for {low} to {high}")
    return problems


_OTHER_FILE = re.compile(r"\\(input|include|InputIfFileExists|subfile|import)\s*\{([^}]*)\}")


def single_file_problems(tex: str) -> list[str]:
    r"""The checks read main.tex only, so text may not come from another file; \input{numbers} (the generated numbers) is the one exception."""
    problems = []
    for match in _OTHER_FILE.finditer(strip_comments(tex)):
        name = match.group(2).strip()
        if match.group(1) == "input" and name in ("numbers", "numbers.tex"):
            continue
        problems.append(
            f"text comes from another file (\\{match.group(1)}{{{name}}}): the gate reads main.tex only, so its sections, captions and citations are not checked"
        )
    return problems


def references_heading_problems(pages: list[str]) -> list[str]:
    """A manual \\section*{References} next to \\bibliography prints the heading twice; first_references_page would take the first and pass."""
    found: list[int] = []
    for number, text in enumerate(pages, start=1):
        found += [number] * sum(_references_headings(line) for line in text.splitlines())
    if len(found) > 1:
        return [f"the heading 'References' is printed {len(found)} times (pages {', '.join(map(str, found))})"]
    return []


def polish_problems(tex: str, pages: list[str], log_text: str, max_overfull_pt: float = MAX_OVERFULL_PT) -> list[str]:
    return layout_problems(log_text, max_overfull_pt) + leftover_template_text(pages) + word_count_problems(tex) + single_file_problems(tex) + references_heading_problems(pages)
