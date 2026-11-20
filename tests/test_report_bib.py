"""Tests for report/check_bib.py and the committed report/references.bib.

The two real entries are pinned to literals copied from the arXiv pages that were opened on 2026-11-19 (arXiv 2404.05405 and 1711.05101): a change to a verified field
fails these tests until someone opens the page again. The unit tests build small entries and break one thing at a time. The integration test runs real
pdflatex and bibtex on a tiny document that cites both entries and reads what comes out (it fails, not skips, when TeX is missing, as in tests/test_report_gate.py).
"""
from __future__ import annotations

import dataclasses
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from report import check_bib as cb

ROOT = Path(__file__).resolve().parents[1]
BIB = ROOT / "report" / "references.bib"

ANCHOR = {
    "author": "Allen-Zhu, Zeyuan and Li, Yuanzhi",
    "title": "{Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws}",
    "year": "2024",
    "howpublished": "arXiv preprint arXiv:2404.05405",
    "eprint": "2404.05405",
    "archiveprefix": "arXiv",
    "primaryclass": "cs.CL",
    "url": "https://arxiv.org/abs/2404.05405",
    "verified": "2026-11-19, https://arxiv.org/abs/2404.05405",
}
ADAMW = {
    "author": "Loshchilov, Ilya and Hutter, Frank",
    "title": "{Decoupled Weight Decay Regularization}",
    "booktitle": "International Conference on Learning Representations",
    "year": "2019",
    "eprint": "1711.05101",
    "archiveprefix": "arXiv",
    "primaryclass": "cs.LG",
    "url": "https://arxiv.org/abs/1711.05101",
    "verified": "2026-11-19, https://arxiv.org/abs/1711.05101",
}
PINNED = {("misc", "allenzhu2024physics"): ANCHOR, ("inproceedings", "loshchilov2019decoupled"): ADAMW}


def entry_text(kind: str, key: str, fields: dict) -> str:
    body = ",\n".join(f"  {name} = {{{value}}}" for name, value in fields.items())
    return f"@{kind}{{{key},\n{body}\n}}\n"


def misc(**changes) -> str:
    fields = {**ANCHOR, **changes}
    return entry_text("misc", "allenzhu2024physics", {k: v for k, v in fields.items() if v is not None})


def inproceedings(**changes) -> str:
    fields = {**ADAMW, **changes}
    return entry_text("inproceedings", "loshchilov2019decoupled", {k: v for k, v in fields.items() if v is not None})


def problems_of(text: str) -> list[str]:
    return cb.check_text(text)


# ---- the parser ------------------------------------------------------------------------------------------------------------------------------
def test_braced_quoted_and_number_values_names_are_lower_cased_and_the_type_too():
    (entry,) = cb.parse_bib('@ARTICLE{key2020a, Author = {A, B}, title = "T x", YEAR = 2020, journal = {J}}')
    assert (entry.type, entry.key) == ("article", "key2020a")
    assert entry.fields == {"author": "A, B", "title": "T x", "year": "2020", "journal": "J"}


def test_nested_braces_are_kept_and_a_quote_inside_braces_does_not_end_a_quoted_value():
    (entry,) = cb.parse_bib('@misc{k2020, title = {{A {B} c}}, note = "x {"} y"}')
    assert entry.fields == {"title": "{A {B} c}", "note": 'x {"} y'}


def test_a_trailing_comma_comment_lines_comment_entries_and_windows_line_ends_are_fine():
    text = "% a comment\r\n@comment{ anything {nested} here }\r\n@misc{k2020,\r\n  title = {T},\r\n}\r\n% end\r\n@misc{k2021, title = {U}}\r\n"
    assert [e.key for e in cb.parse_bib(text)] == ["k2020", "k2021"]


def test_an_empty_file_has_no_entries_and_is_a_problem_not_an_error():
    assert cb.parse_bib("") == [] and cb.parse_bib("% only a comment\n") == []
    assert problems_of("") == ["no entries"]


@pytest.mark.parametrize("text, message", [
    ("junk\n@misc{k2020, title = {T}}", "line 1: text outside an entry"),
    ("@misc{k2020, title = {T}}\n\nmore junk", "line 3: text outside an entry"),
    ("@string{a = {b}}", "@string is not supported"),
    ("@preamble{a}", "@preamble is not supported"),
    ("@misc(k2020, title = {T})", "line 1: an entry must open with a brace"),
    ("@misc{, title = {T}}", "line 1: missing citation key"),
    ("@misc{k2020 title = {T}}", "line 1: expected a comma after the citation key"),
    ("@misc{k2020, title {T}}", "line 1: expected '=' after the field name"),
    ("@misc{k2020, title = {T}", "line 1: the entry is not closed"),
    ("@misc{k2020, title = {T {x}}", "line 1: the entry is not closed"),
    ('@misc{k2020, title = "T}', "line 1: unterminated quoted value"),
    ("@misc{k2020, year = YEAR}", "line 1: a bare word is not allowed as a value"),
    ('@misc{k2020, title = {a} # {b}}', "line 1: '#' concatenation is not supported"),
    ("@misc{k2020, title = {a}, title = {b}}", "line 1: duplicate field 'title'"),
    ("@misc{k2020, title = {a} year = {b}}", "line 1: expected a comma or the end of the entry"),
    ("@misc{k2020, title = }", "line 1: a value is missing"),
    ("@misc{k2020, = {a}}", "line 1: a field name is missing"),
    ("@{k2020, title = {a}}", "line 1: an entry type is missing"),
])
def test_malformed_input_is_refused_with_the_line_number(text, message):
    with pytest.raises(cb.BibError, match=re.escape(message)):
        cb.parse_bib(text)


def test_check_text_turns_a_parse_error_into_one_problem():
    assert problems_of("@misc{k2020, title = {T}") == ["line 1: the entry is not closed"]


def test_the_line_number_counts_lines_inside_earlier_entries():
    text = "@misc{k2020,\n  title = {T},\n  year = {2020}\n}\n\n@misc{k2021, title = }"
    with pytest.raises(cb.BibError, match="line 6: a value is missing"):
        cb.parse_bib(text)


# ---- the rules -------------------------------------------------------------------------------------------------------------------------------
def test_the_two_real_looking_entries_and_both_together_are_clean():
    assert problems_of(misc()) == [] and problems_of(inproceedings()) == [] and problems_of(misc() + "\n" + inproceedings()) == []


@pytest.mark.parametrize("text, expected", [
    (entry_text("book", "allenzhu2024physics", ANCHOR), "allenzhu2024physics: entry type 'book' is not allowed"),
    (misc().replace("allenzhu2024physics", "Allen2024"), "Allen2024: citation key must be lower-case letters, then a four-digit year, then optional lower-case letters or digits"),
    (misc().replace("allenzhu2024physics", "allenzhu"), "allenzhu: citation key must be lower-case letters, then a four-digit year, then optional lower-case letters or digits"),
    (misc().replace("allenzhu2024physics", "allenzhu2019physics"), "allenzhu2019physics: the year in the key (2019) does not match the year field (2024)"),
])
def test_type_and_key_rules(text, expected):
    assert expected in problems_of(text)


@pytest.mark.parametrize("missing", ["author", "title", "year", "howpublished", "verified"])
def test_every_required_field_of_a_misc_entry_is_required(missing):
    assert f"allenzhu2024physics: missing field '{missing}'" in problems_of(misc(**{missing: None}))


@pytest.mark.parametrize("missing", ["author", "title", "booktitle", "year", "verified"])
def test_every_required_field_of_an_inproceedings_entry_is_required(missing):
    assert f"loshchilov2019decoupled: missing field '{missing}'" in problems_of(inproceedings(**{missing: None}))


def test_every_required_field_of_an_article_is_required():
    base = {"author": "A, B", "title": "{T}", "journal": "J", "year": "2020", "verified": "2026-11-19, https://example.org/p"}
    assert problems_of(entry_text("article", "a2020x", base)) == []
    for missing in ("author", "title", "journal", "year", "verified"):
        text = entry_text("article", "a2020x", {k: v for k, v in base.items() if k != missing})
        assert f"a2020x: missing field '{missing}'" in problems_of(text)


@pytest.mark.parametrize("field", ["author", "title", "year", "howpublished", "verified", "url"])
def test_an_empty_field_is_a_problem(field):
    assert f"allenzhu2024physics: field '{field}' is empty" in problems_of(misc(**{field: ""}))
    assert f"allenzhu2024physics: field '{field}' is empty" in problems_of(misc(**{field: "   "}))


@pytest.mark.parametrize("author, expected", [
    ("Zeyuan Allen-Zhu and Yuanzhi Li", "author 'Zeyuan Allen-Zhu' is not in the form 'Last, First'"),
    ("Allen-Zhu, Zeyuan and Li, Yuanzhi and others", "author list ends in 'others'; list every author"),
    ("Allen-Zhu, Zeyuan et al.", "author list contains 'et al'; list every author"),
    ("Allen-Zhu, Zeyuan and ", "author list has an empty author"),
    ("Allen-Zhu, Zeyuan and Li, Yuanzhi, Extra", "author 'Li, Yuanzhi, Extra' is not in the form 'Last, First'"),
    ("Allen-Zhu, and Li, Yuanzhi", "author 'Allen-Zhu,' is not in the form 'Last, First'"),
    (", Zeyuan", "author ', Zeyuan' is not in the form 'Last, First'"),
])
def test_author_rules(author, expected):
    assert f"allenzhu2024physics: {expected}" in problems_of(misc(author=author))


def test_a_braced_organization_name_is_one_author_and_and_inside_braces_is_not_a_separator():
    assert problems_of(misc(author="{Anthropic and Friends}")) == []
    assert problems_of(misc(author="{Anthropic and Friends} and Li, Yuanzhi")) == []


@pytest.mark.parametrize("title, expected", [
    ("Physics of Language Models", "title must be one brace group around the whole title, so BibTeX keeps its capitals"),
    ("{Physics} of {Language} Models", "title must be one brace group around the whole title, so BibTeX keeps its capitals"),
    ("{Physics of Language Models.}", "title ends in a period"),
    ("{ Physics of Language Models}", "title has a space after its opening brace or before its closing brace"),
    ("{Physics of Language Models }", "title has a space after its opening brace or before its closing brace"),
])
def test_title_rules(title, expected):
    assert f"allenzhu2024physics: {expected}" in problems_of(misc(title=title))


@pytest.mark.parametrize("year", ["24", "2024a", "twenty", "1899", "2027", "02024"])
def test_year_rules(year):
    key = "allenzhu2024physics"
    assert f"{key}: year '{year}' is not a four-digit year from 1900 to 2026" in problems_of(misc(year=year))


def test_the_last_allowed_year_is_2026_and_the_first_is_1900():
    assert problems_of(entry_text("misc", "a2026x", {**ANCHOR, "year": "2026"})) == []
    assert problems_of(entry_text("misc", "a1900x", {**ANCHOR, "year": "1900"})) == []


def test_non_ascii_text_and_a_percent_sign_are_problems():
    assert "allenzhu2024physics: field 'author' has a non-ASCII character" in problems_of(misc(author="Allen-Zhu, Zeyuan and Li, Yuanzh" + chr(0xE9)))
    assert "allenzhu2024physics: field 'howpublished' contains a '%'" in problems_of(misc(howpublished="arXiv preprint arXiv:2404.05405 100%"))


def test_an_unknown_field_name_is_a_problem_so_a_typo_cannot_hide():
    assert "allenzhu2024physics: unknown field 'auther'" in problems_of(misc(auther="x"))


@pytest.mark.parametrize("identifier, valid", [
    ("2404.05405", True), ("1611.03530", True), ("1711.05101", True), ("2201.02177", True), ("1501.00001", True), ("1412.6980", True), ("0704.0001", True),
    ("2404.5405", False),          # since January 2015 (1501) the number has five digits: a dropped zero is a typo
    ("1412.69801", False),         # before 1501 it has four
    ("2413.00001", False), ("2400.00001", False),        # month 13 and month 00 do not exist
    ("2404.05405v1", False), ("2404.0540", False), ("404.05405", False), ("2404-05405", False), ("2404.054051", False), ("", False), ("abcd.efghi", False),
])
def test_arxiv_identifier_shapes(identifier, valid):
    assert cb.is_arxiv_id(identifier) is valid


@pytest.mark.parametrize("change, expected", [
    ({"eprint": "2404.5405"}, "eprint '2404.5405' is not an arXiv identifier"),
    ({"eprint": "2404.05405v1"}, "eprint '2404.05405v1' is not an arXiv identifier"),
    ({"archiveprefix": None}, "missing field 'archiveprefix'"),
    ({"archiveprefix": "arxiv"}, "archiveprefix must be 'arXiv'"),
    ({"primaryclass": None}, "missing field 'primaryclass'"),
    ({"primaryclass": "CS.CL"}, "primaryclass 'CS.CL' is not an arXiv category"),
    ({"url": None}, "missing field 'url'"),
    ({"url": "https://arxiv.org/abs/2404.05406"}, "url must be https://arxiv.org/abs/2404.05405"),
    ({"howpublished": "arXiv preprint arXiv:2404.05406"}, "howpublished must be 'arXiv preprint arXiv:2404.05405'"),
])
def test_arxiv_rules(change, expected):
    assert f"allenzhu2024physics: {expected}" in problems_of(misc(**change))


@pytest.mark.parametrize("verified, expected", [
    ("yesterday", "verified must look like 'YYYY-MM-DD, https://...'"),
    ("2026-11-19 https://arxiv.org/abs/2404.05405", "verified must look like 'YYYY-MM-DD, https://...'"),
    ("2026-13-40, https://arxiv.org/abs/2404.05405", "verified has the date '2026-13-40', which is not a calendar date"),
    ("2026-11-19, https://arxiv.org/abs/2404.05406", "verified must name the page https://arxiv.org/abs/2404.05405"),
    ("2026-11-19, http://arxiv.org/abs/2404.05405", "verified must look like 'YYYY-MM-DD, https://...'"),
])
def test_verified_rules(verified, expected):
    assert f"allenzhu2024physics: {expected}" in problems_of(misc(verified=verified))


def test_duplicates_across_entries_are_problems():
    one, two = misc(), inproceedings()
    assert "duplicate citation key 'allenzhu2024physics'" in problems_of(one + one)
    same_title = two.replace("{Decoupled Weight Decay Regularization}", ANCHOR["title"])
    assert any(p.startswith("duplicate title") for p in problems_of(one + same_title))
    same_eprint = two.replace("1711.05101", "2404.05405")
    assert any(p.startswith("duplicate eprint") for p in problems_of(one + same_eprint))


def test_problems_come_in_file_order_one_line_each():
    text = misc(year="2027") + "\n" + inproceedings(author="Hutter")
    problems = problems_of(text)
    assert problems == ["allenzhu2024physics: year '2027' is not a four-digit year from 1900 to 2026",
                        "loshchilov2019decoupled: author 'Hutter' is not in the form 'Last, First'"]


# ---- the committed bibliography --------------------------------------------------------------------------------------------------------------
def test_the_committed_file_has_exactly_the_two_verified_entries_with_exactly_their_fields():
    entries = cb.parse_bib(BIB.read_text(encoding="utf-8"))
    assert {(e.type, e.key): e.fields for e in entries} == PINNED
    assert [e.key for e in entries] == ["allenzhu2024physics", "loshchilov2019decoupled"]


def test_the_committed_file_is_clean_and_ascii():
    text = BIB.read_text(encoding="utf-8")
    assert cb.check_text(text) == [] and text.isascii()       # check_text refuses stray text outside the entries


# ---- the command line ------------------------------------------------------------------------------------------------------------------------
def test_main_exit_codes_and_output(tmp_path, capsys):
    assert cb.main([str(BIB)]) == 0
    assert capsys.readouterr().out.strip() == f"{BIB}: 2 entries, no problems"
    bad = tmp_path / "bad.bib"
    bad.write_text(misc(year="2027"), encoding="utf-8")
    assert cb.main([str(bad)]) == 1
    out = capsys.readouterr().out
    assert "allenzhu2024physics: year '2027' is not a four-digit year from 1900 to 2026" in out and out.count("\n") == 1
    assert cb.main([str(tmp_path / "missing.bib")]) == 2
    assert "cannot read" in capsys.readouterr().err
    notutf = tmp_path / "latin.bib"
    notutf.write_bytes(b"@misc{k2020, title = {\xe9}}")
    assert cb.main([str(notutf)]) == 2
    assert "not UTF-8" in capsys.readouterr().err


def test_main_with_a_tex_file_reports_uncited_entries_and_citations_without_an_entry(tmp_path, capsys):
    tex = tmp_path / "main.tex"
    tex.write_text("Text \\citep{allenzhu2024physics} and \\citet{nobody2020x}. % \\cite{commented2020y}\n", encoding="utf-8")
    assert cb.main([str(BIB), "--tex", str(tex)]) == 1
    out = capsys.readouterr().out
    assert "loshchilov2019decoupled: not cited in " in out and "citation key 'nobody2020x' has no entry" in out
    assert "commented2020y" not in out and "allenzhu2024physics: not cited" not in out
    tex.write_text("\\citep{allenzhu2024physics,loshchilov2019decoupled}", encoding="utf-8")
    assert cb.main([str(BIB), "--tex", str(tex)]) == 0
    assert capsys.readouterr().out.strip() == f"{BIB}: 2 entries, no problems"


def test_main_with_an_unreadable_tex_file_is_exit_2(tmp_path, capsys):
    assert cb.main([str(BIB), "--tex", str(tmp_path / "none.tex")]) == 2
    assert "cannot read" in capsys.readouterr().err


def test_the_module_runs_as_a_command_from_another_folder(tmp_path):
    done = subprocess.run([sys.executable, "-m", "report.check_bib", str(BIB)], cwd=ROOT, capture_output=True, text=True)
    assert done.returncode == 0 and "2 entries, no problems" in done.stdout, done.stdout + done.stderr
    script = subprocess.run([sys.executable, str(ROOT / "report" / "check_bib.py"), str(BIB)], cwd=tmp_path, capture_output=True, text=True)
    assert script.returncode == 0 and "2 entries, no problems" in script.stdout, script.stdout + script.stderr


# ---- gaps found by the planted-bug run -------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("key", ["2024physics", "allenzhu202physics", "allenzhu2024Physics", "allenzhu2024-physics", "allenzhu2024physics!", "Allenzhu2024"])
def test_these_keys_are_refused(key):
    problems = problems_of(misc().replace("allenzhu2024physics", key))
    assert any(p.startswith(f"{key}: citation key must be") for p in problems), problems


@pytest.mark.parametrize("key", ["allenzhu2024a1", "allenzhu2024", "a2024", "allenzhu2024physics3"])
def test_these_keys_are_fine(key):
    assert problems_of(misc().replace("allenzhu2024physics", key)) == []


@pytest.mark.parametrize("category, ok", [
    ("cs.CL", True), ("cs.LG", True), ("stat.ML", True), ("math.OC", True), ("hep-th", True), ("math-ph", True), ("cond-mat.mes-hall", True),
    ("physics.acc-ph", True), ("q-bio.NC", True),
    ("CS.CL", False), ("cs.", False), ("cs.C", False), ("cs.CLX", False), (".CL", False), ("cs CL", False), ("", False), ("cs.CL.X", False),
])
def test_arxiv_categories(category, ok):
    problems = problems_of(misc(primaryclass=category))
    assert any("is not an arXiv category" in p for p in problems) is (not ok), problems


@pytest.mark.parametrize("verified", ["2026-1-5, https://arxiv.org/abs/2404.05405", "2026-11-5, https://arxiv.org/abs/2404.05405"])
def test_verified_needs_a_two_digit_month_and_day(verified):
    assert "allenzhu2024physics: verified must look like 'YYYY-MM-DD, https://...'" in problems_of(misc(verified=verified))


def test_field_names_may_have_dashes_and_underscores_but_not_start_with_a_digit():
    (entry,) = cb.parse_bib("@misc{k2020, primary-class = {x}, a_b = {y}}")
    assert entry.fields == {"primary-class": "x", "a_b": "y"}
    with pytest.raises(cb.BibError, match="line 1: a field name is missing"):
        cb.parse_bib("@misc{k2020, 1st = {x}}")


def test_the_allowed_extra_field_names_are_all_accepted():
    fields = {"author": "A, B", "title": "{T}", "journal": "J", "year": "2020", "volume": "1", "number": "2", "pages": "3--4", "publisher": "P",
              "doi": "10.1/x", "note": "n", "verified": "2026-11-19, https://example.org/p"}
    assert problems_of(entry_text("article", "a2020x", fields)) == []


@pytest.mark.parametrize("text, message", [
    ("\n@misc{k2020, title = }", "line 2: a value is missing"),
    ("@comment{ never closed", "line 1: the entry is not closed"),
    ("@comment text", "line 1: an entry must open with a brace"),
    ("@misc{k2020, title = {T {x}", "line 1: the entry is not closed"),
    ("@misc", "line 1: an entry must open with a brace"),
    ("@misc  ", "line 1: an entry must open with a brace"),
    ("@misc{", "line 1: missing citation key"),
    ("@misc{k2020", "line 1: the entry is not closed"),
    ("@misc{k2020,", "line 1: the entry is not closed"),
    ("@misc{k2020, title", "line 1: the entry is not closed"),
    ("@misc{k2020, title =", "line 1: the entry is not closed"),
    ("@misc{k2020, title = {T}", "line 1: the entry is not closed"),
    ("@misc{k2020, title = {T},", "line 1: the entry is not closed"),
    ("@misc{k2020, title = , year = 2020}", "line 1: a value is missing"),
])
def test_end_of_file_and_other_edges_of_the_parser(text, message):
    with pytest.raises(cb.BibError, match=re.escape(message)):
        cb.parse_bib(text)


def test_quoted_values_empty_or_with_braces_and_quotes_inside():
    (entry,) = cb.parse_bib('@misc{k2020, title = "", note = "{a} {"} b"}')
    assert entry.fields == {"title": "", "note": '{a} {"} b'}


def test_entries_without_fields_and_entries_written_without_spaces():
    assert [(e.key, e.fields) for e in cb.parse_bib("@misc{k2020}@misc{k2021}")] == [("k2020", {}), ("k2021", {})]
    assert [e.key for e in cb.parse_bib("@misc{k2020, title = {T},}@misc{k2021, title = {U}}")] == ["k2020", "k2021"]
    (entry,) = cb.parse_bib("@misc{k2020,title={T},year=2020}")
    assert entry.fields == {"title": "T", "year": "2020"}
    assert problems_of("@misc{k2020}")[:2] == ["k2020: missing field 'author'", "k2020: missing field 'title'"]


def test_a_comment_on_the_last_line_without_a_newline_is_fine():
    assert [e.key for e in cb.parse_bib("@misc{k2020, title = {T}}\n% last line")] == ["k2020"]
    assert cb.parse_bib("% only") == []


def test_an_entry_cannot_be_changed_after_it_is_read():
    (entry,) = cb.parse_bib("@misc{k2020, title = {T}}")
    with pytest.raises(dataclasses.FrozenInstanceError):
        entry.key = "other"


@pytest.mark.parametrize("value, expected", [
    ("A, B and C, D", ["A, B", "C, D"]),
    ("A, B AND C, D", ["A, B", "C, D"]),
    ("A, B\nand\nC, D", ["A, B", "C, D"]),
    ("{X and Y} and Z, W", ["{X and Y}", "Z, W"]),
    ("{X and Y}", ["{X and Y}"]),
    ("Andrew, Brown and Sand, Anders", ["Andrew, Brown", "Sand, Anders"]),
    ("A, B and C, D and E, F", ["A, B", "C, D", "E, F"]),
    ("A, B and ", ["A, B", ""]),
    ("", [""]),
    ("A, B and {C, D}", ["A, B", "{C, D}"]),
    ("Band, Bill", ["Band, Bill"]),
])
def test_split_authors(value, expected):
    assert cb.split_authors(value) == expected


@pytest.mark.parametrize("text, expected", [
    ("{a}", True), ("{{a}}", True), ("{a{b}c}", True), ("{}", True), ("{a} {b}", False), ("{a}{b}", False), ("{a} b", False), ("a {b}", False), ("{a", False),
    ("a}", False), ("", False), ("}{", False), ("{a}}", False), ("{{a}", False),
])
def test_is_single_group(text, expected):
    assert cb.is_single_group(text) is expected


@pytest.mark.parametrize("author, expected", [
    ("Allen-Zhu, Zeyuan and others", ["author list ends in 'others'; list every author"]),
    ("Allen-Zhu, Zeyuan and Others", ["author list ends in 'others'; list every author"]),
    ("Smith, {Jr., John}", []),
    ("Smith, J. et al.", ["author list contains 'et al'; list every author"]),
    ("Smith, J ET AL", ["author list contains 'et al'; list every author"]),
    ("Smith, J et. al", ["author list contains 'et al'; list every author"]),
    ("Smith, Janet Al", []),
    ("Poet, Al", []),
    ("A, B and C, D", []),
])
def test_exact_author_problems(author, expected):
    mine = [p.split(": ", 1)[1] for p in problems_of(misc(author=author)) if "author" in p.split(": ", 1)[1]]
    assert mine == expected


@pytest.mark.parametrize("field", ["title", "author", "year"])
def test_an_empty_or_blank_field_is_exactly_one_problem(field):
    for blank in ("", "   "):
        assert problems_of(misc(**{field: blank})) == [f"allenzhu2024physics: field '{field}' is empty"]


def test_a_repeated_key_is_exactly_one_problem_and_not_also_a_repeated_title_or_eprint():
    assert problems_of(misc() + "\n" + misc()) == ["duplicate citation key 'allenzhu2024physics'"]


@pytest.mark.parametrize("title", [
    "{physics of language models: part 3.3, knowledge capacity scaling laws}",
    "{Physics  of Language   Models: Part 3.3, Knowledge Capacity Scaling Laws}",
    "{Physics of {Language} Models: Part 3.3, Knowledge Capacity Scaling Laws}",
])
def test_titles_that_differ_only_in_case_spacing_or_inner_braces_are_the_same_title(title):
    assert any(p.startswith("duplicate title (keys 'allenzhu2024physics' and 'loshchilov2019decoupled')")
               for p in problems_of(misc() + "\n" + inproceedings(title=title)))


def test_entries_without_a_title_or_an_eprint_are_not_duplicates_of_each_other():
    one = entry_text("misc", "a2020x", {"author": "A, B", "year": "2020", "howpublished": "h", "verified": "2026-11-19, https://example.org/p"})
    two = one.replace("a2020x", "b2020x")
    assert not any(p.startswith("duplicate") for p in problems_of(one + "\n" + two))


def test_howpublished_is_only_held_to_the_arxiv_form_for_a_misc_entry():
    assert problems_of(inproceedings(howpublished="Some text")) == []


def test_a_byte_order_mark_at_the_start_of_the_file_is_ignored(tmp_path, capsys):
    path = tmp_path / "bom.bib"
    path.write_bytes(b"\xef\xbb\xbf" + BIB.read_bytes())
    assert cb.main([str(path)]) == 0
    assert "2 entries, no problems" in capsys.readouterr().out


def test_citations_without_an_entry_are_listed_in_sorted_order(tmp_path, capsys):
    tex = tmp_path / "main.tex"
    keys = ["kkk2020a", "bbb2020a", "ttt2020a", "ddd2020a", "zzz2020a", "mmm2020a", "aaa2020a", "qqq2020a"]     # eight, so that set order is never sorted by luck
    tex.write_text("\\cite{" + ",".join(keys) + "}", encoding="utf-8")
    assert cb.main([str(BIB), "--tex", str(tex)]) == 1
    lines = capsys.readouterr().out.splitlines()
    assert [line.split(":")[0].split("'")[1] if line.startswith("citation") else line.split(":")[0] for line in lines] == [
        "allenzhu2024physics", "loshchilov2019decoupled"] + sorted(keys)


# ---- gaps left by the second planted-bug round -----------------------------------------------------------------------------------------------
def test_an_equals_sign_cannot_be_part_of_a_citation_key():
    with pytest.raises(cb.BibError, match="line 1: expected a comma after the citation key"):
        cb.parse_bib("@misc{k=2020, title = {T}}")


@pytest.mark.parametrize("text, message", [
    ("@misc\n\n", "line 1: an entry must open with a brace"),
    ("@misc  \n\n\n", "line 1: an entry must open with a brace"),
    ("@misc{\n\n", "line 1: missing citation key"),
    ("\n\n@misc{k2020}\n@misc{\n", "line 4: missing citation key"),
])
def test_an_error_at_the_end_of_the_text_names_the_line_of_the_entry_not_of_the_last_newline(text, message):
    with pytest.raises(cb.BibError, match=re.escape(message)):
        cb.parse_bib(text)


@pytest.mark.parametrize("value, expected", [
    ("{A {x} and B} and C, D", ["{A {x} and B}", "C, D"]),
    ("A, B} and C, D", ["A, B}", "C, D"]),
    ("{A {{x}} and B} and C, D", ["{A {{x}} and B}", "C, D"]),
    ("Hand, Brand", ["Hand, Brand"]),
    ("Brand, Hand and Sand, Brand", ["Brand, Hand", "Sand, Brand"]),
])
def test_split_authors_nested_stray_and_and_inside_a_name(value, expected):
    assert cb.split_authors(value) == expected


@pytest.mark.parametrize("author, expected", [
    ("Smith, Etal", []),
    ("Zeyuan and others", ["author list ends in 'others'; list every author", "author 'Zeyuan' is not in the form 'Last, First'"]),
    ("Good, One and Bad and others", ["author list ends in 'others'; list every author", "author 'Bad' is not in the form 'Last, First'"]),
    ("{Smith, J} Doe", ["author '{Smith, J} Doe' is not in the form 'Last, First'"]),
])
def test_more_exact_author_problems(author, expected):
    mine = [p.split(": ", 1)[1] for p in problems_of(misc(author=author)) if "author" in p.split(": ", 1)[1]]
    assert sorted(mine) == sorted(expected)


# ---- a real BibTeX build ---------------------------------------------------------------------------------------------------------------------
TINY = r"""\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{natbib}
\usepackage{url}
\hyphenpenalty=10000 \exhyphenpenalty=10000 \sloppy
\begin{document}
Weight decay follows \citet{loshchilov2019decoupled}; the anchor is \citep{allenzhu2024physics}.
\bibliographystyle{plainnat}
\bibliography{references}
\end{document}
"""


def run_tool(args, cwd):
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=180)


@pytest.mark.integration
def test_pdflatex_and_bibtex_accept_the_file_and_print_both_references_exactly(tmp_path):
    for tool in ("pdflatex", "bibtex", "pdftotext"):
        assert shutil.which(tool), f"{tool} is not on PATH: the bibliography build test needs TeX (see report/README.md)"
    shutil.copy(BIB, tmp_path / "references.bib")
    (tmp_path / "tiny.tex").write_text(TINY, encoding="utf-8")
    assert run_tool(["pdflatex", "-interaction=nonstopmode", "tiny.tex"], tmp_path).returncode == 0
    bib = run_tool(["bibtex", "tiny"], tmp_path)
    assert bib.returncode == 0, bib.stdout
    blg = (tmp_path / "tiny.blg").read_text(encoding="utf-8", errors="replace")
    assert not re.search(r"^(Warning--|I couldn't|I was expecting|---)", blg, flags=re.M), blg
    for _ in range(2):
        assert run_tool(["pdflatex", "-interaction=nonstopmode", "tiny.tex"], tmp_path).returncode == 0
    log = (tmp_path / "tiny.log").read_text(encoding="utf-8", errors="replace")
    assert not re.search(r"Citation .* undefined|There were undefined", log), log
    assert run_tool(["pdftotext", "-layout", "tiny.pdf", "tiny.txt"], tmp_path).returncode == 0
    text = " ".join((tmp_path / "tiny.txt").read_text(encoding="utf-8").split())
    assert text.count("References") == 1
    for expected in (
        "Zeyuan Allen-Zhu and Yuanzhi Li. Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws. arXiv preprint arXiv:2404.05405, 2024.",
        "Ilya Loshchilov and Frank Hutter. Decoupled Weight Decay Regularization. In International Conference on Learning Representations, 2019.",
        "Loshchilov and Hutter [2019]",
        "[Allen-Zhu and Li, 2024]",
    ):
        assert expected in text, (expected, text)
