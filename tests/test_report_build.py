"""Tests for the report gate helpers (report/build_report.py). Nothing here needs a TeX engine."""
from __future__ import annotations

import re

import pytest

from report import build_report as br

AUX_PLAIN = r"\newlabel{endcontent}{{3}{9}}"
AUX_HYPERREF = r"\newlabel{endcontent}{{}{9}{}{Doc-Start}{}}"


# ---- content_last_page: where the end-of-content label landed -----------------------------------------------------

def test_content_last_page_reads_plain_and_hyperref_aux():
    assert br.content_last_page(AUX_PLAIN) == 9
    assert br.content_last_page(AUX_HYPERREF) == 9


def test_content_last_page_ten_is_ten():
    assert br.content_last_page(r"\newlabel{endcontent}{{}{10}{}{x}{}}") == 10


def test_content_last_page_missing_label_raises():
    with pytest.raises(ValueError):
        br.content_last_page(r"\newlabel{other}{{}{3}{}{x}{}}")


def test_content_last_page_does_not_match_a_longer_label_name():
    with pytest.raises(ValueError):
        br.content_last_page(r"\newlabel{endcontent2}{{}{9}{}{x}{}}")


# ---- parse_log -----------------------------------------------------------------------------------------------------

LOG = (
    "This is pdfTeX, Version 3.14159265\n"
    "! Undefined control sequence.\n"
    "LaTeX Warning: Reference `fig:cap' on page 3 undefined on input line 12.\n"
    "Package natbib Warning: Citation `li2018' on page 1 undefined on input line 7.\n"
    "LaTeX Warning: Label `fig:cap' multiply defined.\n"
)


def test_parse_log_finds_each_problem_class():
    assert br.parse_log(LOG) == {
        "errors": ["Undefined control sequence."],
        "undefined_refs": ["fig:cap"],
        "undefined_cites": ["li2018"],
        "multiply_defined": ["fig:cap"],
    }


def test_parse_log_clean_log_is_empty():
    clean = "This is pdfTeX, Version 3.14159265\nOutput written on main.pdf (8 pages).\n"
    assert br.parse_log(clean) == {"errors": [], "undefined_refs": [], "undefined_cites": [], "multiply_defined": []}


def test_parse_log_joins_lines_wrapped_inside_a_label():
    label = "fig:" + "x" * 70
    warning = f"LaTeX Warning: Reference `{label}' on page 3 undefined on input line 12."
    wrapped = warning[:79] + "\n" + warning[79:] + "\n"
    assert len(wrapped.splitlines()[0]) == 79
    assert br.parse_log(wrapped)["undefined_refs"] == [label]


def test_parse_log_keeps_a_final_line_of_exactly_79_characters():
    last = "! " + "y" * 77
    assert len(last) == 79
    assert br.parse_log("ok\n" + last)["errors"] == ["y" * 77]


# ---- missing_sections ----------------------------------------------------------------------------------------------

TEX_OK = (
    "\\begin{abstract}A.\\end{abstract}\n"
    "\\section{Introduction}x\n\\section{Related Work}y\n\\section{Conclusion}z\n"
    "\\section*{References}\n"
    "\\appendix\\section{Use Of Artificial Intelligence}q\n"
)


def test_missing_sections_none_missing():
    assert br.missing_sections(TEX_OK) == []


def test_missing_sections_reports_the_absent_one():
    assert br.missing_sections(TEX_OK.replace("\\section{Related Work}y\n", "")) == ["Related Work"]


def test_missing_sections_ignores_a_commented_out_heading():
    commented = TEX_OK.replace("\\section{Related Work}y\n", "% \\section{Related Work}\n")
    assert br.missing_sections(commented) == ["Related Work"]


def test_missing_sections_accepts_the_bibliography_command_as_references():
    assert br.missing_sections(TEX_OK.replace("\\section*{References}", "\\bibliography{refs}")) == []


def test_missing_sections_needs_the_abstract_environment():
    assert br.missing_sections(TEX_OK.replace("\\begin{abstract}A.\\end{abstract}\n", "")) == ["Abstract"]


# ---- uncited_bib_entries -------------------------------------------------------------------------------------------

BIB = '@article{li2018,\n title={x}\n}\n@misc{robins1995,\n title={y}\n}\n@comment{ignored, here}\n@string{s = "v"}\n'


def test_uncited_bib_entries_lists_only_the_uncited_one():
    assert br.uncited_bib_entries("See \\citep[p.~3]{li2018}.", BIB) == ["robins1995"]


def test_uncited_bib_entries_multi_key_cite_counts_every_key():
    assert br.uncited_bib_entries("\\citet{li2018, robins1995}", BIB) == []


def test_uncited_bib_entries_a_commented_cite_does_not_count():
    assert br.uncited_bib_entries("% \\cite{li2018}\n\\cite{robins1995}\n", BIB) == ["li2018"]


def test_an_escaped_percent_sign_does_not_hide_a_citation_after_it():
    assert br.uncited_bib_entries("Accuracy was 100\\% \\citep{li2018}.", BIB) == ["robins1995"]


def test_a_comment_after_text_on_the_same_line_is_dropped():
    assert br.uncited_bib_entries("\\citep{li2018} % \\cite{robins1995}", BIB) == ["robins1995"]


# ---- float_problems ------------------------------------------------------------------------------------------------

FIG_OK = "\\begin{figure}\\centering\\caption{c}\\label{fig:a}\\end{figure}\nSee Figure~\\ref{fig:a}.\n"


def test_float_problems_clean():
    assert br.float_problems(FIG_OK) == []


def test_float_problems_unreferenced_label():
    assert br.float_problems(FIG_OK.replace("\\ref{fig:a}", "")) == ["label fig:a is never referenced"]


def test_float_problems_a_longer_label_is_not_a_reference_to_a_shorter_one():
    assert br.float_problems(FIG_OK.replace("\\ref{fig:a}", "\\ref{fig:ab}")) == ["label fig:a is never referenced"]


def test_float_problems_missing_caption():
    assert br.float_problems(FIG_OK.replace("\\caption{c}", "")) == ["figure without a caption"]


def test_float_problems_missing_label():
    assert br.float_problems(FIG_OK.replace("\\label{fig:a}", "")) == ["figure without a label"]


def test_float_problems_starred_table_is_checked_too():
    tex = "\\begin{table*}\\label{tab:b}\\end{table*} \\ref{tab:b}"
    assert br.float_problems(tex) == ["table without a caption"]


def test_float_problems_cleveref_with_several_labels():
    tex = "\\begin{table*}\\caption{t}\\label{tab:b}\\end{table*} \\cref{fig:z,tab:b}"
    assert br.float_problems(tex) == []


def test_float_problems_two_floats_are_judged_separately():
    second = "\\begin{figure}\\label{fig:b}\\end{figure} \\ref{fig:b}\n"
    assert br.float_problems(FIG_OK + second) == ["figure without a caption"]


def test_float_problems_a_commented_out_caption_does_not_count():
    tex = FIG_OK.replace("\\caption{c}", "% \\caption{c}\n")
    assert br.float_problems(tex) == ["figure without a caption"]


# ---- page budget: two independent measurements must agree, and the limit is nine inclusive --------------------------

def test_first_references_page_finds_the_heading_line():
    assert br.first_references_page(["Intro", "more", "Conclusion\nReferences\n[1] Li"]) == 3


def test_first_references_page_ignores_the_word_inside_a_sentence():
    assert br.first_references_page(["see References for details", "References\n[1] x"]) == 2


def test_first_references_page_missing_raises():
    with pytest.raises(ValueError):
        br.first_references_page(["a", "b"])


@pytest.mark.parametrize("label_page,references_page,ok", [(9, 9, True), (9, 10, True), (10, 10, False), (10, 11, False)])
def test_page_budget_limit_is_nine_inclusive(label_page, references_page, ok):
    assert (br.check_page_budget(label_page, references_page) == []) is ok


def test_page_budget_disagreeing_measurements_fail_loudly():
    problems = br.check_page_budget(8, 10)
    assert len(problems) == 1 and "disagree" in problems[0]


def test_page_budget_references_before_the_label_is_a_disagreement():
    assert any("disagree" in p for p in br.check_page_budget(9, 8))


# ---- check_report: everything together -----------------------------------------------------------------------------

CLEAN_TEX = (
    "\\begin{abstract}A.\\end{abstract}\n"
    "\\section{Introduction}See Figure~\\ref{fig:a} and \\citep{li2018,robins1995}.\n"
    "\\section{Related Work}x\n"
    "\\begin{figure}\\caption{c}\\label{fig:a}\\end{figure}\n"
    "\\section{Conclusion}y \\label{endcontent}\n"
    "\\section*{References}\n"
    "\\appendix\\section{Use Of Artificial Intelligence}z\n"
)
CLEAN_BIB = "@article{li2018,\n}\n@article{robins1995,\n}\n"
CLEAN_AUX = "\\newlabel{endcontent}{{3}{8}}\n"
CLEAN_PAGES = ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "References\n[1]"]


def run(tex=CLEAN_TEX, bib=CLEAN_BIB, aux=CLEAN_AUX, log="", pages=CLEAN_PAGES):
    return br.check_report(tex, bib, aux, log, pages)


def test_check_report_clean_inputs_have_no_problems():
    assert run() == []


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({"tex": CLEAN_TEX.replace("\\section{Related Work}x\n", "")}, "missing required section: Related Work"),
        ({"bib": CLEAN_BIB + "@misc{unused,\n}\n"}, "bibliography entry never cited: unused"),
        ({"log": "ok\n! Undefined control sequence.\n"}, "LaTeX error: Undefined control sequence."),
        ({"log": "LaTeX Warning: Reference `fig:zzz' on page 3 undefined on input line 4.\n"}, "undefined reference: fig:zzz"),
        ({"log": "Package natbib Warning: Citation `nope2020' on page 1 undefined on input line 4.\n"}, "undefined citation: nope2020"),
        ({"log": "LaTeX Warning: Label `fig:a' multiply defined.\n"}, "label defined twice: fig:a"),
        ({"tex": CLEAN_TEX.replace("Figure~\\ref{fig:a} and ", "")}, "label fig:a is never referenced"),
        (
            {"aux": "\\newlabel{endcontent}{{3}{10}}\n", "pages": CLEAN_PAGES[:8] + ["x"] + ["References"]},
            "content ends on page 10; the limit is 9 content pages",
        ),
    ],
)
def test_check_report_each_defect_class_is_reported_exactly_once(kwargs, expected):
    assert run(**kwargs) == [expected]


def test_check_report_missing_end_label_is_reported_instead_of_passing():
    problems = run(aux="")
    assert len(problems) == 1 and problems[0].startswith("page budget not measured: label 'endcontent' not found")


def test_check_report_missing_references_heading_is_reported_instead_of_passing():
    assert run(pages=CLEAN_PAGES[:8] + ["no heading here"]) == [
        "page budget not measured: no page has a line that is exactly 'References'"
    ]


def test_check_report_lists_every_other_problem_when_the_page_measurements_fail():
    tex = CLEAN_TEX.replace("\\section{Related Work}x\n", "")
    problems = run(tex=tex, aux="", pages=["a", "b"], log="! Undefined control sequence.\n")
    assert [problem.split(":")[0] for problem in problems] == [
        "LaTeX error",
        "missing required section",
        "page budget not measured",
        "page budget not measured",
    ]


# ---- added in W1.3, after reading the helpers against how TeX itself reads a file ---------------------------------
# (Appendix C had check_report raise on an unmeasurable page, which hid every other problem from the command-line wrapper.)

def test_content_last_page_error_says_how_to_add_the_label():
    with pytest.raises(ValueError, match=r"\\label\{endcontent\}"):
        br.content_last_page("")


def test_a_percent_after_a_line_break_command_starts_a_comment():
    # in TeX the two backslashes are one control symbol, so the percent sign after them is a real comment
    tex = "row \\\\% \\cite{li2018}\n\\cite{robins1995}"
    assert br.strip_comments(tex) == "row \\\\\n\\cite{robins1995}"
    assert br.uncited_bib_entries(tex, BIB) == ["li2018"]


def test_a_percent_after_an_odd_number_of_backslashes_is_escaped():
    # backslash-backslash then backslash-percent: the percent is escaped, so the citation after it counts
    tex = "\\\\\\% \\cite{li2018} \\cite{robins1995}"
    assert br.strip_comments(tex) == tex
    assert br.uncited_bib_entries(tex, BIB) == []


@pytest.mark.parametrize("other", ["fig:a-b", "fig:a.b", "fig:a:b", "fig:ab", "xfig:a"])
def test_float_problems_only_the_exact_label_counts_as_a_reference(other):
    tex = FIG_OK.replace("\\ref{fig:a}", "\\ref{" + other + "}")
    assert br.float_problems(tex) == ["label fig:a is never referenced"]


def test_float_problems_spaces_after_the_commas_of_a_reference_list_are_allowed():
    tex = "\\begin{table*}\\caption{t}\\label{tab:b}\\end{table*} \\cref{fig:z, tab:b}"
    assert br.float_problems(tex) == []


# ---- added in W1.3 after the first planted-bug run: behaviours the tests above left unpinned --------------------------

def test_unwrap_joins_a_79_character_line_to_the_next_one():
    assert br._unwrap("a" * 79 + "\nbcd\nnext\n") == "a" * 79 + "bcd\nnext"


def test_unwrap_does_not_join_a_line_longer_than_79_characters():
    assert br._unwrap("y" * 80 + "\nnext") == "y" * 80 + "\nnext"


def test_unwrap_leaves_no_trailing_blank_line():
    assert br._unwrap("ab\ncd\n") == "ab\ncd"


def test_parse_log_ignores_a_bang_line_with_no_message():
    assert br.parse_log("! \nok\n")["errors"] == []


def test_content_last_page_reads_a_first_group_with_nested_braces():
    assert br.content_last_page(r"\newlabel{endcontent}{{\textbf{3}}{9}{}{x}{}}") == 9


def test_missing_sections_reports_references_when_there_is_no_heading_and_no_bibliography():
    assert br.missing_sections(TEX_OK.replace("\\section*{References}\n", "")) == ["References"]


def test_missing_sections_accepts_a_heading_at_any_level():
    assert br.missing_sections(TEX_OK.replace("\\section{Related Work}", "\\subsection{Related Work}")) == []


@pytest.mark.parametrize(
    "command", ["\\Citet{li2018}", "\\citep*{li2018}", "\\citep[see][p.~3]{li2018}", "\\citeauthor{li2018}", "\\citeNP{li2018}"]
)
def test_uncited_bib_entries_every_natbib_cite_form_counts(command):
    assert br.uncited_bib_entries(command, BIB) == ["robins1995"]


def test_bib_keys_entry_types_are_not_case_sensitive():
    assert br.bib_keys("@ARTICLE{a1,\n}\n@Misc{b2,\n}\n") == ["a1", "b2"]


@pytest.mark.parametrize("kind", ["string", "STRING", "Preamble", "comment", "COMMENT"])
def test_bib_keys_skips_string_preamble_and_comment_blocks_in_any_case_even_when_they_hold_commas(kind):
    bib = "@" + kind + '{jan="January", feb="February"}\n@article{k1,\n}\n'
    assert br.bib_keys(bib) == ["k1"]


@pytest.mark.parametrize("command", ["\\ref", "\\autoref", "\\cref", "\\Cref"])
def test_float_problems_every_reference_command_counts(command):
    assert br.float_problems(FIG_OK.replace("\\ref{fig:a}", command + "{fig:a}")) == []


def test_float_problems_the_word_caption_without_the_command_is_not_a_caption():
    assert br.float_problems(FIG_OK.replace("\\caption{c}", "caption")) == ["figure without a caption"]


def test_float_problems_a_float_may_span_several_lines():
    tex = "\\begin{figure}\n\\centering\n\\label{fig:m}\n\\end{figure}\n\\ref{fig:m}\n"
    assert br.float_problems(tex) == ["figure without a caption"]


def test_first_references_page_ignores_spaces_around_the_heading_line():
    assert br.first_references_page(["intro", "  References  \n[1] x"]) == 2


# ---- added in W1.3 after the code review (G5) -----------------------------------------------------------------------

def test_text_above_references_lists_the_non_blank_lines_before_the_heading():
    assert br.text_above_references(["a", "Conclusion\n\n  more text \nReferences\n[1]"], 2) == ["Conclusion", "more text"]
    assert br.text_above_references(["a", "References\n[1]"], 2) == []
    assert br.text_above_references(["a", "\n \nReferences\n[1]"], 2) == []


def test_check_report_content_above_the_heading_on_the_page_after_the_label_means_the_label_is_too_early():
    pages = CLEAN_PAGES[:8] + ["An acknowledgment that spilled over\nReferences\n[1]"]
    assert run(pages=pages) == [
        "content continues on page 9 above the 'References' heading (first line: 'An acknowledgment that spilled over'), "
        "so the end-of-content label on page 8 is too early"
    ]


def test_text_above_references_finds_a_heading_line_with_spaces_around_it():
    assert br.text_above_references(["a", "x\n  References  \n[1]"], 2) == ["x"]


def test_check_report_quotes_the_first_of_several_lines_above_the_heading():
    pages = CLEAN_PAGES[:8] + ["first line\nsecond line\nReferences\n[1]"]
    assert "(first line: 'first line')" in run(pages=pages)[0]


def test_check_report_text_above_the_heading_on_the_same_page_as_the_label_is_fine():
    pages = CLEAN_PAGES[:7] + ["Conclusion text\nReferences\n[1]", "appendix"]
    assert run(pages=pages) == []


def test_check_report_text_above_the_heading_is_not_reported_a_second_time_when_the_pages_disagree():
    pages = CLEAN_PAGES[:7] + ["p8", "p9", "text\nReferences\n[1]"]  # the heading is two pages after the label: one problem, not two
    assert [problem.split(":")[0] for problem in run(pages=pages)] == ["page measurements disagree"]


def test_bib_keys_allows_the_spaces_bibtex_allows():
    assert br.bib_keys("@article{ spaced ,\n}\n@article {other,\n}\n@misc{\n third\n ,\n}\n") == ["spaced", "other", "third"]


def test_cited_and_referenced_keys_are_sets_without_repeats():
    assert br.cited_keys("\\cite{a,b} \\citep{a}") == {"a", "b"}
    assert br.referenced_keys("\\ref{x} \\cref{x,y}") == {"x", "y"}


def test_keys_in_splits_strips_and_drops_empty_keys_and_ignores_comments():
    pattern = re.compile(r"\\x\{([^}]*)\}")
    assert br.keys_in(pattern, "\\x{a, b,,c} % \\x{z}\n\\x{ d }") == ["a", "b", "c", "d"]
