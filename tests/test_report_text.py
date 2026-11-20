"""Tests for report/check_text.py: the mechanical checks on the prose of the report (numbers and claims).

The claim table is the accepted claim ledger (v2, 2026-11-13): each claim's status words, from its row of phonebook/RESULTS.md, and the items of its "Do not say"
line that a program can check. The tests pin the table to literals, so a change to it is a visible decision, and test every rule in both directions.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from report import check_text as ct
from report import make_numbers as mn

ROOT = Path(__file__).resolve().parents[1]
VALUES = {"e4.cap.w64.b1000": "453", "e6.n.w64": "640", "design.bitsentry": "32.575", "e7.next.w64": "905", "design.layers": "2", "e6.bits.w16": "0.566",
          "e4.params.w64": "110,656", "design.hold.window": "100"}

# a paragraph for each claim that satisfies its own rules
GOOD = {
    "CL-1": "Established for this budget, with exact recall of every entry, three book seeds per rung.",
    "CL-2": "Supported; not shown for width 64, whose rung still fails at weight decay 0.1.",
    "CL-3": "Established by the pre-registered rule: at 4000 epochs the next rung passes at three widths.",
    "CL-4": "Not true in E6: at widths 16 and 32 the 100\\% is isolated.",
    "CL-5": "Not established: the rungs above E7's passes were not run.",
    "CL-6": "Every capacity is a lower bound on what the model could store.",
    "CL-7": "Supported: with the decay at 0.0 the rung passes at width 64.",
}


def document(*pairs) -> str:
    """A tex source with a claim paragraph for each (id, text); blank lines between paragraphs."""
    return "\n\n".join(f"% claim: {claim}\n{body}" for claim, body in pairs) + "\n"


def problems(tex: str, final: bool = False) -> list[str]:
    return ct.claim_problems(tex, final=final)


# ---- the table -------------------------------------------------------------------------------------------------------------------------------
def test_the_claims_are_the_ledgers_seven():
    assert list(ct.CLAIMS) == ["CL-1", "CL-2", "CL-3", "CL-4", "CL-5", "CL-6", "CL-7"] and ct.FINAL_CLAIMS == tuple(ct.CLAIMS)


def test_the_required_status_words_are_the_ledgers():
    assert {claim: [label for label, _ in spec.required] for claim, spec in ct.CLAIMS.items()} == {
        "CL-1": ["Established"], "CL-2": ["Supported", "not shown for width 64"], "CL-3": ["Established"], "CL-4": ["Not true in E6"],
        "CL-5": ["Not established"], "CL-6": ["lower bound"], "CL-7": ["Supported"]}


def test_the_forbidden_phrases_of_each_claim_are_the_ledgers_do_not_say_items_that_can_be_checked():
    assert {claim: [label for label, _ in spec.forbidden] for claim, spec in ct.CLAIMS.items()} == {
        "CL-1": ["can store", "has a capacity of"], "CL-2": ["established", "shown"], "CL-3": ["holds", "keeps"], "CL-4": ["cannot hold"], "CL-5": [],
        "CL-6": [], "CL-7": ["established"]}
    assert [label for label, _ in ct.GLOBAL_FORBIDDEN] == ["prove", "impossible", "optimal", "guarantee"]


# ---- finding the claim paragraphs ------------------------------------------------------------------------------------------------------------
def test_a_claim_paragraph_runs_from_its_marker_to_the_next_blank_line():
    tex = "Intro text.\n\n% claim: CL-4\nFirst line\nsecond line.\n\nNot a claim.\n"
    (paragraph,) = ct.claim_paragraphs(tex)
    assert (paragraph.ids, paragraph.line, paragraph.text) == (("CL-4",), 3, "First line\nsecond line.")


def test_stacked_markers_share_the_paragraph_and_a_comma_list_names_several_claims():
    stacked = ct.claim_paragraphs("% claim: CL-2\n% claim: CL-3\nBody.\n")
    assert [(p.ids, p.line, p.text) for p in stacked] == [(("CL-2", "CL-3"), 1, "Body.")]
    listed = ct.claim_paragraphs("% claim: CL-2, CL-3\nBody.\n")
    assert [(p.ids, p.text) for p in listed] == [(("CL-2", "CL-3"), "Body.")]


def test_markers_with_spaces_crlf_and_indentation_are_found():
    tex = "  %   claim:   CL-6  \r\nBody \\textit{text}.\r\n\r\nOther.\r\n"
    (paragraph,) = ct.claim_paragraphs(tex)
    assert (paragraph.ids, paragraph.text) == (("CL-6",), "Body \\textit{text}.")


def test_a_claim_paragraph_at_the_end_of_the_file_without_a_final_newline():
    (paragraph,) = ct.claim_paragraphs("% claim: CL-6\nBody.")
    assert paragraph.text == "Body."


def test_a_comment_that_merely_mentions_claims_is_not_a_marker():
    assert ct.claim_paragraphs("% a claim: CL-4 is discussed here\nBody.\n") == []
    assert ct.claim_paragraphs("% claims: CL-4\nBody.\n") == []


@pytest.mark.parametrize("tex, message", [
    ("% claim: CL-9\nBody.\n", "line 1: unknown claim 'CL-9' (use CL-1 to CL-7)"),
    ("% claim: banana\nBody.\n", "line 1: a claim marker must look like '% claim: CL-4'"),
    ("% claim:\nBody.\n", "line 1: a claim marker must look like '% claim: CL-4'"),
    ("% claim: CL-4\n\nBody.\n", "line 1: claim marker with no paragraph after it"),
    ("% claim: CL-4\n", "line 1: claim marker with no paragraph after it"),
    ("Text before.\n% claim: CL-4\nBody.\n", "line 2: a claim marker must start a paragraph (put a blank line before it)"),
])
def test_a_misplaced_or_malformed_marker_is_a_problem(tex, message):
    assert message in problems(tex)


# ---- the required status words ---------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("claim", sorted(GOOD))
def test_a_paragraph_with_the_status_words_and_nothing_forbidden_is_clean(claim):
    assert problems(document((claim, GOOD[claim]))) == []


@pytest.mark.parametrize("claim, text, missing", [
    ("CL-1", "A model memorizes entries within the budget.", "Established"),
    ("CL-1", "It is not established for this budget.", "Established"),
    ("CL-2", "Supported, but only for three widths.", "not shown for width 64"),
    ("CL-2", "Not shown for width 64, a clear limit.", "Supported"),
    ("CL-2", "It is not supported at width 64; not shown for width 64.", "Supported"),
    ("CL-3", "The next rung passes at three widths.", "Established"),
    ("CL-4", "The 100\\% is isolated at widths 16 and 32.", "Not true in E6"),
    ("CL-4", "Not true at widths 16 and 32 (in E6).", "Not true in E6"),
    ("CL-5", "Established that the rungs above were not run.", "Not established"),
    ("CL-6", "The entries amount to 0.13 to 0.57 bits per parameter.", "lower bound"),
    ("CL-7", "With the decay at 0.0 the rung passes at width 64.", "Supported"),
    ("CL-7", "This is unsupported by the data.", "Supported"),
])
def test_a_missing_status_word_is_a_problem_naming_the_claim_and_the_phrase(claim, text, missing):
    assert f"line 1: {claim} must contain '{missing}'" in problems(document((claim, text)))


@pytest.mark.parametrize("claim, text", [
    ("CL-1", "ESTABLISHED for this budget."),
    ("CL-1", "\\textit{Established} for this budget."),
    ("CL-1", "\\emph{Established}\nfor this budget."),
    ("CL-2", "Supported; not shown\nfor width~64."),
    ("CL-2", "\\textbf{Supported}; Not Shown for width 64."),
    ("CL-4", "Not true in E6."),
    ("CL-4", "Not\ntrue\nin E6."),
    ("CL-6", "Every capacity is a lower-bound estimate."),
    ("CL-6", "These are lower bounds."),
])
def test_status_words_match_whatever_the_case_the_markup_the_line_breaks_and_the_hyphens(claim, text):
    assert [p for p in problems(document((claim, text))) if "must contain" in p] == []


def test_a_paragraph_can_carry_two_claims_and_each_is_checked():
    tex = "% claim: CL-1\n% claim: CL-4\nEstablished for this budget.\n"
    assert problems(tex) == ["line 1: CL-4 must contain 'Not true in E6'"]


# ---- the forbidden phrases -------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("claim, extra, phrase", [
    ("CL-1", "The model can store 453 entries.", "can store"),
    ("CL-1", "At width 64 it has a capacity of 453 entries.", "has a capacity of"),
    ("CL-2", "Training time is established as the limit.", "established"),
    ("CL-2", "This is shown in E6.", "shown"),
    ("CL-3", "At 4000 epochs the model holds its 100\\%.", "holds"),
    ("CL-3", "The width-32 model keeps its 100\\%.", "keeps"),
    ("CL-4", "Wide models cannot hold 100\\%.", "cannot hold"),
    ("CL-7", "The decay is established as the cause.", "established"),
])
def test_a_forbidden_phrase_is_a_problem(claim, extra, phrase):
    assert f"line 1: {claim} must not contain '{phrase}'" in problems(document((claim, GOOD[claim] + " " + extra)))


@pytest.mark.parametrize("claim, extra", [
    ("CL-2", "It is not established that width does not matter."),
    ("CL-7", "It is not established that the decay is the only factor."),
    ("CL-2", "This is not shown for width 64."),
    ("CL-3", "The model hold its place."),
    ("CL-1", "The model can be stored in memory."),
    ("CL-4", "Wide models can hold 100\\% in E7."),
])
def test_the_negated_or_near_miss_forms_are_not_forbidden(claim, extra):
    assert problems(document((claim, GOOD[claim] + " " + extra))) == []


@pytest.mark.parametrize("word", ["prove", "proves", "proved", "proven", "Proof", "impossible", "optimal", "optimally", "guarantee", "guarantees", "guaranteed"])
@pytest.mark.parametrize("claim", ["CL-1", "CL-6"])
def test_the_words_the_ledger_never_allows_are_problems_in_any_claim_paragraph(claim, word):
    expected = {"prove": "prove", "proves": "prove", "proved": "prove", "proven": "prove", "Proof": "prove", "impossible": "impossible", "optimal": "optimal",
                "optimally": "optimal", "guarantee": "guarantee", "guarantees": "guarantee", "guaranteed": "guarantee"}[word]
    assert f"line 1: {claim} must not contain '{expected}'" in problems(document((claim, GOOD[claim] + f" This {word} the point.")))


@pytest.mark.parametrize("word", ["improve", "approve", "improvement", "proofread", "reprove", "suboptimal", "optimality", "guaranty"])
def test_words_that_only_contain_the_forbidden_ones_are_fine(word):
    assert problems(document(("CL-6", GOOD["CL-6"] + f" We {word} it."))) == []


def test_what_is_in_a_comment_or_in_a_command_argument_is_not_prose():
    body = GOOD["CL-6"] + " % this proves it\nSee \\cite{proof2020x} and \\ref{fig:optimal} and \\label{guarantee}.\n"
    assert problems(document(("CL-6", body))) == []


def test_text_outside_the_marked_paragraphs_is_never_checked():
    tex = document(("CL-6", GOOD["CL-6"])) + "\nThis proves, beyond doubt, an optimal and guaranteed impossible thing.\n"
    assert problems(tex) == []


def test_an_unescaped_percent_sign_starts_a_comment_and_an_escaped_one_does_not():
    assert problems(document(("CL-6", GOOD["CL-6"] + " Half of it 50% this proves it."))) == []                 # everything after the % is a comment
    assert problems(document(("CL-6", GOOD["CL-6"] + " Half of it 50\\% this proves it."))) == ["line 1: CL-6 must not contain 'prove'"]


def test_a_numbers_macro_in_a_claim_paragraph_is_not_scanned_for_its_key():
    assert problems(document(("CL-6", GOOD["CL-6"] + " Entries: \\num{e4.cap.w64.b1000}."))) == []


# ---- the whole report ------------------------------------------------------------------------------------------------------------------------
def test_without_final_only_the_marked_claims_are_checked_and_with_final_every_claim_must_appear():
    tex = document(("CL-1", GOOD["CL-1"]), ("CL-6", GOOD["CL-6"]))
    assert problems(tex) == []
    assert problems(tex, final=True) == [f"{claim}: no paragraph is marked '% claim: {claim}'" for claim in ("CL-2", "CL-3", "CL-4", "CL-5", "CL-7")]


def test_a_report_that_marks_all_seven_passes_final():
    assert problems(document(*sorted(GOOD.items())), final=True) == []


def test_final_counts_a_claim_marked_in_a_stacked_or_listed_marker():
    tex = document(("CL-1", GOOD["CL-1"]), ("CL-2", GOOD["CL-2"]), ("CL-3", GOOD["CL-3"]), ("CL-4", GOOD["CL-4"]), ("CL-5", GOOD["CL-5"]),
                   ("CL-6", GOOD["CL-6"])) + "\n% claim: CL-7, CL-6\n" + GOOD["CL-7"] + " It is a lower bound.\n"
    assert problems(tex, final=True) == []


def test_problems_are_listed_in_file_order_with_the_markers_line():
    tex = document(("CL-6", "No status words here."), ("CL-4", GOOD["CL-4"] + " This proves it."))
    assert problems(tex) == ["line 1: CL-6 must contain 'lower bound'", "line 4: CL-4 must not contain 'prove'"]


# ---- numbers ---------------------------------------------------------------------------------------------------------------------------------
def test_an_undefined_number_key_is_a_problem_and_a_defined_one_is_not():
    assert ct.number_problems("Entries: \\num{e4.cap.w64.b1000} and \\num{no.such.key}, again \\num{no.such.key}.", VALUES) == [
        "undefined number key 'no.such.key'", "undefined number key 'no.such.key'"]
    assert ct.number_problems("\\num{e4.cap.w64.b1000}", VALUES) == []


def test_a_key_in_a_comment_is_not_a_use():
    assert ct.number_problems("% \\num{no.such.key}\nText.", VALUES) == []


@pytest.mark.parametrize("text, found", [
    ("It holds 453 entries.", ["453"]),
    ("Width 64 has 110,656 parameters.", ["110,656"]),
    ("That is 32.575 bits per entry.", ["32.575"]),
    ("The rung 905 was not run.", ["905"]),
    ("Layers: 2, entries 20.", []),                  # small whole numbers are structure
    ("Seed 12 and width 64.", []),
    ("A year, 2026, is typed.", []),                 # equals no generated value
    ("Entries: \\num{e4.cap.w64.b1000}.", []),
    ("See Table~\\ref{tab:453}.", []),
])
def test_a_number_typed_in_the_prose_that_equals_a_generated_value_is_a_problem(text, found):
    assert [p.split(" typed")[0].removeprefix("the number ") for p in ct.number_problems(text, VALUES)] == found


def test_the_typed_number_problem_names_the_keys_that_carry_the_value():
    (problem,) = ct.number_problems("Rung 905 was not run.", VALUES)
    assert problem == "the number 905 typed in the text equals a generated value (e7.next.w64): write \\num{e7.next.w64}"
    (problem,) = ct.number_problems("The window is 100 epochs wide.", {"a.x": "100", "b.y": "100"})
    assert problem == "the number 100 typed in the text equals a generated value (a.x, b.y): write \\num{a.x}"


def test_the_notes_list_other_typed_numbers_and_unused_keys_for_the_authors_own_read():
    notes = ct.number_notes("Year 2026 and 17.25 percent; \\num{design.layers}; 453.", VALUES)
    assert notes == ["typed numbers equal to no generated value, to read yourself: 2026, 17.25", "keys never used: 7 of 8"]
    assert ct.number_notes("\\num{design.layers}", VALUES)[-1] == "keys never used: 7 of 8"


def test_the_notes_say_nothing_when_there_is_nothing_to_read():
    values = {"a.x": "453"}
    assert ct.number_notes("Entries: \\num{a.x}.", values) == []


def test_crlf_and_lf_give_the_same_report():
    tex = "% claim: CL-6\nNo words.\n\nRung 905.\n"
    assert ct.check_text(tex, VALUES) == ct.check_text(tex.replace("\n", "\r\n"), VALUES)


# ---- the report object and the command line --------------------------------------------------------------------------------------------------
def test_check_text_collects_number_and_claim_problems_and_notes():
    tex = document(("CL-6", "No words.")) + "\nRung 905 and \\num{x.y}. Year 2026.\n"
    report = ct.check_text(tex, VALUES)
    assert report.problems == ["line 1: CL-6 must contain 'lower bound'", "undefined number key 'x.y'",
                               "the number 905 typed in the text equals a generated value (e7.next.w64): write \\num{e7.next.w64}"]
    assert report.notes[0] == "typed numbers equal to no generated value, to read yourself: 2026"


def write(tmp_path: Path, tex: str) -> Path:
    path = tmp_path / "main.tex"
    path.write_text(tex, encoding="utf-8")
    return path


def test_main_prints_problems_and_notes_and_exits_1(tmp_path, capsys):
    path = write(tmp_path, document(("CL-6", "No words.")) + "\nRung 905.\n")
    assert ct.main([str(path)]) == 1
    out = capsys.readouterr().out
    assert out.splitlines()[0] == f"{path}: 2 problems"
    assert "  line 1: CL-6 must contain 'lower bound'" in out and "  the number 905 typed in the text equals a generated value" in out


def test_main_says_ok_and_exits_0_when_clean(tmp_path, capsys):
    path = write(tmp_path, document(("CL-6", GOOD["CL-6"] + " Entries: \\num{e4.cap.w64.b1000}.")))
    assert ct.main([str(path)]) == 0
    assert capsys.readouterr().out.splitlines()[0] == f"{path}: no problems"


def test_main_final_requires_every_claim(tmp_path, capsys):
    path = write(tmp_path, document(("CL-6", GOOD["CL-6"])))
    assert ct.main([str(path), "--final"]) == 1
    assert "CL-1: no paragraph is marked '% claim: CL-1'" in capsys.readouterr().out


def test_main_exit_2_for_a_missing_file_and_for_numbers_that_cannot_be_built(tmp_path, capsys):
    assert ct.main([str(tmp_path / "none.tex")]) == 2
    assert "cannot read" in capsys.readouterr().err
    path = write(tmp_path, "Text.\n")
    assert ct.main([str(path), "--root", str(tmp_path)]) == 2
    assert "numbers:" in capsys.readouterr().err


def test_main_exit_2_for_a_file_that_is_not_utf8(tmp_path, capsys):
    path = tmp_path / "main.tex"
    path.write_bytes(b"caf\xe9")
    assert ct.main([str(path)]) == 2
    assert "not UTF-8" in capsys.readouterr().err


def test_the_real_numbers_build_defines_every_key_the_tests_use():
    values = mn.build(ROOT)
    assert all(key in values for key in VALUES) and all(values[key] == VALUES[key] for key in ("e4.cap.w64.b1000", "e6.n.w64", "e7.next.w64", "design.hold.window"))


def test_the_module_runs_as_a_command_and_as_a_script_from_another_folder(tmp_path):
    path = write(tmp_path, document(("CL-6", GOOD["CL-6"])))
    done = subprocess.run([sys.executable, "-m", "report.check_text", str(path)], cwd=ROOT, capture_output=True, text=True)
    assert done.returncode == 0 and done.stdout.startswith(f"{path}: no problems"), done.stdout + done.stderr
    script = subprocess.run([sys.executable, str(ROOT / "report" / "check_text.py"), str(path)], cwd=tmp_path, capture_output=True, text=True)
    assert script.returncode == 0 and script.stdout.startswith(f"{path}: no problems"), script.stdout + script.stderr


# ---- found by the planted-bug run of R5: each test below kills a bug that the first 118 tests let through ---------------------------------------------
@pytest.mark.parametrize("claim, text", [
    ("CL-1", "The capacity is unestablished at this budget."),
    ("CL-1", "The capacity is never established at this budget."),
    ("CL-1", "The capacity is not yet established at this budget."),
    ("CL-1", "The capacity has not been established at this budget."),
    ("CL-1", "The capacity has not yet been established at this budget."),
    ("CL-1", "The capacity isn't established at this budget."),
    ("CL-1", "The capacity wasn't established at this budget."),
    ("CL-1", "The capacity is not \\textit{established} at this budget."),
    ("CL-1", "The capacity is not \\cite{k} established at this budget."),
    ("CL-7", "The effect is never supported at width 64."),
    ("CL-7", "The effect isn't supported at width 64."),
])
def test_a_negated_or_misspelled_status_word_is_not_the_status_word(claim, text):
    label = ct.CLAIMS[claim].required[0][0]
    assert problems(document((claim, text))) == [f"line 1: {claim} must contain '{label}'"]


def test_has_been_established_is_the_status_word():
    assert problems(document(("CL-1", "Exact recall has been established at every budget."))) == []


@pytest.mark.parametrize("claim, text, missing", [
    ("CL-2", "Supported; not shown for width 32, whose rung fails.", "not shown for width 64"),
    ("CL-4", "Not true in E7: the 100\\% is isolated.", "Not true in E6"),
    ("CL-4", "True in E6: the 100\\% is isolated.", "Not true in E6"),
    ("CL-6", "A lower value than the largest rung.", "lower bound"),
])
def test_the_required_words_must_match_exactly(claim, text, missing):
    assert problems(document((claim, text))) == [f"line 1: {claim} must contain '{missing}'"]


@pytest.mark.parametrize("text", ["It is not \\textit{established} here.", "It is not \\Emph{established} here.", "It is not \\cite{k} established here.",
                                  "It is not \\emph*{established} here.", "It is not \\num{x.y}established here."])
def test_markup_and_citations_between_not_and_the_word_leave_the_negation_in_place(text):
    assert ct.prose(text) == "it is not established here"


def test_a_negation_split_by_markup_still_guards_a_forbidden_word():
    assert problems(document(("CL-2", "Supported at widths 16 and 32, not \\emph{shown} for width 64."))) == []


# the marker
def test_a_line_that_is_not_a_comment_is_not_a_marker():
    assert problems("Intro.\n\nclaim: CL-4 is what this paragraph says.\nMore.\n") == []


def test_a_marker_may_name_any_number_of_claims_separated_by_commas():
    (paragraph,) = ct.claim_paragraphs("% claim: CL-1, CL-2, CL-3, CL-4\nBody.\n")
    assert paragraph.ids == ("CL-1", "CL-2", "CL-3", "CL-4")


def test_the_claims_of_a_marker_are_separated_by_commas_not_by_spaces():
    assert "line 1: a claim marker must look like '% claim: CL-4'" in problems("% claim: CL-1 CL-2\nBody.\n")


def test_a_two_digit_claim_number_is_unknown_and_not_cut_to_one_digit():
    assert "line 1: unknown claim 'CL-10' (use CL-1 to CL-7)" in problems("% claim: CL-10\nBody.\n")


def test_a_marker_directly_after_a_paragraph_is_a_problem_and_ends_that_paragraph():
    tex = "% claim: CL-1\nEstablished.\n% claim: CL-6\nA lower bound.\n"
    assert problems(tex) == ["line 3: a claim marker must start a paragraph (put a blank line before it)"]
    assert [(p.ids, p.text) for p in ct.claim_paragraphs(tex)] == [(("CL-1",), "Established."), (("CL-6",), "A lower bound.")]


def test_a_marker_that_names_no_valid_claim_makes_no_paragraph():
    assert ct.claim_paragraphs("% claim: banana\nBody.\n") == []


@pytest.mark.parametrize("tex", ["Text.\n\n% claim: CL-4", "% claim: CL-4"])
def test_a_marker_on_the_very_last_line_has_no_paragraph(tex):
    assert any("claim marker with no paragraph after it" in problem for problem in problems(tex))


@pytest.mark.parametrize("newline", ["\r\n", "\r"])
def test_every_kind_of_line_ending_gives_the_same_paragraphs(newline):
    tex = "Intro.\n\n% claim: CL-4\nFirst line\nsecond line.\n\nOther."
    assert ct.claim_paragraphs(tex.replace("\n", newline)) == ct.claim_paragraphs(tex)


@pytest.mark.parametrize("newline", ["\r\n", "\r"])
def test_a_comment_ends_at_its_line_whatever_the_line_ending(newline):
    tex = "% a comment with 453\nRung 905 and \\num{no.such.key}.\n\n" + document(("CL-6", "No words."))
    assert ct.check_text(tex.replace("\n", newline), VALUES) == ct.check_text(tex, VALUES)
    assert len(ct.check_text(tex.replace("\n", newline), VALUES).problems) == 3


# the report
def test_claim_problems_and_check_text_are_not_final_by_default():
    tex = document(("CL-6", GOOD["CL-6"]))
    assert ct.claim_problems(tex) == [] and ct.check_text(tex, VALUES).problems == []


def test_a_marker_problem_later_in_the_file_than_a_claim_problem_is_listed_after_it():
    tex = "% claim: CL-6\nNo words.\n\nText.\n% claim: CL-4\nNot true in E6.\n"
    assert problems(tex) == ["line 1: CL-6 must contain 'lower bound'", "line 5: a claim marker must start a paragraph (put a blank line before it)"]


def test_the_keys_that_carry_a_value_are_listed_in_sorted_order_whatever_order_the_values_came_in():
    (problem,) = ct.number_problems("The window is 100 epochs wide.", {"b.y": "100", "a.x": "100"})
    assert problem == "the number 100 typed in the text equals a generated value (a.x, b.y): write \\num{a.x}"


def test_a_typed_number_with_a_thousands_comma_that_equals_a_value_is_not_an_unknown_number():
    assert ct.number_notes("Width 64 has 110,656 parameters.", VALUES) == ["keys never used: 8 of 8"]


def test_a_number_typed_twice_is_listed_once_in_the_notes():
    assert ct.number_notes("Years 2026 and 2026, then 17.25.", VALUES)[0] == "typed numbers equal to no generated value, to read yourself: 2026, 17.25"


def test_prose_is_exactly_the_words_lower_case_with_single_spaces():
    assert ct.prose("A \\textbf{Strong}\n  claim, 100\\% sure; \\cite{k} and \\num{a.b} % hidden\nend.") == "a strong claim 100 sure and end"


# the command line
def test_main_says_one_problem_not_one_problems(tmp_path, capsys):
    path = write(tmp_path, "Year 2026 and rung 905.\n")
    assert ct.main([str(path)]) == 1
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == f"{path}: 1 problem"


def test_main_prints_the_notes_under_their_heading_after_the_problems(tmp_path, capsys):
    path = write(tmp_path, "Year 2026 and rung 905.\n")
    ct.main([str(path)])
    lines = capsys.readouterr().out.splitlines()
    heading = lines.index("for your own read (not problems):")
    assert lines[heading + 1] == "  typed numbers equal to no generated value, to read yourself: 2026"
    assert lines[heading + 2].startswith("  keys never used: ")
    assert heading == 2 and len(lines) == heading + 3


def test_main_prints_no_heading_when_there_are_no_notes(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(ct.mn, "build", lambda root: {"a.x": "453"})
    path = write(tmp_path, "Entries: \\num{a.x}.\n")
    assert ct.main([str(path)]) == 0
    assert capsys.readouterr().out == f"{path}: no problems\n"


def test_main_reads_a_file_that_starts_with_a_byte_order_mark(tmp_path, capsys):
    path = tmp_path / "main.tex"
    path.write_bytes(b"\xef\xbb\xbf" + document(("CL-6", "No words.")).encode("utf-8"))
    assert ct.main([str(path)]) == 1
    assert "  line 1: CL-6 must contain 'lower bound'" in capsys.readouterr().out


def test_the_help_describes_the_arguments(capsys, monkeypatch):
    monkeypatch.setenv("COLUMNS", "300")
    with pytest.raises(SystemExit) as stop:
        ct.main(["--help"])
    assert stop.value.code == 0
    out = capsys.readouterr().out
    for words in ("the report's main .tex file", "every claim CL-1 to CL-7 must be marked at least once", "the repository the numbers are built from (default: this repository)"):
        assert words in out
