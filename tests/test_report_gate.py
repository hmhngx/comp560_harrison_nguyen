"""Tests for the report gate wrapper (report/gate.py).

The tests in the first half run against FakeTeX, which stands in for subprocess.run and writes the files a TeX run would write, so they
need no TeX. The integration tests at the end build the lab's unmodified template with the installed TeX; they must FAIL, not skip, when
TeX or pdftotext is missing.
"""
from __future__ import annotations

import hashlib
import io
import ntpath
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from report import gate

ROOT = Path(__file__).resolve().parents[1]

GOOD_TEX = (
    "\\begin{abstract}A.\\end{abstract}\n"
    "\\section{Introduction}x\n\\section{Related Work}y\n\\section{Conclusion}z \\label{endcontent}\n"
    "\\section*{References}\n"
    "\\appendix\\section{Use Of Artificial Intelligence}q\n"
)
GOOD_AUX = "\\newlabel{endcontent}{{3}{2}}\n"
GOOD_TEXT = "page one\fConclusion\nReferences\n[1] Li\f"  # what pdftotext prints: two pages, 'References' on page 2
BIB_AUX = "\\bibdata{refs}\n"


def which_everything(name: str) -> str:
    return f"/fake/bin/{name}"


def which_except(*missing: str):
    return lambda name: None if name in missing else which_everything(name)


class FakeTeX:
    """Stands in for subprocess.run: records every command and writes the files a TeX run would write."""

    def __init__(self, aux=(GOOD_AUX, GOOD_AUX), log="", pdf=True, write_outputs=True, pdflatex_code=0, bibtex_code=0,
                 pdftotext_out=GOOD_TEXT, pdftotext_code=0):
        self.aux_by_pass = list(aux)
        self.log, self.pdf, self.write_outputs = log, pdf, write_outputs
        self.pdflatex_code, self.bibtex_code = pdflatex_code, bibtex_code
        self.pdftotext_out, self.pdftotext_code = pdftotext_out, pdftotext_code
        self.calls: list[tuple[list[str], dict]] = []
        self.pdflatex_runs = 0

    def tools(self) -> list[str]:
        return [Path(command[0]).name for command, _ in self.calls]

    def __call__(self, command, **kwargs):
        self.calls.append((list(command), kwargs))
        tool = Path(command[0]).name
        if tool == "pdflatex":
            out = Path(next(arg for arg in command if arg.startswith("-output-directory=")).split("=", 1)[1])
            index = min(self.pdflatex_runs, len(self.aux_by_pass) - 1)
            self.pdflatex_runs += 1
            if self.write_outputs:
                (out / "main.aux").write_text(self.aux_by_pass[index], encoding="utf-8")
                (out / "main.log").write_text(self.log, encoding="utf-8")
                if self.pdf:
                    (out / "main.pdf").write_bytes(b"%PDF-fake")
            return subprocess.CompletedProcess(command, self.pdflatex_code, "", "")
        if tool == "bibtex":
            return subprocess.CompletedProcess(command, self.bibtex_code, "", "")
        if tool == "pdftotext":
            return subprocess.CompletedProcess(command, self.pdftotext_code, self.pdftotext_out, "pdftotext complaint")
        raise AssertionError(f"unexpected command {command}")


def project(tmp_path: Path, tex: str = GOOD_TEX) -> tuple[Path, Path]:
    (tmp_path / "main.tex").write_text(tex, encoding="utf-8")
    return tmp_path / "main.tex", tmp_path / "build"


# ---- find_tool ------------------------------------------------------------------------------------------------------

def test_find_tool_returns_the_path_that_which_found():
    assert gate.find_tool("pdflatex", which=lambda name: "/x/" + name) == "/x/pdflatex"


def test_find_tool_names_the_missing_tool_and_says_how_to_get_it():
    with pytest.raises(gate.GateError) as caught:
        gate.find_tool("pdflatex", which=lambda name: None)
    assert str(caught.value) == (
        "pdflatex not found on PATH: install a TeX distribution (MiKTeX or TeX Live), then open a new terminal so that PATH is refreshed"
    )


# ---- bib_paths ------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize(
    "tex,names",
    [
        ("\\bibliography{refs}", ["refs.bib"]),
        ("\\bibliography{a, b}", ["a.bib", "b.bib"]),
        ("\\bibliography{refs.bib}", ["refs.bib"]),
        ("% \\bibliography{old}\n\\bibliography{new}", ["new.bib"]),
        ("no bibliography here", []),
    ],
)
def test_bib_paths_reads_the_bibliography_command(tex, names, tmp_path):
    assert gate.bib_paths(tex, tmp_path) == [tmp_path / name for name in names]


# ---- the path bibtex is given ----------------------------------------------------------------------------------------

def test_relative_gives_the_relative_path_when_there_is_one(tmp_path):
    assert gate._relative(tmp_path / "build", tmp_path) == Path("build")


def test_relative_gives_the_path_itself_when_it_is_on_another_drive():
    assert gate._relative(Path("D:/out"), Path("C:/src"), relpath=ntpath.relpath) == Path("D:/out")


# ---- pdf_pages ------------------------------------------------------------------------------------------------------

def test_pdf_pages_splits_on_form_feeds_and_drops_the_empty_piece_after_the_last_page():
    fake = FakeTeX(pdftotext_out="page one\nline\fpage two\f")
    assert gate.pdf_pages(Path("x.pdf"), run=fake, which=which_everything) == ["page one\nline", "page two"]


def test_pdf_pages_keeps_blank_pages_in_the_middle_and_at_the_end():
    fake = FakeTeX(pdftotext_out="a\f\fc\f\f")
    assert gate.pdf_pages(Path("x.pdf"), run=fake, which=which_everything) == ["a", "", "c", ""]


def test_pdf_pages_keeps_the_last_page_when_the_tool_leaves_out_the_final_form_feed():
    fake = FakeTeX(pdftotext_out="page one\fpage two")
    assert gate.pdf_pages(Path("x.pdf"), run=fake, which=which_everything) == ["page one", "page two"]


def test_pdf_pages_asks_pdftotext_for_utf8_text_on_stdout():
    fake = FakeTeX()
    gate.pdf_pages(Path("x.pdf"), run=fake, which=which_everything)
    assert fake.calls[0][0] == ["/fake/bin/pdftotext", "-enc", "UTF-8", "x.pdf", "-"]


def test_pdf_pages_failure_is_a_gate_error_with_the_tools_message():
    fake = FakeTeX(pdftotext_code=1)
    with pytest.raises(gate.GateError, match="pdftotext failed.*pdftotext complaint"):
        gate.pdf_pages(Path("x.pdf"), run=fake, which=which_everything)


def test_pdf_pages_missing_tool_is_a_gate_error():
    with pytest.raises(gate.GateError, match="pdftotext"):
        gate.pdf_pages(Path("x.pdf"), run=FakeTeX(), which=which_except("pdftotext"))


# ---- build_pdf: how many passes, in which order -----------------------------------------------------------------------

def test_build_stops_after_two_passes_when_the_aux_file_does_not_change(tmp_path):
    tex, out = project(tmp_path)
    fake = FakeTeX(aux=["same", "same"])
    build = gate.build_pdf(tex, out, run=fake, which=which_everything)
    assert fake.tools() == ["pdflatex", "pdflatex"]
    assert (build.passes, build.notes) == (2, [])
    assert build.pdf == out / "main.pdf" and build.aux_text == "same"


def test_build_repeats_pdflatex_until_the_aux_file_stops_changing(tmp_path):
    tex, out = project(tmp_path)
    fake = FakeTeX(aux=["a", "b", "b"])
    build = gate.build_pdf(tex, out, run=fake, which=which_everything)
    assert fake.tools() == ["pdflatex"] * 3
    assert (build.passes, build.notes) == (3, [])


def test_build_gives_up_after_the_maximum_number_of_passes_and_says_so(tmp_path):
    tex, out = project(tmp_path)
    fake = FakeTeX(aux=[str(number) for number in range(20)])
    build = gate.build_pdf(tex, out, run=fake, which=which_everything)
    assert fake.tools() == ["pdflatex"] * gate.MAX_PASSES
    assert build.passes == gate.MAX_PASSES
    assert len(build.notes) == 1 and "still changing after 5 passes" in build.notes[0]


def test_build_runs_bibtex_once_after_the_first_pass_when_the_document_has_a_bibliography(tmp_path):
    tex, out = project(tmp_path)
    fake = FakeTeX(aux=[BIB_AUX, BIB_AUX + "\\bibcite{a}{1}\n", BIB_AUX + "\\bibcite{a}{1}\n"])
    build = gate.build_pdf(tex, out, run=fake, which=which_everything)
    assert fake.tools() == ["pdflatex", "bibtex", "pdflatex", "pdflatex"]
    assert (build.passes, build.notes) == (3, [])


def test_build_does_not_run_bibtex_when_there_is_no_bibdata(tmp_path):
    tex, out = project(tmp_path)
    fake = FakeTeX(aux=["\\citation{a}\n", "\\citation{a}\n"])
    gate.build_pdf(tex, out, run=fake, which=which_everything)
    assert "bibtex" not in fake.tools()


# ---- build_pdf: the command lines ----------------------------------------------------------------------------------

def test_build_calls_pdflatex_in_the_sources_folder_and_writes_into_the_output_folder(tmp_path):
    tex, out = project(tmp_path)
    fake = FakeTeX()
    gate.build_pdf(tex, out, run=fake, which=which_everything)
    command, kwargs = fake.calls[0]
    assert command == ["/fake/bin/pdflatex", "-interaction=nonstopmode", f"-output-directory={out}", "main.tex"]
    assert Path(kwargs["cwd"]) == tmp_path
    assert kwargs["timeout"] == gate.TIMEOUT_SECONDS and kwargs["capture_output"] is True and kwargs["check"] is False
    assert (kwargs["encoding"], kwargs["errors"]) == ("utf-8", "replace")


def test_build_calls_bibtex_in_the_sources_folder_on_the_jobname_inside_the_output_folder(tmp_path):
    tex, out = project(tmp_path)
    fake = FakeTeX(aux=[BIB_AUX, "x", "x"])
    gate.build_pdf(tex, out, run=fake, which=which_everything)
    command, kwargs = fake.calls[1]
    assert command == ["/fake/bin/bibtex", str(Path("build") / "main")]
    assert Path(kwargs["cwd"]) == tmp_path


def test_build_creates_the_output_folder(tmp_path):
    tex, out = project(tmp_path)
    assert not out.exists()
    gate.build_pdf(tex, out, run=FakeTeX(), which=which_everything)
    assert out.is_dir()


def test_build_creates_a_missing_parent_of_the_output_folder_too(tmp_path):
    tex, _ = project(tmp_path)
    out = tmp_path / "a" / "b" / "build"
    gate.build_pdf(tex, out, run=FakeTeX(), which=which_everything)
    assert out.is_dir()


# ---- build_pdf: failures --------------------------------------------------------------------------------------------

def test_build_removes_old_outputs_first_so_a_failed_build_cannot_pass_on_stale_files(tmp_path):
    tex, out = project(tmp_path)
    out.mkdir()
    for suffix in (".aux", ".log", ".pdf", ".out", ".bbl", ".blg", ".toc", ".lof", ".lot"):
        (out / f"main{suffix}").write_text("stale", encoding="utf-8")
    (out / "other.aux").write_text("not ours", encoding="utf-8")
    fake = FakeTeX(write_outputs=False, pdflatex_code=1)
    build = gate.build_pdf(tex, out, run=fake, which=which_everything)
    assert build.pdf is None and build.aux_text == "" and build.log_text == ""
    assert sorted(path.name for path in out.iterdir()) == ["other.aux"]


def test_build_without_a_pdf_stops_after_the_first_pass_and_says_so(tmp_path):
    tex, out = project(tmp_path)
    fake = FakeTeX(pdf=False, pdflatex_code=7)
    build = gate.build_pdf(tex, out, run=fake, which=which_everything)
    assert fake.tools() == ["pdflatex"]
    assert build.pdf is None and build.passes == 1
    assert build.notes == ["pdflatex produced no PDF (exit status 7)"]


def test_build_notes_a_failing_bibtex_run(tmp_path):
    tex, out = project(tmp_path)
    fake = FakeTeX(aux=[BIB_AUX, "x", "x"], bibtex_code=2)
    build = gate.build_pdf(tex, out, run=fake, which=which_everything)
    assert build.notes == ["bibtex failed (exit status 2)"]
    assert build.pdf is not None


def test_build_does_not_treat_bibtex_warnings_as_a_failure(tmp_path):
    tex, out = project(tmp_path)
    build = gate.build_pdf(tex, out, run=FakeTeX(aux=[BIB_AUX, "x", "x"], bibtex_code=1), which=which_everything)
    assert build.notes == []


def test_build_reads_the_log_back_as_text(tmp_path):
    tex, out = project(tmp_path)
    build = gate.build_pdf(tex, out, run=FakeTeX(log="Overfull \\hbox\n! Oops.\n"), which=which_everything)
    assert build.log_text == "Overfull \\hbox\n! Oops.\n"


def test_build_with_pdflatex_missing_is_a_gate_error(tmp_path):
    tex, out = project(tmp_path)
    with pytest.raises(gate.GateError, match="pdflatex not found"):
        gate.build_pdf(tex, out, run=FakeTeX(), which=which_except("pdflatex"))


def test_tools_are_run_without_a_terminal_to_wait_on(tmp_path):
    tex, out = project(tmp_path)
    fake = FakeTeX()
    gate.build_pdf(tex, out, run=fake, which=which_everything)
    gate.pdf_pages(out / "main.pdf", run=fake, which=which_everything)
    assert {tool for tool in fake.tools()} == {"pdflatex", "pdftotext"}
    assert all(kwargs["stdin"] is subprocess.DEVNULL for _, kwargs in fake.calls)


def test_an_old_output_that_cannot_be_removed_is_a_gate_error(tmp_path):
    tex, out = project(tmp_path)
    (out / "main.pdf").mkdir(parents=True)  # a folder where the old PDF should be: removing it fails, as it does while a viewer holds the file open
    with pytest.raises(gate.GateError, match=r"cannot remove .*main\.pdf"):
        gate.build_pdf(tex, out, run=FakeTeX(), which=which_everything)


def test_a_tool_that_does_not_finish_is_a_gate_error(tmp_path):
    tex, out = project(tmp_path)

    def hangs(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    with pytest.raises(gate.GateError) as caught:
        gate.build_pdf(tex, out, run=hangs, which=which_everything)
    assert str(caught.value) == (
        "pdflatex did not finish within 300 seconds "
        "(MiKTeX may be waiting to install a missing package: build once by hand, let it install, then run the gate again)"
    )


def test_a_timeout_of_another_tool_does_not_blame_the_miktex_install_prompt():
    def hangs(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    with pytest.raises(gate.GateError) as caught:
        gate.pdf_pages(Path("x.pdf"), run=hangs, which=which_everything)
    assert str(caught.value) == "pdftotext did not finish within 300 seconds"


def test_an_output_folder_that_cannot_be_created_is_a_gate_error(tmp_path, capsys):
    tex, out = project(tmp_path)
    out.write_text("a file where the folder should be", encoding="utf-8")
    with pytest.raises(gate.GateError, match=r"cannot create .*build"):
        gate.build_pdf(tex, out, run=FakeTeX(), which=which_everything)
    assert gate.main([str(tex)], run=FakeTeX(), which=which_everything) == 2
    assert "gate could not run: cannot create" in capsys.readouterr().err


def test_a_tool_that_cannot_start_is_a_gate_error(tmp_path):
    tex, out = project(tmp_path)

    def broken(command, **kwargs):
        raise OSError("exec format error")

    with pytest.raises(gate.GateError, match=r"could not start pdflatex: exec format error"):
        gate.build_pdf(tex, out, run=broken, which=which_everything)


# ---- check_build: the pieces put together -------------------------------------------------------------------------

def test_check_build_clean_document_has_no_problems_and_reports_the_measurements(tmp_path):
    tex, out = project(tmp_path)
    result = gate.check_build(tex, out, run=FakeTeX(), which=which_everything)
    assert result.problems == []
    assert (result.label_page, result.references_page, result.pages, result.passes) == (2, 2, 2, 2)


def test_check_build_reports_a_defect_found_in_the_log(tmp_path):
    tex, out = project(tmp_path)
    log = "LaTeX Warning: Reference `fig:x' on page 1 undefined on input line 3.\n"
    result = gate.check_build(tex, out, run=FakeTeX(log=log), which=which_everything)
    assert result.problems == ["undefined reference: fig:x"]


def test_check_build_reads_the_bibliography_files_named_in_the_document(tmp_path):
    tex, out = project(tmp_path, GOOD_TEX + "\\cite{used}\\bibliography{refs}\n")
    (tmp_path / "refs.bib").write_text("@article{used,\n}\n@article{unused,\n}\n", encoding="utf-8")
    fake = FakeTeX(aux=[BIB_AUX + GOOD_AUX, BIB_AUX + GOOD_AUX + "x", BIB_AUX + GOOD_AUX + "x"])
    result = gate.check_build(tex, out, run=fake, which=which_everything)
    assert result.problems == ["bibliography entry never cited: unused"]


def test_files_that_are_not_valid_utf8_are_read_with_a_replacement_character(tmp_path):
    odd = tmp_path / "odd.txt"
    odd.write_bytes(b"caf\xe9")
    assert gate._read(odd) == "caf" + chr(0xFFFD)
    assert gate._read(tmp_path / "missing.txt") == ""


def test_check_build_reads_a_source_and_a_bibliography_that_are_not_valid_utf8(tmp_path):
    tex, out = project(tmp_path, GOOD_TEX + "\\cite{used}\\bibliography{refs}\n")
    tex.write_bytes(tex.read_bytes() + b"% caf\xe9\n")
    (tmp_path / "refs.bib").write_bytes(b"@article{used,\n title = {caf\xe9}\n}\n@article{unused,\n}\n")
    fake = FakeTeX(aux=[BIB_AUX + GOOD_AUX, BIB_AUX + GOOD_AUX + "x", BIB_AUX + GOOD_AUX + "x"])
    result = gate.check_build(tex, out, run=fake, which=which_everything)
    assert result.problems == ["bibliography entry never cited: unused"]


def test_check_build_joins_bibliography_files_with_a_newline(tmp_path):
    tex, out = project(tmp_path, GOOD_TEX + "\\cite{a}\\bibliography{one,two}\n")
    (tmp_path / "one.bib").write_bytes(b"@article{a,\n}")  # no newline at the end of the file
    (tmp_path / "two.bib").write_bytes(b"@article{b,\n}\n")
    fake = FakeTeX(aux=[BIB_AUX + GOOD_AUX, BIB_AUX + GOOD_AUX + "x", BIB_AUX + GOOD_AUX + "x"])
    assert gate.check_build(tex, out, run=fake, which=which_everything).problems == ["bibliography entry never cited: b"]


def test_check_build_names_a_bibliography_file_that_is_missing(tmp_path):
    tex, out = project(tmp_path, GOOD_TEX + "\\cite{a}\\bibliography{refs}\n")
    result = gate.check_build(tex, out, run=FakeTeX(aux=[BIB_AUX + GOOD_AUX] * 3), which=which_everything)
    assert "bibliography file not found: refs.bib" in result.problems


def test_check_build_includes_what_the_build_noticed(tmp_path):
    tex, out = project(tmp_path)
    result = gate.check_build(tex, out, run=FakeTeX(aux=[str(n) for n in range(9)]), which=which_everything)
    assert any("still changing" in problem for problem in result.problems)


def test_check_build_without_a_pdf_lists_the_latex_errors_and_does_not_read_pages(tmp_path):
    tex, out = project(tmp_path)
    fake = FakeTeX(pdf=False, pdflatex_code=1, log="! Undefined control sequence.\n")
    result = gate.check_build(tex, out, run=fake, which=which_everything)
    assert result.problems == ["pdflatex produced no PDF (exit status 1)", "LaTeX error: Undefined control sequence."]
    assert (result.pages, result.label_page, result.references_page) == (None, None, None)
    assert fake.tools() == ["pdflatex"]


def test_check_build_lists_a_missing_end_label_and_still_reports_the_other_measurement(tmp_path):
    tex, out = project(tmp_path)
    result = gate.check_build(tex, out, run=FakeTeX(aux=["", ""]), which=which_everything)
    assert len(result.problems) == 1 and result.problems[0].startswith("page budget not measured: label 'endcontent'")
    assert (result.label_page, result.references_page) == (None, 2)


def test_check_build_unreadable_source_is_a_gate_error(tmp_path):
    with pytest.raises(gate.GateError, match="cannot read"):
        gate.check_build(tmp_path / "missing.tex", tmp_path / "build", run=FakeTeX(), which=which_everything)


# ---- main: exit status and what is printed ---------------------------------------------------------------------------

def test_main_passes_with_exit_status_0_and_prints_the_measurements(tmp_path, capsys):
    tex, _ = project(tmp_path)
    assert gate.main([str(tex)], run=FakeTeX(), which=which_everything) == 0
    out = capsys.readouterr().out
    assert "PASS" in out and "end-of-content label: page 2 (limit 9)" in out and "'References' heading: page 2" in out
    assert "pdflatex passes: 2" in out and "pages: 2" in out


def test_main_says_not_found_for_a_measurement_it_could_not_make(tmp_path, capsys):
    tex, _ = project(tmp_path)
    assert gate.main([str(tex)], run=FakeTeX(aux=["", ""]), which=which_everything) == 1
    out = capsys.readouterr().out
    assert "end-of-content label: not found (limit 9)" in out and "'References' heading: page 2" in out


def test_main_prints_the_pass_count_and_the_page_count_separately(tmp_path, capsys):
    tex, _ = project(tmp_path)
    assert gate.main([str(tex)], run=FakeTeX(aux=[GOOD_AUX, GOOD_AUX + "\n", GOOD_AUX + "\n"]), which=which_everything) == 0
    assert f"{tex}: pdflatex passes: 3; pages: 2\n" in capsys.readouterr().out


def test_main_prints_to_the_streams_it_is_given(tmp_path):
    tex, _ = project(tmp_path)
    out, err = io.StringIO(), io.StringIO()
    assert gate.main([str(tex)], run=FakeTeX(), which=which_everything, stdout=out, stderr=err) == 0
    assert out.getvalue().splitlines()[-1] == "PASS: every gate passed" and err.getvalue() == ""
    out, err = io.StringIO(), io.StringIO()
    assert gate.main([str(tex)], run=FakeTeX(), which=which_except("pdflatex"), stdout=out, stderr=err) == 2
    assert out.getvalue() == "" and err.getvalue().startswith("gate could not run: pdflatex not found")


def test_main_guards_the_console_it_prints_to_by_default(tmp_path, monkeypatch):
    tex, _ = project(tmp_path)
    stream = io.TextIOWrapper(io.BytesIO(), encoding="ascii")
    monkeypatch.setattr(sys, "stdout", stream)
    assert gate.main([str(tex)], run=FakeTeX(log="! Caf" + chr(0xE9) + "\n"), which=which_everything) == 1
    stream.flush()
    assert "LaTeX error: Caf?" in stream.buffer.getvalue().decode("ascii")


def test_main_without_a_pdf_prints_no_measurements(tmp_path, capsys):
    tex, _ = project(tmp_path)
    assert gate.main([str(tex)], run=FakeTeX(pdf=False, pdflatex_code=1, log="! Oops.\n"), which=which_everything) == 1
    out = capsys.readouterr().out
    assert out.splitlines()[0] == f"{tex}: pdflatex passes: 1" and "end-of-content" not in out


def test_main_says_not_found_for_a_missing_references_heading(tmp_path, capsys):
    tex, _ = project(tmp_path)
    assert gate.main([str(tex)], run=FakeTeX(pdftotext_out="one\ftwo\f"), which=which_everything) == 1
    assert "  end-of-content label: page 2 (limit 9); 'References' heading: not found\n" in capsys.readouterr().out


def test_main_fails_with_exit_status_1_and_lists_every_problem(tmp_path, capsys):
    tex, _ = project(tmp_path)
    log = "! Oops.\nLaTeX Warning: Reference `x' on page 1 undefined on input line 3.\n"
    assert gate.main([str(tex)], run=FakeTeX(log=log), which=which_everything) == 1
    out = capsys.readouterr().out
    assert "FAIL: 2 problems" in out and "  - LaTeX error: Oops." in out and "  - undefined reference: x" in out
    assert "PASS" not in out


def test_main_says_problem_in_the_singular_for_one_problem(tmp_path, capsys):
    tex, _ = project(tmp_path)
    log = "! Oops.\n"
    assert gate.main([str(tex)], run=FakeTeX(log=log), which=which_everything) == 1
    assert "FAIL: 1 problem\n" in capsys.readouterr().out


def test_main_with_a_missing_tool_exits_with_status_2_and_explains_on_stderr(tmp_path, capsys):
    tex, _ = project(tmp_path)
    assert gate.main([str(tex)], run=FakeTeX(), which=which_except("pdflatex")) == 2
    captured = capsys.readouterr()
    assert "gate could not run" in captured.err and "pdflatex not found" in captured.err and captured.out == ""


def test_main_with_an_unreadable_source_exits_with_status_2(tmp_path, capsys):
    assert gate.main([str(tmp_path / "missing.tex")], run=FakeTeX(), which=which_everything) == 2
    assert "cannot read" in capsys.readouterr().err


def test_main_does_not_crash_on_characters_the_console_cannot_show(tmp_path):
    tex, _ = project(tmp_path)
    stream = io.TextIOWrapper(io.BytesIO(), encoding="ascii")
    assert gate.main([str(tex)], run=FakeTeX(log="! Caf" + chr(0xE9) + " " + chr(0x2192) + " error\n"), which=which_everything, stdout=stream) == 1
    stream.flush()
    assert "LaTeX error: Caf? ? error" in stream.buffer.getvalue().decode("ascii")


def test_main_builds_into_a_build_folder_next_to_the_source_by_default(tmp_path):
    tex, out = project(tmp_path)
    fake = FakeTeX()
    gate.main([str(tex)], run=fake, which=which_everything)
    assert f"-output-directory={out}" in fake.calls[0][0]


def test_main_builds_into_the_folder_given_with_dash_dash_out(tmp_path):
    tex, _ = project(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    fake = FakeTeX()
    gate.main([str(tex), "--out", str(elsewhere)], run=fake, which=which_everything)
    assert f"-output-directory={elsewhere}" in fake.calls[0][0]
    assert (elsewhere / "main.pdf").exists() and not (tmp_path / "build").exists()


def test_the_module_runs_with_dash_m_and_shows_its_usage():
    done = subprocess.run([sys.executable, "-m", "report.gate", "--help"], cwd=ROOT, capture_output=True, text=True)
    assert done.returncode == 0 and done.stdout.startswith("usage: python -m report.gate [-h] [--out OUT] tex")


def test_the_module_run_without_arguments_exits_with_status_2():
    done = subprocess.run([sys.executable, "-m", "report.gate"], cwd=ROOT, capture_output=True, text=True)
    assert done.returncode == 2 and "required" in done.stderr


# ---- the lab's template, and real builds of it -------------------------------------------------------------------------
# These need pdflatex, bibtex and pdftotext on PATH. There is no skip: a machine without TeX must fail them (see report/README.md).

TEMPLATE_DIR = ROOT / "report" / "template"
TEMPLATE_SHA256 = {
    "dnu-report.tex": "905463f2bd68acb39ca95c3b84040261bd12e6b9b7d5a97e0b2900cfddd2eb68",
    "dnu.sty": "0386fd83577bf60067a0e0f0ad210ad077667086ffc0ea8879146060ad148ffe",
}
LAST_PARAGRAPH = "Optionally, acknowledge contributions, funding, or other support as appropriate."
END_LABEL = "\\label{endcontent}"
BIB = (
    "@article{alpha2020,\n  author = {Test, Ann},\n  title = {A Test Entry One},\n  journal = {Journal of Tests},\n  year = {2020}\n}\n"
    "@article{beta2021,\n  author = {Test, Bo},\n  title = {A Test Entry Two},\n  journal = {Journal of Tests},\n  year = {2021}\n}\n"
)


@pytest.mark.parametrize("name", sorted(TEMPLATE_SHA256))
def test_the_template_files_are_the_lab_files_unmodified(name):
    # line endings are normalised first: git for Windows may check the files out with CRLF
    data = (TEMPLATE_DIR / name).read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(data).hexdigest() == TEMPLATE_SHA256[name]


def template_build(tmp_path: Path, edit=lambda tex: tex) -> tuple[Path, Path]:
    """Copy the template to tmp_path/src as main.tex beside dnu.sty, with the end label added and edit(tex) applied; return (main.tex, build folder)."""
    source = tmp_path / "src"
    source.mkdir()
    shutil.copy(TEMPLATE_DIR / "dnu.sty", source / "dnu.sty")
    tex = (TEMPLATE_DIR / "dnu-report.tex").read_text(encoding="utf-8")
    assert tex.count(LAST_PARAGRAPH) == 1
    (source / "main.tex").write_text(edit(tex.replace(LAST_PARAGRAPH, LAST_PARAGRAPH + END_LABEL)), encoding="utf-8")
    return source / "main.tex", source / "build"


@pytest.mark.integration
def test_the_template_with_an_end_label_passes_every_gate_on_a_real_build(tmp_path):
    tex, out = template_build(tmp_path)
    result = gate.check_build(tex, out)
    assert result.problems == []
    pages = gate.pdf_pages(out / "main.pdf")
    # the page the .aux file gives for the label is the page that holds the text the label follows
    holder = next(number for number, text in enumerate(pages, start=1) if "support as appropriate" in " ".join(text.split()))
    assert result.label_page == holder
    assert result.references_page - result.label_page in (0, 1)
    assert (result.pages, result.passes) == (len(pages), 2)


@pytest.mark.integration
def test_a_defective_document_is_flagged_from_the_real_log(tmp_path):
    defects = "See \\ref{nowhere}, \\citep{ghost}, \\notamacro and \\label{dup}\\label{dup}.\n\n"
    tex, out = template_build(tmp_path, lambda text: text.replace("\\section{Conclusion}", defects + "\\section{Conclusion}"))
    assert gate.check_build(tex, out).problems == [
        "LaTeX error: Undefined control sequence.",
        "undefined reference: nowhere",
        "undefined citation: ghost",
        "label defined twice: dup",
    ]


@pytest.mark.integration
def test_a_bibliography_is_built_and_every_entry_is_checked_on_a_real_build(tmp_path):
    def with_bibliography(text: str) -> str:
        placeholder = text[text.index("\\section*{References}"): text.index("\\appendix")]
        text = text.replace(placeholder, "\\bibliographystyle{plainnat}\n\\bibliography{refs}\n\n")
        return text.replace("\\section{Related Work}", "\\section{Related Work}\nBy \\citet{alpha2020}.\n")

    tex, out = template_build(tmp_path, with_bibliography)
    (tex.parent / "refs.bib").write_text(BIB, encoding="utf-8")
    result = gate.check_build(tex, out)
    assert result.problems == ["bibliography entry never cited: beta2021"]
    assert result.passes == 3
    # the reference list was typeset (plainnat lowercases the title), so bibtex ran and the second pass read its .bbl file
    assert "a test entry one" in " ".join(gate.pdf_pages(out / "main.pdf")).lower()
