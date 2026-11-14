r"""Command-line wrapper for the report gate: build the report with TeX, then check the result with report.build_report.

    python -m report.gate report/main.tex [--out report/build]

It runs pdflatex (and bibtex once, when the document has a bibliography) and repeats pdflatex until the .aux file stops changing. It then
extracts the text of every page with pdftotext and gives the source, the .bib files, the .aux and .log files and the page texts to
build_report.check_report. The checks read the main .tex file only: files it pulls in with \input or \include are not followed.

Exit status: 0 when every gate passed; 1 when the gate ran and found problems (a build that produced no PDF counts as a problem); 2 when
the gate could not run (a tool is missing, a tool did not finish, or the source cannot be read). A missing TeX is never skipped.

Needs pdflatex, bibtex and pdftotext on PATH; MiKTeX ships all three.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from report import build_report as br

MAX_PASSES = 5
TIMEOUT_SECONDS = 300
STALE_SUFFIXES = (".aux", ".log", ".pdf", ".out", ".bbl", ".blg", ".toc", ".lof", ".lot")
INSTALL_HINT = "install a TeX distribution (MiKTeX or TeX Live), then open a new terminal so that PATH is refreshed"

Run = Callable[..., subprocess.CompletedProcess]
Which = Callable[[str], "str | None"]  # quoted: this line runs at import, and "X | None" needs Python 3.10 unless it is a string
_BIBLIOGRAPHY = re.compile(r"\\bibliography\{([^}]*)\}")


class GateError(RuntimeError):
    """The gate could not run, which is not the same thing as the report having problems."""


def find_tool(name: str, which: Which = shutil.which) -> str:
    path = which(name)
    if path is None:
        raise GateError(f"{name} not found on PATH: {INSTALL_HINT}")
    return path


def _run(command: list[str], cwd: Path | None, run: Run) -> subprocess.CompletedProcess:
    tool = Path(command[0]).stem
    try:
        return run(command, cwd=cwd, stdin=subprocess.DEVNULL, capture_output=True, encoding="utf-8", errors="replace",
                   timeout=TIMEOUT_SECONDS, check=False)
    except subprocess.TimeoutExpired as error:
        hint = " (MiKTeX may be waiting to install a missing package: build once by hand, let it install, then run the gate again)"
        raise GateError(f"{tool} did not finish within {TIMEOUT_SECONDS} seconds{hint if tool == 'pdflatex' else ''}") from error
    except OSError as error:
        raise GateError(f"could not start {tool}: {error}") from error


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""


def _relative(path: Path, start: Path, relpath: Callable[[Path, Path], str] = os.path.relpath) -> Path:
    try:
        return Path(relpath(path, start))
    except ValueError:  # on Windows, a path on another drive has no relative form
        return path


def bib_paths(tex: str, directory: Path) -> list[Path]:
    r"""The .bib files named by \bibliography{...} (comments ignored), looked for next to the main file."""
    return [directory / (name if name.endswith(".bib") else name + ".bib") for name in br.keys_in(_BIBLIOGRAPHY, tex)]


def pdf_pages(pdf: Path, run: Run = subprocess.run, which: Which = shutil.which) -> list[str]:
    """The text of every page of a PDF, from pdftotext, which ends each page with a form feed."""
    done = _run([find_tool("pdftotext", which), "-enc", "UTF-8", str(pdf), "-"], None, run)
    if done.returncode != 0:
        raise GateError(f"pdftotext failed (exit status {done.returncode}): {done.stderr.strip()}")
    pages = done.stdout.split("\f")
    if pages[-1] == "":
        pages.pop()
    return pages


@dataclass
class Build:
    pdf: Path | None
    aux_text: str
    log_text: str
    passes: int
    notes: list[str] = field(default_factory=list)


def build_pdf(tex: Path, out: Path, run: Run = subprocess.run, which: Which = shutil.which) -> Build:
    """pdflatex, then bibtex once when the document has a bibliography, then pdflatex again until the .aux file stops changing."""
    pdflatex = find_tool("pdflatex", which)
    tex_dir, out, job = tex.absolute().parent, out.absolute(), tex.stem
    try:
        out.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise GateError(f"cannot create the build folder {out}: {error}") from error
    for suffix in STALE_SUFFIXES:
        old = out / f"{job}{suffix}"
        try:
            old.unlink(missing_ok=True)
        except OSError as error:
            raise GateError(f"cannot remove {old}: {error} (is another program, such as a PDF viewer, using it?)") from error
    pdf, aux = out / f"{job}.pdf", out / f"{job}.aux"
    command = [pdflatex, "-interaction=nonstopmode", f"-output-directory={out}", tex.name]
    notes: list[str] = []
    previous: str | None = None
    bibtex_done = False
    passes = 0
    while passes < MAX_PASSES:
        done = _run(command, tex_dir, run)
        passes += 1
        if not pdf.is_file():
            notes.append(f"pdflatex produced no PDF (exit status {done.returncode})")
            break
        current = _read(aux)
        if "\\bibdata" in current and not bibtex_done:
            bibtex = _run([find_tool("bibtex", which), str(_relative(out, tex_dir) / job)], tex_dir, run)
            bibtex_done = True
            if bibtex.returncode >= 2:  # 0 is clean and 1 is warnings only
                notes.append(f"bibtex failed (exit status {bibtex.returncode})")
            continue
        if current == previous:
            break
        previous = current
    else:
        notes.append(f"the .aux file was still changing after {MAX_PASSES} passes; page numbers may be unreliable")
    return Build(pdf if pdf.is_file() else None, _read(aux), _read(out / f"{job}.log"), passes, notes)


@dataclass
class Result:
    problems: list[str]
    passes: int
    label_page: int | None = None
    references_page: int | None = None
    pages: int | None = None


def _measure(function: Callable[[object], int], argument: object) -> int | None:
    try:
        return function(argument)
    except ValueError:
        return None


def _read_bibliography(tex: str, directory: Path) -> tuple[str, list[str]]:
    texts: list[str] = []
    problems: list[str] = []
    for path in bib_paths(tex, directory):
        if path.is_file():
            texts.append(path.read_text(encoding="utf-8", errors="replace"))
        else:
            problems.append(f"bibliography file not found: {path.name}")
    return "\n".join(texts), problems


def check_build(tex_path: Path, out: Path, run: Run = subprocess.run, which: Which = shutil.which) -> Result:
    try:
        tex = tex_path.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        raise GateError(f"cannot read {tex_path}: {error}") from error
    bib_text, problems = _read_bibliography(tex, tex_path.absolute().parent)
    build = build_pdf(tex_path, out, run, which)
    problems += build.notes
    if build.pdf is None:
        problems += [f"LaTeX error: {error}" for error in br.parse_log(build.log_text)["errors"]]
        return Result(problems, build.passes)
    pages = pdf_pages(build.pdf, run, which)
    problems += br.check_report(tex, bib_text, build.aux_text, build.log_text, pages)
    return Result(
        problems,
        build.passes,
        label_page=_measure(br.content_last_page, build.aux_text),
        references_page=_measure(br.first_references_page, pages),
        pages=len(pages),
    )


def main(argv: Sequence[str] | None = None, run: Run = subprocess.run, which: Which = shutil.which, stdout=None, stderr=None) -> int:
    stdout = sys.stdout if stdout is None else stdout
    stderr = sys.stderr if stderr is None else stderr
    for stream in (stdout, stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")  # a console that cannot show a character must not turn into a crash
    parser = argparse.ArgumentParser(prog="python -m report.gate", description="Build the report with TeX and run the report gate checks.")
    parser.add_argument("tex", type=Path, help="the report's main .tex file")
    parser.add_argument("--out", type=Path, help="folder for the build files (default: build/ next to the .tex file, which git ignores)")
    args = parser.parse_args(argv)
    out = args.out if args.out is not None else args.tex.parent / "build"
    try:
        result = check_build(args.tex, out, run, which)
    except GateError as error:
        print(f"gate could not run: {error}", file=stderr)
        return 2
    pages = "" if result.pages is None else f"; pages: {result.pages}"
    print(f"{args.tex}: pdflatex passes: {result.passes}{pages}", file=stdout)
    if result.pages is not None:
        label = "not found" if result.label_page is None else f"page {result.label_page}"
        heading = "not found" if result.references_page is None else f"page {result.references_page}"
        print(f"  end-of-content label: {label} (limit {br.MAX_CONTENT_PAGES}); 'References' heading: {heading}", file=stdout)
    if result.problems:
        count = len(result.problems)
        print(f"FAIL: {count} problem{'' if count == 1 else 's'}", file=stdout)
        for problem in result.problems:
            print(f"  - {problem}", file=stdout)
        return 1
    print("PASS: every gate passed", file=stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
