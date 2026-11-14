# report/

The build-and-check gate for the DNU-style report. Nothing here is part of the experiments.

| Path | What it is |
|---|---|
| `build_report.py` | Pure-Python helpers that check one requirement each: the nine-content-page limit, the required sections, a caption and a reference for every figure and table, every bibliography entry cited, and the problems LaTeX reports in its log. They need no TeX. |
| `gate.py` | The command-line wrapper: builds a report with pdflatex and bibtex, reads the text of every page with pdftotext, runs the helpers, and sets the exit status. |
| `template/` | The lab's report template, unmodified. The integration tests build it. |

Tests: `tests/test_report_build.py` (the helpers) and `tests/test_report_gate.py` (the wrapper, and real builds of the template).

## Run the gate

```bash
python -m report.gate report/main.tex
```

It builds into `report/build/` (git ignores `build/`), runs pdflatex until the `.aux` file stops changing (bibtex once, when the document has a bibliography), and prints where the content ends and where the References heading is. Pass `--out DIR` to build somewhere else.

| Exit status | Meaning |
|---|---|
| 0 | every gate passed |
| 1 | the gate ran and found problems; each is printed (a build that produced no PDF counts) |
| 2 | the gate could not run: a tool is missing or did not finish, or a file or folder cannot be read, created or removed |

The page limit is measured two ways that must agree: the page that `\label{endcontent}` lands on, read from the `.aux` file, and the first page whose text has a line that is exactly `References`, read with pdftotext. Put `\label{endcontent}` as the last token of the last content paragraph (the Acknowledgments paragraph if there is one). The References heading may be on the same page as the label or on the next one; if it is on the next page, nothing may sit above it there, because text above it means content continues after the label and the label is too early. The checks read the main `.tex` file only; files it pulls in with `\input` or `\include` are not followed.

## Needs TeX

pdflatex, bibtex and pdftotext must be on PATH. MiKTeX provides all three (`winget install MiKTeX.MiKTeX`, then open a new terminal so that PATH is refreshed). The three real-build tests in `tests/test_report_gate.py` are marked `integration` and **fail**, they do not skip, when TeX is missing, so a green full suite means the report really builds here. `python -m pytest -m "not integration"` leaves them out.

A report that uses a package MiKTeX has not installed yet can stop at MiKTeX's install prompt, which the gate cannot answer; the gate gives up after 300 seconds and says so. Build once by hand in `report/` (`pdflatex main.tex`), let MiKTeX install the package, then run the gate again.

## Where `template/` came from

`dnu.sty` and `dnu-report.tex` are the lab's files, unmodified, downloaded on 2026-11-13 from `dnulab/dnulab-web` at commit `c7c702dcec5ec91c6b64867f36b224a44ac86c8e` (`public/internal/report-template/`). `tests/test_report_gate.py` checks their SHA-256 (line endings normalised):

| File | SHA-256 |
|---|---|
| `dnu-report.tex` | `905463f2bd68acb39ca95c3b84040261bd12e6b9b7d5a97e0b2900cfddd2eb68` |
| `dnu.sty` | `0386fd83577bf60067a0e0f0ad210ad077667086ffc0ea8879146060ad148ffe` |
