"""Every text file whose exact bytes are hashed or locked must be pinned to LF.

Git for Windows defaults to core.autocrlf=true, which rewrites LF to CRLF on checkout and changes the
SHA-256 of any file not pinned by .gitattributes. A fresh Windows clone then fails the lock test even
though the committed blob is intact.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = ("phonebook/preregistration/*.json", "phonebook/specs/*.json", "phonebook/inputs/*.txt")


@pytest.mark.skipif(shutil.which("git") is None or not (ROOT / ".git").exists(), reason="needs a git checkout")
def test_hash_locked_text_files_are_pinned_to_lf() -> None:
    files = sorted(path.relative_to(ROOT).as_posix() for pattern in PATTERNS for path in ROOT.glob(pattern))
    assert files, "no locked files found; the patterns are stale"
    out = subprocess.run(["git", "check-attr", "eol", "--", *files], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout
    unpinned = [line for line in out.splitlines() if not line.endswith(": eol: lf")]
    assert not unpinned, "not pinned to LF in .gitattributes:\n" + "\n".join(unpinned[:10])
