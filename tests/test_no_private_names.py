"""recordstream names neither helios nor waivefront: its code, tests and docs illustrate with a general taidal project.

recordstream serves every domain, where the signal domain (waivefront) and a private bridge (helios) mean nothing.
Root ``AGENTS.md`` → "General Projects Name Neither Helios Nor Waivefront" (user ruling 2026-10-19).
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Dict, List

PROJECT = Path(__file__).resolve().parents[1]
FORBIDDEN = re.compile("helios|waivefront", re.IGNORECASE)
#: Files that must spell a forbidden word, and why.
ALLOWED: Dict[str, str] = {}


def _project_files() -> List[Path]:
    """Every file git tracks in recordstream, and every new one it does not ignore."""
    listed = subprocess.run(
        ["git", "-C", str(PROJECT), "ls-files", "--cached", "--others", "--exclude-standard"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    return [PROJECT / name for name in listed if (PROJECT / name).is_file()]


def test_no_file_of_recordstream_names_helios_or_waivefront() -> None:
    this_file = Path(__file__).resolve()
    hits = []
    for path in _project_files():
        if path.resolve() == this_file or path.relative_to(PROJECT).as_posix() in ALLOWED:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for number, line in enumerate(text.splitlines(), start=1):
            if FORBIDDEN.search(line):
                hits.append(f"{path.relative_to(PROJECT)}:{number}: {line.strip()}")
    assert not hits, "recordstream names helios or waivefront — use a general taidal project instead:\n" + "\n".join(
        hits
    )
