"""The documentation's internal links resolve.

A dead docs link is the kind of rot nobody notices until a reader hits one, and
it is created by exactly the ordinary edits this project does constantly: renaming
a heading, splitting a page, moving a section between guides. Nothing else checks
it — CI runs the test suite and the examples, and a broken `[text](other.md#x)`
breaks neither.

Three rules, one per failure mode:

* a relative link names a file that exists,
* an `#anchor` names a heading that exists in that file,
* every docs page is reachable from the README index.

This project's README links to docs RELATIVELY, which is correct for a project not
published to PyPI — the absolute-URL rule exists because a PyPI landing page cannot
resolve a relative link. Those relative links are still covered by the two rules
above, since the README is in the scanned set.

Fenced code blocks are excluded from the scan — see :func:`_strip_code`.

The CI workflow is generated and must not be hand-edited, so these live as tests
— which is also where they belong: they run locally on the same command.
"""

import re
from pathlib import Path
from typing import List, Set

import pytest

_REPO = Path(__file__).resolve().parent.parent
_DOCS = _REPO / "docs"
_README = _REPO / "README.md"

#: `[text](target)` where the target is a relative path — absolute URLs are excluded
#: by requiring the target not to contain `://`.
_RELATIVE_LINK = re.compile(r"\[[^\]]*\]\((?!\w+://)([^)\s]+)\)")


def _slug(heading: str) -> str:
    """GitHub's anchor slug for a heading: lowercase, punctuation dropped, spaces hyphenated.

    Punctuation removal is what makes `## Registering a class you don't own` reachable
    as `#registering-a-class-you-dont-own` and a backticked `## \\`flow()\\` finishes...`
    reachable without the backticks or parens.
    """
    return re.sub(r"[^\w\s-]", "", heading.lower()).replace(" ", "-")


def _headings(path: Path) -> Set[str]:
    return {_slug(line.lstrip("#").strip()) for line in path.read_text().splitlines() if line.startswith("#")}


def _markdown_files() -> List[Path]:
    return sorted(_DOCS.glob("*.md")) + [_README]


def _strip_code(text: str) -> str:
    """Blank out fenced code blocks before scanning for links.

    Python subscript-then-call — `PartialClass[Metric](SomeClass)` — is
    indistinguishable from a markdown link to a regex, so a code sample containing
    one gets reported as a link to a file named `SomeClass`. Found exactly that way:
    the first run of this check against another project's docs produced a false
    positive, which in a link checker is worse than a miss, because it trains the
    reader to ignore it.
    """
    out, fenced = [], False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            fenced = not fenced
            out.append("")
            continue
        out.append("" if fenced else line)
    return "\n".join(out)


def _links(path: Path) -> List[str]:
    return _RELATIVE_LINK.findall(_strip_code(path.read_text()))


@pytest.mark.parametrize("path", _markdown_files(), ids=lambda p: p.name)
def test_relative_links_name_a_file_that_exists(path: Path) -> None:
    """A `[text](other.md)` must point at a real file.

    Catches a page that was renamed or deleted while something still pointed at it.
    """
    missing = []
    for target in _links(path):
        if target.startswith("#"):
            continue  # same-page anchor, checked below
        file_part = target.split("#", 1)[0]
        if not (path.parent / file_part).resolve().exists():
            missing.append(target)

    assert not missing, f"{path.name} links to files that do not exist: {missing}"


@pytest.mark.parametrize("path", _markdown_files(), ids=lambda p: p.name)
def test_anchors_name_a_heading_that_exists(path: Path) -> None:
    """A `#anchor` must match a heading in the file it points at.

    This is the half that rots silently: renaming a heading leaves every link to it
    pointing at the top of the page instead of failing, so the reader lands
    somewhere plausible and never reports it.
    """
    dead = []
    for target in _links(path):
        if "#" not in target:
            continue
        file_part, _, anchor = target.partition("#")
        if not anchor:
            continue
        target_file = path if not file_part else (path.parent / file_part).resolve()
        if not target_file.exists():
            continue  # reported by the test above; don't double-fail
        if anchor not in _headings(target_file):
            dead.append(target)

    assert not dead, f"{path.name} links to headings that do not exist: {dead}"


def test_every_docs_page_is_listed_in_the_readme_index() -> None:
    """The README is the docs index; a page missing from it is a page nobody finds.

    `architecture.md` is exempt: it is the rationale record for maintainers, and the
    README index is the user-facing table.
    """
    exempt = {"architecture.md"}
    listed = set(re.findall(r"docs/([a-z0-9_-]+\.md)", _README.read_text()))
    present = {p.name for p in _DOCS.glob("*.md")} - exempt

    assert not (present - listed), f"docs pages missing from the README index: {sorted(present - listed)}"


def test_the_slug_rule_matches_githubs() -> None:
    """Guard the slug helper itself — a wrong rule makes every test above vacuous.

    If `_slug` stopped stripping punctuation, every anchor would fail to match and
    the suite would report dead links everywhere; if it over-stripped, nothing would
    match and everything would pass. Either way the failure is in the helper the
    other tests trust, so it needs assertions that do not depend on them.

    Deliberately literal rather than read from this repo's headings: the same file
    is used across projects, and a fixture that names a particular page would make
    it non-portable for no gain.

    The three cases are the punctuation classes that actually occur in headings —
    an apostrophe, code backticks with parens, and a comma plus an em dash (which
    leaves a double hyphen, matching GitHub).
    """
    assert _slug("Registering a class you don't own") == "registering-a-class-you-dont-own"
    assert _slug("`flow()` finishes the object") == "flow-finishes-the-object"
    assert _slug("Bare, addressed, glob — the scoping model") == "bare-addressed-glob--the-scoping-model"
