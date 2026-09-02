"""Shared fixtures for the one-file report.

The reference record is the fixture the schema ships; everything else is built from it so that a
test that inflates a record never loses the shape the renderer has to read.
"""

from __future__ import annotations

import copy
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "schema" / "assay" / "1.0" / "fixtures" / "reference.json"

JSON_BLOCK = re.compile(
    r'<script type="application/json" id="assay">(.*?)</script>', re.DOTALL
)

CHROME_CANDIDATES = (
    "google-chrome-stable",
    "google-chrome",
    "chromium",
    "chromium-browser",
    "chrome",
)


@pytest.fixture(scope="session")
def reference() -> dict:
    return json.loads(REFERENCE.read_text(encoding="utf-8"))


@pytest.fixture()
def record(reference: dict) -> dict:
    return copy.deepcopy(reference)


def embedded_record(html: bytes) -> dict:
    """The record as a browser would read it: the block's text, parsed as JSON."""
    match = JSON_BLOCK.search(html.decode("utf-8"))
    assert match is not None, "no application/json block carrying the record"
    return json.loads(match.group(1))


def inflate(record: dict, *, entries: int, filler: int = 900) -> dict:
    """Copy one entry `entries` times with distinct ids so the record grows past a tier line."""
    template = record["measurement"]["soundness"][0]
    grown = record["measurement"]["soundness"]
    for i in range(entries):
        entry = copy.deepcopy(template)
        entry["entry_id"] = f"soundness.filler_{i:06d}"
        entry["limitations"] = ["x" * filler]
        grown.append(entry)
    return record


def declare_tables(record: dict, ids: list[str], rows: int = 25000, inline: int = 12) -> dict:
    """Declare tables the bundle holds, each carrying `inline` of its rows as D-16's samples."""
    digest = "sha256:" + "1" * 64
    record["tables"] = [
        {
            "id": t,
            "digest": digest,
            "rows": rows,
            "schema": f"schemas/{t}.json",
            **({"inline_rows": sample_rows(t, inline)} if inline else {}),
        }
        for t in ids
    ]
    return record


def chrome() -> str:
    for name in CHROME_CANDIDATES:
        found = shutil.which(name)
        if found:
            return found
    raise AssertionError(
        "no Chrome or Chromium on PATH; the file:// first-screen check needs one "
        f"of {CHROME_CANDIDATES}"
    )


def dump_dom(html_path: Path, profile: Path) -> str:
    """Load the file from file:// in a browser that cannot resolve a name, and dump the DOM.

    `MAP * ~NOTFOUND` sends every hostname to nothing, so a page that needed the network renders
    without whatever it wanted. Together with the file:// origin this is the stranger test.
    """
    profile.mkdir(parents=True, exist_ok=True)
    argv = [
        chrome(),
        "--headless=new",
        "--no-sandbox",
        "--disable-gpu",
        "--no-proxy-server",
        "--host-resolver-rules=MAP * ~NOTFOUND",
        "--disable-background-networking",
        "--disable-component-update",
        "--disable-sync",
        f"--user-data-dir={profile}",
        "--virtual-time-budget=8000",
        "--dump-dom",
        html_path.resolve().as_uri(),
    ]
    done = subprocess.run(argv, capture_output=True, text=True, timeout=180)
    assert done.returncode == 0, f"chrome exited {done.returncode}: {done.stderr[-2000:]}"
    return done.stdout


class Node:
    """The smallest tree a test needs: tag, attributes, children, and the text under it."""

    __slots__ = ("tag", "attrs", "children", "parent", "_text")

    def __init__(self, tag: str, attrs: dict[str, str], parent: "Node | None" = None) -> None:
        self.tag = tag
        self.attrs = attrs
        self.children: list[Node] = []
        self.parent = parent
        self._text: list[str] = []

    @property
    def text(self) -> str:
        out = "".join(self._text)
        for child in self.children:
            out += child.text
        return " ".join(out.split())

    def find_all(self, *, tag: str | None = None, attr: str | None = None, cls: str | None = None):
        hits = []
        for child in self.children:
            ok = (tag is None or child.tag == tag) and (attr is None or attr in child.attrs)
            if ok and cls is not None:
                ok = cls in child.attrs.get("class", "").split()
            if ok:
                hits.append(child)
            hits.extend(child.find_all(tag=tag, attr=attr, cls=cls))
        return hits


VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "wbr"}


def parse(dom: str) -> Node:
    from html.parser import HTMLParser

    root = Node("#root", {})

    class Builder(HTMLParser):
        def __init__(self) -> None:
            super().__init__(convert_charrefs=True)
            self.cursor = root

        def handle_starttag(self, tag, attrs):
            node = Node(tag, {k: (v or "") for k, v in attrs}, self.cursor)
            self.cursor.children.append(node)
            if tag not in VOID:
                self.cursor = node

        def handle_startendtag(self, tag, attrs):
            self.cursor.children.append(Node(tag, {k: (v or "") for k, v in attrs}, self.cursor))

        def handle_endtag(self, tag):
            node = self.cursor
            while node is not root and node.tag != tag:
                node = node.parent
            if node is not root:
                self.cursor = node.parent

        def handle_data(self, data):
            if self.cursor.tag not in ("script", "style"):
                self.cursor._text.append(data)

    builder = Builder()
    builder.feed(dom)
    return root


def sample_rows(table_id: str, n: int = 12) -> list[dict]:
    """Rows of the shape a table carries in the record: `tables[].inline_rows`, D-16's samples."""
    return [
        {"row": i, "table": table_id, "score": round(0.1 * i, 4), "note": f"{table_id}-{i:04d}"}
        for i in range(n)
    ]


def seal(record: dict) -> dict:
    """Seal through the store's own seal: `assay_id` is the digest of what the record now holds.

    `reward_lens.store.project` is imported for its `digest` and its `Assay`, so this is the same
    two lines `Reuse.apply` ends on (project.py, `sealed["assay_id"] = digest(sealed)`) rather than
    a digest computed some other way. A test that sealed its own records another way would prove
    nothing about the records a reader gets.
    """
    from reward_lens.store.project import Assay, digest

    sealed = Assay.model_validate(record).to_dict()
    sealed["assay_id"] = digest(sealed)
    return sealed


def plan_and_seal(
    record: dict,
    *,
    bundle_manifest_digest: str | None = None,
    tables: dict | None = None,
) -> dict:
    """The pipeline P-AUDIT-1 runs: plan the embedding, write it in, then seal (D-80, unit 6).

    `tables` is the table source the render will be given. It belongs here rather than only at
    the render because what the plan can embed is what decides the tier and what the record then
    declares under `omitted_tables`: a record planned without the payloads and rendered with them
    would declare tables absent that the file could carry.
    """
    from reward_lens.contracts.models import Assay
    from reward_lens.render.report import plan_embedding

    normalised = Assay.model_validate(record).to_dict()
    plan = plan_embedding(
        normalised, bundle_manifest_digest=bundle_manifest_digest, tables=tables
    )
    normalised["embedding"] = plan.to_dict()
    return seal(normalised)


def tune_to(reference: dict, slack: int) -> dict:
    """A record whose tier-B file lands `slack` bytes under the ceiling, or past it if negative.

    One filler character is one byte of canonical JSON and one byte of the file, so a single
    measured pass fixes the length: measure, then add the difference. The line is found by
    `plan_embedding`, never by moving a constant.
    """
    import copy as _copy

    from reward_lens.render.report import plan_embedding, tiers

    record = _copy.deepcopy(reference)
    declare_tables(record, ["samples", "groups"])
    inflate(record, entries=120, filler=200_000)
    tuning = _copy.deepcopy(record["measurement"]["soundness"][0])
    tuning["entry_id"] = "soundness.tuning"
    tuning["limitations"] = ["y"]
    record["measurement"]["soundness"].append(tuning)
    probe = plan_embedding(record)
    assert probe.tier == "B", "the bulk record has to be a tier B before it is tuned"
    tuning["limitations"] = ["y" * (1 + tiers.CEILING_B - probe.html_bytes - slack)]
    return record


@pytest.fixture(scope="session")
def under_the_ceiling(reference: dict) -> dict:
    return plan_and_seal(tune_to(reference, slack=64))


@pytest.fixture(scope="session")
def over_the_ceiling(reference: dict) -> dict:
    """The tuned record whose tier-B file is past the ceiling, so the plan sends it to tier C.

    Session scope because building it costs a 26 MB compose on every pass of the plan, and both
    the tier module and the browser module read the same record off it.
    """
    return plan_and_seal(tune_to(reference, slack=-256))


def block_rows(html: bytes, ident: str) -> int:
    """How many rows the page's Parquet block for `ident` actually holds, read back with pyarrow."""
    import base64
    import io

    import pyarrow.parquet as pq

    pattern = re.compile(
        r'<script type="application/octet-stream" id="table:'
        + re.escape(ident)
        + r'">(.*?)</script>',
        re.DOTALL,
    )
    match = pattern.search(html.decode("utf-8"))
    assert match is not None, f"no Parquet block for table {ident}"
    return pq.read_table(io.BytesIO(base64.b64decode(match.group(1)))).num_rows


def has_block(html: bytes, ident: str) -> bool:
    return f'<script type="application/octet-stream" id="table:{ident}">' in html.decode("utf-8")
