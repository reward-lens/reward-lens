"""The portable report: one self-contained HTML file that renders an embedded JSON record.

D-15. The record is the truth and the page is a rendering of it, so no statistic is computed in
the browser and, the point this module was rewritten on, **the renderer never edits what it
embeds**. The block carries the sealed record byte for byte: unescape it, canonicalise it, and the
digest is the record's own `assay_id`. A renderer that wrote its own measurements into the record
on the way out produced an embedded record whose digest no longer matched its id, which is an
invalid record sitting inside a page that claims to carry a valid one. So the tier is decided
before the seal, by `plan_embedding`, and the renderer only honours what the record already
declares, or refuses.

A page opened from `file://` has an opaque origin, which takes away `fetch`, external modules,
`import()`, workers and streaming WebAssembly; what is left is one classic script with everything
inlined, on a system font stack, with `print-color-adjust: exact` so a printed copy keeps its
status colours. None of the neighbouring tools ships this, which is why the stranger test is the
differentiator and why it constrains the architecture.

D-16. The HTML is derived and the record never references it. The bundle's manifest digest is
shown in the page so a reader can tie what they are reading to what was signed; it is never
written into the record.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from reward_lens.contracts import canonical_bytes, validate_record
from reward_lens.contracts.errors import RecordInvalid

from . import parquet, tiers
from .embed import escape_attribute, escape_json_text, escape_text
from .tiers import CEILING_B, HARD_A, SOFT_A, Embedding

__all__ = [
    "Embedding",
    "embed",
    "parquet",
    "omission_reason",
    "plan_embedding",
    "render",
    "render_to",
    "tiers",
]

ASSETS = Path(__file__).parent / "assets"

MAX_PASSES = 5
"""The plan iterates to a fixed point; more than this and something is not converging."""


def _as_dict(record: Any) -> dict[str, Any]:
    if isinstance(record, Mapping):
        return json.loads(json.dumps(record))
    to_dict = getattr(record, "to_dict", None)
    if to_dict is None:
        raise TypeError(f"render expects an Assay or a dict, not {type(record).__name__}")
    return to_dict()


def _assets() -> tuple[str, str]:
    js = ASSETS / "report.js"
    css = ASSETS / "report.css"
    if not js.is_file() or not css.is_file():
        raise FileNotFoundError(
            f"the report bundle is missing from {ASSETS}; build it with "
            "`npm run build:report` in the frontend workspace (D-58)"
        )
    return js.read_text(encoding="utf-8"), css.read_text(encoding="utf-8")


# --------------------------------------------------------------------------------------------
# The page
# --------------------------------------------------------------------------------------------


def omission_reason(tier: str, omitted: Any) -> str | None:
    """Why the tables are not in the file, worded once so the plan and the render agree.

    `html_bytes` is the size of the file `render_to` will write, so every byte of the notice has
    to be the same in both. A sentence the plan phrases one way and the renderer another puts the
    recorded size a few bytes off the file's, which is the kind of drift that makes a measurement
    unusable.
    """
    if tier != "C" or not list(omitted):
        return None
    if not parquet.available():
        return parquet.UNAVAILABLE
    return f"Carrying them would put the file past the {CEILING_B:,}-byte ceiling."


def _notice(document: dict[str, Any], plan: Embedding) -> str:
    """What the page says about its own size, written by the renderer rather than by the script.

    A truncated report that does not say it is truncated is the failure mode that matters, so the
    sentence is in the HTML and survives a script that never runs.
    """
    total = len(document.get("tables", []))
    omitted = list(plan.omitted_tables)
    if plan.tier == "A":
        if not plan.past_soft_limit:
            return ""
        head = "This record is past the size a page carries comfortably"
        body = (
            f"Its canonical JSON is {plan.record_bytes:,} bytes, past the {SOFT_A:,}-byte comfort"
            " line. Everything is still on this page."
        )
    elif plan.tier == "B" and not omitted:
        head = "Everything is here, and the tables are not JSON"
        body = (
            f"The record is too large to inline its tables as JSON, so all {total} of them are"
            " carried in this file as Parquet blocks and read in the page itself. Nothing is"
            " missing and nothing is fetched."
        )
    elif plan.tier == "B":
        carried = total - len(omitted)
        rest = (
            f" The other {carried} are carried in this file as Parquet blocks."
            if carried
            else ""
        )
        head = "Not every table is on this page"
        body = (
            f"{len(omitted)} of {total} tables are in the bundle, not here: the record refers to"
            f" them by digest and row count.{rest}"
        )
    else:
        head = "Not everything is on this page"
        why = f" {escape_text(plan.reason)}" if plan.reason else ""
        body = (
            f"{len(omitted)} of {total} tables are in the bundle, not here: the record refers to"
            f" them by digest and row count.{why}"
        )
    notes = dict(plan.omission_notes)
    items = "".join(
        f'<li data-omitted="omitted-table:{escape_attribute(name)}">{escape_text(name)}'
        + (f" ({escape_text(notes[name])})" if name in notes else "")
        + "</li>"
        for name in omitted
    )
    truncated = f' data-truncated="{plan.tier}"' if omitted else ""
    return (
        f'<aside class="truncation" data-tier-notice="{plan.tier}"{truncated} role="note">'
        f"<h2>{head}</h2><p>{body}</p><ul>{items}</ul></aside>"
    )


def _compose(
    document: dict[str, Any],
    plan: Embedding,
    *,
    blocks: dict[str, str],
    bundle_manifest_digest: str | None,
) -> bytes:
    """The whole file, for a record the caller has already decided the tier of."""
    js, css = _assets()
    payload = escape_json_text(json.dumps(document, separators=(",", ":"), ensure_ascii=False))
    subject = document.get("subject", {}).get("reward_system", {})
    title = subject.get("name") or subject.get("id") or "reward system"
    version = document.get("subject", {}).get("version", {}).get("id", "")
    digest_meta = (
        f'<meta name="bundle-manifest-digest" content="'
        f'{escape_attribute(bundle_manifest_digest)}">\n'
        if bundle_manifest_digest
        else ""
    )
    digest_attr = (
        f' data-bundle-manifest-digest="{escape_attribute(bundle_manifest_digest)}"'
        if bundle_manifest_digest
        else ""
    )
    tables = "".join(
        f'<script type="application/octet-stream" id="table:{escape_attribute(ident)}">'
        f"{b64}</script>\n"
        for ident, b64 in blocks.items()
    )
    html = (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        '<meta name="generator" content="reward-lens">\n'
        f"{digest_meta}"
        f"<title>reward-lens assay: {escape_text(title)} {escape_text(version)}</title>\n"
        f"<style>\n{css}\n</style>\n"
        f"</head>\n<body{digest_attr}>\n"
        f"{_notice(document, plan)}\n"
        '<div id="rl-report"></div>\n'
        "<noscript>This report renders the record embedded below; its script has to run once."
        "</noscript>\n"
        f'<script type="application/json" id="assay">{payload}</script>\n'
        f"{tables}"
        f"<script>\n{js}\n</script>\n"
        "</body>\n</html>\n"
    )
    return html.encode("utf-8")


# --------------------------------------------------------------------------------------------
# The plan: run before the seal, so the record declares a tier the file can honour
# --------------------------------------------------------------------------------------------


def _band(document: dict[str, Any], record_bytes: int) -> tuple[str, list[str], str | None]:
    """Which band this record falls in, and the reason if the tables cannot travel with it."""
    if record_bytes <= HARD_A:
        return "A", [], None
    names = [table["id"] for table in document.get("tables", [])]
    if not parquet.available():
        return "C", names, omission_reason("C", names)
    return "B", [], None


def plan_embedding(
    record: Any,
    *,
    bundle_manifest_digest: str | None = None,
    tables: Mapping[str, Any] | None = None,
) -> Embedding:
    """Decide the tier, and measure the two objects the tiers are measured on.

    The record handed in is the one about to be sealed: every field but `embedding` (an `embedding`
    already present is ignored and recomputed, so planning is idempotent). Nothing is written back;
    the caller puts `plan.to_dict()` into the record and then seals it, which is what makes the id
    the digest of a record that already declares its own tier.

    The measurement is a fixed point. `record_bytes` and `html_bytes` are written into the record,
    so writing them changes both, and a decimal digit more or fewer moves each by one byte. The
    loop re-measures with the values it just produced until two passes agree, which is what makes
    the number in the record the number of the file the renderer will write.
    """
    document = _as_dict(record)
    document.pop("embedding", None)
    plan = Embedding(tier="A")
    for attempt in range(1, MAX_PASSES + 1):
        candidate = dict(document, embedding=plan.to_dict())
        record_bytes = len(canonical_bytes(candidate))
        tier, omitted, reason = _band(candidate, record_bytes)
        blocks: dict[str, str] = {}
        notes: dict[str, str] = {}
        if tier == "B":
            blocks, notes = _blocks(candidate, tables)
            omitted = [t["id"] for t in candidate.get("tables", []) if t["id"] in notes]
        measured = Embedding(
            tier=tier,
            omitted_tables=tuple(omitted),
            record_bytes=record_bytes,
            reason=reason,
            omission_notes=tuple(sorted(notes.items())),
            past_soft_limit=tier == "A" and record_bytes > SOFT_A,
        )
        html_bytes = len(
            _compose(
                candidate,
                measured,
                blocks=blocks,
                bundle_manifest_digest=bundle_manifest_digest,
            )
        )
        if tier == "B" and html_bytes > CEILING_B:
            names = [table["id"] for table in candidate.get("tables", [])]
            measured = Embedding(
                tier="C",
                omitted_tables=tuple(names),
                record_bytes=record_bytes,
                reason=omission_reason("C", names),
            )
            html_bytes = len(
                _compose(
                    candidate,
                    measured,
                    blocks={},
                    bundle_manifest_digest=bundle_manifest_digest,
                )
            )
        settled = Embedding(
            tier=measured.tier,
            omitted_tables=measured.omitted_tables,
            record_bytes=record_bytes,
            html_bytes=html_bytes,
            reason=measured.reason,
            omission_notes=measured.omission_notes,
            past_soft_limit=measured.past_soft_limit,
            passes=attempt,
        )
        if settled == plan:
            return settled
        plan = settled
    raise RuntimeError(
        f"the embedding plan did not settle in {MAX_PASSES} passes; the last was {plan}"
    )


# --------------------------------------------------------------------------------------------
# The render: honour the declared tier, or refuse
# --------------------------------------------------------------------------------------------


def _refuse(tier: str, what: str, ceiling: int, measured: int) -> RecordInvalid:
    return RecordInvalid(
        message=(
            f"the record declares embedding tier {tier}, whose limit is {ceiling:,} bytes of"
            f" {what}, and it measures {measured:,} bytes"
        ),
        remediation=(
            "run `plan_embedding(record, bundle_manifest_digest=...)` and seal the record with"
            " its result; the renderer honours the tier the record declares and never re-tiers"
            " a sealed record silently"
        ),
        context={
            "declared_tier": tier,
            "measured_on": what,
            "ceiling_bytes": ceiling,
            "measured_bytes": measured,
        },
    )


def _disagrees(field: str, what: str, declared: int, measured: int) -> RecordInvalid:
    """The record's own number put against the object that number is about.

    A sealed record's sizes are claims, and a claim the renderer does not check is a claim its
    author gets to choose. Reading `record_bytes` off the record and believing it let a record
    declare tier A on a `record_bytes` of 1 and render forty megabytes as an A, under the very
    line the tier exists to hold. So both objects are weighed, and a declared number that does
    not survive the weighing is refused the way a declared tier whose object does not fit is
    refused, with the record's number and the measurement both on the error.
    """
    return RecordInvalid(
        message=(
            f"the record declares {field} of {declared:,} bytes, and {what} measures"
            f" {measured:,} bytes"
        ),
        remediation=(
            "run `plan_embedding(record, bundle_manifest_digest=...)` and seal the record with"
            " its result; the renderer weighs both objects for itself and will not render a"
            " record whose declared size is not the size it has"
        ),
        context={
            "declared_field": field,
            "measured_on": what,
            "declared_bytes": declared,
            "measured_bytes": measured,
        },
    )


def _payload_note(table: dict[str, Any]) -> str:
    """Why a declared table has no payload to embed, in the words the page prints beside it."""
    preview = len(table.get("inline_rows") or [])
    if preview:
        return f"preview only: {preview} of {table.get('rows', 0):,} rows"
    return "payload not supplied"


def _rows_refused(ident: str, declared: int, found: int) -> RecordInvalid:
    return RecordInvalid(
        message=(
            f"the record declares table {ident} with {declared:,} rows, and the payload supplied"
            f" for it holds {found:,}"
        ),
        remediation=(
            "pass the complete payload the record's digest and row count are about; the sample"
            " rows under `tables[].inline_rows` are a preview and are never embedded as the table"
        ),
        context={"table": ident, "declared_rows": declared, "found_rows": found},
    )


def _digest_refused(ident: str, declared: str, found: str) -> RecordInvalid:
    return RecordInvalid(
        message=(
            f"the record declares table {ident} with digest {declared}, and the payload supplied"
            f" for it digests to {found}"
        ),
        remediation=(
            "pass the bytes the record was sealed against; a payload that digests to something"
            " else is a different table however many rows it has"
        ),
        context={"table": ident, "declared_digest": declared, "found_digest": found},
    )


def _claimed_absent(idents: list[str], notes: dict[str, str]) -> RecordInvalid:
    listed = ", ".join(f"{i} ({notes.get(i, 'payload not supplied')})" for i in idents)
    return RecordInvalid(
        message=(
            "the record declares embedding tier B, which carries every table it does not name"
            f" under `omitted_tables` as a Parquet block, and no complete payload was supplied"
            f" for {listed}"
        ),
        remediation=(
            "pass the tables' complete payloads as `render(record, tables=...)`, or re-plan and"
            " seal the record without them, which names them in the record and prints on the page"
            " what is in the bundle rather than in the file"
        ),
        context={"declared_tier": "B", "tables": list(idents)},
    )


def _blocks(
    document: dict[str, Any], tables: Mapping[str, Any] | None
) -> tuple[dict[str, str], dict[str, str]]:
    """A Parquet block for every declared table a complete payload was supplied for, and why not.

    A table is embedded only from its own complete payload, checked against what the record says
    about it: the row count always, and the digest when the payload arrives as the bundle's own
    bytes, which are the only bytes a digest can be taken of. The sample rows under
    `tables[].inline_rows` are a preview and are never a payload. That is the whole point of this
    function: a table declaring twenty-five thousand rows and carrying twelve sample rows used to
    travel as a twelve-row block, under a page whose notice said nothing was missing, and a table
    carrying no preview at all travelled as a block of no rows at all.

    Returns the blocks, and for every table without one the reason the page states beside it.
    """
    source = dict(tables or {})
    blocks: dict[str, str] = {}
    notes: dict[str, str] = {}
    for table in document.get("tables", []):
        ident = table["id"]
        payload = source.get(ident)
        if payload is None:
            notes[ident] = _payload_note(table)
            continue
        declared_digest = table.get("digest")
        if isinstance(payload, (bytes, bytearray, memoryview)):
            found = "sha256:" + hashlib.sha256(bytes(payload)).hexdigest()
            if declared_digest and found != declared_digest:
                raise _digest_refused(ident, declared_digest, found)
        block, rows = parquet.encode_payload(payload)
        declared_rows = table.get("rows", 0)
        if rows != declared_rows:
            raise _rows_refused(ident, declared_rows, rows)
        blocks[ident] = block
    return blocks, notes


def _declared(document: dict[str, Any]) -> Embedding:
    """The plan the record declares, weighed against the record in hand.

    The canonical JSON is measured here every time rather than read off the record, because the
    tier the renderer honours is only worth what the number beside it is worth. A record that
    declares no number has nothing to disagree with, and the measurement stands on its own.
    """
    block = document.get("embedding") or {}
    tier = block.get("tier", "A")
    omitted = tuple(block.get("omitted_tables", ()))
    measured = len(canonical_bytes(document))
    claimed = block.get("record_bytes")
    if claimed and claimed != measured:
        raise _disagrees("record_bytes", "the canonical JSON of the record", claimed, measured)
    return Embedding(
        tier=tier,
        omitted_tables=omitted,
        record_bytes=measured,
        html_bytes=block.get("html_bytes") or 0,
        reason=omission_reason(tier, omitted),
        past_soft_limit=tier == "A" and measured > SOFT_A,
    )


def render(
    record: Any,
    *,
    bundle_manifest_digest: str | None = None,
    tables: Mapping[str, Any] | None = None,
) -> bytes:
    """One self-contained HTML file rendering `record`, exactly as the record declares.

    The record is embedded byte for byte: the block's text, unescaped and canonicalised, digests
    to the record's own `assay_id`. Raises `RecordInvalid` (RL0604) before producing any HTML if
    the record does not validate, or if the file the declared tier asks for would not fit under
    that tier's ceiling.

    The sizes the record declares are weighed rather than trusted. The canonical JSON is measured
    before anything is composed and the file is measured once it is, and either number the record
    states disagreeing with its measurement is the same refusal, because a size nobody checks is
    a size the record can choose.
    """
    document = _as_dict(record)
    validate_record(document)
    plan = _declared(document)
    blocks: dict[str, str] = {}
    if plan.tier == "A" and plan.record_bytes > HARD_A:
        raise _refuse("A", "the canonical JSON of the record", HARD_A, plan.record_bytes)
    if plan.tier == "B":
        if not parquet.available():
            raise RecordInvalid(
                message=(
                    "the record declares embedding tier B, which carries every table as a Parquet"
                    f" block, and {parquet.UNAVAILABLE}"
                ),
                remediation=(
                    "install the `trace` extra so pyarrow is importable, or re-plan the record:"
                    " without a writer the plan chooses tier C and the page says why"
                ),
                context={"declared_tier": "B", "ceiling_bytes": CEILING_B},
            )
        blocks, notes = _blocks(document, tables)
        claimed = [
            table["id"]
            for table in document.get("tables", [])
            if table["id"] not in plan.omitted_tables
        ]
        absent = [ident for ident in claimed if ident not in blocks]
        if absent:
            raise _claimed_absent(absent, notes)
        blocks = {ident: blocks[ident] for ident in claimed}
        plan = replace(plan, omission_notes=tuple(sorted(notes.items())))
    html = _compose(
        document, plan, blocks=blocks, bundle_manifest_digest=bundle_manifest_digest
    )
    if plan.tier == "B" and len(html) > CEILING_B:
        raise _refuse("B", "the whole HTML file", CEILING_B, len(html))
    if plan.html_bytes and plan.html_bytes != len(html):
        raise _disagrees("html_bytes", "the whole HTML file", plan.html_bytes, len(html))
    return html


def render_to(
    record: Any,
    path: str | Path,
    *,
    bundle_manifest_digest: str | None = None,
    tables: Mapping[str, Any] | None = None,
) -> Path:
    """Render and write. Nothing is written if the record is refused."""
    html = render(record, bundle_manifest_digest=bundle_manifest_digest, tables=tables)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(html)
    return out
