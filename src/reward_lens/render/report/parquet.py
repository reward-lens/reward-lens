"""Tier B's per-row tables: Parquet written here, read in the browser by hyparquet (D-15).

A record refers to the bundle's tables by digest and row count and carries at most fifty sample
rows under `tables[].inline_rows`. Those rows are a preview and nothing more. At tier B a table
travels as its own base64 block, written here from the table's **complete payload**, which the
caller supplies; a table with no payload is disclosed as absent rather than approximated, because
a block written from the preview is a table of twelve rows wearing the name of a table of
twenty-five thousand. Uncompressed, because hyparquet reads that with nothing else inlined beside
it; snappy would work too and would cost a decompressor in the bundle for a saving the base64
gives back at four bytes for three. Bytes handed in are read and written again rather than passed
through, so whatever the bundle compressed them with, what lands in the page is what the page can
read.

`pyarrow` lives in the `trace` extra, so it is not always there. Its absence is a fact about the
installation, not an error: the plan chooses tier C and says why, and the page prints the reason
beside the tables it could not carry. Nothing here writes a table into the record, and nothing
here is called when the record declares tier A or C.
"""

from __future__ import annotations

import base64
import io
import json
from typing import Any

UNAVAILABLE = (
    "no Parquet writer is installed: tier B carries each table as a Parquet block, which needs "
    "pyarrow (the `trace` extra)"
)


def _pyarrow() -> Any:
    try:
        import pyarrow  # noqa: PLC0415
        import pyarrow.parquet  # noqa: PLC0415
    except Exception:  # pragma: no cover - exercised by the absence test
        return None
    return pyarrow


def available() -> bool:
    """Whether tier B can be written at all in this installation."""
    return _pyarrow() is not None


def preview_rows(table: dict[str, Any]) -> list[dict[str, Any]]:
    """The sample rows the record itself carries for a table. A preview, never the table (D-16)."""
    return [dict(row) for row in table.get("inline_rows") or []]


def _arrow_table(pa: Any, rows: list[dict[str, Any]]) -> Any:
    """Rows as columns where they infer, and as one JSON column where they do not.

    A row is `{"type": "object"}` in the frozen schema, with nothing said about its fields, so a
    table whose rows disagree about a column's type is a legal record. Falling back to one text
    column keeps such a table readable on the page instead of failing the render; the column name
    says what happened, so no reader mistakes it for the table's own shape.
    """
    try:
        return pa.Table.from_pylist(rows)
    except (pa.ArrowInvalid, pa.ArrowTypeError, pa.ArrowNotImplementedError):
        return pa.table({"row_json": [json.dumps(row, sort_keys=True) for row in rows]})


def encode_payload(payload: Any) -> tuple[str, int]:
    """One table's complete payload as base64 Parquet, and the number of rows it turned out to be.

    `payload` is either the bundle's own Parquet bytes or the rows themselves. The row count comes
    back so the caller can hold it against what the record declares about the table: this function
    writes what it is given and makes no claim that the two agree. Raises if no writer is
    installed, which at tier B the caller has already checked.
    """
    pa = _pyarrow()
    if pa is None:
        raise RuntimeError(UNAVAILABLE)
    import pyarrow.parquet as pq  # noqa: PLC0415

    if isinstance(payload, (bytes, bytearray, memoryview)):
        table = pq.read_table(io.BytesIO(bytes(payload)))
    else:
        table = _arrow_table(pa, [dict(row) for row in payload])
    buffer = io.BytesIO()
    pq.write_table(
        table,
        buffer,
        compression="none",
        version="2.6",
        write_statistics=False,
        store_schema=False,
    )
    return base64.b64encode(buffer.getvalue()).decode("ascii"), table.num_rows
