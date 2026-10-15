"""Tournament edge extraction for TOPO-HODGE.

Two real k-wise sources feed the topology card: Nectar rows carry seven responses ranked
1..7 (edges follow the rank order), and UltraFeedback rows carry four completions with a
GPT-4 overall score each (edges follow score comparisons, tied pairs dropped). The loaders
build ``reward_lens.data.schema.Tournament`` objects from those rows using the edge builders
here; this module also flattens a tournament view into the exact matrix
``campaign.topo.resolve_subjects`` reads back out of the store:

- ``edges``: an (E, 5) float array of rows ``[t_idx, i, j, wins_i, wins_j]``
- ``n_items``: one response count per tournament, indexed by ``t_idx``

``matrix_payload`` wraps that in the MatrixPayload the arcs record under
``evidence_keys.OBS_TOURNAMENTS`` with ``roster_key=DATA_KEY``, and ``record_tournaments``
appends both intermediates to a store from the cached slices.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

from reward_lens.data.schema import EdgeObs, Tournament


def edges_from_ranks(
    ranks: Sequence[float], *, judge_id: str | None = None, annotator_id: str | None = None
) -> tuple[EdgeObs, ...]:
    """Pairwise edges from a rank vector: a lower rank beats a higher one, ties dropped."""
    edges: list[EdgeObs] = []
    for i in range(len(ranks)):
        for j in range(i + 1, len(ranks)):
            if ranks[i] == ranks[j]:
                continue
            wins_i = 1 if ranks[i] < ranks[j] else 0
            edges.append(
                EdgeObs(
                    i=i,
                    j=j,
                    wins_i=wins_i,
                    wins_j=1 - wins_i,
                    annotator_id=annotator_id,
                    judge_id=judge_id,
                )
            )
    return tuple(edges)


def edges_from_scores(
    scores: Sequence[float], *, judge_id: str | None = None, annotator_id: str | None = None
) -> tuple[EdgeObs, ...]:
    """Pairwise edges from a score vector: a higher score beats a lower one, ties dropped."""
    edges: list[EdgeObs] = []
    for i in range(len(scores)):
        for j in range(i + 1, len(scores)):
            if scores[i] == scores[j]:
                continue
            wins_i = 1 if scores[i] > scores[j] else 0
            edges.append(
                EdgeObs(
                    i=i,
                    j=j,
                    wins_i=wins_i,
                    wins_j=1 - wins_i,
                    annotator_id=annotator_id,
                    judge_id=judge_id,
                )
            )
    return tuple(edges)


def edge_matrix(tournaments: Iterable[Tournament]) -> tuple[np.ndarray, list[int]]:
    """Flatten tournaments into the (E, 5) matrix and per-tournament item counts.

    Row format is ``[t_idx, i, j, wins_i, wins_j]`` with ``t_idx`` the tournament's position
    in the iteration order, which is the format ``campaign.topo`` rebuilds from. Tournaments
    with no edges keep their ``n_items`` entry so indices stay aligned; the rebuild skips
    them, exactly as it skips them here at decomposition time.
    """
    rows: list[list[float]] = []
    n_items: list[int] = []
    for t_idx, t in enumerate(tournaments):
        n_items.append(len(t.responses))
        for e in t.edges:
            rows.append([float(t_idx), float(e.i), float(e.j), float(e.wins_i), float(e.wins_j)])
    matrix = np.asarray(rows, dtype=np.float64).reshape(len(rows), 5)
    return matrix, n_items


def subjects_entry(tournaments: Iterable[Tournament]) -> dict[str, Any]:
    """The payload-ready dict one dataset key of ``campaign.topo`` consumes."""
    matrix, n_items = edge_matrix(tournaments)
    return {"edges": matrix, "n_items": n_items}


def matrix_payload(slice_name: str, tournaments: Iterable[Tournament]) -> Any:
    """The MatrixPayload an arc records under OBS_TOURNAMENTS for this slice."""
    from campaign.payloads import MatrixPayload

    matrix, n_items = edge_matrix(tournaments)
    return MatrixPayload(
        name=f"tournaments:{slice_name}",
        matrix=matrix,
        col_labels=["t_idx", "i", "j", "wins_i", "wins_j"],
        meta={"n_items": n_items, "slice": slice_name},
    )


def record_tournaments(
    store: Any,
    slice_names: tuple[str, ...] = ("nectar-tournaments", "ultrafeedback-tournaments"),
) -> list[Any]:
    """Record the tournament intermediates for TOPO-HODGE from the cached slices.

    Reads each slice through ``load_slice`` (cache first), builds its MatrixPayload, and
    appends a ``campaign.tournaments`` intermediate under the model-free join key. Returns
    the Evidence objects in slice order. This is the ingest-side half of the seam that
    ``campaign.topo.resolve_subjects`` reads.
    """
    from campaign import evidence_keys as ek
    from campaign.data.loaders import load_slice

    out = []
    for name in slice_names:
        result = load_slice(name)
        ev = ek.record_intermediate(
            observable=ek.OBS_TOURNAMENTS,
            roster_key=ek.DATA_KEY,
            slice_name=name,
            value=matrix_payload(name, result.view),
            dataset_id=str(result.view.checksum()),
        )
        store.append(ev)
        out.append(ev)
    return out


__all__ = [
    "edges_from_ranks",
    "edges_from_scores",
    "edge_matrix",
    "subjects_entry",
    "matrix_payload",
    "record_tournaments",
]
