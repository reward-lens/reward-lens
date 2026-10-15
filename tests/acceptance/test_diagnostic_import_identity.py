"""The legacy diagnostic import must resolve to the one canonical dataset builder."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import pytest


def _fresh_import(order: tuple[str, str]) -> dict[str, object]:
    """Import the two public paths in a fresh isolated process and return observed identity."""
    body = f"""
        import importlib
        import json

        from reward_lens.data.registry import _LOADERS, get_card, load_dataset
        from reward_lens.reference.load import load, load_diagnostic_v3_material

        first = importlib.import_module({order[0]!r})
        second = importlib.import_module({order[1]!r})
        canonical = importlib.import_module("reward_lens.data.builtin.diagnostic_v3")
        legacy = importlib.import_module("reward_lens.reference.materials.diagnostic_v3")

        canonical_views = canonical.load_diagnostic_v3()
        legacy_views = legacy.load_diagnostic_v3()
        assert tuple(canonical_views) == tuple(legacy_views)
        assert canonical.all_pairs().checksum() == legacy.all_pairs().checksum()
        canonical_matched = canonical.matched_prompt_views()
        legacy_matched = legacy.matched_prompt_views()
        assert tuple(canonical_matched) == tuple(legacy_matched)
        assert all(
            canonical_matched[name].checksum() == legacy_matched[name].checksum()
            and canonical_matched[name].effective_n() == legacy_matched[name].effective_n()
            for name in canonical_matched
        )
        assert all(
            canonical_views[name].checksum() == legacy_views[name].checksum()
            and canonical_views[name].effective_n() == legacy_views[name].effective_n()
            for name in canonical_views
        )

        loaded = load_dataset(get_card("diagnostic_v3"))
        adopted = load_diagnostic_v3_material(load("diagnostic-v3"))
        assert adopted.content_checksum == canonical.all_pairs().checksum()
        assert adopted.pair_count == len(canonical.all_pairs())
        assert adopted.dimensions == tuple(canonical_views)
        print(json.dumps({{
            "first": first.__name__,
            "second": second.__name__,
            "canonical_origin": canonical.__file__,
            "legacy_origin": legacy.__file__,
            "loader_module": _LOADERS["diagnostic_v3"].__module__,
            "loader_count": sum(key in _LOADERS for key in ("diagnostic_v3", "diagnostic_v3_matched")),
            "all_pairs_checksum": canonical.all_pairs().checksum(),
            "loaded_checksum": loaded.checksum(),
            "adopted_checksum": adopted.content_checksum,
            "adopted_pair_count": adopted.pair_count,
            "dimensions": {{name: [len(view), view.effective_n(), view.checksum()] for name, view in canonical_views.items()}},
        }}, sort_keys=True))
    """
    result = subprocess.run(
        [sys.executable, "-I", "-c", textwrap.dedent(body)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return json.loads(result.stdout)


@pytest.mark.parametrize(
    "order",
    [
        ("reward_lens.data.builtin.diagnostic_v3", "reward_lens.reference.materials.diagnostic_v3"),
        ("reward_lens.reference.materials.diagnostic_v3", "reward_lens.data.builtin.diagnostic_v3"),
    ],
)
def test_legacy_and_canonical_import_orders_share_one_registered_builder(
    order: tuple[str, str],
) -> None:
    """Both public paths preserve the exact data without attempting duplicate registration."""
    result = _fresh_import(order)

    assert result["loader_module"] == "reward_lens.data.builtin.diagnostic_v3"
    assert result["loader_count"] == 2
    assert result["all_pairs_checksum"] == result["loaded_checksum"]
    assert result["all_pairs_checksum"] == result["adopted_checksum"]
    assert result["adopted_pair_count"] > 0
    assert result["dimensions"]
