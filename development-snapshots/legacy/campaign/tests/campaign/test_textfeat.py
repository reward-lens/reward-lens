"""Surface feature tests on fixed strings.

The features feed the susceptibility banks, so what matters is that each named column moves
in the right direction on text that manipulates exactly that attribute, and that the whole
map is deterministic. All strings here are fixed; there is no randomness to seed.
"""

from __future__ import annotations

import numpy as np

from campaign.textfeat import FEATURE_NAMES, surface_features

PLAIN = "Tides are caused mainly by the gravitational pull of the Moon."
MARKDOWN = "## Answer\n\n- **Tides** are caused by the `Moon`.\n- See [tides](https://x.y).\n\n| a | b |"
HEDGED = (
    "I think tides are perhaps caused by the Moon, though I am not sure; it seems likely, "
    "and it could be the Sun plays some role too, possibly."
)
POSITIVE = "This is a great, clear, accurate and helpful answer with excellent, reliable advice."
NEGATIVE = "This is a bad, wrong, misleading answer full of errors, problems and harmful advice."
SYCOPHANTIC = (
    "What a brilliant question, you're right to ask! I completely agree, your insight is "
    "amazing and honestly this shows impressive, incredible thinking. Great point."
)
STUFFED = (
    "Tides depend on gravity. Keywords: gravity, gravity, gravity, gravity, tides, tides, "
    "tides, tides, moon, moon, moon, moon."
)


def _col(name: str) -> int:
    return FEATURE_NAMES.index(name)


def _row(text: str) -> np.ndarray:
    _, m = surface_features([text])
    return m[0]


def test_shape_and_names():
    names, m = surface_features([PLAIN, MARKDOWN, HEDGED])
    assert names == list(FEATURE_NAMES)
    assert m.shape == (3, len(FEATURE_NAMES))
    assert m.dtype == np.float64
    assert np.all(np.isfinite(m))


def test_deterministic():
    _, a = surface_features([PLAIN, MARKDOWN, HEDGED, SYCOPHANTIC])
    _, b = surface_features([PLAIN, MARKDOWN, HEDGED, SYCOPHANTIC])
    np.testing.assert_array_equal(a, b)


def test_length_orders_by_text_length():
    short, long = _row(PLAIN), _row(PLAIN + " " + HEDGED)
    assert long[_col("len_chars")] > short[_col("len_chars")]
    assert long[_col("len_tokens")] > short[_col("len_tokens")]


def test_markdown_density_separates_formatting():
    assert _row(MARKDOWN)[_col("markdown_density")] > _row(PLAIN)[_col("markdown_density")]
    assert _row(PLAIN)[_col("markdown_density")] == 0.0


def test_hedging_rate_separates_hedged_text():
    assert _row(HEDGED)[_col("hedging_rate")] > _row(PLAIN)[_col("hedging_rate")]


def test_sentiment_signs():
    assert _row(POSITIVE)[_col("sentiment")] > 0.0
    assert _row(NEGATIVE)[_col("sentiment")] < 0.0
    assert _row(POSITIVE)[_col("sentiment")] > _row(NEGATIVE)[_col("sentiment")]


def test_keyword_stuffing_concentration():
    assert _row(STUFFED)[_col("keyword_stuffing")] > _row(PLAIN)[_col("keyword_stuffing")]


def test_sycophancy_rate():
    assert _row(SYCOPHANTIC)[_col("sycophancy_rate")] > _row(PLAIN)[_col("sycophancy_rate")]
    assert _row(PLAIN)[_col("sycophancy_rate")] == 0.0


def test_empty_and_degenerate_strings_are_safe():
    names, m = surface_features(["", " ", "a"])
    assert m.shape == (3, len(names))
    assert np.all(np.isfinite(m))
    assert m[0, _col("len_tokens")] == 0.0


def test_probe_variants_move_their_own_feature():
    # The hack-probe templates must move the matching feature column, family by family.
    from campaign.data.loaders import PROBE_RECORDS, _probe_variant

    prompt, answer = PROBE_RECORDS[0]
    by_family = {
        "length": "len_tokens",
        "format": "markdown_density",
        "sycophancy": "sycophancy_rate",
        "keyword-stuffing": "keyword_stuffing",
    }
    for family, feature in by_family.items():
        clean = _row(answer)
        manipulated = _row(_probe_variant(family, prompt, answer))
        assert manipulated[_col(feature)] > clean[_col(feature)], family
