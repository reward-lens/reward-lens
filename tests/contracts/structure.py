"""Keyword-level structural comparison of two JSON Schemas (D-66, note 25 (d) assertion 3).

Parity is never byte equality. This walks both schemas after inlining `$ref`, and maps each
JSON Pointer to the keywords that carry meaning, ignoring `title`, `description`, `default`,
`$comment` and key order. `anyOf`/`oneOf` of exactly two branches, one of them `{"type": "null"}`,
collapses to the other branch plus a `nullable` marker, because pydantic writes optionals that way
and the hand schema writes them the other way round.
"""

from __future__ import annotations

from typing import Any

KEYWORDS = (
    "type",
    "required",
    "enum",
    "const",
    "additionalProperties",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "minLength",
    "maxLength",
    "minItems",
    "maxItems",
    "pattern",
    "x-scale",
    "nullable",
)


def inline(schema: Any, defs: dict, depth: int = 0) -> Any:
    """Resolve every local `$ref` against `$defs`. The assay schema has no recursive $defs."""
    if depth > 40:
        return schema
    if isinstance(schema, list):
        return [inline(s, defs, depth + 1) for s in schema]
    if not isinstance(schema, dict):
        return schema
    if "$ref" in schema:
        name = schema["$ref"].rsplit("/", 1)[-1]
        target = defs.get(name)
        if target is None:
            return schema
        merged = {**inline(target, defs, depth + 1)}
        merged.update({k: inline(v, defs, depth + 1) for k, v in schema.items() if k != "$ref"})
        return merged
    return {k: inline(v, defs, depth + 1) for k, v in schema.items() if k != "$defs"}


def collapse_nullable(node: dict) -> dict:
    for key in ("oneOf", "anyOf"):
        branches = node.get(key)
        if not isinstance(branches, list) or len(branches) != 2:
            continue
        nulls = [b for b in branches if isinstance(b, dict) and b.get("type") == "null"]
        rest = [b for b in branches if not (isinstance(b, dict) and b.get("type") == "null")]
        if len(nulls) == 1 and len(rest) == 1 and isinstance(rest[0], dict):
            merged = {k: v for k, v in node.items() if k != key}
            merged.update(rest[0])
            merged["nullable"] = True
            return collapse_nullable(merged)
    return node


def keymap(schema: dict, defs: dict | None = None) -> dict[str, dict]:
    """Map JSON Pointer -> the meaningful keywords at that pointer."""
    defs = schema.get("$defs", {}) if defs is None else defs
    resolved = inline(schema, defs)
    out: dict[str, dict] = {}

    def walk(node: Any, pointer: str) -> None:
        if not isinstance(node, dict):
            return
        node = collapse_nullable(node)
        entry = {}
        for kw in KEYWORDS:
            if kw not in node:
                continue
            value = node[kw]
            if kw == "required":
                value = sorted(value)
            elif kw == "enum":
                value = sorted(value, key=repr)
            elif kw == "additionalProperties":
                if value is True:
                    # `additionalProperties: true` is what an absent keyword already means.
                    continue
                value = value if isinstance(value, bool) else "schema"
            elif kw == "type" and isinstance(value, list):
                value = sorted(value)
            entry[kw] = value
        if ("enum" in entry or "const" in entry) and entry.get("type") == "string":
            # The hand schema writes `{"enum": [...]}` bare where pydantic writes the redundant
            # `"type": "string"` beside it. The members carry the type, so this is notation.
            entry.pop("type")
        if entry:
            out[pointer] = entry
        for name, sub in (node.get("properties") or {}).items():
            walk(sub, f"{pointer}/{name}")
        items = node.get("items")
        if isinstance(items, dict):
            walk(items, f"{pointer}/[]")
        extra = node.get("additionalProperties")
        if isinstance(extra, dict):
            walk(extra, f"{pointer}/*")
        for key in ("oneOf", "anyOf", "allOf"):
            for n, branch in enumerate(node.get(key) or []):
                walk(branch, f"{pointer}/{key}[{n}]")

    walk(resolved, "")
    return out


def diff(left: dict[str, dict], right: dict[str, dict]) -> list[str]:
    """Readable differences, left being the specification."""
    problems = []
    for pointer in sorted(set(left) | set(right)):
        a, b = left.get(pointer), right.get(pointer)
        if a is None:
            problems.append(f"{pointer}: only the model declares it ({b})")
        elif b is None:
            problems.append(f"{pointer}: only the schema declares it ({a})")
        elif a != b:
            for kw in sorted(set(a) | set(b)):
                if a.get(kw) != b.get(kw):
                    problems.append(f"{pointer}.{kw}: schema {a.get(kw)!r} vs model {b.get(kw)!r}")
    return problems


def strip_annotations(node: Any) -> Any:
    """Drop `x-` keywords and `format` so a generator that cannot read them still runs."""
    if isinstance(node, list):
        return [strip_annotations(n) for n in node]
    if not isinstance(node, dict):
        return node
    return {
        k: strip_annotations(v)
        for k, v in node.items()
        if not k.startswith("x-") and k != "format"
    }


# A-003 step 2. `hypothesis-jsonschema` compiles every `pattern` into `st.from_regex`, and one
# assay record reaches sixteen of them, two of which are expensive on every draw rather than once:
# `^sha256:[0-9a-f]{64}$` and the 200-character `ident`. Measured on this worktree, one example of
# the whole-record strategy cost 36.7s and ten cost more than 190s, so the cost is per example.
#
# Every pattern is therefore pinned to a few legal values before the generator sees the schema.
# The gate loses nothing: no rule of the record turns on *which* digest or ident it carries,
# pattern parity between the two schemas is assertion 3's job (it compares the `pattern` keyword
# at every pointer), and rejection of illegal values is the differential corpus's job. What
# generation is for is the shape of the record, and pinning is what buys the examples to reach it.
PATTERN_SAMPLES: dict[str, tuple[str, ...]] = {
    # digest, ident, timestamp, money: the four the brief names.
    r"^sha256:[0-9a-f]{64}$": ("sha256:" + "0" * 64, "sha256:" + "ab" * 32),
    r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,199}$": ("a", "reward-lens/assay:1", "S0"),
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$": ("2026-09-13T00:00:00Z", "2025-01-31T12:34:56Z"),
    r"^-?[0-9]+\.[0-9]{2}$": ("0.00", "12.50", "-3.25"),
    # the twelve others the schema carries, pinned for the same reason.
    r"^[0-9]+\.[0-9]+\.[0-9]+([-.+][0-9A-Za-z.]+)?$": ("3.1.0", "1.2.3-rc.1"),
    r"^(blocking_finding|required_missing|required_stale|required_invalid|required_inconclusive"
    r"|constraint_violated|tradeoff_unsettled|policy_absent|outcome_unqualified)"
    r":[A-Za-z0-9_.:/-]+$": (
        "blocking_finding:RGX-2026-0001",
        "required_missing:validity.scope",
        "policy_absent:refusal",
    ),
    r"^\d{4}-\d{2}-\d{2}$": ("2026-09-13", "2025-01-31"),
    r"^schemas/[A-Za-z0-9_.-]+\.json$": ("schemas/assay-1.0.json", "schemas/rows.json"),
    r"^run-[0-9a-f]{8,32}$": ("run-0123abcd", "run-0123456789abcdef"),
    r"^https?://[^\s]+$": ("https://example.org/x", "http://example.org/y"),
    r"^(validity|soundness|reach|exploits|framing|reward_statistics|signal|trace|forecast"
    r"|calibration)\.[a-z0-9_.]+$": (
        "validity.scope",
        "forecast.lead_time",
        "reward_statistics.spread",
    ),
    r"^[a-z_]+\.[a-z0-9_.]+$": ("validity.scope_declared", "core.rule_1"),
    r"^(RGX-[0-9]{4}-[0-9]{4}|RGX-local-[0-9]{4})$": ("RGX-2026-0001", "RGX-local-0007"),
    r"^(T0|L[0-3]|M[01]|W[0-2])$": ("T0", "L2", "W1"),
    r"^digest:(source|environment|scorer_config|task_distribution|samples|outcome_protocol"
    r"|policy|training_semantics|instrument_method)$": (
        "digest:source",
        "digest:policy",
        "digest:samples",
    ),
    r"^RL[0-9]{4}$": ("RL0601", "RL0000"),
}


SUBSCHEMA_MAPS = ("properties", "$defs", "definitions", "patternProperties", "dependentSchemas")
UNINVERTIBLE = ("if", "then", "else", "not")


def relax_conditionals(node: Any) -> Any:
    """Drop `if`/`then`/`else`/`not` from the schema the generator draws from.

    `hypothesis-jsonschema` cannot invert these into a strategy. It generates against what is
    left and then filters the result through the whole schema, so with the ten conditionals the
    assay record carries, most draws are thrown away and one example costs about eighteen
    seconds. Dropping them costs the gate nothing, because the generated instance is still
    judged by the *full* schema (`rs_validator`) and by the models: relaxing changes which
    instances are drawn, not which verdict either side gives. It is in fact the better
    distribution, since a strategy that satisfies every conditional by construction never puts
    the two sides on opposite ends of one.
    """
    if isinstance(node, list):
        return [relax_conditionals(n) for n in node]
    if not isinstance(node, dict):
        return node
    out: dict[str, Any] = {}
    for key, value in node.items():
        if key in UNINVERTIBLE:
            continue
        if key in SUBSCHEMA_MAPS and isinstance(value, dict):
            out[key] = {name: relax_conditionals(sub) for name, sub in value.items()}
        else:
            out[key] = relax_conditionals(value)
    return out


ROUND_TRIP_DIGITS = 15


def bound_quantised_numbers(node: Any) -> Any:
    """Bound every `x-scale` number to the range where the schema and the models agree.

    A number written at scale `s` must fit in the fifteen significant digits a double round-trips,
    or it belongs in a string (D-10). The models enforce that; JSON Schema cannot say it, so
    `1e12` at scale 3 is valid against the schema and refused by the models. That divergence is
    real and is pinned by `test_the_quantiser_layer_is_the_one_divergence`; what the generator
    must not do is spend its ten examples rediscovering it instead of exploring the record. The
    bound applies to the generation schema only. Run on the raw schema, before `x-` is stripped.
    """
    if isinstance(node, list):
        return [bound_quantised_numbers(n) for n in node]
    if not isinstance(node, dict):
        return node
    out: dict[str, Any] = {}
    for key, value in node.items():
        if key in SUBSCHEMA_MAPS and isinstance(value, dict):
            out[key] = {name: bound_quantised_numbers(sub) for name, sub in value.items()}
        else:
            out[key] = bound_quantised_numbers(value)
    scale = node.get("x-scale")
    if isinstance(scale, int) and node.get("type") in ("number", "integer"):
        limit = 10 ** (ROUND_TRIP_DIGITS - scale) - 1
        out["maximum"] = min(out.get("maximum", limit), limit)
        out["minimum"] = max(out.get("minimum", -limit), -limit)
    return out


def patterns_in(node: Any, found: set[str] | None = None) -> set[str]:
    """Every `pattern` value anywhere in a schema, so the table above can be checked for rot."""
    found = set() if found is None else found
    if isinstance(node, list):
        for item in node:
            patterns_in(item, found)
    elif isinstance(node, dict):
        if isinstance(node.get("pattern"), str):
            found.add(node["pattern"])
        for value in node.values():
            patterns_in(value, found)
    return found


def pin_patterns(node: Any) -> Any:
    """Replace every `pattern` with an `enum` of legal values, for the generator only.

    Raises `KeyError` on a pattern the table does not cover, so a new one in the schema stops the
    gate rather than quietly costing it half a minute an example.
    """
    if isinstance(node, list):
        return [pin_patterns(n) for n in node]
    if not isinstance(node, dict):
        return node
    out = {k: pin_patterns(v) for k, v in node.items() if k != "pattern"}
    pattern = node.get("pattern")
    if isinstance(pattern, str):
        if pattern not in PATTERN_SAMPLES:
            raise KeyError(f"no pinned samples for {pattern!r}; add them to PATTERN_SAMPLES")
        if "enum" in out:
            raise ValueError(f"{pattern!r} sits beside an enum; pinning would drop a constraint")
        out["enum"] = list(PATTERN_SAMPLES[pattern])
    return out
