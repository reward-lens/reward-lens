"""The declared-scale rule of D-10.

Every measured quantity in the record is a JSON number quantised at the record boundary with
`Decimal.quantize(..., ROUND_HALF_EVEN)` to the scale the schema declares for that field
(`x-scale`). That, not the choice of JSON type, is what makes the bytes reproducible across
machines: RFC 8785 renders a double as the shortest decimal that round-trips, which is a pure
function of the 64 bits, and IEEE 754 guarantees the round trip for at most 15 significant digits.

Two things live here. `quantise()` and the `quantised()` annotated type are the boundary: models
round on construction. `scale_violations()` is the rule read the other way, a walker over an
instance and the schema together, which is what the runtime validator uses to refuse a number that
arrived carrying more digits than its scale.
"""

from __future__ import annotations

import math
from decimal import ROUND_HALF_EVEN, Decimal
from typing import Annotated, Any

from pydantic import AfterValidator, BeforeValidator, Field

__all__ = [
    "MAX_SIGNIFICANT_DIGITS",
    "decimal_places",
    "quantise",
    "quantised",
    "scale_violations",
    "JsonInt",
]

#: IEEE 754 round-trips decimal -> binary64 -> decimal for at most this many significant digits.
MAX_SIGNIFICANT_DIGITS = 15


def decimal_places(value: float | int) -> int:
    """Decimal places in the shortest representation of `value`, which is what JCS will write."""
    exponent = Decimal(repr(value)).as_tuple().exponent
    return -exponent if isinstance(exponent, int) and exponent < 0 else 0


def quantise(value: Any, scale: int) -> float:
    """Round `value` half-to-even to `scale` decimal places and return it as a float.

    Raises `ValueError` for NaN, either infinity, and any result that would need more than 15
    significant digits, because outside that bound the JCS rendering is no longer the decimal that
    went in.
    """
    if isinstance(value, bool):
        raise ValueError("a boolean is not a measured quantity")
    if isinstance(value, Decimal):
        number = value
        if not number.is_finite():
            raise ValueError(f"{value!r} is not finite; NaN and Infinity are refused (D-10)")
    else:
        as_float = float(value)
        if math.isnan(as_float) or math.isinf(as_float):
            raise ValueError(f"{value!r} is not finite; NaN and Infinity are refused (D-10)")
        number = Decimal(repr(as_float))
    rounded = number.quantize(Decimal(1).scaleb(-scale), rounding=ROUND_HALF_EVEN)
    digits = len(rounded.as_tuple().digits)
    if digits > MAX_SIGNIFICANT_DIGITS:
        raise ValueError(
            f"{value!r} needs {digits} significant digits at scale {scale}; at most "
            f"{MAX_SIGNIFICANT_DIGITS} round-trip through a double, so this belongs in a string"
        )
    return float(rounded)


def _to_int(value: Any) -> Any:
    """JSON Schema calls 1.0 an integer; Python does not. Accept it, reject 1.5 and booleans."""
    if isinstance(value, bool):
        return value
    if isinstance(value, float) and value.is_integer() and abs(value) <= 2**53 - 1:
        return int(value)
    return value


#: An integer field, spelled the way JSON Schema 2020-12 means it.
JsonInt = Annotated[int, BeforeValidator(_to_int)]


def quantised(scale: int, *, ge: float | None = None, le: float | None = None) -> Any:
    """A numeric field at a declared scale: quantised on construction, `x-scale` in the schema."""

    def _round(value: float) -> float:
        return quantise(value, scale)

    constraints: dict[str, Any] = {
        "allow_inf_nan": False,
        "json_schema_extra": {"x-scale": scale},
    }
    if ge is not None:
        constraints["ge"] = ge
    if le is not None:
        constraints["le"] = le
    return Annotated[float, Field(**constraints), AfterValidator(_round)]


def scale_violations(instance: Any, schema: dict, path: str = "") -> list[str]:
    """Walk an instance and the schema together; report numbers whose digits exceed `x-scale`.

    This is the same walk the freeze checker applies (`schema/assay/1.0/check_fixtures.py:13`), kept
    here because it is a rule of the record and not of the checker.
    """
    defs = schema.get("$defs", {})
    out: list[str] = []

    def walk(inst: Any, sch: Any, where: str) -> None:
        if not isinstance(sch, dict):
            return
        if "$ref" in sch:
            name = sch["$ref"].rsplit("/", 1)[-1]
            target = defs.get(name)
            if target is not None:
                sch = {**target, **{k: v for k, v in sch.items() if k != "$ref"}}
        for alternative in list(sch.get("oneOf", [])) + list(sch.get("anyOf", [])):
            walk(inst, alternative, where)
        if isinstance(inst, (int, float)) and not isinstance(inst, bool) and "x-scale" in sch:
            places = decimal_places(inst)
            if places > sch["x-scale"]:
                out.append(f"{where}: {inst} has {places} decimals, scale {sch['x-scale']}")
        if isinstance(inst, dict):
            for key, value in inst.items():
                sub = (sch.get("properties") or {}).get(key)
                if sub is not None:
                    walk(value, sub, f"{where}.{key}" if where else key)
                elif isinstance(sch.get("additionalProperties"), dict):
                    walk(value, sch["additionalProperties"], f"{where}.{key}" if where else key)
        if isinstance(inst, list) and isinstance(sch.get("items"), dict):
            for n, value in enumerate(inst):
                walk(value, sch["items"], f"{where}[{n}]")

    walk(instance, schema, path)
    return out
