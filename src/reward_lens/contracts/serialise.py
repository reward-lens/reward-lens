"""One serialisation rule, derived from the schema that governs each model.

Every `to_dict()` in this package obeys the same three rules:

1. A property the governing schema lists under `required` is always written, `null` when its
   value is None.
2. A property the schema does not require is omitted when its value is None. An optional
   non-nullable property is therefore never written as `null`.
3. Everything else is written as its JSON value.

The set a model may omit is read out of the schema itself, never out of the model's own field
defaults. A default is a convenience for callers; `required` is the contract, and the two drift
the moment someone gives a field a default without moving the schema. `bind()` walks the schema
and the model graph together once, from the root object and from each named `$def`, and keeps a
reference to the live schema node for every model it reaches. `omitted(model)` then reads
`properties - required` off that node at call time, so a schema changed in memory is a serialiser
changed in memory, with nothing to invalidate.

A model the walk cannot reach through the schema has no governing node. Its omission set falls
back to its own optional-and-None fields, and `unbound()` names it, so a model that drops out of
the correspondence is visible rather than silently self-governed.
"""

from __future__ import annotations

from typing import Any, Iterator, Mapping, get_args

from pydantic import BaseModel

__all__ = ["SchemaRule", "bind"]

_MAX_HOPS = 8


def _resolve(schema: Mapping[str, Any], node: Any) -> dict[str, Any] | None:
    """Follow `$ref`, `items` and the non-null branch of a union to the object node beneath."""
    for _ in range(_MAX_HOPS):
        if not isinstance(node, dict):
            return None
        if "properties" in node:
            return node
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            node = schema.get("$defs", {}).get(ref.rsplit("/", 1)[-1])
            continue
        if "items" in node:
            node = node["items"]
            continue
        branch = _non_null_branch(node)
        if branch is None:
            return None
        node = branch
    return None


def _non_null_branch(node: Mapping[str, Any]) -> Any:
    for key in ("anyOf", "oneOf", "allOf"):
        members = node.get(key)
        if not isinstance(members, list):
            continue
        for member in members:
            if isinstance(member, dict) and member.get("type") != "null":
                return member
    return None


def _models_in(annotation: Any) -> Iterator[type[BaseModel]]:
    """Every model class inside an annotation: `X`, `X | None`, `list[X]`, `dict[str, X]`."""
    stack = [annotation]
    while stack:
        item = stack.pop()
        if isinstance(item, type) and issubclass(item, BaseModel):
            yield item
            continue
        stack.extend(get_args(item))


class SchemaRule:
    """The correspondence between a schema and the models that implement it."""

    def __init__(self, schema: Mapping[str, Any], seeds: Mapping[type[BaseModel], Any]) -> None:
        self.schema = schema
        self.nodes: dict[type[BaseModel], dict[str, Any]] = {}
        self.unbound: set[type[BaseModel]] = set()
        queue: list[tuple[type[BaseModel], dict[str, Any]]] = []
        for model, node in seeds.items():
            resolved = _resolve(schema, node)
            if resolved is not None:
                queue.append((model, resolved))
        while queue:
            model, node = queue.pop(0)
            if model in self.nodes:
                continue
            self.nodes[model] = node
            properties = node.get("properties", {})
            for name, field in model.model_fields.items():
                child = properties.get(field.alias or name)
                if child is None:
                    continue
                resolved = _resolve(schema, child)
                if resolved is None:
                    continue
                for sub in _models_in(field.annotation):
                    if sub not in self.nodes:
                        queue.append((sub, resolved))

    def omitted(self, model: type[BaseModel]) -> frozenset[str]:
        """The keys `model` omits when they hold None: the properties the schema leaves optional.

        Read live off the schema node, so this follows a `required` list edited in memory. Both
        the field name and its alias are returned, because a dump may be keyed by either.
        """
        node = self.nodes.get(model)
        if node is None:
            self.unbound.add(model)
            return _by_default(model)
        required = set(node.get("required", ()))
        keys: set[str] = set()
        for name, field in model.model_fields.items():
            alias = field.alias or name
            if alias in required or name in required:
                continue
            keys.add(name)
            keys.add(alias)
        return frozenset(keys)


def _by_default(model: type[BaseModel]) -> frozenset[str]:
    """The fallback for a model no schema node governs: its own optional-and-None fields."""
    keys: set[str] = set()
    for name, field in model.model_fields.items():
        if not field.is_required() and field.default is None:
            keys.add(name)
            if field.alias:
                keys.add(field.alias)
    return frozenset(keys)


_BOUND: dict[int, SchemaRule] = {}


def bind(schema: Mapping[str, Any], seeds: Mapping[type[BaseModel], Any]) -> SchemaRule:
    """The rule for this schema object, walked once and kept while the object lives."""
    rule = _BOUND.get(id(schema))
    if rule is None or rule.schema is not schema:
        rule = SchemaRule(schema, seeds)
        _BOUND[id(schema)] = rule
    return rule
