"""`rewardlens.yaml` as a typed contract, with its own hand-authored JSON Schema."""

from __future__ import annotations

import pytest
import yaml

from conftest import PROJECT_SCHEMA, load

from reward_lens.contracts import ProjectConfig, RewardLensError
from reward_lens.contracts.project import project_schema as frozen_project_schema

VALID = """
reward:
  kind: plain
  entry: grader.py:score
  shape: plain
  watch: [none_to_nan, exception_to_zero, bool_as_score]
  trainer: trl
tasks:
  path: tasks.jsonl
  prompt: prompt
  reference: reference
  tests: tests
responses:
  path: responses.jsonl
outcome:
  kind: protected_test_suite
  path: outcome/
success: more held-out task success at a fixed training budget
constraints:
  - legitimate solutions are kept
version:
  id: v1
  parents: []
"""

MINIMAL = """
reward:
  entry: grader.py:score
tasks:
  path: tasks.jsonl
  prompt: prompt
outcome: null
"""


@pytest.fixture(scope="session")
def project_schema():
    return load(PROJECT_SCHEMA)


@pytest.fixture(scope="session")
def project_validator(project_schema):
    import jsonschema_rs

    return jsonschema_rs.validator_for(project_schema)


@pytest.mark.parametrize("document", [VALID, MINIMAL])
def test_a_valid_project_file_loads(document, project_validator):
    raw = yaml.safe_load(document)
    config = ProjectConfig.model_validate(raw)
    assert config.reward.entry == "grader.py:score"
    assert project_validator.is_valid(raw), list(project_validator.iter_errors(raw))


def test_defaults_match_the_schema(project_schema):
    config = ProjectConfig.model_validate(yaml.safe_load(MINIMAL))
    assert config.reward.kind == "plain"
    assert config.reward.shape is None
    assert config.reward.watch == []
    assert config.outcome is None
    assert project_schema["$id"].endswith("/schema/project/1.0/rewardlens.schema.json")


def test_an_unknown_key_is_a_field_level_error(project_validator):
    raw = yaml.safe_load(MINIMAL)
    raw["rewrad"] = {}
    with pytest.raises(Exception) as excinfo:
        ProjectConfig.model_validate(raw)
    assert "rewrad" in str(excinfo.value)
    assert not project_validator.is_valid(raw)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d["reward"].__setitem__("kind", "exotic"),
        lambda d: d["reward"].__setitem__("watch", ["something_else"]),
        lambda d: d["reward"].__setitem__("shape", "keras"),
        lambda d: d["reward"].__setitem__("trainer", "keras"),
        lambda d: d.__setitem__("outcome", {"kind": "vibes", "path": "outcome/"}),
        lambda d: d["tasks"].__delitem__("path"),
        lambda d: d.__setitem__("constraints", "not a list"),
        lambda d: d.__setitem__("name", ""),
        lambda d: d.__setitem__("name", "   "),
        lambda d: d.__setitem__("name", "x" * 201),
        lambda d: d.__setitem__("name", 7),
        lambda d: d.__setitem__("name", "code-reward"),
    ],
)
def test_the_two_layers_refuse_the_same_project_files(mutate, project_validator):
    raw = yaml.safe_load(VALID)
    mutate(raw)
    model_ok = True
    try:
        ProjectConfig.model_validate(raw)
    except Exception:
        model_ok = False
    assert project_validator.is_valid(raw) == model_ok


def test_a_field_level_error_names_its_location():
    """RL0003 names the field in its own message, and pydantic's every location is the cause.

    A person edits `rewardlens.yaml` by hand, so what they need first is the key that was wrong.
    `make("RL0003", field=...)` puts it in the rendered line and in `context["field"]`; the full
    list, for a tool rather than a person, is on `__cause__`."""
    raw = yaml.safe_load(VALID)
    raw["reward"]["kind"] = "exotic"
    with pytest.raises(RewardLensError) as excinfo:
        ProjectConfig.model_validate(raw)
    assert excinfo.value.code == "RL0003"
    assert excinfo.value.context["field"] == "reward.kind"
    assert "reward.kind" in str(excinfo.value)
    errors = excinfo.value.__cause__.errors()
    assert any(e["loc"][:2] == ("reward", "kind") for e in errors), errors


def test_a_field_the_project_schema_requires_is_written_even_when_it_is_none():
    """The record's rule governs `rewardlens.yaml` too, and the project schema is what says so.

    As frozen, the project schema requires only `reward`, `tasks` and `outcome`, and none of the
    three admits null, so the property is shown where it can be shown: an optional field the
    config leaves empty is moved into `required` in memory and the serialiser writes it as null
    instead of dropping it. `to_dict()` used to be `exclude_unset`, under which the same config
    wrote different keys depending on how it was built.
    """
    optional = [
        name for name, spec in ProjectConfig.model_fields.items() if not spec.is_required()
    ]
    raw = yaml.safe_load(VALID)
    field = next(name for name in optional if raw.get(name) is not None)
    raw.pop(field)
    config = ProjectConfig.model_validate(raw)
    assert getattr(config, field) is None
    assert field not in config.to_dict()
    schema = frozen_project_schema()
    kept = list(schema["required"])
    try:
        schema["required"] = kept + [field]
        written = config.to_dict()
        assert field in written and written[field] is None
    finally:
        schema["required"] = kept
    assert field not in config.to_dict()


NAMED = VALID + "name: code-reward\n"


def test_a_declared_name_is_exposed(project_validator):
    """The reward system's own name, carried verbatim, and the schema agrees it belongs here.

    Nothing is derived from it and nothing normalises it: what the author wrote is what the
    record's Subject line has to be able to print.
    """
    raw = yaml.safe_load(NAMED)
    config = ProjectConfig.model_validate(raw)
    assert config.name == "code-reward"
    assert config.to_dict()["name"] == "code-reward"
    assert project_validator.is_valid(raw), list(project_validator.iter_errors(raw))


@pytest.mark.parametrize("document", [VALID, MINIMAL])
def test_a_config_that_declares_no_name_is_unchanged(document, project_validator):
    """The field is additive: a project file written before it existed loads and dumps as before.

    `name` is optional in the project schema, so the serialiser drops it rather than writing null,
    and `to_dict()` is the input to `runtime/fingerprint.py`: an unnamed config keeps its digest.
    """
    raw = yaml.safe_load(document)
    config = ProjectConfig.model_validate(raw)
    assert config.name is None
    assert "name" not in config.to_dict()
    assert project_validator.is_valid(raw), list(project_validator.iter_errors(raw))


@pytest.mark.parametrize("empty", ["", " ", "   ", "\t", "\n", " \t\n "])
def test_a_name_that_is_empty_or_whitespace_is_refused(empty, project_validator):
    """A name of blanks is not a name, and it is refused the way every other bad field is.

    The assay schema's `minLength: 1` would admit `"  "`; both layers here carry the `\\S` that
    does not, and the refusal is RL0003 naming `name`, not a pydantic traceback.
    """
    raw = yaml.safe_load(VALID)
    raw["name"] = empty
    with pytest.raises(RewardLensError) as excinfo:
        ProjectConfig.model_validate(raw)
    assert excinfo.value.code == "RL0003"
    assert excinfo.value.context["field"] == "name"
    assert "name" in str(excinfo.value)
    assert not project_validator.is_valid(raw)


def test_a_name_is_prose_where_a_version_id_is_an_identifier():
    """`Ident` governs `version.id` and deliberately does not govern `name`.

    The record carries the two beside each other (`code-reward` beside `code-reward-v1`), and the
    schema constrains only the second's characters. A name with a space in it is a real name.
    """
    raw = yaml.safe_load(VALID)
    raw["name"] = "Code reward (v1 line)"
    assert ProjectConfig.model_validate(raw).name == "Code reward (v1 line)"
    raw["version"]["id"] = "not an ident"
    with pytest.raises(RewardLensError) as excinfo:
        ProjectConfig.model_validate(raw)
    assert excinfo.value.context["field"] == "version.id"
