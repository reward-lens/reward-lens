"""The surface itself: every name section 8.0 promises, with a typed request and a typed result."""

from __future__ import annotations

import inspect

import pytest
from pydantic import BaseModel

from reward_lens import api, contracts

from .conftest import SECTION_8_0

REQUEST_TYPES = (
    "AuditRequest",
    "TraceRequest",
    "CompareRequest",
    "ForecastIssueRequest",
    "ForecastResolveRequest",
    "ForecastLedgerRequest",
    "ImproveRequest",
    "ExportRequest",
    "DoctorRequest",
)

RESULT_TYPES = ("Plan", "Capabilities", "Capability", "InstallInfo")


@pytest.mark.parametrize("name", SECTION_8_0)
def test_every_function_of_section_8_0_exists_and_is_exported(name: str) -> None:
    assert name in api.__all__, f"{name} is not exported from reward_lens.api"
    assert callable(getattr(api, name))


@pytest.mark.parametrize("name", REQUEST_TYPES + RESULT_TYPES)
def test_every_request_and_result_type_exists_and_is_a_strict_model(name: str) -> None:
    model = getattr(api, name)
    assert issubclass(model, BaseModel)
    assert model.model_config.get("strict") is True
    assert model.model_config.get("extra") == "forbid"
    assert name in api.__all__


@pytest.mark.parametrize(
    "verb,request_type",
    [
        ("audit", "AuditRequest"),
        ("trace", "TraceRequest"),
        ("compare", "CompareRequest"),
        ("forecast_issue", "ForecastIssueRequest"),
        ("forecast_resolve", "ForecastResolveRequest"),
        ("forecast_ledger", "ForecastLedgerRequest"),
        ("improve", "ImproveRequest"),
        ("export", "ExportRequest"),
    ],
)
def test_each_verb_takes_its_own_typed_request(verb: str, request_type: str) -> None:
    signature = inspect.signature(getattr(api, verb))
    first = next(iter(signature.parameters.values()))
    assert first.annotation is getattr(api, request_type), (
        f"{verb} does not take a {request_type} as its first parameter"
    )


def test_audit_and_the_record_verbs_are_annotated_to_return_a_contract_type() -> None:
    for verb in ("audit", "trace", "compare", "improve", "open_record"):
        assert inspect.signature(getattr(api, verb)).return_annotation is contracts.Assay


def test_doctor_returns_capabilities_and_dry_run_returns_a_plan() -> None:
    assert inspect.signature(api.doctor).return_annotation is api.Capabilities
    assert inspect.signature(api.dry_run).return_annotation is api.Plan


def test_the_module_exports_nothing_it_did_not_declare() -> None:
    public = {n for n in vars(api) if not n.startswith("_") and not inspect.ismodule(getattr(api, n))}
    assert public == set(api.__all__)
