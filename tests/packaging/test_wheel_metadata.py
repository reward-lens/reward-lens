"""What the built wheel declares, read back out of the wheel rather than out of `pyproject.toml`.

Reading the source file back would test that I can parse what I wrote. Reading the wheel tests that
the build backend agreed with me, which is where the interesting failures live: a package-data glob
that matched nothing, an extra that hatchling dropped, an entry point that never made it into
`entry_points.txt`.
"""

from __future__ import annotations

from email.parser import Parser

import pytest

EXPECTED_EXTRAS = {
    # 3.1, per D-58
    "trace",
    "judge",
    "mcp",
    "train",
    "sigstore",
    "trackers",
    # carried from v3 because inherited code still needs them
    "verifier",
    "white-box",
    "organisms",
    "viz",
    "record",
    "openenv",
    # the aggregate
    "all",
}

# These are checked against a built wheel's METADATA. They deliberately name the
# two non-base integration lanes that were absent from EXPECTED_EXTRAS, so changing
# source declarations without rebuilding cannot satisfy the control.
EXPECTED_INTEGRATION_EXTRA_REQUIREMENTS = {
    "mcp": {
        "anyio>=4.9",
        "httpx2>=2.5.0",
        "jsonschema>=4.20.0",
        "mcp-types==2.2.0",
        "opentelemetry-api>=1.28.0",
        "pydantic>=2.12.0",
        "pyjwt[crypto]>=2.10.1",
        "python-multipart>=0.0.9",
        "sse-starlette>=3.0.0",
        "starlette>=0.27",
        "typing-extensions>=4.13.0",
        "typing-inspection>=0.4.1",
        "uvicorn>=0.31.1",
    },
    # Hatchling normalises compatible-release specifier order in METADATA.
    "trackers": {"mlflow<4,>=3.11", "wandb<1,>=0.23"},
}


@pytest.fixture(scope="session")
def metadata(dist_info: dict[str, str]):
    assert "METADATA" in dist_info, sorted(dist_info)
    return Parser().parsestr(dist_info["METADATA"])


def test_name_version_and_python_floor(metadata) -> None:
    assert metadata["Name"] == "reward-lens"
    assert metadata["Version"] == "3.1.0"
    assert metadata["Requires-Python"] == ">=3.11"


def test_build_backend_is_hatchling(dist_info: dict[str, str]) -> None:
    """D-58: hatchling, because its build hook is what produces the frontend bundle inside
    `uv build`, which `uv_build` cannot do."""
    generator = dist_info.get("WHEEL", "")
    assert "hatchling" in generator, generator


def test_both_console_scripts_resolve_to_one_entry_point(dist_info: dict[str, str]) -> None:
    """D-04 and D-05: one distribution, one import name, two names for one entry point."""
    eps = dist_info.get("entry_points.txt", "")
    assert "reward-lens = reward_lens.cli.main:main" in eps, eps
    assert "rlens = reward_lens.cli.main:main" in eps, eps


def test_reward_lens_claims_is_gone(dist_info: dict[str, str]) -> None:
    """Appendix C: `reward-lens-claims` becomes `reward-lens-commons claims`, one implementation
    rather than two with one name."""
    eps = dist_info.get("entry_points.txt", "")
    assert "reward-lens-claims" not in eps, eps


def test_base_dependencies_are_the_declared_closure(metadata) -> None:
    """`Requires-Dist` without an extra marker is the base closure as declared."""
    base = sorted(
        d.split(";")[0].split(" ")[0].split(">")[0].split("=")[0].split("<")[0].strip().lower()
        for d in metadata.get_all("Requires-Dist") or []
        if "extra ==" not in d
    )
    assert base == [
        "click",
        "coverage",  # A-002
        "cryptography",  # D-10 offline export signatures
        "jsonschema-rs",
        "libcst",  # A-001
        "pydantic",
        "pydantic-settings",  # A-003 public evidence-store settings
        "pyyaml",
        "rfc8785",
    ], base


def test_extras_are_the_declared_set(metadata) -> None:
    provided = set(metadata.get_all("Provides-Extra") or [])
    assert provided == EXPECTED_EXTRAS, {
        "unexpected": sorted(provided - EXPECTED_EXTRAS),
        "missing": sorted(EXPECTED_EXTRAS - provided),
    }


def _requirements_for_extra(metadata, extra: str) -> set[str]:
    marker = f"extra == '{extra}'"
    return {
        requirement.strip()
        for line in metadata.get_all("Requires-Dist") or []
        for requirement, _, declared_marker in [line.partition(";")]
        if declared_marker.strip() == marker
    }


def test_mcp_and_trackers_extra_requirements_are_in_the_built_wheel(metadata) -> None:
    """The optional integration lanes are publishable metadata, not source-only plans."""
    actual = {
        extra: _requirements_for_extra(metadata, extra)
        for extra in EXPECTED_INTEGRATION_EXTRA_REQUIREMENTS
    }
    assert actual == EXPECTED_INTEGRATION_EXTRA_REQUIREMENTS


def test_all_includes_every_extra(metadata) -> None:
    """D-58's forbidden clause, and the v3 defect it names: `all` omitted `record`, so an `[all]`
    install could not read a parquet.

    Hatchling resolves a self-referential extra (`reward-lens[record]`) into the packages it names
    before it writes the metadata, so the check cannot look for extra *names* under `extra == all`;
    there are none to find. It compares requirement sets instead, which is the stronger statement
    anyway: everything every other extra asks for, `all` asks for too.
    """
    by_extra: dict[str, set[str]] = {}
    for line in metadata.get_all("Requires-Dist") or []:
        requirement, _, marker = line.partition(";")
        name = marker.strip().removeprefix("extra ==").strip().strip("'\"")
        if name:
            by_extra.setdefault(name, set()).add(requirement.strip())
    union = set().union(*(v for k, v in by_extra.items() if k != "all"))
    missing = sorted(union - by_extra.get("all", set()))
    assert missing == [], f"`all` omits {missing}"


def test_wheel_carries_the_packaged_data(wheel_names: list[str]) -> None:
    """Interfaces section 7: the wheel includes `reward_lens/examples/**` and the schema, the root
    `schema/` tree relocated under the import package so it is addressable from an installed
    distribution rather than only from a checkout."""
    assert "reward_lens/examples/__init__.py" in wheel_names
    assert "reward_lens/schema/assay/1.0/assay.schema.json" in wheel_names


def test_wheel_carries_py_typed_and_the_quantity_catalogue(wheel_names: list[str]) -> None:
    """The registry loads the catalogue at import, so a wheel without it has an empty registry and
    every instrument fails the "quantity not registered" check. `py.typed` is what makes the
    annotations real downstream."""
    assert "reward_lens/py.typed" in wheel_names
    assert any(n.startswith("reward_lens/spec/") and n.endswith(".json") for n in wheel_names), [
        n for n in wheel_names if "/spec/" in n
    ][:10]


def test_report_assets_ship_when_they_exist_the_conditional_half(wheel_names: list[str]) -> None:
    """The conditional half of the bundle question: whatever assets are on disk are also in the
    wheel. That is a claim about the build configuration and it is checkable today, on a tree where
    P-RENDER has not yet committed `src/reward_lens/render/report/assets/report.js` and the
    directory does not exist. The unconditional half, that the bundle exists at all, is the wave-1
    exit check in `test_shipped_assets.py`, which is where a missing bundle is caught."""
    from pathlib import Path

    assets = Path(__file__).resolve().parents[2] / "src/reward_lens/render/report/assets"
    on_disk = sorted(p.name for p in assets.glob("*")) if assets.is_dir() else []
    prefix = "reward_lens/render/report/assets/"
    in_wheel = sorted(n.rsplit("/", 1)[1] for n in wheel_names if n.startswith(prefix))
    assert in_wheel == on_disk, {"on_disk": on_disk, "in_wheel": in_wheel}
