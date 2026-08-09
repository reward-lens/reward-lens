"""D-29: flags, then environment, then project file, then user file, then system.

`--no-config` ignores every file. Nothing here imports the user's Python and nothing here requires
an environment variable; a missing file is simply a level that has no opinion.
"""

from __future__ import annotations

import os
import pathlib

PRECEDENCE = ("flag", "environment", "rewardlens.yaml", "user file", "system file")

PRECEDENCE_LINE = (
    "Configuration: flags, then environment, then ./rewardlens.yaml, then the user file, then the "
    "system file. --no-config ignores every file."
)

#: Setting name -> environment variable. Printed inline in help as uv does it: [env: NAME=].
ENV: dict[str, str] = {
    "format": "REWARD_LENS_FORMAT",
    "color": "REWARD_LENS_COLOR",
    "no_progress": "REWARD_LENS_NO_PROGRESS",
    "offline": "REWARD_LENS_OFFLINE",
    "max_budget_usd": "REWARD_LENS_MAX_BUDGET_USD",
    "non_interactive": "REWARD_LENS_NON_INTERACTIVE",
}

CONFIG_NAME = "rewardlens.yaml"
USER_DIR = "reward-lens"


def env_note(setting: str) -> str:
    """The `[env: NAME=]` fragment for a flag's help line."""
    name = ENV.get(setting)
    return f"  [env: {name}=]" if name else ""


def user_file() -> pathlib.Path:
    root = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    return pathlib.Path(root) / USER_DIR / "config.yaml"


def system_file() -> pathlib.Path:
    return pathlib.Path("/etc") / USER_DIR / "config.yaml"


def project_file(start: pathlib.Path | None = None) -> pathlib.Path | None:
    """The nearest `rewardlens.yaml` at or above `start`."""
    here = pathlib.Path(start or pathlib.Path.cwd()).resolve()
    if here.is_file():
        here = here.parent
    for directory in (here, *here.parents):
        candidate = directory / CONFIG_NAME
        if candidate.is_file():
            return candidate
    return None


def _load(path: pathlib.Path | None) -> dict:
    if path is None or not path.is_file():
        return {}
    import yaml

    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as bad:  # a malformed file is a configuration error, not a crash
        from reward_lens import errors

        raise errors.make("RL0003", field=str(path), detail=str(bad).splitlines()[0]) from bad
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        from reward_lens import errors

        raise errors.make("RL0003", field=str(path), detail="the file is not a mapping")
    return loaded.get("cli", loaded) if isinstance(loaded.get("cli", None), dict) else loaded


def files(*, start: pathlib.Path | None = None, no_config: bool = False) -> list[pathlib.Path]:
    if no_config:
        return []
    found = [project_file(start), user_file(), system_file()]
    return [path for path in found if path is not None and path.is_file()]


def resolve(
    setting: str,
    flag_value=None,
    *,
    default=None,
    start: pathlib.Path | None = None,
    no_config: bool = False,
    cast=None,
):
    """The first level with an opinion wins, in the order of `PRECEDENCE`."""
    if flag_value is not None:
        return flag_value
    name = ENV.get(setting)
    if name:
        raw = os.environ.get(name)
        if raw not in (None, ""):
            return cast(raw) if cast else raw
    for path in files(start=start, no_config=no_config):
        loaded = _load(path)
        if setting in loaded:
            value = loaded[setting]
            return cast(value) if cast else value
    return default


def truthy(raw) -> bool:
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() not in ("", "0", "false", "no", "off")
