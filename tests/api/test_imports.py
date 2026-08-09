"""What `import reward_lens.api` costs: nothing numeric, no HTTP client, no engine."""

from __future__ import annotations

import subprocess
import sys

FORBIDDEN = (
    "numpy",
    "scipy",
    "pandas",
    "torch",
    "httpx",
    "requests",
    "urllib3",
    "aiohttp",
)


def _in_a_fresh_interpreter(script: str) -> str:
    done = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
    )
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


def test_importing_the_sdk_leaves_nothing_numeric_and_no_http_client_loaded() -> None:
    script = (
        "import sys\n"
        "import reward_lens.api\n"
        f"forbidden = {FORBIDDEN!r}\n"
        "print(','.join(sorted(n for n in forbidden if n in sys.modules)))\n"
    )
    assert _in_a_fresh_interpreter(script) == ""


def test_importing_the_sdk_imports_no_engine_module() -> None:
    script = (
        "import sys\n"
        "import reward_lens.api\n"
        "print(','.join(sorted(n for n in sys.modules if n.startswith('reward_lens.product'))))\n"
    )
    assert _in_a_fresh_interpreter(script) == ""


def test_the_sdk_imports_the_contracts_and_the_standard_library_and_that_is_all() -> None:
    script = (
        "import sys\n"
        "before = set(sys.modules)\n"
        "import reward_lens.api\n"
        "added = {n.split('.')[0] for n in set(sys.modules) - before}\n"
        "print(','.join(sorted(added)))\n"
    )
    added = set(_in_a_fresh_interpreter(script).split(","))
    assert "reward_lens" in added
    assert not added & set(FORBIDDEN)
