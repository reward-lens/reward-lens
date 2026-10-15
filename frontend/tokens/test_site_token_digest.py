"""Repository-contained proof for the immutable site-token export.

The test binds the export to a hash-identified copy of the C-019 site source.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


PRODUCT_ROOT = Path(__file__).resolve().parents[2]
CHECKER = Path(__file__).with_name("check_site_token_digest.py")
SOURCE_ROOT = PRODUCT_ROOT / "frontend" / "tokens" / "site-source"


def check(site_root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CHECKER), "--site-root", str(site_root)],
        cwd=PRODUCT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


class SiteTokenDigestTests(unittest.TestCase):
    def test_exact_source_and_export_are_bound(self) -> None:
        result = check(SOURCE_ROOT)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("site token manifest: verified", result.stdout)

    def test_source_drift_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "site"
            source = root / "src/styles/tokens.css"
            source.parent.mkdir(parents=True)
            shutil.copy2(SOURCE_ROOT / "src/styles/tokens.css", source)
            source.write_text(f"{source.read_text(encoding='utf-8')}\n/* drift */\n", encoding="utf-8")
            result = check(root)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("source digest", result.stderr)

    def test_relocated_matching_source_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "site"
            source = root / "src/styles/tokens.css"
            source.parent.mkdir(parents=True)
            shutil.copy2(SOURCE_ROOT / "src/styles/tokens.css", source)
            result = check(root)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("site root differs", result.stderr)


if __name__ == "__main__":
    unittest.main()
