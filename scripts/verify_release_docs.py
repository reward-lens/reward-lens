#!/usr/bin/env python3
"""Check generated local documentation links and maintained install wording."""

from __future__ import annotations

import pathlib
from html.parser import HTMLParser
from urllib.parse import unquote, urlsplit

SITE = pathlib.Path("docs/_site").resolve()


class Links(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag not in {"a", "img", "script", "link", "source"}:
            return
        for key, value in attrs:
            if key in {"href", "src"} and value:
                self.links.append(value)


def main() -> None:
    assert SITE.is_dir(), SITE
    pages = list(SITE.rglob("*.html"))
    assert pages, "MkDocs generated no HTML"
    failures = []
    checked = 0
    for page in pages:
        links = Links()
        links.feed(page.read_text(errors="replace"))
        for link in links.links:
            parsed = urlsplit(link)
            if parsed.scheme or parsed.netloc or not parsed.path or link.startswith(("#", "data:")):
                continue
            path = unquote(parsed.path)
            if path.startswith("/reward-lens/"):
                dest = SITE / path.removeprefix("/reward-lens/")
            elif path.startswith("/"):
                continue  # Origin-root links belong to the hosting site's own namespace.
            else:
                dest = page.parent / path
            dest = dest.resolve()
            if not dest.is_relative_to(SITE):
                failures.append((str(page.relative_to(SITE)), link, "outside generated site"))
                continue
            if dest.is_dir():
                dest = dest / "index.html"
            checked += 1
            if not dest.is_file():
                failures.append((str(page.relative_to(SITE)), link, "missing local target"))
    assert not failures, failures[:30]
    install = SITE / "getting-started/install/index.html"
    assert install.is_file(), install
    text = install.read_text()
    assert "reward-lens" in text and "reward-lens==3.0.0" not in text
    print(f"PASS: {len(pages)} generated pages, {checked} local link targets")


if __name__ == "__main__":
    main()
