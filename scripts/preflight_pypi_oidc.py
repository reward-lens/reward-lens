#!/usr/bin/env python3
"""Test the configured PyPI trusted publisher without uploading a distribution.

PyPI documents this exchange as an implementation detail. Keep this script only
for the guarded one-time release preflight; actual upload uses PyPA's action.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request


def request(url: str, *, token: str | None = None, body: bytes | None = None) -> dict:
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = "bearer " + token
    if body is not None:
        headers["Content-Type"] = "application/json"
    try:
        with urllib.request.urlopen(
            urllib.request.Request(url, data=body, headers=headers), timeout=20
        ) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"PyPI OIDC preflight failed with HTTP {exc.code}") from None


def main() -> None:
    audience = request("https://pypi.org/_/oidc/audience")["audience"]
    assert audience == "pypi", "unexpected PyPI OIDC audience"
    identity_url = os.environ["ACTIONS_ID_TOKEN_REQUEST_URL"]
    separator = "&" if "?" in identity_url else "?"
    oidc = request(
        identity_url + separator + "audience=pypi",
        token=os.environ["ACTIONS_ID_TOKEN_REQUEST_TOKEN"],
    )["value"]
    assert isinstance(oidc, str) and oidc
    print("::add-mask::" + oidc, flush=True)
    minted = request(
        "https://pypi.org/_/oidc/mint-token", body=json.dumps({"token": oidc}).encode()
    )["token"]
    assert isinstance(minted, str) and minted.startswith("pypi-")
    print("::add-mask::" + minted, flush=True)
    print(
        "PASS: this workflow/environment OIDC identity minted a short-lived PyPI token; no upload attempted"
    )


if __name__ == "__main__":
    main()
