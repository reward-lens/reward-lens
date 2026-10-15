"""Offline command for writing a canonical E1 assay report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .build import build, persist_assay_bundle
from .counts import counts


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m reward_lens.examples.e1")
    parser.add_argument(
        "--source-root",
        type=Path,
        help="E1 public artifact root containing e1_analysis and .e1-cache",
    )
    parser.add_argument(
        "--derivation-receipt",
        type=Path,
        help="caller-supplied receipt.json; checks output/analysis hashes and counts, not upstream archives",
    )
    parser.add_argument(
        "--output", type=Path, help="directory for assay.json, manifest.json, and report.html"
    )
    parser.add_argument("--field")
    args = parser.parse_args()
    try:
        assay = build(source_root=args.source_root, derivation_receipt=args.derivation_receipt)
    except FileNotFoundError as error:
        parser.error(str(error))
    measured = counts(assay).as_dict()
    if args.field:
        if args.field not in measured:
            parser.error(f"unknown E1 count: {args.field}")
        print(measured[args.field])
    elif args.output:
        print(persist_assay_bundle(assay, args.output).assay_json)
    else:
        import rfc8785

        sys.stdout.buffer.write(rfc8785.dumps(assay.contract_assay().to_dict()) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
