"""Verify or recover selected development source without importing it."""
import argparse
import hashlib
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", help="Snapshot group listed in development-snapshots/README.md")
    parser.add_argument("--out", type=Path, help="New directory for selected full files")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "development-snapshots/MANIFEST.json").read_text())
    selected = [x for x in manifest["files"] if args.group is None or x["group"] == args.group]
    if not selected:
        parser.error("Unknown or empty group")
    if args.out and not args.group:
        parser.error("--out requires one --group so alternative files cannot collide")
    for item in selected:
        rel = Path(item["path"])
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("Unsafe manifest path")
        path = root / rel
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError("Snapshot link escapes repository")
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("Snapshot hash mismatch: " + item["path"])
    if args.out:
        out = args.out.resolve()
        out.mkdir(parents=False, exist_ok=False)
        for item in selected:
            rel = Path(item["recovery_path"])
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError("Unsafe recovery path")
            dest = out / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("xb") as handle:
                handle.write((root / item["path"]).read_bytes())
    print(json.dumps({"verified_files": len(selected), "group": args.group, "recovered": bool(args.out)}))

if __name__ == "__main__":
    main()
