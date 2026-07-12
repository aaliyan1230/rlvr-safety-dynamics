from __future__ import annotations

import argparse
from pathlib import Path

from ..provenance import verify_manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args(argv)
    manifest = verify_manifest(args.manifest)
    print(f"Verified {len(manifest['files'])} artifacts from {args.manifest}")


if __name__ == "__main__":
    main()
