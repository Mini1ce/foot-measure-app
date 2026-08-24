"""Command-line entry point: ``python -m ml.footdata``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .archive import ArchiveLimits, safe_extract_zip
from .errors import FootDataError
from .importers import import_archive, import_dataset
from .manifest import read_manifest, validate_manifest, write_manifest
from .sources import SOURCES, list_sources
from .split import SplitRatios, split_records


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ml.footdata",
        description="Safe, reproducible tooling for foot segmentation datasets.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("sources", help="List pinned source metadata and acquisition policy")

    extract = commands.add_parser("extract", help="Inspect and safely extract a local ZIP")
    extract.add_argument("archive", type=Path)
    extract.add_argument("destination", type=Path)
    extract.add_argument("--max-total-gib", type=float, default=100.0)

    importer = commands.add_parser("import", help="Import a local FootNet or SynFoot release")
    importer.add_argument("source", choices=sorted(SOURCES))
    input_kind = importer.add_mutually_exclusive_group(required=True)
    input_kind.add_argument("--archive", type=Path)
    input_kind.add_argument("--input-dir", type=Path)
    importer.add_argument("--output-root", type=Path, required=True)
    importer.add_argument(
        "--manifest", type=Path, help="Optional manifest path directly under --output-root"
    )
    importer.add_argument(
        "--lock", type=Path, help="Optional dataset-lock path directly under --output-root"
    )
    importer.add_argument("--group-overrides", type=Path)
    importer.add_argument(
        "--group-regex",
        help="Regex with named group 'subject' or one capture group for real datasets",
    )
    importer.add_argument("--expected-sha256")
    importer.add_argument("--accept-license", action="store_true")

    validate = commands.add_parser("validate", help="Validate manifest structure and leakage")
    validate.add_argument("manifest", type=Path)
    validate.add_argument("--data-root", type=Path)
    validate.add_argument("--verify-hashes", action="store_true")
    validate.add_argument("--strict", action="store_true", help="Treat warnings as a failure")

    split = commands.add_parser("split", help="Assign deterministic, group-safe splits")
    split.add_argument("manifest", type=Path)
    split.add_argument(
        "--output", type=Path, required=True, help="Output directly beside the input manifest"
    )
    split.add_argument(
        "--lock", type=Path, required=True, help="Split lock directly beside the input manifest"
    )
    split.add_argument("--seed", type=int, default=20260822)
    split.add_argument("--train", type=float, default=0.70)
    split.add_argument("--val", type=float, default=0.15)
    split.add_argument("--test", type=float, default=0.15)
    split.add_argument(
        "--allow-unresolved-eval",
        action="store_true",
        help=(
            "Development only: allow unresolved real samples in val/test, mark them "
            "sample-only, and pin this weaker policy in the split lock"
        ),
    )
    return parser


def _print_json(value: object) -> None:
    # Windows shells can expose a legacy cp1252 stream even when paths contain
    # Thai or other Unicode characters. JSON escapes preserve the exact value
    # without crashing after a long import has already completed.
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True))


def _print_error(error: object) -> None:
    message = f"error: {error}"
    encoding = getattr(sys.stderr, "encoding", None) or "utf-8"
    safe_message = message.encode(encoding, errors="backslashreplace").decode(encoding)
    print(safe_message, file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "sources":
            _print_json(list_sources())
            return 0

        if args.command == "extract":
            destination = safe_extract_zip(
                args.archive,
                args.destination,
                ArchiveLimits(max_total_bytes=int(args.max_total_gib * 1024**3)),
            )
            _print_json({"extracted_to": str(destination)})
            return 0

        if args.command == "import":
            options = {
                "manifest_path": args.manifest,
                "lock_path": args.lock,
                "group_overrides": args.group_overrides,
                "group_regex": args.group_regex,
                "license_accepted": args.accept_license,
            }
            if args.archive:
                result = import_archive(
                    args.source,
                    args.archive,
                    args.output_root,
                    expected_sha256=args.expected_sha256,
                    **options,
                )
            else:
                if args.expected_sha256:
                    raise FootDataError("--expected-sha256 only applies to --archive")
                result = import_dataset(
                    args.source,
                    args.input_dir,
                    args.output_root,
                    **options,
                )
            _print_json(
                {
                    "records": len(result.records),
                    "valid_records": result.valid_count,
                    "manifest": str(result.manifest_path),
                    "lock": str(result.lock_path),
                    "errors": result.errors,
                    "warnings": result.warnings,
                }
            )
            return 1 if result.errors else 0

        if args.command == "validate":
            if args.verify_hashes and args.data_root is None:
                raise FootDataError("--verify-hashes requires --data-root")
            report = validate_manifest(
                read_manifest(args.manifest),
                data_root=args.data_root,
                verify_hashes=args.verify_hashes,
            )
            _print_json(report)
            return 1 if not report["valid"] or (args.strict and report["warnings"]) else 0

        if args.command == "split":
            output_path = args.output.resolve()
            lock_path = args.lock.resolve()
            manifest_path = args.manifest.resolve()
            if output_path == lock_path:
                raise FootDataError("Split output and lock must use different paths")
            if manifest_path == lock_path:
                raise FootDataError("Input manifest and split lock must use different paths")
            if output_path.parent != manifest_path.parent:
                raise FootDataError(
                    "Split output must be directly beside the input manifest so relative "
                    f"asset paths keep the same base: {output_path}"
                )
            if lock_path.parent != manifest_path.parent:
                raise FootDataError(
                    "Split lock must be directly beside the input manifest: "
                    f"{lock_path}"
                )
            records, _, summary = split_records(
                read_manifest(args.manifest),
                seed=args.seed,
                ratios=SplitRatios(args.train, args.val, args.test),
                lock_path=args.lock,
                allow_unresolved_eval=args.allow_unresolved_eval,
            )
            write_manifest(records, args.output)
            _print_json(
                {
                    "manifest": str(args.output),
                    "lock": str(args.lock),
                    "summary": summary,
                }
            )
            return 0
    except (FootDataError, OSError, UnicodeError, json.JSONDecodeError) as error:
        _print_error(error)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
