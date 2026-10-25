"""Download and pin official documentation sources for APILens.

Run without ``--download`` to review the pinned releases. Pass ``--download``
only after that list has been approved.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
import subprocess
import tempfile
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

try:
    import sphobjinv
except ImportError as error:  # pragma: no cover - exercised only in an incomplete environment
    raise SystemExit(
        "sphobjinv is required. Install the pinned project dependencies before downloading."
    ) from error


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
TEXT_SUFFIXES = {".rst", ".md", ".txt"}


@dataclass(frozen=True)
class Source:
    """A version-pinned upstream documentation source."""

    library: str
    repository: str
    tag: str
    docs_subdir: str
    conf_path: str
    license_path: str
    inventory_url: str


SOURCES = (
    Source(
        "numpy",
        "https://github.com/numpy/numpy.git",
        "v2.5.4",
        "doc/source",
        "doc/source/conf.py",
        "LICENSE.txt",
        "https://numpy.org/doc/2.5/objects.inv",
    ),
    Source(
        "scipy",
        "https://github.com/scipy/scipy.git",
        "v1.18.0",
        "doc/source",
        "doc/source/conf.py",
        "LICENSE.txt",
        "https://docs.scipy.org/doc/scipy-1.18.0/objects.inv",
    ),
    Source(
        "pandas",
        "https://github.com/pandas-dev/pandas.git",
        "v3.1.0",
        "doc/source",
        "doc/source/conf.py",
        "LICENSE",
        "https://pandas.pydata.org/pandas-docs/version/3.1/objects.inv",
    ),
    Source(
        "scikit-learn",
        "https://github.com/scikit-learn/scikit-learn.git",
        "1.9.1",
        "doc",
        "doc/conf.py",
        "COPYING",
        "https://scikit-learn.org/1.9/objects.inv",
    ),
    Source(
        "matplotlib",
        "https://github.com/matplotlib/matplotlib.git",
        "v3.11.2",
        "doc",
        "doc/conf.py",
        "LICENSE/LICENSE",
        "https://matplotlib.org/3.11.2/objects.inv",
    ),
    Source(
        "pytorch",
        "https://github.com/pytorch/pytorch.git",
        "v2.14.1",
        "docs/source",
        "docs/source/conf.py",
        "LICENSE",
        "https://docs.pytorch.org/docs/2.14/objects.inv",
    ),
)
PYTHON_INVENTORY = (
    "python",
    "3.15",
    "https://docs.python.org/3.15/objects.inv",
)


def run(command: list[str], cwd: Path | None = None) -> str:
    """Run a command and return stripped standard output."""
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
    )
    return completed.stdout.strip()


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_if_changed(source: Path, destination: Path) -> bool:
    """Copy a file only when its content differs; return whether it changed."""
    if destination.exists() and sha256(source) == sha256(destination):
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return True


def download_if_changed(url: str, destination: Path) -> bool:
    """Download a URL and replace the destination only when bytes differ."""
    request = urllib.request.Request(url, headers={"User-Agent": "APILens/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
    incoming_hash = hashlib.sha256(payload).hexdigest()
    if destination.exists() and sha256(destination) == incoming_hash:
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    return True


def clone_sparse(source: Source, destination: Path) -> tuple[Path, str]:
    """Create a shallow, blob-filtered sparse clone and return its commit."""
    checkout = destination / source.library
    run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--filter=blob:none",
            "--sparse",
            "--branch",
            source.tag,
            source.repository,
            str(checkout),
        ]
    )
    sparse_directories = [source.docs_subdir]
    license_parent = Path(source.license_path).parent.as_posix()
    if license_parent != ".":
        sparse_directories.append(license_parent)
    run(["git", "sparse-checkout", "set", *sparse_directories], cwd=checkout)
    return checkout, run(["git", "rev-parse", "HEAD"], cwd=checkout)


def static_default_role(conf_path: Path) -> str | None:
    """Read a literal ``default_role`` assignment without executing conf.py."""
    tree = ast.parse(conf_path.read_text(encoding="utf-8"), filename=str(conf_path))
    for node in tree.body:
        value: ast.expr | None = None
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "default_role" for target in node.targets
        ):
            value = node.value
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "default_role"
        ):
            value = node.value
        if value is not None:
            try:
                result = ast.literal_eval(value)
            except (TypeError, ValueError):
                return None
            return result if isinstance(result, str) else None
    return None


def expected_library_files(checkout: Path, source: Source) -> list[Path]:
    """Return allowed documentation text plus the required conf and license."""
    docs_root = checkout / source.docs_subdir
    files = [path for path in docs_root.rglob("*") if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES]
    for required in (checkout / source.conf_path, checkout / source.license_path):
        if not required.is_file():
            raise FileNotFoundError(f"required upstream file missing: {required.relative_to(checkout)}")
        files.append(required)
    return sorted(set(files), key=lambda path: path.relative_to(checkout).as_posix())


def sync_library(checkout: Path, source: Source) -> tuple[int, int, int]:
    """Synchronize one library and return total, changed, and removed counts."""
    output_root = RAW_DIR / source.library
    expected_destinations: set[Path] = set()
    changed = 0
    for upstream in expected_library_files(checkout, source):
        relative = upstream.relative_to(checkout)
        destination = output_root / ("LICENSE" if relative.as_posix() == source.license_path else relative)
        expected_destinations.add(destination.resolve())
        changed += int(copy_if_changed(upstream, destination))

    removed = 0
    if output_root.exists():
        for existing in sorted(output_root.rglob("*"), reverse=True):
            if existing.is_file() and existing.resolve() not in expected_destinations:
                existing.unlink()
                removed += 1
            elif existing.is_dir() and not any(existing.iterdir()):
                existing.rmdir()
    return len(expected_destinations), changed, removed


def inventory_object_count(path: Path) -> int:
    """Load a Sphinx inventory and return its object count."""
    inventory = sphobjinv.Inventory(str(path))
    return len(inventory.objects)


def iter_raw_files() -> Iterable[Path]:
    """Yield checksum-controlled raw files in stable path order."""
    yield from sorted(
        (path for path in RAW_DIR.rglob("*") if path.is_file() and path.name != "checksums.txt"),
        key=lambda path: path.relative_to(ROOT).as_posix(),
    )


def write_checksums() -> None:
    """Write GNU sha256sum-compatible checksums, excluding the manifest itself."""
    lines = [f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}" for path in iter_raw_files()]
    (RAW_DIR / "checksums.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def show_pins() -> None:
    """Print the release review gate before any network writes."""
    print("Pinned releases (review before using --download):")
    print(f"{'library':<15} {'tag':<10} inventory")
    for source in SOURCES:
        print(f"{source.library:<15} {source.tag:<10} {source.inventory_url}")
    name, version, url = PYTHON_INVENTORY
    print(f"{name:<15} {version:<10} {url}")


def download_all() -> None:
    """Download sources and inventories and write reproducibility metadata."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    metadata: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []

    with tempfile.TemporaryDirectory(prefix="apilens-download-") as temporary:
        temporary_root = Path(temporary)
        for source in SOURCES:
            print(f"\n[{source.library}] sparse cloning {source.tag}...", flush=True)
            checkout, commit = clone_sparse(source, temporary_root)
            total, changed, removed = sync_library(checkout, source)
            conf_path = checkout / source.conf_path
            default_role = static_default_role(conf_path)

            inventory_path = RAW_DIR / "inventories" / f"{source.library}.inv"
            inventory_changed = download_if_changed(source.inventory_url, inventory_path)
            object_count = inventory_object_count(inventory_path)
            library_bytes = sum(path.stat().st_size for path in (RAW_DIR / source.library).rglob("*") if path.is_file())

            entry = asdict(source)
            entry.update(
                {
                    "commit": commit,
                    "default_role": default_role,
                    "inventory_objects": object_count,
                }
            )
            metadata.append(entry)
            summaries.append(
                {
                    "library": source.library,
                    "files": total,
                    "mb": library_bytes / (1024 * 1024),
                    "inventory_objects": object_count,
                    "changed": changed + int(inventory_changed),
                    "removed": removed,
                }
            )

    python_name, python_version, python_url = PYTHON_INVENTORY
    python_path = RAW_DIR / "inventories" / f"{python_name}.inv"
    python_changed = download_if_changed(python_url, python_path)
    python_objects = inventory_object_count(python_path)
    metadata.append(
        {
            "library": python_name,
            "tag": python_version,
            "repository": None,
            "commit": None,
            "docs_subdir": None,
            "conf_path": None,
            "license_path": None,
            "inventory_url": python_url,
            "default_role": None,
            "inventory_objects": python_objects,
        }
    )
    summaries.append(
        {
            "library": python_name,
            "files": 0,
            "mb": 0.0,
            "inventory_objects": python_objects,
            "changed": int(python_changed),
            "removed": 0,
        }
    )

    sources_path = RAW_DIR / "SOURCES.json"
    sources_payload = {"schema_version": 1, "sources": metadata}
    sources_path.write_text(
        json.dumps(sources_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    write_checksums()

    print("\nSummary")
    print(f"{'library':<15} {'files':>7} {'MB':>10} {'objects':>10} {'changed':>9} {'removed':>9}")
    for summary in summaries:
        print(
            f"{summary['library']:<15} {summary['files']:>7} {summary['mb']:>10.2f} "
            f"{summary['inventory_objects']:>10} {summary['changed']:>9} {summary['removed']:>9}"
        )


def main() -> None:
    """Parse arguments and run the explicit review/download workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--download",
        action="store_true",
        help="download the reviewed pins into data/raw (default: list pins only)",
    )
    args = parser.parse_args()
    show_pins()
    if args.download:
        download_all()
    else:
        print("\nNothing downloaded. Re-run with --download after human approval.")


if __name__ == "__main__":
    main()
