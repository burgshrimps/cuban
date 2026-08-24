"""Locating, loading and fetching the RepeatMasker annotation."""

import gzip
import os
import shutil
import sys
import urllib.request
import warnings
from pathlib import Path

import pandas as pd

REPEATS_FILENAME = "hg38_repeatmasker.4col.tsv.gz"
REPEATS_URL = (
    "https://github.com/burgshrimps/cuban/releases/latest/download/"
    + REPEATS_FILENAME
)
REPEATS_COLUMNS = ["genoName", "genoStart", "genoEnd", "repClass"]
REPEATS_DTYPES = {
    "genoName": "category",
    "genoStart": "int32",
    "genoEnd": "int32",
    "repClass": "category",
}


def _repo_root():
    """The repo checkout the package runs from, or None for a site-packages install."""
    root = Path(__file__).resolve().parent.parent
    if (root / "pyproject.toml").is_file() and os.access(root, os.W_OK):
        return root
    return None


def annot_dir():
    """Where the repeat annotation is downloaded to: $CUBAN_DATA_DIR if set,
    annot/ in the repo root when running from a checkout, ~/.cuban otherwise."""
    env_dir = os.environ.get("CUBAN_DATA_DIR")
    if env_dir:
        return Path(env_dir)
    root = _repo_root()
    if root is not None:
        return root / "annot"
    return Path.home() / ".cuban"


def default_repeats_path():
    """Return the first existing default location, or None."""
    repo_resources = Path(__file__).resolve().parent.parent / "resources"
    candidates = [
        annot_dir() / REPEATS_FILENAME,
        Path.home() / ".cuban" / REPEATS_FILENAME,
        repo_resources / REPEATS_FILENAME,
        repo_resources / "hg38_repeatmasker.tsv",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def empty_repeats():
    return pd.DataFrame({c: pd.Series(dtype=d) for c, d in REPEATS_DTYPES.items()})


def load_repeats(path=None, auto_fetch=True):
    """Load the repeat annotation as a DataFrame.

    With path=None, searches the default locations; if nothing is found,
    the hg38 table is downloaded automatically (one time, ~40 MB). If the
    download is impossible, warns and returns an empty table (the repeat
    track renders empty).
    """
    if path is None:
        path = default_repeats_path()
        if path is None and auto_fetch:
            print(
                "[cuban] no repeat annotation found; downloading the hg38 "
                "table (~40 MB, one time)...", file=sys.stderr,
            )
            try:
                path = fetch_repeats(quiet=True)
            except SystemExit as e:
                warnings.warn(
                    f"automatic download failed ({e}); the repeat track will "
                    "be empty. Run 'cuban-fetch-repeats' once you are online, "
                    "or pass --repeats /path/to/file.tsv[.gz]."
                )
                return empty_repeats()
        elif path is None:
            return empty_repeats()
    if not os.path.isfile(path):
        raise SystemExit(f"repeats file not found: {path}")
    if os.path.getsize(path) > 1_000_000:
        print(f"[cuban] loading repeat annotation from {path}...", file=sys.stderr)
    try:
        return pd.read_csv(path, sep="\t", usecols=REPEATS_COLUMNS,
                           dtype=REPEATS_DTYPES)
    except ValueError as e:
        raise SystemExit(
            f"repeats file {path} is missing required columns "
            f"{REPEATS_COLUMNS} (UCSC rmsk names): {e}"
        ) from e


def fetch_repeats(dest=None, url=None, quiet=False):
    """Download the repeat annotation to dest (default: the annot directory)."""
    url = url or REPEATS_URL
    dest = Path(dest) if dest else annot_dir() / REPEATS_FILENAME
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    if not quiet:
        print(f"downloading {url}\n        -> {dest}", file=sys.stderr)
    try:
        with urllib.request.urlopen(url) as resp, open(tmp, "wb") as out:
            shutil.copyfileobj(resp, out)
    except OSError as e:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"download failed: {e}") from e
    # Integrity check: must be readable gzip with the expected header.
    try:
        with gzip.open(tmp, "rt") as fh:
            header = fh.readline().split()
        if header != REPEATS_COLUMNS:
            raise ValueError(f"unexpected header: {header}")
    except (OSError, ValueError) as e:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"downloaded file failed validation: {e}") from e
    tmp.replace(dest)
    if not quiet:
        print(f"done ({dest.stat().st_size / 1e6:.0f} MB)", file=sys.stderr)
    return dest


def fetch_main(argv=None):
    import argparse

    p = argparse.ArgumentParser(
        prog="cuban-fetch-repeats",
        description="Download the hg38 RepeatMasker annotation used by cuban "
                    f"(~40 MB) to {annot_dir()}/ (override with --dest or the "
                    "CUBAN_DATA_DIR environment variable).",
    )
    p.add_argument("--dest", help="write the file to this path instead")
    p.add_argument("--url", default=REPEATS_URL, help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    fetch_repeats(dest=args.dest, url=args.url)
