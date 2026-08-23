"""Shared fixtures for the cuban test suite.

All fixtures here are read-only or write exclusively into a tmp_path-derived
scratch directory - no test may write into the user's real ~/.cuban.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "tests" / "data"

# tests/ has no __init__.py (it's not meant to be a package), so make_test_data
# is imported by adding tests/ itself to sys.path rather than via a package path.
sys.path.insert(0, str(REPO_ROOT / "tests"))
import make_test_data  # noqa: E402

REQUIRED_DATA_FILES = [
    "test.bam", "test.bam.bai",
    "test_hp.bam", "test_hp.bam.bai",
    "test_large.bam", "test_large.bam.bai",
    "repeats.tsv", "test.vcf",
]


@pytest.fixture(scope="session")
def data_dir():
    """tests/data, regenerated on the fly if any expected file is missing."""
    if not all((DATA_DIR / f).is_file() for f in REQUIRED_DATA_FILES):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        make_test_data.main(str(DATA_DIR))
    return DATA_DIR


@pytest.fixture
def test_bam(data_dir):
    return data_dir / "test.bam"


@pytest.fixture
def test_hp_bam(data_dir):
    return data_dir / "test_hp.bam"


@pytest.fixture
def test_large_bam(data_dir):
    return data_dir / "test_large.bam"


@pytest.fixture
def repeats_tsv(data_dir):
    return data_dir / "repeats.tsv"


@pytest.fixture
def test_vcf(data_dir):
    return data_dir / "test.vcf"


# Geometry constants shared with the generator, re-exported as fixtures so
# tests never hardcode a second copy of the deletion coordinates.
@pytest.fixture
def contig():
    return make_test_data.CONTIG


@pytest.fixture
def del_start():
    return make_test_data.DEL_START


@pytest.fixture
def del_end():
    return make_test_data.DEL_END


@pytest.fixture
def large_contig():
    return make_test_data.LARGE_CONTIG


@pytest.fixture
def cache_dir(tmp_path):
    """A fresh, isolated --cache-dir / get_coverage(cache_dir=...) target per test,
    so mosdepth cache state never leaks between tests or touches ~/.cuban."""
    d = tmp_path / "cuban-cache"
    d.mkdir()
    return d


@pytest.fixture(scope="session")
def shared_cache_dir(tmp_path_factory):
    """A session-scoped cache dir for CLI-driving tests: reused across tests so
    repeated mosdepth runs on the same fixture BAMs hit the cache instead of
    re-running the binary, keeping the suite fast. Never the real ~/.cuban."""
    return tmp_path_factory.mktemp("cuban-cli-cache")
