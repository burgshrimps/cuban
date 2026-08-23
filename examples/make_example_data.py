"""Regenerate the example fixtures shipped in examples/data/ for the README
quickstart.

Thin wrapper around tests/make_test_data.py: same synthetic-BAM generator,
same fixed seed, just written under examples/data/ with README-friendly
filenames (example.bam instead of test.bam, etc) so users can regenerate the
quickstart data for themselves without digging through tests/.

Usage: python examples/make_example_data.py [outdir]   (default: examples/data)
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# tests/ has no __init__.py, so import make_test_data by adding tests/ itself
# to sys.path rather than via a package path.
sys.path.insert(0, str(REPO_ROOT / "tests"))
import make_test_data  # noqa: E402


def main(outdir="examples/data"):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    bam = make_test_data.make_bam(outdir, name="example.bam")
    hp_bam = make_test_data.make_bam(outdir, hp=True, name="example_hp.bam")
    rep = make_test_data.make_repeats(outdir)

    vcf = make_test_data.make_vcf(outdir)
    example_vcf = outdir / "example.vcf"
    vcf.replace(example_vcf)

    print(f"wrote {bam}, {hp_bam}, {rep}, {example_vcf}")


if __name__ == "__main__":
    main(*sys.argv[1:2])
