"""Coverage backend for cuban.

Resolves per-sample coverage (and per-chromosome baseline coverage) using,
in order of preference:

  (a) precomputed mosdepth output directory (``coverage_dir``), read directly
      via tabix, dev-repo layout: mosdepth.q0.per-base.bed.gz,
      mosdepth.q{minq}.per-base.bed.gz, mosdepth.q{minq}.mosdepth.summary.txt
  (b) the ``mosdepth`` binary on PATH, run restricted to the chromosome in
      question and cached under ``cache_dir``
  (c) the existing pysam-based fallback (:func:`cuban.utils.compute_cov_df` /
      :func:`cuban.utils.compute_baseline_cov`), with a warning that mosdepth
      was not found and pysam is slower.
"""

import hashlib
import os
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pysam

from .utils import compute_cov_df, compute_baseline_cov

DEFAULT_MINQ = 20

_MOSDEPTH_MISSING_MSG = (
    "mosdepth was not found on PATH; falling back to pysam for coverage "
    "computation, which is slower."
)


def _find_mosdepth():
    """ Locates the mosdepth binary: next to the running interpreter first
    (so the console script works without the environment on PATH), then PATH. """
    candidate = Path(sys.executable).parent / 'mosdepth'
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate)
    return shutil.which('mosdepth')


def _cache_root(cache_dir=None):
    """ Resolves the root directory under which mosdepth outputs are cached. """
    if cache_dir is not None:
        return Path(cache_dir)
    env_dir = os.environ.get('CUBAN_DATA_DIR')
    if env_dir:
        return Path(env_dir) / 'coverage'
    return Path.home() / '.cuban' / 'coverage'


def _bam_cache_key(bam_path):
    """ Stable cache key for a BAM file: sha1 of abspath+mtime+size, first 16 hex chars. """
    st = os.stat(bam_path)
    raw = f'{os.path.abspath(bam_path)}:{st.st_mtime}:{st.st_size}'
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def _mosdepth_prefix(cache_dir, bam_path, chrom, minq):
    """ Cache-dir prefix (without suffixes) that mosdepth output for (bam, chrom, minq) lives under. """
    root = _cache_root(cache_dir) / _bam_cache_key(bam_path)
    root.mkdir(parents=True, exist_ok=True)
    return root / f'{chrom}.q{minq}'


def _run_mosdepth(mosdepth_bin, prefix, bam_path, chrom, minq):
    """ Runs mosdepth restricted to `chrom` into `prefix`, unless cached outputs already exist. """
    per_base = Path(f'{prefix}.per-base.bed.gz')
    summary = Path(f'{prefix}.mosdepth.summary.txt')
    if per_base.exists() and summary.exists():
        return  # cache hit, reuse silently

    print(f'[cuban] running mosdepth on {bam_path} ({chrom})...', file=sys.stderr)
    cmd = [mosdepth_bin, '--chrom', chrom, '--mapq', str(minq), str(prefix), str(bam_path)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not per_base.exists():
        raise RuntimeError(f'mosdepth did not produce expected output {per_base}')


def _read_mosdepth_region(bed_path, chrom, start, stop):
    """ Per-base depth array for [start, stop] (1-based, inclusive). Length == stop - start + 1.
    mosdepth BED is 0-based half-open and densely tiles the chromosome (every position belongs
    to exactly one row, including zero-depth runs), so a zeros-initialised array filled by
    overlapping rows recovers the depth track exactly.
    """
    n = stop - start + 1
    depths = np.zeros(n, dtype=np.int64)
    # mosdepth emits .csi by default; pysam.tabix_index emits .tbi. Support both.
    csi = bed_path + '.csi'
    index = csi if os.path.exists(csi) else None
    with pysam.TabixFile(bed_path, parser=pysam.asTuple(), index=index) as tbx:
        for row in tbx.fetch(chrom, start - 1, stop):
            rs, re, d = int(row[1]), int(row[2]), int(row[3])
            if d == 0:
                continue
            s = max(rs, start - 1) - (start - 1)
            e = min(re, stop) - (start - 1)
            depths[s:e] = d
    return depths


def _cov_df_from_bed(bed_path, chrom, start, stop):
    """ Builds a coverage DataFrame with the same column semantics as compute_cov_df:
    col 0: chrom, col 1: position (0-based offset from start), col 2: depth. """
    n = stop - start + 1
    depths = _read_mosdepth_region(bed_path, chrom, start, stop)
    pos = np.arange(n, dtype=np.int64)
    chroms = np.full(n, chrom, dtype=object)
    return pd.DataFrame({0: chroms, 1: pos, 2: depths})


def _baseline_from_summary(summary_path, chrom):
    """ Reads the per-chromosome mean depth out of a mosdepth .mosdepth.summary.txt file. """
    df = pd.read_csv(summary_path, sep='\t')
    row = df.loc[df['chrom'] == chrom]
    if row.empty:
        raise ValueError(f"chromosome {chrom!r} not found in mosdepth summary '{summary_path}'")
    return float(row['mean'].iloc[0])


def get_coverage(bam_path, chrom, start, stop, coverage_dir=None, cache_dir=None, minq=DEFAULT_MINQ):
    """ Computes coverage for a region, once unfiltered and once at mapping quality >= minq.

    Returns (cov, cov_minq) with exactly the same DataFrame shape as compute_cov_df.
    """
    if coverage_dir is not None:
        q0_path = os.path.join(coverage_dir, 'mosdepth.q0.per-base.bed.gz')
        qn_path = os.path.join(coverage_dir, f'mosdepth.q{minq}.per-base.bed.gz')
        if not os.path.isfile(q0_path) or not os.path.isfile(qn_path):
            raise FileNotFoundError(
                f"coverage_dir '{coverage_dir}' is missing mosdepth per-base BED(s) for minq={minq} "
                f"(expected 'mosdepth.q0.per-base.bed.gz' and 'mosdepth.q{minq}.per-base.bed.gz')"
            )
        return _cov_df_from_bed(q0_path, chrom, start, stop), _cov_df_from_bed(qn_path, chrom, start, stop)

    mosdepth_bin = _find_mosdepth()
    if mosdepth_bin is not None:
        prefix0 = _mosdepth_prefix(cache_dir, bam_path, chrom, 0)
        prefixn = _mosdepth_prefix(cache_dir, bam_path, chrom, minq)
        _run_mosdepth(mosdepth_bin, prefix0, bam_path, chrom, 0)
        _run_mosdepth(mosdepth_bin, prefixn, bam_path, chrom, minq)
        q0_path = f'{prefix0}.per-base.bed.gz'
        qn_path = f'{prefixn}.per-base.bed.gz'
        return _cov_df_from_bed(q0_path, chrom, start, stop), _cov_df_from_bed(qn_path, chrom, start, stop)

    warnings.warn(_MOSDEPTH_MISSING_MSG)
    return compute_cov_df(bam_path, chrom, start, stop, minq=minq)


def get_baseline(bam_path, chrom, coverage_dir=None, cache_dir=None):
    """ Resolves 'auto' baseline coverage for a chromosome: the whole-chromosome mean depth
    from the q0 mosdepth pass, or the seeded pysam sampling fallback if mosdepth is unavailable. """
    if coverage_dir is not None:
        summary_path = os.path.join(coverage_dir, 'mosdepth.q0.mosdepth.summary.txt')
        if not os.path.isfile(summary_path):
            raise FileNotFoundError(
                f"coverage_dir '{coverage_dir}' is missing 'mosdepth.q0.mosdepth.summary.txt'"
            )
        return _baseline_from_summary(summary_path, chrom)

    mosdepth_bin = _find_mosdepth()
    if mosdepth_bin is not None:
        prefix0 = _mosdepth_prefix(cache_dir, bam_path, chrom, 0)
        _run_mosdepth(mosdepth_bin, prefix0, bam_path, chrom, 0)
        return _baseline_from_summary(f'{prefix0}.mosdepth.summary.txt', chrom)

    warnings.warn(_MOSDEPTH_MISSING_MSG)
    return compute_baseline_cov(bam_path, chrom)
