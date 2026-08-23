"""Tests for cuban/coverage.py: the mosdepth-backed coverage/baseline resolver
and its pysam fallback."""

import pytest

from cuban import coverage


def test_get_coverage_mosdepth_matches_expected_depth(test_bam, cache_dir, contig, del_start, del_end):
    """test.bam is built for ~12x coverage outside the DEL and 0x inside it
    (see tests/make_test_data.py); a window straddling both must show that
    contrast through the mosdepth path."""
    window_start, window_stop = del_start - 5_000, del_end + 5_000
    cov, cov_minq = coverage.get_coverage(str(test_bam), contig, window_start, window_stop,
                                          cache_dir=str(cache_dir))

    offset = cov[1].to_numpy()
    depths = cov[2].to_numpy()
    # Stay 500bp clear of both breakpoints: the split reads planted there
    # deliberately dangle a few soft-clipped bases across the boundary, so the
    # true 0x/~12x contrast only holds once you're clear of that fringe.
    inside = (offset >= (del_start - window_start + 500)) & (offset < (del_end - window_start - 500))
    outside = (offset < (del_start - window_start - 500)) | (offset >= (del_end - window_start + 500))

    assert depths[inside].mean() == 0
    mean_outside = depths[outside].mean()
    assert 8 <= mean_outside <= 16  # nominal 12x with slack for edge fragmentation

    # minq=20 pass drops the 5 planted low-MAPQ (mapq=5) reads, so it can only
    # be <= the unfiltered pass, never higher.
    assert cov_minq[2].to_numpy().mean() <= mean_outside


def test_get_coverage_pysam_fallback_warns_and_matches_shape(test_bam, cache_dir, contig,
                                                              del_start, monkeypatch):
    window_start, window_stop = del_start - 500, del_start + 500
    cov_mos, covq_mos = coverage.get_coverage(str(test_bam), contig, window_start, window_stop,
                                              cache_dir=str(cache_dir))

    monkeypatch.setattr(coverage, "_find_mosdepth", lambda: None)
    with pytest.warns(UserWarning, match="mosdepth"):
        cov_py, covq_py = coverage.get_coverage(str(test_bam), contig, window_start, window_stop,
                                                cache_dir=str(cache_dir))

    assert cov_py.shape == cov_mos.shape
    assert covq_py.shape == covq_mos.shape
    # pysam's pileup and mosdepth can disagree slightly at the margins (overlap/
    # orphan handling), but over this window the means must land close together.
    assert abs(cov_py[2].mean() - cov_mos[2].mean()) < 2
    assert abs(covq_py[2].mean() - covq_mos[2].mean()) < 2


def test_get_coverage_bin_size_bins_rows(test_large_bam, cache_dir, large_contig):
    bin_size = 1_000
    start, stop = 1, 50_000  # exact multiple of bin_size, 0-based-aligned
    cov, cov_minq = coverage.get_coverage(str(test_large_bam), large_contig, start, stop,
                                          cache_dir=str(cache_dir), bin_size=bin_size)

    expected_bins = (stop - start + 1) / bin_size
    assert abs(len(cov) - expected_bins) <= 1
    assert abs(len(cov_minq) - expected_bins) <= 1
    assert len(cov) < (stop - start + 1)  # actually binned down, not per-base


def test_get_baseline_auto_returns_positive_float(test_bam, cache_dir, contig):
    baseline = coverage.get_baseline(str(test_bam), contig, cache_dir=str(cache_dir))
    assert isinstance(baseline, float)
    assert baseline > 0


def test_get_coverage_cache_reuse_skips_subprocess_on_second_call(test_bam, cache_dir, contig,
                                                                   del_start, monkeypatch):
    window_start, window_stop = del_start - 500, del_start + 500
    # Prime the cache with a real mosdepth run.
    coverage.get_coverage(str(test_bam), contig, window_start, window_stop, cache_dir=str(cache_dir))

    def _boom(*args, **kwargs):
        raise AssertionError("subprocess.run must not be called on a mosdepth cache hit")

    monkeypatch.setattr(coverage.subprocess, "run", _boom)

    # Identical (bam, chrom, minq, bin_size) -> cache hit, no subprocess call.
    coverage.get_coverage(str(test_bam), contig, window_start, window_stop, cache_dir=str(cache_dir))
