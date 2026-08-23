"""Unit tests for the BAM-facing helpers in cuban/utils.py."""

from pathlib import Path

import numpy as np
import pysam
import pytest

from cuban import utils


# ---------------------------------------------------------------------------
# compute_aln_matrix - duplicate-read row alignment regression
# ---------------------------------------------------------------------------

def test_compute_aln_matrix_excludes_duplicates_and_keeps_rows_aligned(test_bam, del_start):
    """REGRESSION: duplicate-flagged reads must be dropped, and every remaining
    row must stay in lockstep across the matrix and every aux_dict list keyed
    by row index. test.bam plants 5 duplicate-flagged reads around 19,000-19,262
    (just left of the DEL_START breakpoint at 20,000), so a window spanning
    that region is what actually exercises the exclusion.
    """
    bam = pysam.AlignmentFile(str(test_bam), "rb")
    try:
        aln_matrix, aux = utils.compute_aln_matrix(bam, "chr1", 18_900, del_start + 100)
    finally:
        bam.close()

    n_rows = aln_matrix.shape[0]
    assert n_rows > 0
    assert len(aux["name"]) == n_rows
    assert len(aux["haplotag_idx"]) == n_rows
    assert not any(name.startswith("dup") for name in aux["name"])

    for key in ("split_idx", "low_mapq_idx", "reverse_idx",
                "discordant_idx_ff", "discordant_idx_rr",
                "discordant_idx_rf", "discordant_idx_tx"):
        assert all(0 <= i < n_rows for i in aux[key])


# ---------------------------------------------------------------------------
# compute_aln_matrix - downsampling
# ---------------------------------------------------------------------------

def test_compute_aln_matrix_max_reads_caps_row_count(test_bam, contig):
    bam = pysam.AlignmentFile(str(test_bam), "rb")
    try:
        aln_matrix, aux = utils.compute_aln_matrix(
            bam, contig, 0, 50_000, max_reads=20, downsample="early_stop")
    finally:
        bam.close()
    assert aln_matrix.shape[0] == 20
    assert len(aux["name"]) == 20


def test_compute_aln_matrix_random_downsample_is_deterministic(test_bam, contig):
    def _run():
        bam = pysam.AlignmentFile(str(test_bam), "rb")
        try:
            return utils.compute_aln_matrix(
                bam, contig, 0, 50_000, max_reads=15, downsample="random")
        finally:
            bam.close()

    _, aux1 = _run()
    _, aux2 = _run()
    assert len(aux1["name"]) == 15
    assert aux1["name"] == aux2["name"]  # same seeded draw both times


# ---------------------------------------------------------------------------
# reorder_by_hp
# ---------------------------------------------------------------------------

def test_reorder_by_hp_bands_padding_and_index_remap(test_hp_bam, contig, del_start, del_end):
    bam = pysam.AlignmentFile(str(test_hp_bam), "rb")
    try:
        aln_left, aux_left = utils.compute_aln_matrix(bam, contig, del_start - 100, del_start + 100)
        aln_right, aux_right = utils.compute_aln_matrix(bam, contig, del_end - 100, del_end + 100)
    finally:
        bam.close()

    # Mirrors the pre-padding/trim-to-raw dance visualize.plot_cigar performs
    # before handing matrices to reorder_by_hp.
    aln_left, aln_right = utils.pad_alignment_matrices(aln_left, aln_right)
    aln_left = aln_left[:len(aux_left["haplotag_idx"])]
    aln_right = aln_right[:len(aux_right["haplotag_idx"])]

    new_left, new_aux_left, new_right, new_aux_right, bands = utils.reorder_by_hp(
        aln_left, aux_left, aln_right, aux_right)

    # Band order is HP1, HP2, untagged - whichever of those are actually present.
    hp_values = [b[0] for b in bands]
    assert hp_values, "test_hp.bam should contain at least one HP group"
    canonical_order = [hp for hp in (1, 2, -1) if hp in set(hp_values)]
    assert hp_values == canonical_order

    # Both sides end up with identical row counts.
    assert new_left.shape[0] == new_right.shape[0]
    assert new_left.shape[0] == len(new_aux_left["name"]) == len(new_aux_left["haplotag_idx"])
    assert new_right.shape[0] == len(new_aux_right["name"]) == len(new_aux_right["haplotag_idx"])

    n_rows = new_left.shape[0]
    for aux in (new_aux_left, new_aux_right):
        # Sentinel pad names must be unique so the name-merge used for
        # read-connection lines can never accidentally pair two padding rows.
        pad_names = [n for n in aux["name"] if isinstance(n, str) and n.startswith("__pad")]
        assert len(pad_names) == len(set(pad_names))

        # Remapped index-list keys stay in bounds of the new row count.
        for key in ("split_idx", "low_mapq_idx", "reverse_idx",
                    "discordant_idx_ff", "discordant_idx_rr",
                    "discordant_idx_rf", "discordant_idx_tx"):
            assert all(0 <= i < n_rows for i in aux[key])


# ---------------------------------------------------------------------------
# compute_aln_matrix - insertion-at-CIGAR-position-0 masking edge case
# ---------------------------------------------------------------------------

def _write_single_read_bam(path: Path, cigartuples, reference_start, contig="chr1", contig_len=200):
    header = {"HD": {"VN": "1.6", "SO": "coordinate"}, "SQ": [{"SN": contig, "LN": contig_len}]}
    seq_len = sum(length for op, length in cigartuples if op in (0, 1, 4, 7, 8))
    with pysam.AlignmentFile(str(path), "wb", header=header) as bam:
        read = pysam.AlignedSegment()
        read.query_name = "ins0"
        read.reference_id = 0
        read.reference_start = reference_start
        read.mapping_quality = 60
        read.query_sequence = "A" * seq_len
        read.query_qualities = pysam.qualitystring_to_array("I" * seq_len)
        read.cigartuples = cigartuples
        read.flag = 0
        bam.write(read)
    pysam.index(str(path))


def test_compute_aln_matrix_insertion_at_position_zero_does_not_corrupt_last_column(tmp_path):
    """collapse_ins's masking builds `mask[ins_idx[ins_idx > 0] - 1] = 0` to blank
    the column before an insertion run, specifically guarded with `ins_idx > 0` so
    a run starting at CIGAR position 0 never computes index -1. Without that
    guard, numpy's negative-index wraparound (`mask[-1] = 0`) would silently
    zero out the LAST column of the array instead of doing nothing (there is no
    column before position 0). This constructs a read whose CIGAR opens with an
    insertion (2I) and closes with a deletion (1D), so a corrupted last column
    is directly observable: it would read back as the -1/M sentinel instead of
    the deletion op.
    """
    bam_path = tmp_path / "ins0.bam"
    ref_start = 50
    # 2 bases inserted right at CIGAR position 0, 10 matches, 1bp deletion as
    # the very last op.
    cigartuples = [(1, 2), (0, 10), (2, 1)]
    _write_single_read_bam(bam_path, cigartuples, ref_start)

    ref_end = ref_start + 10 + 1  # 10M + 1D advance the reference; I does not
    region_start, region_stop = ref_start, ref_end + 1  # window wide enough that nothing truncates

    bam = pysam.AlignmentFile(str(bam_path), "rb")
    try:
        aln_matrix, aux = utils.compute_aln_matrix(bam, "chr1", region_start, region_stop, collapse_ins=True)
    finally:
        bam.close()

    assert aln_matrix.shape == (1, region_stop - region_start)
    assert len(aux["name"]) == 1
    # The final reference-adjacent column must carry the deletion op (2), not a
    # corrupted sentinel/M value.
    assert aln_matrix[0, -1] == 2
