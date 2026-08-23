# Architecture

## Data Flow

```
SV definition (type, chrom, start, end) + BAM files
  │
  ▼
gather_data()                          [visualize.py]
  ├── compute_aln_matrix()             [utils.py]    → CIGAR integer matrix (reads × positions)
  ├── pad_alignment_matrices()         [utils.py]    → equalize left/right matrix heights
  ├── get_coverage()                   [coverage.py] → depth per position (all reads + MAPQ≥20)
  ├── compute_rep_df()                 [utils.py]    → repeat elements in region
  └── compute_isize_orientation_dict() [utils.py]    → insert size outliers + discordant orientations
  │
  ▼
cuban() or cuban_bnd()                 [visualize.py]
  ├── plot_cov()      — coverage line plot with baseline reference
  ├── plot_rep()      — repeat annotations as colored horizontal lines
  ├── plot_isize()    — insert size outlier histograms
  ├── plot_orient()   — discordant orientation histograms
  ├── plot_cigar()    — CIGAR alignment heatmap
  ├── add_splitread_overlay()  — hatch marks on split reads
  ├── add_disco_overlay()      — hatch marks on discordant pairs
  └── add_mapq_overlay()       — grey overlay on low MAPQ reads
  │
  ▼
Publication-quality PNG (matplotlib figure)
```

## Sample Dict Format

The primary API contract. Passed to `cuban()` and `cuban_bnd()`:

```python
samples = {
    'sample_name': {
        'family_status': 'index' | 'mother' | 'father' | 'control',
        'disease_status': 'affected' | 'unaffected',
        'technology': 'ill' | 'pb',        # short-read or long-read
        'bam_name': '/path/to/file.bam',   # must be indexed (.bai)
        'baseline_cov': 32.5,              # or 'auto' (derived from the BAM)
        # optional: directory of precomputed mosdepth output for this sample
        # 'coverage_dir': '/path/to/mosdepth/output',
    }
}
```

With `baseline_cov='auto'`, the per-chromosome mean coverage is derived from
the sample's own BAM at render time (see coverage backend below), so no
pre-computed baseline table is needed.

## Coverage Backend

`cuban/coverage.py` resolves coverage per sample in this order:

1. `coverage_dir` supplied → read precomputed
   [mosdepth](https://github.com/brentp/mosdepth) per-base output directly.
2. mosdepth binary on `PATH` → run mosdepth restricted to the needed
   chromosome (two passes: all reads, and MAPQ ≥ 20) and cache the output
   under `~/.cuban/coverage/` (override with `--cache-dir` or
   `CUBAN_DATA_DIR`), so repeated renders — for example VCF batch mode —
   reuse it.
3. Otherwise → fall back to per-region `pysam` depth with a warning
   (slower, no caching, but no external dependency).

## Visualization Layout

Built with matplotlib GridSpec. Figure size: 25 × 11 inches per sample.

**Illumina samples** — 5 rows (height ratios 1:3:0.5:0.5:7):
1. Title (sample name, family/disease status, SV coordinates)
2. Coverage + repeat annotations
3. Insert size outlier histogram
4. Discordant orientation histogram
5. CIGAR alignment heatmap with split/discordant/MAPQ overlays

**Long-read samples** — 3 rows (height ratios 1:3:7):
1. Title
2. Coverage + repeat annotations
3. CIGAR alignment heatmap with split/discordant/MAPQ overlays

Long-read figures omit insert size and orientation rows (not applicable).

GridSpec uses 4 columns. For DEL/DUP/INV the CIGAR row spans all 4 columns
(left breakpoint | gap | right breakpoint). For BND, `cuban_bnd()` renders
two independent loci side by side (2 columns each).

## Module Responsibilities

**`cuban/utils.py`** — BAM I/O and data extraction. Reads BAM files via
pysam, computes CIGAR matrices, repeat overlaps, and insert
size/orientation statistics. Pure data processing, no plotting.

**`cuban/coverage.py`** — the coverage backend (see above), including
baseline-coverage derivation.

**`cuban/visualize.py`** — orchestration and rendering. `gather_data()`
assembles a result dict; `cuban()` and `cuban_bnd()` build the multi-panel
matplotlib figure; individual `plot_*` functions render one panel each.

**`cuban/cli.py`** — the `cuban` command-line interface: argument parsing,
sample-spec handling, single-SV and VCF batch modes.

**`cuban/repeats.py`** — locating, loading, and downloading the
RepeatMasker annotation (`cuban-fetch-repeats`).
