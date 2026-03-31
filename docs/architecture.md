# Architecture

## Data Flow

```
SV definition (type, chrom, start, end) + BAM files
  │
  ▼
gather_data()                          [visualize.py]
  ├── compute_aln_matrix()             [utils.py] → CIGAR integer matrix (reads × positions)
  ├── pad_alignment_matrices()         [utils.py] → equalize left/right matrix heights
  ├── compute_cov_df()                 [utils.py] → depth per position (all + MAPQ≥30)
  ├── compute_rep_df()                 [utils.py] → repeat elements in region
  └── compute_isize_orientation_dict() [utils.py] → insert size outliers + discordant orientations
  │
  ▼
cuban() or cuban_bnd()                 [visualize.py]
  ├── plot_cov()      — coverage line plot with baseline reference
  ├── plot_rep()      — repeat annotations as colored horizontal lines
  ├── plot_isize()    — insert size outlier histograms
  ├── plot_orient()   — discordant orientation histograms
  ├── plot_cigar()    — CIGAR alignment heatmap
  ├── add_splitread_overlay()  — hatch marks on split reads
  ├── add_disco_overlay()     — hatch marks on discordant pairs
  └── add_mapq_overlay()      — grey overlay on low MAPQ reads
  │
  ▼
Publication-quality PNG (matplotlib figure)
```

## Sample Dict Format

The primary API contract. Passed to `cuban()` and `cuban_bnd()`:

```python
samples = {
    'sample_name': {
        'family_status': 'index' | 'control',
        'disease_status': 'affected' | 'unaffected',
        'technology': 'ill' | 'pb',       # Illumina or PacBio
        'bam_name': '/path/to/file.bam',  # must be indexed (.bai)
        'baseline_cov': 32.5              # per-chromosome median coverage
    }
}
```

Baseline coverage values come from `resources/baseline_cov_ill.json` and `resources/baseline_cov_pb.json` (pre-computed per-chromosome medians).

## Visualization Layout

Built with matplotlib GridSpec. Figure size: 25 × 11 inches per sample.

**Illumina samples** — 5 rows (height ratios 1:3:0.5:0.5:7):
1. Title (sample name, family/disease status, SV coordinates)
2. Coverage + repeat annotations
3. Insert size outlier histogram
4. Discordant orientation histogram
5. CIGAR alignment heatmap with split/discordant/MAPQ overlays

**PacBio samples** — 3 rows (height ratios 1:3:7):
1. Title
2. Coverage + repeat annotations
3. CIGAR alignment heatmap with split/discordant/MAPQ overlays

PacBio omits insert size and orientation rows (not applicable to long reads).

GridSpec uses 4 columns. For DEL/DUP/INV the CIGAR row spans all 4 columns (left breakpoint | gap | right breakpoint). For BND, `cuban_bnd()` renders two independent loci side by side (2 columns each).

## Module Responsibilities

**`cuban_lib/utils.py`** — All BAM I/O and data extraction. Reads BAM files via pysam, computes CIGAR matrices, coverage depth, repeat overlaps, and insert size/orientation statistics. Pure data processing, no plotting.

**`cuban_lib/visualize.py`** — Orchestration and rendering. `gather_data()` calls utils functions and assembles a result dict. `cuban()` and `cuban_bnd()` build the multi-panel matplotlib figure. Individual `plot_*` functions render one panel each. Also contains dotplot functions (`acc_dot`, `plot_dotplots`, `plot_dotplot_multi_sample`).

**`cuban_lib/docs.py`** — Standalone script that generates a CIGAR color legend as PNG.

## Web App (`app/`)

Static HTML/JS interface for manual variant curation:
1. User loads a CSV/VCF file with variant IDs and a directory of PNG images
2. Variants are displayed one at a time on an HTML5 canvas
3. Curator marks each as Confirm / Discard / Back / Stop (keyboard arrows supported)
4. Output: original file with an added evaluation column, downloaded as `filename.curator_name_curated.tsv/vcf`
