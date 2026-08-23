# cuban — Structural-variant visualization

**cuban** is a standalone Python library that renders publication-quality
multi-panel PNGs of structural variants from BAM files. Given an SV
definition (type + coordinates) and a `samples` dict pointing at indexed
BAMs, it produces a figure with coverage, repeat overlay, insert-size
outliers, discordant orientations, and a CIGAR alignment heatmap.

cuban is **caller-, cohort-, and reference-agnostic**: any indexed BAM and
any RepeatMasker TSV will do. The tGenVar paper uses cuban to power the
manual-curation web app and the SV detail figures.

This is the fourth of five sub-archives that accompany the paper:

```
00_ground_truth/        SV ground-truth construction
01_variant_database/    SQLite DB + dicast model training
02_dicast/              dicast library + trained models
03_cuban/               ← you are here
04_figures/             paper figures
```

## Layout

```
.
├── README.md
├── environment.yml             conda environment
├── cuban.py                    CLI entry point — render a figure from the shell
├── cuban_lib/
│   ├── visualize.py            Python API: cuban(), cuban_bnd(), gather_data()
│   └── utils.py                BAM I/O — CIGAR matrices, coverage, isize/orientation
├── app/                        static web frontend for manual variant curation
│   ├── index.html
│   ├── styles.css
│   └── utils.js
└── resources/
    ├── baseline_cov_ill.json   per-chromosome median Illumina coverage (pre-computed)
    ├── baseline_cov_pb.json    per-chromosome median PacBio coverage (pre-computed)
    └── hg38_repeatmasker.tsv   RepeatMasker output (chrom/start/end/repclass)
```

## Dependencies

- Python 3.10 with: pandas, numpy, scipy, matplotlib, seaborn, pyyaml, pysam,
  pybedtools, pyBigWig, bioframe.

`environment.yml` ships a minimal conda spec; recreate with
`conda env create -f environment.yml`.

## Quickstart — render one SV (CLI)

```bash
python cuban.py \
    --sv-type DEL --chrom chr1 --start 1234500 --end 1239800 \
    --sample SAMPLE_001:ill:/path/to/SAMPLE_001.bam \
    --out SAMPLE_001_chr1_1234500_DEL.png
```

The RepeatMasker TSV ships with the archive at
`resources/hg38_repeatmasker.tsv` and is picked up automatically; pass
`--repeats /path/to/your.tsv` to override.

Each `--sample` is a colon-separated spec
`name:tech:bam[:baseline_cov[:family_status[:disease_status]]]`. `tech` is
`ill` or `pb`. If `baseline_cov` is omitted (or set to `auto`), cuban reads
the per-chromosome median for the SV's chromosome from
`resources/baseline_cov_<tech>.json`. Repeat `--sample` for each member of
a trio or larger family. For BNDs, pass `--bnd` together with `--chrom-b`,
`--start-b`, and `--end-b`. Run `python cuban.py --help` for the full
argument reference.

## Python API

```python
import pandas as pd
from cuban_lib.visualize import cuban

rep_df = pd.read_csv('/path/to/hg38_repeatmasker.tsv', sep='\t')
samples = {
    'SAMPLE_001': {
        'family_status':   'index',
        'disease_status':  'affected',
        'technology':      'ill',
        'bam_name':        '/path/to/SAMPLE_001.bam',  # must be indexed (.bai)
        'baseline_cov':    32.5,
    },
}
cuban(samples, rep_df, sv_type='DEL',
      chrom='chr1', start=1_234_500, end=1_239_800,
      outfile='SAMPLE_001_chr1_1234500_DEL.png')
```

`cuban_bnd()` works analogously for BNDs (two independent loci side by side).

## Web app — manual curation

The static frontend at `app/index.html` is a single-page HTML5 canvas tool
for blinded manual curation:

1. Load a CSV/VCF of variant IDs + a directory of cuban-rendered PNGs.
2. Variants are presented one at a time on the canvas.
3. Curator marks each as **Confirm / Discard / Back / Stop** (keyboard arrows).
4. Output: original file with an added evaluation column, downloaded as
   `<filename>.<curator_name>_curated.{tsv,vcf}`.

No backend — open `index.html` directly in a browser.

## Reference annotations

cuban reads a single TSV: `hg38_repeatmasker.tsv` (RepeatMasker
output filtered to chrom/start/end/repclass columns). It ships with the
archive at `resources/hg38_repeatmasker.tsv`; the CLI defaults to that
copy, and the Python API expects it as a `pandas.read_csv` DataFrame.
Per-chromosome baseline coverage JSONs in `resources/` can be regenerated
for any cohort by sampling 1000 random 1000-bp regions and taking the
median.
