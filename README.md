![cuban](docs/img/cuban-banner.png)

# cuban: Scalable structural variant visualization for manual inspection

[![CI](https://github.com/burgshrimps/cuban/actions/workflows/ci.yml/badge.svg)](https://github.com/burgshrimps/cuban/actions/workflows/ci.yml)
[![License: GPL-3.0](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

**cuban** is a tool for manually inspecting the sequencing evidence behind
structural variant (SV) calls. Given an SV (type + coordinates) and one or
more BAM files, it renders all the read-level evidence for the call in a
single multi-panel image: coverage, repeat elements, insert-size outliers,
discordant read pairs, and the individual read alignments around both
breakpoints, including haplotype grouping for phased BAMs. cuban is
caller-agnostic (works from coordinates or any SV VCF), supports all five
SV types (DEL, DUP, INS, INV, BND), and stacks any number of samples,
short-read and long-read together, so trios and cohorts can be reviewed
side by side.

![anatomy of a cuban figure](docs/img/figure-anatomy.png)

## Installation

With conda (recommended, includes mosdepth):

```bash
git clone https://github.com/burgshrimps/cuban.git
cd cuban
conda env create -f environment.yml
conda activate cuban
```

Or with pip (Python 3.10+; install
[mosdepth](https://github.com/brentp/mosdepth) separately for fast
coverage, otherwise cuban falls back to a slower built-in method):

```bash
pip install git+https://github.com/burgshrimps/cuban.git
```

## Usage

*On first use cuban asks where to store the RepeatMasker table it
downloads once (~40 MB; press Enter to accept the suggested location,
the repo's `annot/` folder) and reuses it from then on.*

**Single variant, single sample:**

```bash
cuban --sv-type DEL --chrom chr2 --start 1234500 --end 1239800 \
      --sample proband:proband.bam \
      --out proband_del.png
```

**Single variant, multiple samples** (repeat `--sample`; each sample
becomes one block of the figure):

```bash
cuban --sv-type DEL --chrom chr2 --start 1234500 --end 1239800 \
      --sample proband:proband.bam \
      --sample mother:mother.bam \
      --sample father:father.bam \
      --out trio_del.png
```

**VCF, single sample** (one PNG per record, named `<ID>.png`;
already-rendered variants are skipped, so a batch can resume):

```bash
cuban --vcf variants.vcf --outdir plots/ \
      --sample proband:proband.bam
```

**VCF, multiple samples:**

```bash
cuban --vcf variants.vcf --outdir plots/ \
      --sample proband:proband.bam \
      --sample mother:mother.bam \
      --sample father:father.bam
```

**Visualizing BNDs**: breakpoint junctions render as two independent loci
side by side. In VCF mode, BND records (all four breakend bracket
notations) are handled automatically; in single variant mode, pass both
loci explicitly:

```bash
cuban --bnd --chrom chr1 --start 20000 --end 20001 \
      --chrom-b chr5 --start-b 90000 --end-b 90001 \
      --sample proband:proband.bam --out bnd.png
```

For detailed descriptions of all parameters and flags and their default
values, run `cuban --help`.

## Interpreting cuban plots

Every figure stacks several evidence tracks for one SV, one block per
sample. For Illumina samples there are five tracks: coverage (with repeat
elements), insert size outliers, discordant read pairs, and the read
alignments around both breakpoints. Long-read samples omit the insert-size
and discordant-pair tracks, which only apply to paired-end data.

### Coverage

Read coverage across the whole SV with padding on both sides. Breakpoints
are indicated by vertical black dashed lines. The average coverage of the
chromosome is indicated by a horizontal red dashed line. The gray area
counts all reads; the black line counts only reads with mapping quality of
20 or more, so a gap between the two exposes regions supported only by
ambiguous alignments. A deletion shows as a drop below the baseline between
the breakpoints, a duplication as a gain; inversions and insertions usually
leave coverage flat.

### Repeat elements

Repetitive sequence affects read alignment and variant detection, so all
annotated repeat elements near the SV are drawn under the coverage track,
one row per class.

### Insert size outliers

The insert size is the distance between the two mates of a read pair. Read
pairs spanning a deletion (or duplication) appear to have a larger insert
than the sequencing library was prepared with, so peaks of insert-size
outliers (insert > 1 kb) at both breakpoints support a DEL/DUP call. This
signal is only informative for variants larger than roughly 400-500 bp.

### Discordant read pairs

With paired-end sequencing the first read of a pair is expected to align to
the forward strand and its mate to the reverse strand. Deviations indicate
an SV. The track shows the distribution of read pairs aligning in
reverse-forward (dark blue), reverse-reverse (orange), and forward-forward
(cadet blue) orientation; the dashed red line shows reads whose mate maps
to a different chromosome. Duplications typically produce reverse-forward
pairs around the breakpoints; inversions produce reverse-reverse and
forward-forward pairs.

### Read alignments

The bottom track shows the individual reads in two windows (default
100 bp) around the two breakpoints. When reads carry `HP` tags (phased
BAMs), they are grouped into labelled HP:1 / HP:2 / unassigned haplotype
bands, and a per-read strand barcode (blue forward, red reverse) is drawn
next to each panel.

**Colors** encode the CIGAR operation of each read segment:

<table>
  <tr>
    <td align="center"><img src="docs/img/read-normal.png" width="230"><br>normally aligned read</td>
    <td align="center"><img src="docs/img/read-low-mapq.png" width="230"><br>low mapping quality (&lt; 30)</td>
    <td align="center"><img src="docs/img/read-deletion.png" width="230"><br>deleted segment</td>
  </tr>
  <tr>
    <td align="center"><img src="docs/img/read-insertion.png" width="230"><br>inserted segment</td>
    <td align="center"><img src="docs/img/read-soft-clipped.png" width="230"><br>soft-clipped read</td>
    <td align="center"><img src="docs/img/read-hard-clipped.png" width="230"><br>hard-clipped read</td>
  </tr>
</table>

**Overlays** are drawn on top of reads and mark split alignments and
read-pair orientation:

<table>
  <tr>
    <td colspan="2" align="center"><img src="docs/img/read-split-a.png" width="230"><br>split read</td>
    <td colspan="2" align="center"><img src="docs/img/overlay-reverse-forward.png" width="230"><br>reverse-forward pair</td>
    <td colspan="2" align="center"><img src="docs/img/overlay-reverse-reverse.png" width="230"><br>reverse-reverse pair</td>
  </tr>
  <tr>
    <td colspan="2"></td>
    <td colspan="2" align="center"><img src="docs/img/overlay-forward-forward.png" width="230"><br>forward-forward pair</td>
    <td colspan="2"></td>
  </tr>
</table>

Clipped (soft or hard) read ends mean only part of the read aligns to the
reference, which can indicate a breakpoint; hard-clipped bases are not
stored in the BAM. Split reads align in two different places and carry the
black grid overlay on both segments.

### Read connections

Dashed lines drawn between the two breakpoint windows connect related
reads. Black dashed lines join the two segments of one split read; split
reads bridging both breakpoints are strong evidence that the breakpoints
are placed correctly. Red dashed lines join the two mates of a read pair
(or the same read appearing in both windows when the variant is small
enough for the windows to overlap); for larger variants they play the same
role as the insert-size track, showing pairs that span the variant.

## Python API

```python
from pathlib import Path
from cuban import cuban
from cuban.repeats import load_repeats

rep_df = load_repeats()          # auto-downloads the hg38 table on first use
samples = {
    "SAMPLE_001": {
        "technology": "ill",             # short-read 'ill' / long-read 'pb';
                                         # cuban.utils.infer_technology(bam) infers it
        "bam_name": "SAMPLE_001.bam",    # must be indexed
        "baseline_cov": "auto",          # or an explicit float
    },
}
cuban(samples, rep_df, sv_type="DEL",
      chrom="chr1", start=1_234_500, end=1_239_800,
      padding=1500, outfile="SAMPLE_001_DEL.png")
```

`cuban_bnd()` works analogously for breakpoint pairs. A worked, executed
notebook is at [examples/cuban_examples.ipynb](examples/cuban_examples.ipynb).
