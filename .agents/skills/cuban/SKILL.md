---
name: cuban
description: Visualize the read-level evidence behind structural variant (SV) calls with cuban. Use when asked to plot, render, inspect, review, or check a deletion, duplication, insertion, inversion, or breakend/translocation (DEL, DUP, INS, INV, BND) from one or more BAM files, for a single variant given as coordinates or for many variants from an SV VCF, optionally across several samples (trio, cohort). Produces one multi-panel PNG per variant. Not for calling SVs or for non-SV variants (SNVs, indels below ~50 bp).
---

# cuban: SV evidence plots from BAMs

cuban renders coverage, repeat elements, insert-size outliers, discordant
read pairs, and the read alignments around both breakpoints of one SV into a
single PNG, one block per sample. It is caller-agnostic: it only needs the SV
type plus coordinates (or a VCF) and indexed BAMs.

## 1. Preflight

```bash
cuban --version
```

If the command is missing, install it (conda is preferred because it brings
`mosdepth`, the fast coverage backend; without it cuban falls back to a slower
built-in method):

```bash
git clone https://github.com/burgshrimps/cuban.git && cd cuban
conda env create -f environment.yml && conda activate cuban
# or: pip install git+https://github.com/burgshrimps/cuban.git
```

Then check the inputs:

- Every BAM needs a `.bai`/`.csi` index next to it (`samtools index x.bam`).
- BAM paths must not contain a colon (the `--sample NAME:BAM` syntax splits on it).
- Chromosome names must match the BAM header (`chr1` vs `1`); cuban aborts with
  the BAM's contig names if they don't, so fix the coordinates, not the BAM.
- Coordinates are 1-based, `--end` inclusive.

Repeat annotation: on first use cuban downloads a ~40 MB hg38 RepeatMasker
table. In a non-interactive shell it picks the location itself (the repo's
`annot/` folder, or `~/.cuban`) and prints where; set `CUBAN_DATA_DIR` to
control it. For non-hg38 genomes pass your own table with `--repeats` (UCSC
rmsk columns genoName/genoStart/genoEnd/repClass) or `--no-repeats`.

## 2. Pick the mode

| Situation | Mode |
|---|---|
| One SV, or a handful, given as type + coordinates | single-variant call per SV (`--sv-type ... --out x.png`) |
| A breakend / translocation with two loci | `--bnd` with `--chrom-b/--start-b/--end-b` |
| An SV VCF, or more than a handful of SVs | batch (`--vcf calls.vcf --outdir plots/`) |
| Several samples (trio, cohort, tumor/normal) | repeat `--sample` in any of the above; each sample becomes one block of the same figure |

Never re-implement the figure or parse BAMs yourself; always call the CLI.
Use the Python API (`from cuban import cuban`, see README) only when the user
explicitly wants programmatic use.

## 3. Commands

Single variant, one sample:

```bash
cuban --sv-type DEL --chrom chr2 --start 1234500 --end 1239800 \
      --sample proband:/path/proband.bam \
      --out proband_del.png
```

Single variant, several samples (one figure, one block per sample; put the
proband/tumor first so it is on top):

```bash
cuban --sv-type DUP --chrom chr7 --start 5500000 --end 5620000 \
      --sample proband:/path/proband.bam \
      --sample mother:/path/mother.bam \
      --sample father:/path/father.bam \
      --out trio_dup.png
```

Breakend (two independent loci rendered side by side):

```bash
cuban --bnd --chrom chr1 --start 20000 --end 20001 \
      --chrom-b chr5 --start-b 90000 --end-b 90001 \
      --sample proband:/path/proband.bam --out bnd.png
```

VCF batch (one PNG per record named `<ID>.png`, or `<chrom>_<pos>_<type>.png`
when ID is `.`; existing PNGs are skipped so a run can be resumed; BND
records with any breakend bracket notation are handled automatically):

```bash
cuban --vcf calls.vcf --outdir plots/ \
      --sample proband:/path/proband.bam
```

For a subset of a large VCF, filter first (`bcftools view -i 'SVLEN<-50000'`,
`bcftools view -r chr1`, or an ID list) and pass the filtered file.

Insertions: give the insertion point as `--start N --end N+1`; the inserted
sequence itself is invisible in the reference, so the evidence is clipped and
split reads in the alignment track.

## 4. Useful flags

| Flag | When to use |
|---|---|
| `--tech SAMPLE:sr\|lr` | cuban infers short- vs long-read from read lengths and prints its guess; override if wrong. Long-read blocks omit the insert-size and discordant-pair tracks. |
| `--baseline-cov SAMPLE:30.5` | override the red baseline line (default: chromosome mean depth via mosdepth, cached in `cuban_coverage/` next to the BAM). |
| `--padding N` | context around the SV in bp. Default adaptive: max(1500, size/10). Increase for cleaner coverage context, decrease for tiny variants. |
| `--window N` | bp of read alignment shown around each breakpoint (default 100). Widen if breakpoints are imprecise (large CIPOS). |
| `--max-reads N` | reads per alignment panel before deterministic downsampling (default 5000). Raise for very deep targeted data. |
| `--bin-size N` | coverage bin in bp; auto-binned above 100 kb. |
| `--cache-dir DIR` | where mosdepth output is cached when the BAM directory is read-only. |
| `--no-collapse-ins` | show inserted bases at full width instead of a collapsed column. |

`cuban --help` lists everything with defaults.

## 5. After rendering

1. Confirm the PNG exists and is non-trivial (`ls -l`), and for batch runs
   read the final `[cuban] rendered N, skipped N, failed N` line; a non-zero
   `failed` count exits 1 and each failure is printed with its record ID and
   reason (usually missing SVTYPE or an unparsable BND ALT).
2. Look at the image (open or attach it) and interpret it using
   [reference/interpretation.md](reference/interpretation.md): which tracks
   support the call, whether the breakpoints look correctly placed, and
   whether repeats or low mapping quality explain the signal.
3. Report the verdict per variant in plain language (supported / not
   supported / ambiguous) with the specific evidence, not just the file path.
