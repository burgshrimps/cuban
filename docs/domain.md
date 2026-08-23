# Domain Rules

## BAM & Alignment

- BAM files **must be indexed** (`.bai`) for `pysam.fetch()` to work
- Chromosome naming: both numeric (`1`-`22`) and prefixed (`chrX`, `chrY`, `chrM`) formats are supported
- Reads that are unmapped, duplicate, or QC-fail are excluded from the alignment matrix
- To bound memory on deep data, at most 5000 reads are drawn per breakpoint window
  (configurable via `max_reads`; downsampling is seeded and reproducible)
- MAPQ thresholds: the filtered **coverage** line counts reads with
  `mapping_quality >= 20`; in the read-alignment track, reads with
  `mapping_quality < 30` are flagged as low quality and overlaid in grey

### CIGAR Integer Encoding (pysam convention)

| Code | Meaning     | Display Color        |
|------|-------------|----------------------|
| -1   | No coverage | white                |
| 0    | Match       | lightgrey            |
| 1    | Insertion   | tan (`#b7954b`)      |
| 2    | Deletion    | blue (`#5066a2`)     |
| 3    | Ref skip    | orange (`#f0b6a0`)   |
| 4    | Soft clip   | teal (`#6ac0b7`)     |
| 5    | Hard clip   | red (`#df624c`)      |
| 6    | Padding     | lightgrey            |
| 7    | Seq match   | lightgrey            |
| 8    | Seq mismatch| lightgrey            |

## SV Types and Processing

Five supported types: **DEL**, **DUP**, **INV**, **INS**, **BND**

**DEL / DUP / INV** — Two-breakpoint visualization:
- Computes separate alignment matrices for left (`start - window` to `start + window`) and right (`end - window` to `end + window`) breakpoints
- Matrices are height-padded to match, then concatenated with a 50-column gap
- Coverage padding: `max(padding, 20% of SV size)`

**INS / BND** — Single-region visualization:
- One alignment matrix from `start - window` to `end + window`
- Coverage padding equals the `padding` parameter directly

**BND specifics** — `cuban_bnd()` takes two independent loci (chromA/startA/endA, chromB/startB/endB) and renders them side by side. Each locus gets the baseline coverage of its own chromosome.

The CLI default padding is adaptive: `max(1500, SV size / 10)`.

## Coverage

- Computed with [mosdepth](https://github.com/brentp/mosdepth) when available
  (per-base depth for all reads and for MAPQ≥20 reads separately), falling
  back to per-region `pysam` depth otherwise — see
  [architecture.md](architecture.md#coverage-backend)
- **SVs > 200 Mb**: coverage computation is skipped entirely (memory guard)
- **SVs 1 kb–100 kb**: Savitzky-Golay smoothing is applied (window = 5% of SV size, polynomial order 3)
- **INS/BND**: fixed smoothing window of 25, polynomial order 3
- Baseline coverage (`'auto'`): the mean depth of the SV's chromosome, taken
  from the sample's own mosdepth summary (or, in the pysam fallback, the
  median of 1000 randomly sampled 1000 bp regions, seeded)

## Insert Size & Orientation

- Default thresholds: `thr_min=50`, `thr_max=1000`
- Reads with `template_length > thr_max` are flagged as insert size outliers
- Only computed when coverage is computed (skipped for SVs > 200 Mb)

### Discordant Orientation Categories

| Category | Meaning                              | Color         |
|----------|--------------------------------------|---------------|
| RR       | Both reads on reverse strand         | sandybrown    |
| FF       | Both reads on forward strand         | cadetblue     |
| RF       | Reverse-forward (unexpected)         | midnightblue  |
| TX       | Mates on different chromosomes       | red           |

## Repeat Annotations

Loaded from a RepeatMasker table with UCSC `rmsk` column names
(`genoName`, `genoStart`, `genoEnd`, `repClass`), filtered by overlap with
the SV region. Download the prepared hg38 table with `cuban-fetch-repeats`,
or build one for any genome with `scripts/build_repeats.py`.

| Repeat Class    | Color                    |
|-----------------|--------------------------|
| LINE            | teal (`#6ac0b7`)         |
| SINE            | tan (`#b7954b`)          |
| LTR             | orange (`#f0b6a0`)       |
| DNA             | blue (`#5066a2`)         |
| Simple_repeat   | purple (`#504669`)       |
| Satellite       | red (`#df624c`)          |
| Low_complexity  | green (`#61856b`)        |
| Retroposon      | dark green (`#2f7155`)   |

Only these eight classes are rendered; rows with other classes are ignored.
