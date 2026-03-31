# Domain Rules

## BAM & Alignment

- BAM files **must be indexed** (`.bai`) for `pysam.fetch()` to work
- Chromosome naming: both numeric (`1`-`22`) and prefixed (`chrX`, `chrY`, `chrM`) formats are supported
- Reads that are unmapped, duplicate, or QC-fail are skipped during matrix computation
- MAPQ threshold: reads with `mapping_quality < 30` are flagged as low quality and overlaid in grey

### CIGAR Integer Encoding (pysam convention)

| Code | Meaning     | Display Color        |
|------|-------------|----------------------|
| -1   | No coverage | white                |
| 0    | Match       | white                |
| 1    | Insertion   | orange (`#f0b6a0`)   |
| 2    | Deletion    | blue (`#5066a2`)     |
| 3    | Ref skip    | teal (`#6ac0b7`)     |
| 4    | Soft clip   | (overlay hatch `||`) |
| 5    | Hard clip   | (overlay hatch `||`) |
| 6    | Padding     | red (`#df624c`)      |
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

**BND specifics** — `cuban_bnd()` takes two independent loci (chromA/startA/endA, chromB/startB/endB) and renders them side by side.

## Coverage

- Computed via `pysam depth` — returns per-position depth for all reads and MAPQ≥30 reads separately
- **SVs > 200MB**: coverage computation is skipped entirely (memory guard)
- **SVs 1KB–100KB**: Savitzky-Golay smoothing is applied (window = 5% of SV size, polynomial order 3)
- **INS/BND**: fixed smoothing window of 25, polynomial order 3
- Baseline coverage: pre-computed by sampling 1000 random 1000bp regions and taking the median. Stored in `resources/baseline_cov_*.json`

## Insert Size & Orientation

- Default thresholds: `thr_min=50`, `thr_max=1000`
- Reads with `template_length > thr_max` are flagged as insert size outliers
- Only computed when coverage is computed (skipped for SVs > 200MB)

### Discordant Orientation Categories

| Category | Meaning                              | Color         |
|----------|--------------------------------------|---------------|
| RR       | Both reads on reverse strand         | sandybrown    |
| FF       | Both reads on forward strand         | cadetblue     |
| RF       | Reverse-forward (unexpected)         | midnightblue  |
| TX       | Mates on different chromosomes       | red           |

## Repeat Annotations

Loaded from an external RepeatMasker TSV (`hg38_repeatmasker.tsv`). Filtered by overlap with the SV region.

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
