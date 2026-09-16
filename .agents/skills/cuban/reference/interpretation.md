# Reading a cuban figure

One figure per SV, one block per sample (top to bottom in `--sample` order).
Illumina blocks have five tracks; long-read blocks omit the insert-size and
discordant-pair tracks (they only apply to paired-end data). BND figures show
the two loci side by side.

## Tracks

**Coverage** (top). Depth across the SV plus padding. Black dashed verticals
mark the breakpoints; the red dashed horizontal is the sample's baseline
(chromosome mean, or `--baseline-cov`). Gray area = all reads, black line =
reads with mapping quality >= 20. A gap between gray and black means the
region is supported only by ambiguous alignments.

**Repeat elements** (under coverage). Annotated RepeatMasker elements, one row
per class. Breakpoints inside the same repeat family (e.g. two Alus) are a
classic source of false calls and mis-placed breakpoints.

**Insert size outliers** (Illumina only). Count of read pairs with insert
> 1 kb. Only informative for SVs larger than ~400-500 bp.

**Discordant read pairs** (Illumina only). Pairs whose orientation is not the
expected forward-reverse: reverse-forward (dark blue), reverse-reverse
(orange), forward-forward (cadet blue); dashed red = mate on another
chromosome.

**Read alignments** (bottom). Individual reads in a window (default 100 bp)
around each breakpoint. Colors encode the CIGAR operation: normal alignment,
low mapping quality (< 30), deletion, insertion, soft clip, hard clip. A black
grid overlay marks split reads; colored overlays mark pair orientation. With
phased BAMs (`HP` tags) reads are grouped into HP:1 / HP:2 / unassigned bands
with a blue/red strand barcode.

**Read connections**. Dashed lines between the two breakpoint windows: black
joins the two segments of one split read (strong evidence that both
breakpoints are correctly placed); red joins the two mates of a pair (or the
same read appearing in both windows for small SVs).

## Expected signatures

| SV type | Coverage | Insert size | Discordant pairs | Read alignments |
|---|---|---|---|---|
| DEL | drop between breakpoints (to ~half for het, ~zero for hom) | peaks at both breakpoints (if > ~400 bp) | none specific | soft-clipped / split reads at both breakpoints, black dashed connections between windows; long reads show a deletion segment |
| DUP (tandem) | gain between breakpoints | peaks at both breakpoints | reverse-forward (dark blue) around breakpoints | clipped / split reads at both breakpoints |
| INV | flat | none | reverse-reverse (orange) at one breakpoint, forward-forward (cadet blue) at the other | clipped / split reads at both breakpoints |
| INS | flat | none (or slight increase) | none | clipped reads pointing at the insertion point; long reads show an insertion segment |
| BND | flat at both loci, or a drop/gain if part of a larger event | none | dashed red (mate on another chromosome) at both loci | clipped / split reads at both loci, black connections between the two loci |

## Verdict checklist

Call a variant **supported** when at least two independent tracks agree and
the breakpoints line up: e.g. a coverage drop whose edges coincide with the
insert-size peaks and with clipped reads, or split reads bridging both
windows.

Treat it as **ambiguous / likely artifact** when:

- the coverage signal exists only in the gray (all reads) area, not the black
  MAPQ >= 20 line;
- the breakpoints fall inside repeats of the same class and the supporting
  reads are mostly low-MAPQ (colored as such in the alignment track);
- clipped reads pile up at only one breakpoint and nothing else agrees
  (possible mis-placed second breakpoint: consider re-running with a larger
  `--window` or `--padding`);
- the signal appears in all samples of a trio/cohort, including controls,
  with identical breakpoints (reference artifact or common variant rather
  than a de novo event).

Call it **not supported** when coverage is flat where a DEL/DUP is expected,
no clipped/split reads sit at the breakpoints, and the pair-based tracks are
quiet.

For trios, compare the proband block against the parents: a real de novo
event shows the signature only in the proband; an inherited one shows it in
one parent as well.
