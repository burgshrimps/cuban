# Interpreting cuban figures

Every cuban figure stacks several evidence tracks for one structural variant
(SV), one block per sample. This page explains what each track shows and how
to read it.

![Anatomy of a cuban figure](img/figure-anatomy.png)

For Illumina samples the figure has five tracks: coverage (with repeat
elements), insert size outliers, discordant read pairs, and the read
alignments around both breakpoints. Long-read samples omit the insert-size
and discordant-pair tracks, which only apply to paired-end data.

## Coverage

Read coverage across the whole SV with padding on both sides. Breakpoints
are indicated by vertical black dashed lines. The average coverage of the
chromosome is indicated by a horizontal red dashed line. For the gray
coverage area, all reads are considered; the black line overlay shows the
coverage when only reads with mapping quality ≥ 20 are counted, so a gap
between the two exposes regions supported only by ambiguous alignments.

A deletion shows as a drop below the baseline between the breakpoints, a
duplication as a gain; inversions and insertions usually leave coverage flat.

## Repeat Elements

Repetitive sequence affects read alignment and variant detection, so all
annotated repeat elements near the SV are drawn under the coverage track,
one row per class:

| Repeat class    | Color                    |
|-----------------|--------------------------|
| LINE            | teal (`#6ac0b7`)         |
| SINE            | tan (`#b7954b`)          |
| LTR             | orange (`#f0b6a0`)       |
| DNA             | blue (`#5066a2`)         |
| Simple_repeat   | purple (`#504669`)       |
| Satellite       | red (`#df624c`)          |
| Low_complexity  | green (`#61856b`)        |
| Retroposon      | dark green (`#2f7155`)   |

## Insert Size Outliers

The insert size is the distance between the two mates of a read pair. Read
pairs spanning a deletion (or duplication) appear to have a larger insert
than the sequencing library was prepared with, so peaks of insert-size
outliers (insert > 1 kb) at both breakpoints support a DEL/DUP call. This
signal is only informative for variants larger than roughly 400–500 bp.

![How discordant pairs and insert size outliers arise](img/signal-schematics.png)

## Discordant Read Pairs

With paired-end sequencing the first read of a pair is expected to align to
the forward strand and its mate to the reverse strand. Deviations indicate
an SV. The track shows the distribution of read pairs aligning in
reverse-forward (dark blue), reverse-reverse (orange), and forward-forward
(cadet blue) orientation; the dashed red line shows reads whose mate maps to
a different chromosome. Duplications typically produce reverse-forward
pairs around the breakpoints; inversions produce reverse-reverse and
forward-forward pairs. For deletions this track is not informative.

## Read Alignments

The bottom track shows the individual reads in two windows (default
±100 bp) around the two breakpoints. Colors encode the CIGAR operations of
each read:

| Appearance | Meaning |
|---|---|
| ![normal](img/read-normal.png) light gray | normally aligned read |
| ![low mapq](img/read-low-mapq.png) dark gray | mapping quality < 30 — the read aligns to multiple positions in the reference |
| ![deletion](img/read-deletion.png) blue | small deleted segment within the read |
| ![insertion](img/read-insertion.png) yellow | small inserted segment within the read |
| ![soft clip](img/read-soft-clipped.png) green | soft-clipped read — only part of the read aligns; the clipped part (green) can indicate a breakpoint |
| ![hard clip](img/read-hard-clipped.png) red | hard-clipped read — as above, but the clipped bases are not stored in the BAM |
| ![split a](img/read-split-a.png) ![split b](img/read-split-b.png) black grid overlay | split read — parts of the read align to two different positions |
| ![rf](img/overlay-reverse-forward.png) dark blue overlay | read pair in reverse-forward orientation |
| ![rr](img/overlay-reverse-reverse.png) orange overlay | read pair in reverse-reverse orientation |
| ![ff](img/overlay-forward-forward.png) cadet blue overlay | read pair in forward-forward orientation |

## Read Connections

Dashed lines drawn between the two breakpoint windows connect related
reads. Black dashed lines join the two segments of one split read — split
reads bridging both breakpoints are strong evidence that the breakpoints
are placed correctly. Red dashed lines join the two mates of a read pair
(or the same read appearing in both windows when the variant is small
enough for the windows to overlap); for larger variants they play the same
role as the insert-size track, showing pairs that span the variant.
