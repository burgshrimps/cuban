"""Generate deterministic synthetic test data for cuban.

Creates, in a target directory:
  - test.bam / test.bam.bai   paired-end reads on one 50 kb contig ("chr1")
                              simulating a homozygous 5 kb deletion at
                              20,000-25,000: coverage drop, insert-size
                              outliers, discordant pairs and soft-clipped
                              split reads at both breakpoints
  - repeats.tsv               minimal RepeatMasker table (UCSC rmsk column
                              names: genoName/genoStart/genoEnd/repClass)
  - test.vcf                  three DEL records for VCF-mode tests

Usage: python tests/make_test_data.py [outdir]   (default: tests/data)
"""

import random
import sys
from pathlib import Path

import pysam

CONTIG = "chr1"
CONTIG_LEN = 50_000
DEL_START = 20_000
DEL_END = 25_000
READ_LEN = 150
INSERT = 400
SEED = 42


def _seq(rng, n):
    return "".join(rng.choice("ACGT") for _ in range(n))


def _pair(rng, name, pos1, pos2, flags=None, cigar1=None, cigar2=None,
          mapq=60, sa1=None, sa2=None):
    """Build a properly-flagged read pair; returns two AlignedSegments."""
    a = pysam.AlignedSegment()
    b = pysam.AlignedSegment()
    for seg, pos, cigar, sa in ((a, pos1, cigar1, sa1), (b, pos2, cigar2, sa2)):
        seg.query_name = name
        seg.reference_id = 0
        seg.reference_start = pos
        seg.mapping_quality = mapq
        seg.query_sequence = _seq(rng, READ_LEN)
        seg.query_qualities = pysam.qualitystring_to_array("I" * READ_LEN)
        seg.cigarstring = cigar or f"{READ_LEN}M"
        if sa:
            seg.set_tag("SA", sa)
    default = (99, 147)  # proper pair, R1 fwd / R2 rev
    f1, f2 = flags or default
    a.flag, b.flag = f1, f2
    a.next_reference_id = b.next_reference_id = 0
    a.next_reference_start = pos2
    b.next_reference_start = pos1
    tlen = (pos2 + READ_LEN) - pos1
    a.template_length = tlen
    b.template_length = -tlen
    return a, b


def make_bam(outdir: Path) -> Path:
    rng = random.Random(SEED)
    header = {"HD": {"VN": "1.6", "SO": "coordinate"},
              "SQ": [{"SN": CONTIG, "LN": CONTIG_LEN}]}
    reads = []

    # Normal pairs tiling the contig, skipping the deleted interval
    # (homozygous DEL: no reads start inside it).
    n = 0
    for start in range(0, CONTIG_LEN - INSERT - READ_LEN, 25):
        if DEL_START - READ_LEN < start < DEL_END:
            continue
        pos2 = start + INSERT - READ_LEN
        if DEL_START - READ_LEN < pos2 < DEL_END:
            continue
        jitter = rng.randint(-8, 8)
        reads.extend(_pair(rng, f"norm{n}", start, pos2 + jitter))
        n += 1

    # Pairs spanning the deletion: mate jumps across -> big insert.
    for i in range(30):
        p1 = DEL_START - INSERT + rng.randint(-60, 40)
        p2 = DEL_END + rng.randint(0, 100)
        reads.extend(_pair(rng, f"span{i}", p1, p2))

    # Soft-clipped split reads at both breakpoints with SA tags.
    half = READ_LEN // 2
    for i in range(12):
        off = rng.randint(-15, -1)
        p_left = DEL_START - half + off
        p_right = DEL_END + off
        sa_right = f"{CONTIG},{DEL_END + 1},+,{half}S{half}M,60,0;"
        sa_left = f"{CONTIG},{p_left + 1},+,{half}M{half}S,60,0;"
        a, b = _pair(rng, f"split{i}", p_left, p_right,
                     cigar1=f"{half}M{half}S", cigar2=f"{half}S{half}M",
                     sa1=sa_right, sa2=sa_left)
        reads.extend((a, b))

    # A few duplicate-flagged reads (regression: must not desync rows).
    for i in range(5):
        a, b = _pair(rng, f"dup{i}", 19_000 + i * 3, 19_000 + i * 3 + INSERT - READ_LEN)
        a.flag |= 1024
        b.flag |= 1024
        reads.extend((a, b))

    # A few low-MAPQ reads.
    for i in range(5):
        reads.extend(_pair(rng, f"lowq{i}", 18_500 + i * 7,
                           18_500 + i * 7 + INSERT - READ_LEN, mapq=5))

    unsorted = outdir / "test.unsorted.bam"
    with pysam.AlignmentFile(unsorted, "wb", header=header) as bam:
        for r in sorted(reads, key=lambda r: r.reference_start):
            bam.write(r)
    out = outdir / "test.bam"
    pysam.sort("-o", str(out), str(unsorted))
    unsorted.unlink()
    pysam.index(str(out))
    return out


def make_repeats(outdir: Path) -> Path:
    rows = [
        (CONTIG, 19_500, 19_800, "LINE"),
        (CONTIG, 19_900, 20_100, "SINE"),
        (CONTIG, 24_800, 25_300, "Simple_repeat"),
        (CONTIG, 30_000, 30_500, "LTR"),
    ]
    out = outdir / "repeats.tsv"
    with open(out, "w") as fh:
        fh.write("genoName\tgenoStart\tgenoEnd\trepClass\n")
        for r in rows:
            fh.write("\t".join(map(str, r)) + "\n")
    return out


def make_vcf(outdir: Path) -> Path:
    out = outdir / "test.vcf"
    with open(out, "w") as fh:
        fh.write("##fileformat=VCFv4.2\n")
        fh.write(f"##contig=<ID={CONTIG},length={CONTIG_LEN}>\n")
        fh.write('##INFO=<ID=SVTYPE,Number=1,Type=String,Description="SV type">\n')
        fh.write('##INFO=<ID=END,Number=1,Type=Integer,Description="SV end">\n')
        fh.write('##INFO=<ID=SVLEN,Number=1,Type=Integer,Description="SV length">\n')
        fh.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        recs = [
            (CONTIG, DEL_START, "del1", "N", "<DEL>",
             f"SVTYPE=DEL;END={DEL_END};SVLEN=-{DEL_END - DEL_START}"),
            (CONTIG, 10_000, "del2", "N", "<DEL>", "SVTYPE=DEL;END=11000;SVLEN=-1000"),
            (CONTIG, 35_000, "dup1", "N", "<DUP>", "SVTYPE=DUP;END=36500;SVLEN=1500"),
        ]
        for chrom, pos, vid, ref, alt, info in recs:
            fh.write(f"{chrom}\t{pos}\t{vid}\t{ref}\t{alt}\t.\tPASS\t{info}\n")
    return out


def main(outdir="tests/data"):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    bam = make_bam(outdir)
    rep = make_repeats(outdir)
    vcf = make_vcf(outdir)
    print(f"wrote {bam}, {rep}, {vcf}")


if __name__ == "__main__":
    main(*sys.argv[1:2])
