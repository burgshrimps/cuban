"""Build the slim RepeatMasker annotation used by cuban.

Produces a 4-column, gzip-compressed TSV (genoName, genoStart, genoEnd,
repClass) from the UCSC RepeatMasker track. cuban only needs these four
columns; the slim file is ~10x smaller than the raw table and much faster
to load.

Usage:
    # download the raw table from UCSC and convert (default: hg38)
    python scripts/build_repeats.py --genome hg38 -o resources/hg38_repeatmasker.4col.tsv.gz

    # or convert an already-downloaded raw file (rmsk.txt.gz from UCSC
    # goldenPath, or a headered TSV export of the rmsk table)
    python scripts/build_repeats.py --input /path/to/rmsk.txt.gz -o resources/hg38_repeatmasker.4col.tsv.gz
"""

import argparse
import gzip
import io
import sys
import urllib.request

UCSC_URL = "https://hgdownload.soe.ucsc.edu/goldenPath/{genome}/database/rmsk.txt.gz"

# Column order of the raw (headerless) UCSC rmsk.txt dump.
RMSK_COLUMNS = [
    "bin", "swScore", "milliDiv", "milliDel", "milliIns", "genoName",
    "genoStart", "genoEnd", "genoLeft", "strand", "repName", "repClass",
    "repFamily", "repStart", "repEnd", "repLeft", "id",
]
KEEP = ["genoName", "genoStart", "genoEnd", "repClass"]


def _open_maybe_gzip(path):
    fh = open(path, "rb")
    if fh.read(2) == b"\x1f\x8b":
        fh.seek(0)
        return io.TextIOWrapper(gzip.GzipFile(fileobj=fh))
    fh.seek(0)
    return io.TextIOWrapper(fh)


def convert(reader, out_path):
    first = reader.readline()
    if not first:
        sys.exit("error: input is empty")
    fields = first.rstrip("\n").split("\t")
    if "genoName" in fields:
        # Headered export: map column positions from the header row.
        try:
            idx = [fields.index(c) for c in KEEP]
        except ValueError as e:
            sys.exit(f"error: header is missing a required column: {e}")
        pending = None
    else:
        # Raw headerless goldenPath dump: fixed column order.
        if len(fields) != len(RMSK_COLUMNS):
            sys.exit(
                f"error: expected {len(RMSK_COLUMNS)} columns in raw rmsk "
                f"table, got {len(fields)} - is this a UCSC rmsk file?"
            )
        idx = [RMSK_COLUMNS.index(c) for c in KEEP]
        pending = fields  # first line is data, not a header

    n = 0
    with gzip.open(out_path, "wt", compresslevel=6) as out:
        out.write("\t".join(KEEP) + "\n")
        if pending is not None:
            out.write("\t".join(pending[i] for i in idx) + "\n")
            n += 1
        for line in reader:
            fields = line.rstrip("\n").split("\t")
            out.write("\t".join(fields[i] for i in idx) + "\n")
            n += 1
            if n % 1_000_000 == 0:
                print(f"  ...{n:,} rows", file=sys.stderr)
    print(f"wrote {n:,} rows to {out_path}", file=sys.stderr)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = p.add_mutually_exclusive_group()
    src.add_argument("--genome", default="hg38",
                     help="UCSC genome to download rmsk for (default: hg38)")
    src.add_argument("--input",
                     help="local raw rmsk file (.txt.gz or headered TSV) "
                          "instead of downloading")
    p.add_argument("-o", "--output", required=True,
                   help="output path (.tsv.gz)")
    args = p.parse_args()

    if args.input:
        reader = _open_maybe_gzip(args.input)
    else:
        url = UCSC_URL.format(genome=args.genome)
        print(f"downloading {url} ...", file=sys.stderr)
        resp = urllib.request.urlopen(url)
        reader = io.TextIOWrapper(gzip.GzipFile(fileobj=io.BufferedReader(resp)))
    convert(reader, args.output)


if __name__ == "__main__":
    main()
