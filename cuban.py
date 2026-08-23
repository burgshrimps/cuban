#!/usr/bin/env python3
"""cuban: render structural-variant read-support plots (coverage, CIGAR,
insert size, orientation) from BAMs around a breakpoint or breakpoint pair."""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

CUBAN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CUBAN_DIR))
from cuban_lib.visualize import cuban, cuban_bnd  # noqa: E402


SV_TYPES = ('DEL', 'DUP', 'INS', 'INV', 'BND')
TECHS = ('ill', 'pb')

BASELINE_COV_FILES = {
    'ill': CUBAN_DIR / 'resources' / 'baseline_cov_ill.json',
    'pb':  CUBAN_DIR / 'resources' / 'baseline_cov_pb.json',
}
DEFAULT_REPEATS_TSV = CUBAN_DIR / 'resources' / 'hg38_repeatmasker.tsv'


def _parse_sample_spec(spec, chrom):
    """Parse `name:tech:bam[:baseline_cov[:family[:disease]]]`."""
    parts = spec.split(':')
    if len(parts) < 3:
        raise argparse.ArgumentTypeError(
            f"--sample {spec!r}: need at least name:tech:bam, got {len(parts)} field(s)"
        )
    name, tech, bam = parts[0], parts[1], parts[2]
    baseline_field = parts[3] if len(parts) > 3 and parts[3] else 'auto'
    family_status = parts[4] if len(parts) > 4 and parts[4] else 'index'
    disease_status = parts[5] if len(parts) > 5 and parts[5] else 'affected'

    if tech not in TECHS:
        raise argparse.ArgumentTypeError(
            f"--sample {name}: technology must be one of {TECHS}, got {tech!r}"
        )
    if not os.path.isfile(bam):
        raise argparse.ArgumentTypeError(f"--sample {name}: BAM not found: {bam}")

    if baseline_field == 'auto':
        baseline_cov = _resolve_baseline(tech, chrom, name)
    else:
        try:
            baseline_cov = float(baseline_field)
        except ValueError as e:
            raise argparse.ArgumentTypeError(
                f"--sample {name}: baseline_cov must be a float or 'auto', got {baseline_field!r}"
            ) from e

    return name, {
        'family_status': family_status,
        'disease_status': disease_status,
        'technology': tech,
        'bam_name': bam,
        'baseline_cov': baseline_cov,
    }


def _resolve_baseline(tech, chrom, sample_name):
    path = BASELINE_COV_FILES.get(tech)
    if path is None or not path.is_file():
        raise SystemExit(
            f"sample {sample_name}: baseline_cov='auto' but no shipped table for technology={tech}"
        )
    with open(path) as fh:
        data = json.load(fh)
    if chrom not in data:
        raise SystemExit(
            f"sample {sample_name}: baseline_cov='auto' but chromosome {chrom!r} not in {path.name}; "
            "pass an explicit float in the --sample spec."
        )
    return float(data[chrom])


def _build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--sv-type', choices=SV_TYPES,
                        help='SV type. BND requires --chrom-b/--start-b/--end-b '
                             '(or pass --bnd instead, which implies BND).')
    parser.add_argument('--bnd', action='store_true',
                        help='render two independent loci side by side (BND mode). '
                             'Implies --sv-type BND; conflicts with any other --sv-type.')
    parser.add_argument('--chrom', required=True, help='chromosome of the SV (or first locus for --bnd).')
    parser.add_argument('--start', type=int, required=True, help='start position (1-based).')
    parser.add_argument('--end', type=int, required=True, help='end position (1-based, inclusive).')
    parser.add_argument('--chrom-b', help='chromosome of the second locus (--bnd only).')
    parser.add_argument('--start-b', type=int, help='start position of the second locus (--bnd only).')
    parser.add_argument('--end-b', type=int, help='end position of the second locus (--bnd only).')

    parser.add_argument('--repeats', default=str(DEFAULT_REPEATS_TSV),
                        help='RepeatMasker TSV (chrom/start/end/repclass columns). '
                             'Defaults to the bundled resources/hg38_repeatmasker.tsv.')
    parser.add_argument('--sample', action='append', required=True,
                        help='sample spec (repeatable): name:tech:bam[:baseline_cov[:family[:disease]]]. '
                             'Fields are colon-separated, so the bam path must not contain a colon.')

    parser.add_argument('-o', '--out', required=True, help='output PNG path.')
    parser.add_argument('--padding', type=int, default=1500,
                        help='context window around the SV (bp). Default 1500.')
    parser.add_argument('--window', type=int, default=100,
                        help='CIGAR window around each breakpoint (bp). Default 100.')
    parser.add_argument('--no-collapse-ins', dest='collapse_ins', action='store_false',
                        help='do not collapse insertion runs into a single column.')
    parser.add_argument('--sv-len', type=int, default=None,
                        help='explicit SV length (used in the title; non-BND only).')

    return parser


def main(argv=None):
    args = _build_parser().parse_args(argv)

    if args.bnd:
        if args.sv_type not in (None, 'BND'):
            raise SystemExit(f'--bnd conflicts with --sv-type {args.sv_type} (--bnd implies BND).')
        sv_type = 'BND'
    else:
        if args.sv_type is None:
            raise SystemExit('--sv-type is required (or pass --bnd for translocations).')
        sv_type = args.sv_type

    if sv_type == 'BND':
        missing = [flag for flag, val in (
            ('--chrom-b', args.chrom_b), ('--start-b', args.start_b), ('--end-b', args.end_b),
        ) if val is None]
        if missing:
            raise SystemExit(f"--sv-type BND requires {', '.join(missing)}.")

    if not os.path.isfile(args.repeats):
        raise SystemExit(f'repeats TSV not found: {args.repeats}')
    rep_df = pd.read_csv(args.repeats, sep='\t')

    samples = {}
    for spec in args.sample:
        name, sample_dict = _parse_sample_spec(spec, args.chrom)
        if name in samples:
            raise SystemExit(f'duplicate --sample name: {name}')
        samples[name] = sample_dict

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if sv_type == 'BND':
        cuban_bnd(
            samples=samples,
            rep_df=rep_df,
            chromA=args.chrom, startA=args.start, endA=args.end,
            chromB=args.chrom_b, startB=args.start_b, endB=args.end_b,
            padding=args.padding, window=args.window,
            collapse_ins=args.collapse_ins, outfile=args.out,
        )
    else:
        cuban(
            samples=samples,
            rep_df=rep_df,
            sv_type=sv_type,
            chrom=args.chrom, start=args.start, end=args.end,
            padding=args.padding, window=args.window,
            collapse_ins=args.collapse_ins, outfile=args.out,
            sv_len=args.sv_len,
        )

    print(f'[cuban] wrote {args.out}')


if __name__ == '__main__':
    main()
