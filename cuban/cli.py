"""cuban: render structural-variant read-support plots (coverage, CIGAR,
insert size, orientation) from BAMs around a breakpoint or breakpoint pair."""

import argparse
import os
import re

import pysam

from .repeats import empty_repeats, load_repeats
from .utils import infer_technology
from .visualize import cuban, cuban_bnd

_BND_MATE_RE = re.compile(r'[\[\]]([^\[\]:]+):(\d+)[\[\]]')

SV_TYPES = ('DEL', 'DUP', 'INS', 'INV', 'BND')
# Public technology vocabulary (sr/lr) mapped to the internal 'ill'/'pb' values.
TECH_ALIASES = {'sr': 'ill', 'ill': 'ill', 'lr': 'pb', 'pb': 'pb'}


def _bam_index_missing(bam):
    """True if `bam` has no discoverable .bai/.csi index next to it."""
    candidates = (bam + '.bai', bam + '.csi',
                  os.path.splitext(bam)[0] + '.bai', os.path.splitext(bam)[0] + '.csi')
    return not any(os.path.isfile(c) for c in candidates)


def _parse_sample_spec(spec, chrom):
    """Parse `name:bam`."""
    parts = spec.split(':')
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"--sample {spec!r}: expected name:bam (the bam path must not contain a colon)"
        )
    name, bam = parts

    if not os.path.isfile(bam):
        raise argparse.ArgumentTypeError(f"--sample {name}: BAM not found: {bam}")
    if _bam_index_missing(bam):
        raise argparse.ArgumentTypeError(
            f"--sample {name}: no index found for {bam} (looked for .bai/.csi). "
            f"Run: samtools index {bam}"
        )

    return name, {
        'technology': None,   # resolved later: --tech override or inference
        'bam_name': bam,
        'baseline_cov': 'auto',  # resolved later: --baseline-cov override or mosdepth
    }


_EXAMPLES = """\
EXAMPLES:
  # Single deletion, one sample
  cuban --sv-type DEL --chrom chr1 --start 20000 --end 25000 \\
        --sample proband:sample.bam -o out.png

  # Trio (one figure block per sample)
  cuban --sv-type DEL --chrom chr1 --start 20000 --end 25000 \\
        --sample proband:proband.bam \\
        --sample mother:mother.bam \\
        --sample father:father.bam \\
        -o trio.png

  # Breakpoint junction (BND), two independent loci
  cuban --bnd --chrom chr1 --start 20000 --end 20001 \\
        --chrom-b chr5 --start-b 90000 --end-b 90001 \\
        --sample proband:sample.bam -o bnd.png

  # VCF batch mode: one PNG per record in --outdir
  cuban --vcf variants.vcf --outdir out/ \\
        --sample proband:sample.bam
"""


def _build_parser():
    from . import __version__

    parser = argparse.ArgumentParser(
        prog='cuban',
        description=__doc__,
        epilog=_EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--version', action='version', version=f'cuban {__version__}')

    parser.add_argument('--sv-type', choices=SV_TYPES,
                        help='SV type. Required unless --vcf is given. BND requires '
                             '--chrom-b/--start-b/--end-b (or pass --bnd instead, '
                             'which implies BND).')
    parser.add_argument('--bnd', action='store_true',
                        help='render two independent loci side by side (BND mode). '
                             'Implies --sv-type BND; conflicts with any other --sv-type.')
    parser.add_argument('--chrom', help='chromosome of the SV (or first locus for --bnd). '
                        'Required unless --vcf is given.')
    parser.add_argument('--start', type=int, help='start position (1-based). '
                        'Required unless --vcf is given.')
    parser.add_argument('--end', type=int, help='end position (1-based, inclusive). '
                        'Required unless --vcf is given.')
    parser.add_argument('--chrom-b', help='chromosome of the second locus. Required in BND mode.')
    parser.add_argument('--start-b', type=int, help='start position of the second locus (1-based). Required in BND mode.')
    parser.add_argument('--end-b', type=int, help='end position of the second locus (1-based, inclusive). Required in BND mode.')

    parser.add_argument('--vcf', help='VCF/BCF of structural variants to batch-render, one PNG per '
                        'record. Requires --outdir and at least one --sample; mutually exclusive '
                        'with --chrom/--start/--end/--chrom-b/--start-b/--end-b/--sv-type/--bnd/--out.')
    parser.add_argument('--outdir', help='output directory for --vcf batch mode.')

    parser.add_argument('--repeats',
                        help='RepeatMasker TSV[.gz] with UCSC rmsk columns '
                             '(genoName/genoStart/genoEnd/repClass). Without this '
                             'flag the hg38 table is downloaded automatically on '
                             'first use (~40 MB, kept in annot/ in the repo, or '
                             '~/.cuban for installs without a checkout).')
    parser.add_argument('--no-repeats', action='store_true',
                        help='render the repeat track empty without warning.')
    parser.add_argument('--sample', action='append', required=True, metavar='NAME:BAM',
                        help='sample to render (required; repeat for multiple samples, one '
                             'figure block each). The BAM must be indexed (.bai/.csi) and its '
                             'path must not contain a colon. The sequencing technology '
                             '(short-read vs long-read) is inferred from the read lengths; '
                             'use --tech to set it explicitly.')
    parser.add_argument('--tech', action='append', metavar='SAMPLE:TECH',
                        help='set the sequencing technology for a sample explicitly instead of '
                             'inferring it: SAMPLE:sr (short-read) or SAMPLE:lr (long-read). '
                             'Repeatable, one per sample.')
    parser.add_argument('--baseline-cov', action='append', metavar='SAMPLE:COV',
                        help='override the baseline coverage (the horizontal reference line) '
                             'for a sample, e.g. PROBAND:32.5. Repeatable, one per sample. '
                             "By default it is the chromosome's mean depth from the sample's "
                             'own BAM (computed via mosdepth and cached).')

    parser.add_argument('-o', '--out', help='output PNG path. Required unless --vcf is given.')
    parser.add_argument('--padding', type=int, default=None,
                        help='context window around the SV (bp). Default: adaptive, '
                             'max(1500, round((end-start)/10)) (plain 1500 for --bnd).')
    parser.add_argument('--window', type=int, default=100,
                        help='CIGAR window around each breakpoint (bp). Default 100.')
    parser.add_argument('--no-collapse-ins', dest='collapse_ins', action='store_false',
                        help='do not collapse insertion runs into a single column of the read '
                             'panel (collapsed by default so large insertions do not stretch it).')
    parser.add_argument('--sv-len', type=int, default=None,
                        help='explicit SV length shown in the figure title (non-BND only). '
                             'Default: derived from --start/--end.')
    parser.add_argument('--cache-dir', default=None,
                        help='directory where mosdepth coverage output is cached. Default: a '
                             "cuban_coverage/ folder next to each BAM (like a .bai index), or "
                             '~/.cuban/coverage when the BAM directory is not writable; '
                             '$CUBAN_DATA_DIR/coverage when that variable is set.')
    parser.add_argument('--max-reads', type=int, default=5000,
                        help='maximum number of reads per read panel; deeper regions are '
                             'downsampled deterministically. Default 5000.')
    parser.add_argument('--bin-size', type=int, default=None,
                        help='coverage bin size in bp. Default: auto (1 for SVs <= 100 kb, '
                             'else ~size/2000; always 1 for BND).')

    return parser


def _resolve_bin_size(bin_size_arg, sv_type, size=None):
    """ Resolves the coverage bin size: explicit --bin-size wins; otherwise auto: always 1 for
    BND (two independent breakpoints, not a sized interval), 1 for SVs <= 100 kb, else
    size // 2000 (minimum 1). """
    if bin_size_arg is not None:
        return bin_size_arg
    if sv_type == 'BND':
        return 1
    return 1 if size <= 100_000 else max(1, size // 2000)


def _check_chroms_in_samples(chroms, samples):
    """Verify each name in `chroms` is a contig in every sample's BAM header.

    Raises SystemExit with a friendly message (listing example contig names
    from the offending BAM, to help catch chr1-vs-1 style mismatches) on the
    first sample/chrom combination that doesn't match.
    """
    for name, sample in samples.items():
        bam = sample['bam_name']
        with pysam.AlignmentFile(bam) as af:
            refs = af.references
        missing = sorted(c for c in chroms if c not in refs)
        if missing:
            examples = ', '.join(refs[:5])
            raise SystemExit(
                f"--sample {name} ({bam}): chromosome(s) {', '.join(missing)} not found in "
                f"this BAM's header. Example contigs in this BAM: {examples}"
            )


def _vcf_unique_chroms(records):
    """Unique chromosome names referenced by `records` (both loci for BND)."""
    chroms = set()
    for record in records:
        chroms.add(record.chrom)
        sv_type = _sv_type_from_record(record)
        if sv_type == 'BND' and record.alts:
            mate = _BND_MATE_RE.search(record.alts[0])
            if mate is not None:
                chroms.add(mate.group(1))
    return chroms


def _sv_type_from_record(record):
    """SV type from INFO/SVTYPE, falling back to a symbolic ALT like <DEL> or <DEL:ME>."""
    sv_type = record.info.get('SVTYPE')
    if not sv_type and record.alts and len(record.alts) == 1:
        alt = record.alts[0]
        if alt.startswith('<') and alt.endswith('>'):
            sv_type = alt[1:-1].split(':')[0]
    return sv_type


def _record_outfile(record, sv_type, outdir):
    rid = record.id
    if rid and rid != '.':
        return os.path.join(outdir, f'{rid}.png')
    return os.path.join(outdir, f'{record.chrom}_{record.pos}_{sv_type}.png')


def _render_vcf_record(record, samples, rep_df, args):
    """Render one VCF record with `cuban`/`cuban_bnd`.

    Returns (outfile, rendered) where `rendered` is False if the PNG already
    existed and was left in place. Raises ValueError if the record's SV type
    or (for BND) mate locus can't be determined.
    """
    sv_type = _sv_type_from_record(record)
    if sv_type not in SV_TYPES:
        raise ValueError(f'missing/unrecognized SVTYPE ({sv_type!r})')

    outfile = _record_outfile(record, sv_type, args.outdir)
    if os.path.exists(outfile):
        return outfile, False

    if sv_type == 'BND':
        mate = _BND_MATE_RE.search(record.alts[0]) if record.alts else None
        if mate is None:
            raise ValueError(f'could not parse BND mate locus from ALT {record.alts!r}')
        mate_chrom, mate_pos = mate.group(1), int(mate.group(2))
        padding = args.padding if args.padding is not None else 1500
        bin_size = _resolve_bin_size(args.bin_size, 'BND')
        cuban_bnd(
            samples=samples,
            rep_df=rep_df,
            chromA=record.chrom, startA=record.pos, endA=record.pos + 1,
            chromB=mate_chrom, startB=mate_pos, endB=mate_pos + 1,
            padding=padding, window=args.window,
            collapse_ins=args.collapse_ins, outfile=outfile,
            cache_dir=args.cache_dir, max_reads=args.max_reads, bin_size=bin_size,
        )
    else:
        start, end = record.pos, record.stop
        if sv_type == 'INS' and end <= start:
            end = start + 1
        padding = args.padding if args.padding is not None else max(1500, round((end - start) / 10))
        bin_size = _resolve_bin_size(args.bin_size, sv_type, end - start)
        cuban(
            samples=samples,
            rep_df=rep_df,
            sv_type=sv_type,
            chrom=record.chrom, start=start, end=end,
            padding=padding, window=args.window,
            collapse_ins=args.collapse_ins, outfile=outfile,
            sv_len=None,
            cache_dir=args.cache_dir, max_reads=args.max_reads, bin_size=bin_size,
        )
    return outfile, True


def _run_vcf_batch(args, samples, rep_df):
    os.makedirs(args.outdir, exist_ok=True)

    with pysam.VariantFile(args.vcf) as vcf_in:
        records = list(vcf_in)
    n = len(records)

    _check_chroms_in_samples(_vcf_unique_chroms(records), samples)

    n_rendered = n_skipped = n_failed = 0
    for i, record in enumerate(records, start=1):
        rid = record.id if record.id else '.'
        try:
            outfile, rendered = _render_vcf_record(record, samples, rep_df, args)
        except Exception as e:
            print(f'[cuban] ({i}/{n}) {rid}: WARNING skipping - {e}')
            n_failed += 1
            continue
        if rendered:
            print(f'[cuban] ({i}/{n}) {rid} -> {outfile}')
            n_rendered += 1
        else:
            print(f'[cuban] ({i}/{n}) {rid} -> {outfile} ... skipped (exists)')
            n_skipped += 1

    print(f'[cuban] rendered {n_rendered}, skipped {n_skipped}, failed {n_failed}')
    if n_failed:
        raise SystemExit(1)


def main(argv=None):
    args = _build_parser().parse_args(argv)

    if args.vcf is not None:
        conflicting = [flag for flag, val in (
            ('--chrom', args.chrom), ('--start', args.start), ('--end', args.end),
            ('--chrom-b', args.chrom_b), ('--start-b', args.start_b), ('--end-b', args.end_b),
            ('--sv-type', args.sv_type), ('--out', args.out),
        ) if val is not None]
        if args.bnd:
            conflicting.append('--bnd')
        if conflicting:
            raise SystemExit(f"--vcf cannot be combined with {', '.join(conflicting)}.")
        if args.outdir is None:
            raise SystemExit('--vcf requires --outdir.')
    else:
        if args.outdir is not None:
            raise SystemExit('--outdir requires --vcf.')
        if args.chrom is None or args.start is None or args.end is None:
            raise SystemExit('--chrom/--start/--end are required (or use --vcf for batch mode).')
        if args.out is None:
            raise SystemExit('--out is required (or use --vcf/--outdir for batch mode).')

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

    if args.no_repeats:
        rep_df = empty_repeats()
    else:
        rep_df = load_repeats(args.repeats)

    samples = {}
    for spec in args.sample:
        try:
            name, sample_dict = _parse_sample_spec(spec, args.chrom)
        except argparse.ArgumentTypeError as e:
            raise SystemExit(str(e)) from e
        if name in samples:
            raise SystemExit(f'duplicate --sample name: {name}')
        samples[name] = sample_dict

    for item in args.baseline_cov or []:
        sample_name, sep, cov = item.rpartition(':')
        if not sep or sample_name not in samples:
            raise SystemExit(
                f"--baseline-cov {item!r}: expected SAMPLE:COV with SAMPLE one of "
                f"{', '.join(samples)}"
            )
        try:
            samples[sample_name]['baseline_cov'] = float(cov)
        except ValueError as e:
            raise SystemExit(
                f"--baseline-cov {item!r}: COV must be a number, got {cov!r}"
            ) from e

    tech_overrides = {}
    for item in args.tech or []:
        sample_name, sep, tech = item.rpartition(':')
        if not sep or sample_name not in samples:
            raise SystemExit(
                f"--tech {item!r}: expected SAMPLE:TECH with SAMPLE one of "
                f"{', '.join(samples)}"
            )
        if tech.lower() not in TECH_ALIASES:
            raise SystemExit(
                f"--tech {item!r}: technology must be 'sr' (short-read) or 'lr' (long-read)"
            )
        tech_overrides[sample_name] = TECH_ALIASES[tech.lower()]

    for name, sample in samples.items():
        if name in tech_overrides:
            sample['technology'] = tech_overrides[name]
        else:
            sample['technology'] = infer_technology(sample['bam_name'])
            label = 'long-read (lr)' if sample['technology'] == 'pb' else 'short-read (sr)'
            print(f'[cuban] {name}: inferred {label} technology '
                  f'(override with --tech {name}:sr|lr)')

    if args.vcf is not None:
        _run_vcf_batch(args, samples, rep_df)
        return

    chroms = {args.chrom}
    if args.chrom_b is not None:
        chroms.add(args.chrom_b)
    _check_chroms_in_samples(chroms, samples)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    n_samples = len(samples)
    if sv_type == 'BND':
        print(f'[cuban] rendering BND {args.chrom}:{args.start:,}-{args.end:,} <-> '
              f'{args.chrom_b}:{args.start_b:,}-{args.end_b:,} ({n_samples} sample(s))...')
    else:
        print(f'[cuban] rendering {sv_type} {args.chrom}:{args.start:,}-{args.end:,} '
              f'({n_samples} sample(s))...')

    if sv_type == 'BND':
        padding = args.padding if args.padding is not None else 1500
        bin_size = _resolve_bin_size(args.bin_size, 'BND')
        cuban_bnd(
            samples=samples,
            rep_df=rep_df,
            chromA=args.chrom, startA=args.start, endA=args.end,
            chromB=args.chrom_b, startB=args.start_b, endB=args.end_b,
            padding=padding, window=args.window,
            collapse_ins=args.collapse_ins, outfile=args.out,
            cache_dir=args.cache_dir, max_reads=args.max_reads, bin_size=bin_size,
        )
    else:
        padding = args.padding if args.padding is not None else max(1500, round((args.end - args.start) / 10))
        bin_size = _resolve_bin_size(args.bin_size, sv_type, args.end - args.start)
        cuban(
            samples=samples,
            rep_df=rep_df,
            sv_type=sv_type,
            chrom=args.chrom, start=args.start, end=args.end,
            padding=padding, window=args.window,
            collapse_ins=args.collapse_ins, outfile=args.out,
            sv_len=args.sv_len,
            cache_dir=args.cache_dir, max_reads=args.max_reads, bin_size=bin_size,
        )

    print(f'[cuban] wrote {args.out}')
