"""CLI-level tests, driven in-process via cuban.cli.main(argv).

main() reports errors by raising SystemExit(message) and successes by simply
returning, so `run_cli` normalizes both into an exit-code-like return value
(mirroring how the `cuban` console script would actually exit).
"""

import cuban
from cuban.cli import main as cuban_main


def run_cli(argv):
    try:
        cuban_main(argv)
    except SystemExit as exc:
        return exc.code
    return 0


def test_single_sv_del_render_writes_png(tmp_path, test_bam, repeats_tsv, shared_cache_dir,
                                          contig, del_start, del_end):
    out_png = tmp_path / "del.png"
    argv = [
        "--sv-type", "DEL",
        "--chrom", contig, "--start", str(del_start), "--end", str(del_end),
        "--sample", f"S1:ill:{test_bam}",
        "--repeats", str(repeats_tsv),
        "--out", str(out_png),
        "--cache-dir", str(shared_cache_dir),
    ]
    code = run_cli(argv)
    assert code == 0
    assert out_png.is_file()
    assert out_png.stat().st_size > 20_000


def test_vcf_batch_renders_three_and_reruns_skip(tmp_path, test_bam, repeats_tsv, test_vcf,
                                                  shared_cache_dir, capsys):
    outdir = tmp_path / "vcf_out"
    argv = [
        "--vcf", str(test_vcf),
        "--outdir", str(outdir),
        "--sample", f"S1:ill:{test_bam}",
        "--repeats", str(repeats_tsv),
        "--cache-dir", str(shared_cache_dir),
    ]

    code = run_cli(argv)
    assert code == 0
    captured = capsys.readouterr()
    assert "rendered 3, skipped 0, failed 0" in captured.out

    expected = ["del1.png", "del2.png", "dup1.png"]
    written = {}
    for name in expected:
        png = outdir / name
        assert png.is_file()
        assert png.stat().st_size > 5_000
        written[name] = png.stat().st_mtime

    # Rerun: every record's output already exists, so nothing should be
    # re-rendered (resumability) - verified both by the summary line and by
    # the PNGs' mtimes being untouched.
    code2 = run_cli(argv)
    assert code2 == 0
    captured2 = capsys.readouterr()
    assert "rendered 0, skipped 3, failed 0" in captured2.out
    for name in expected:
        assert (outdir / name).stat().st_mtime == written[name]


def test_bnd_without_b_locus_flags_names_missing_flags(tmp_path, test_bam, contig, del_start):
    argv = [
        "--sv-type", "BND",
        "--chrom", contig, "--start", str(del_start), "--end", str(del_start),
        "--sample", f"S1:ill:{test_bam}",
        "--out", str(tmp_path / "bnd.png"),
    ]
    code = run_cli(argv)
    assert code is not None and code != 0
    assert "--chrom-b" in code
    assert "--start-b" in code
    assert "--end-b" in code


def test_bnd_conflicts_with_sv_type(tmp_path, test_bam, contig, del_start, del_end):
    argv = [
        "--bnd", "--sv-type", "DEL",
        "--chrom", contig, "--start", str(del_start), "--end", str(del_end),
        "--chrom-b", contig, "--start-b", str(del_end), "--end-b", str(del_end),
        "--sample", f"S1:ill:{test_bam}",
        "--out", str(tmp_path / "bnd.png"),
    ]
    code = run_cli(argv)
    assert code is not None and code != 0
    assert "--bnd" in code
    assert "--sv-type" in code


def test_vcf_conflicts_with_chrom(tmp_path, test_bam, test_vcf, contig):
    argv = [
        "--vcf", str(test_vcf),
        "--outdir", str(tmp_path / "out"),
        "--chrom", contig,
        "--sample", f"S1:ill:{test_bam}",
    ]
    code = run_cli(argv)
    assert code is not None and code != 0
    assert "--chrom" in code


def test_no_repeats_renders_without_missing_repeats_warning(tmp_path, test_bam, shared_cache_dir,
                                                              contig, del_start, del_end, recwarn):
    out_png = tmp_path / "del_norep.png"
    argv = [
        "--sv-type", "DEL",
        "--chrom", contig, "--start", str(del_start), "--end", str(del_end),
        "--sample", f"S1:ill:{test_bam}",
        "--no-repeats",
        "--out", str(out_png),
        "--cache-dir", str(shared_cache_dir),
    ]
    code = run_cli(argv)
    assert code == 0
    assert out_png.is_file()
    assert not any("repeat annotation" in str(w.message) for w in recwarn.list)


def test_version_flag_prints_version(capsys):
    code = run_cli(["--version"])
    assert code == 0
    captured = capsys.readouterr()
    assert cuban.__version__ in captured.out
