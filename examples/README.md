# cuban examples

Synthetic fixtures for trying cuban without a real BAM: `example.bam`/`example_hp.bam`
(a 50 kb `chr1` region with a homozygous 5 kb deletion, the latter HP-tagged), `example.vcf`
(three DEL/DUP records for batch mode), and `repeats.tsv`. Regenerate them with
`python examples/make_example_data.py`. `example_figure.png` was rendered
from `example_hp.bam` with the command below.

```bash
# Single SV on example.bam
cuban --sv-type DEL --chrom chr1 --start 20000 --end 25000 \
    --sample HG002:examples/data/example.bam \
    --repeats examples/data/repeats.tsv \
    --out examples/output/example.png

# Batch mode: one PNG per record in example.vcf
cuban --vcf examples/data/example.vcf --outdir examples/output \
    --sample HG002:examples/data/example.bam \
    --repeats examples/data/repeats.tsv
```
