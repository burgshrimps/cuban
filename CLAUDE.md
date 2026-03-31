# CLAUDE.md

CUBAN visualizes structural variants (SVs) from BAM files as publication-quality multi-panel PNGs.

## Environment

- Python 3.10 via Conda (env: `svlearn`)
- Setup: `conda env create -f environment.yml`

## Project Structure

- `cuban_lib/` — Core library (utils.py for BAM processing, visualize.py for plotting)
- `app/` — Static web frontend for manual variant curation
- `sv-visualize.ipynb` — Main notebook entry point
- `resources/` — Per-chromosome baseline coverage JSONs (Illumina, PacBio)

## Verification

After modifying `.py` files: `python3 -m py_compile <file>`

## Docs

- `docs/architecture.md` — Data flow, module responsibilities, visualization layout, sample dict format
- `docs/domain.md` — BAM constraints, SV-type differences, color schemes, thresholds

## Obsidian

- company: lucid
- project: cuban
- tag: #lucid
- todoist_project: Lucid Genomics
- todoist_section: Cuban
