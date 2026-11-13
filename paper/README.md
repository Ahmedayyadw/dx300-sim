# Paper: ICEEE 2027, Track 2

IEEE conference format (`IEEEtran`, `[conference]`), 10 pt, double column, 4-8 pages.

## Files

- `main.tex`: paper source. Red `TODO:` lines mark content still to write.
- `table1.tex`: Table 1, placeholder plant results (from `rl/08b_eval_n10.csv`).
- `refs.bib`: references.
- `figs/`: figures.

## Build

From `paper/`:

```bash
latexmk -pdf main.tex
```

Clean build files: `latexmk -c`. Needs a TeX distribution with `IEEEtran`
(e.g. `texlive-publishers` on Debian/Ubuntu).
