# OGD invariant-geometry submission package

This package contains a **personal academic research manuscript** and a fully reproducible synthetic falsification suite.

## Core claim boundary
The computational study tests whether an invariant-reduced `M4 x S3` Abelian gauge compactification is internally transformation-selective, reconstructible within specified comparator families, and falsifiable under perturbations. It **does not provide empirical evidence that photons have extra dimensions**, does not establish a Standard-Model extension, and does not derive collider or dark-matter phenomenology.

## Files
- `docs/manuscript.pdf` / `docs/manuscript.tex` - main journal-style manuscript.
- `docs/supplement.pdf` / `docs/supplement.tex` - full supplementary appendix and protocols.
- `docs/references.bib` - bibliography.
- `code/run_confirmatory.py` - complete confirmatory simulation.
- `results/confirmatory_config.json` - frozen confirmatory settings and seed.
- `results/gate_results.json` - pass/fail audit of the prespecified gates.
- `results/*.csv`, `results/*.json` - machine-readable results.
- `figures/*.png` - 300-dpi publication figures.
- `requirements.txt` - minimal Python dependencies.
- `run_all.sh` - one-command rerun and LaTeX build.
- `SUBMISSION_CHECKLIST.md` - pre-submission checks.
- `AUTHOR_DECLARATIONS_TEMPLATE.md` - declarations to complete truthfully for a chosen journal.

## Exact rerun
From the package root:

```bash
python code/run_confirmatory.py
```

The confirmatory random seed is `84320711`. The code makes no network requests and uses no external dataset.

To rebuild PDFs after rerunning:

```bash
cd docs
pdflatex manuscript.tex
pdflatex manuscript.tex
pdflatex manuscript.tex
pdflatex supplement.tex
pdflatex supplement.tex
pdflatex supplement.tex
```

## Confirmatory headline results
- All 9 prospectively fixed confirmatory gates passed.
- Five-mode spherical-dimension recovery at 2% mass noise: 93.06% (10,000 trials).
- Held-out prediction after fitting scale only on modes 1-2: 97.80% correct dimension at 2% noise.
- Joint eigenvalue + multiplicity reconstruction: 98.13% at 2% mass noise and 70% state detection.
- Near-round failure boundary: Berger anisotropy `a/b = 1.02` is detected only 50.14% of the time at 2% noise.
- Finite S3/RP3/T3 benchmark: S3 classified correctly in 4995/5000 trials, RP3 and T3 in 5000/5000 each.

## Submission note
The manuscript is intentionally conservative. Before submission, the author should independently verify the derivations, rerun the code in a clean environment, select a specific target journal, conform formatting to that journal, and complete authorship/funding/conflict/AI-use declarations according to the journal's current policy.
