#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
python "$ROOT/code/run_confirmatory.py"
cd "$ROOT/docs"
pdflatex -interaction=nonstopmode manuscript.tex >/dev/null
pdflatex -interaction=nonstopmode manuscript.tex >/dev/null
pdflatex -interaction=nonstopmode manuscript.tex >/dev/null
pdflatex -interaction=nonstopmode supplement.tex >/dev/null
pdflatex -interaction=nonstopmode supplement.tex >/dev/null
pdflatex -interaction=nonstopmode supplement.tex >/dev/null
echo "Reproducibility run and PDF build complete."
