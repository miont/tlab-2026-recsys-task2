#!/usr/bin/env bash
# Build report.pdf from report.md with pandoc + XeLaTeX (settings in latex.yaml).
set -euo pipefail
cd "$(dirname "$0")"
pandoc report.md --defaults latex.yaml -o report.pdf
echo "docs/report/report.pdf"
