#!/usr/bin/env bash
# Build docs/report/report.pdf from report.md: pandoc -> self-contained HTML -> headless Chrome.
set -euo pipefail
cd "$(dirname "$0")"
pandoc report.md --standalone --embed-resources --mathml --css report.css \
  --toc --toc-depth=2 -o report.html
"${CHROME:-google-chrome}" --headless --disable-gpu --no-pdf-header-footer \
  --print-to-pdf=report.pdf "file://$PWD/report.html" 2>/dev/null
rm report.html
echo "docs/report/report.pdf"
