#!/usr/bin/env bash
# Regenerate dissertation PDF and DOCX from markdown.
set -euo pipefail
cd "$(dirname "$0")/.."
BASE="Hybrid Machine Learning Model for Early Diabetes Prediction Using Lifestyle and Clinical Data"

pandoc "${BASE}.md" -o "${BASE}.docx" \
  --from markdown \
  --to docx \
  --resource-path=.:figures \
  --standalone

pandoc "${BASE}.md" -o "${BASE}.pdf" \
  --from markdown \
  --pdf-engine=tectonic \
  --resource-path=.:figures \
  -V geometry:margin=1in \
  -V fontsize=11pt \
  -V documentclass=article

echo "Wrote ${BASE}.docx and ${BASE}.pdf"
