#!/usr/bin/env bash
# Build a self-contained Overleaf upload bundle from docs/.
#
# Overleaf needs only the sources the manuscript actually compiles from:
# the two .tex files, the .bib, and the figure PDFs. This script copies
# exactly those into overleaf_bundle/ and zips it, so you can drag the
# single zip into a new Overleaf project ("New Project -> Upload Project").
#
# Usage:  bash make_overleaf.sh
# Output: overleaf_bundle/  and  overleaf_bundle.zip
set -euo pipefail
cd "$(dirname "$0")"

OUT=overleaf_bundle
rm -rf "$OUT" "$OUT.zip"
mkdir -p "$OUT/figures"

cp docs/paper.tex docs/supplementary.tex docs/ref.bib "$OUT/"
# only the figures referenced by \includegraphics (all live in docs/figures)
cp docs/figures/*.pdf "$OUT/figures/" 2>/dev/null || true

cat > "$OUT/README.txt" <<'TXT'
Overleaf upload bundle.

1. Overleaf -> New Project -> Upload Project -> select overleaf_bundle.zip
2. Set the main document to paper.tex (Menu -> Main document).
3. Compiler: pdfLaTeX. Requires the revtex4-2 class (present on Overleaf).
4. supplementary.tex is a second compilable document in the same project.

Files:
  paper.tex, supplementary.tex, ref.bib, figures/*.pdf
Nothing else is needed to compile.
TXT

if command -v zip >/dev/null 2>&1; then
  ( cd "$OUT" && zip -qr "../$OUT.zip" . )
  echo "wrote $OUT/ and $OUT.zip"
else
  # 'zip' is absent on many Windows Git Bash installs; python is a project
  # requirement, so fall back to the stdlib zipfile module.
  PY="$(command -v python3 || command -v python || true)"
  if [ -n "$PY" ]; then
    "$PY" - "$OUT" <<'PYEOF'
import os, sys, zipfile
out = sys.argv[1]
with zipfile.ZipFile(out + ".zip", "w", zipfile.ZIP_DEFLATED) as z:
    for root, _, files in os.walk(out):
        for f in sorted(files):
            p = os.path.join(root, f)
            z.write(p, os.path.relpath(p, out))
PYEOF
    echo "wrote $OUT/ and $OUT.zip (via python zipfile)"
  else
    echo "wrote $OUT/ (no 'zip' or python found; zip the folder contents yourself -- files must sit at the zip root)"
  fi
fi
