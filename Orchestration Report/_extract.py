"""Extract text from an Orchestration Report PDF.

Usage:
    python _extract.py <input.pdf> [output.txt]

If output is omitted, writes alongside the input as <basename>.txt.
"""
import sys, io
from pypdf import PdfReader

if len(sys.argv) < 2:
    sys.exit("Usage: python _extract.py <input.pdf> [output.txt]")

src = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else src.rsplit('.', 1)[0] + '.txt'

r = PdfReader(src)
with io.open(out, 'w', encoding='utf-8') as f:
    f.write(f'PAGES: {len(r.pages)}\n\n')
    for i, p in enumerate(r.pages):
        try:
            t = p.extract_text() or ''
        except Exception as e:
            t = f'[[extract error: {e}]]'
        f.write(f'\n===== PAGE {i} =====\n')
        f.write(t)
print('Wrote', out)
