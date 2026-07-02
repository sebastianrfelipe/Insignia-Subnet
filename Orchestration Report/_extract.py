import sys, io
from pypdf import PdfReader

src = r'c:\Projects\Insignia-Subnet\Orchestration Report\Orchestration Report — 2026-07-02T01-58-22.pdf'
out = r'c:\Projects\Insignia-Subnet\Orchestration Report\_extracted_2026-07-02.txt'

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
