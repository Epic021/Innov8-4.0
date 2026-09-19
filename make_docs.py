"""
documentation.md -> documentation.pdf  (not shipped in the zip; tooling only)

    python make_docs.py [documentation.md] [documentation.pdf]

Tries weasyprint first, then headless Microsoft Edge, then a plain fpdf fallback.
Prints the page count at the end (must be <= 4).
"""
import os
import subprocess
import sys

src = sys.argv[1] if len(sys.argv) > 1 else "documentation.md"
dst = sys.argv[2] if len(sys.argv) > 2 else "documentation.pdf"

import markdown  # noqa: E402

md = open(src, encoding="utf-8").read()
body = markdown.markdown(md, extensions=["tables", "fenced_code"])
CSS = """
@page { size: A4; margin: 14mm 15mm; }
body { font-family: Calibri, Arial, Helvetica, sans-serif; font-size: 9.6pt; line-height: 1.28; color: #111; }
h1 { font-size: 15pt; margin: 0 0 4pt 0; }
h2 { font-size: 11.5pt; margin: 9pt 0 3pt 0; border-bottom: 1px solid #999; padding-bottom: 1pt; }
p { margin: 2pt 0 4pt 0; }
ul, ol { margin: 2pt 0 4pt 16pt; padding: 0; }
li { margin: 0 0 1.5pt 0; }
table { border-collapse: collapse; margin: 3pt 0 5pt 0; font-size: 8.8pt; }
th, td { border: 1px solid #aaa; padding: 1.5pt 5pt; text-align: left; vertical-align: top; }
th { background: #eee; }
code { font-family: Consolas, "Courier New", monospace; font-size: 8.6pt; background: #f3f3f3; padding: 0 2px; }
pre { font-family: Consolas, "Courier New", monospace; font-size: 8.4pt; background: #f3f3f3; padding: 4pt; margin: 3pt 0; }
"""
html = "<html><head><meta charset='utf-8'><style>%s</style></head><body>%s</body></html>" % (CSS, body)
html_path = os.path.abspath("documentation.html")
open(html_path, "w", encoding="utf-8").write(html)

done = False
try:
    from weasyprint import HTML
    HTML(string=html, base_url=os.getcwd()).write_pdf(dst)
    done = True
    print("rendered with weasyprint")
except Exception as e:  # noqa: BLE001
    print("weasyprint unavailable:", type(e).__name__, str(e)[:120])

if not done:
    edge = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    if os.path.exists(edge):
        r = subprocess.run([edge, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                            "--print-to-pdf=%s" % os.path.abspath(dst), "file:///" + html_path.replace("\\", "/")],
                           capture_output=True, timeout=120)
        done = os.path.exists(dst)
        print("rendered with Edge headless" if done else "Edge failed: %s" % r.stderr[:200])

if not done:
    from fpdf import FPDF
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", size=8)
    for line in md.splitlines():
        pdf.set_x(pdf.l_margin)
        text = line.encode("latin-1", "replace").decode("latin-1")
        if not text.strip():
            pdf.ln(3)
            continue
        pdf.multi_cell(0, 3.8, text)
    pdf.output(dst)
    print("rendered with fpdf (plain)")

from pypdf import PdfReader  # noqa: E402
n = len(PdfReader(dst).pages)
print("%s: %d page(s)%s" % (dst, n, "" if n <= 4 else "   <-- TOO LONG, must be <= 4"))
