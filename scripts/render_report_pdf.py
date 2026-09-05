"""Render a vault markdown report (Obsidian flavour) to PDF next to it.

Handles Obsidian's `![[figure.png]]` embeds (resolved against the vault's
experiments/figures) and `[[note|alias]]` links (rendered as plain text), converts the
markdown to HTML with tables, and prints the HTML to PDF with the headless Edge/Chrome
found on the machine. Korean text uses Apple SD Gothic Neo.

    python scripts/render_report_pdf.py ~/Vaults/Research/GeoIndex/planning/weekly-reports/2026-W36.md
    python scripts/render_report_pdf.py <a.md> <b.md> ...        # one PDF per file, same folder
"""
import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown

BROWSERS = [
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
]
CSS = """
@page { size: A4; margin: 18mm 16mm; }
body { font-family: -apple-system, "Helvetica Neue", "Apple SD Gothic Neo", "Noto Sans KR", sans-serif;
       font-size: 10.5pt; line-height: 1.45; color: #111; max-width: 100%; }
h1 { font-size: 18pt; margin: 0 0 8pt; }
h3 { font-size: 13pt; margin: 18pt 0 6pt; border-bottom: 1px solid #999; padding-bottom: 2pt; }
h4 { font-size: 11.5pt; margin: 14pt 0 4pt; }
p, li { margin: 4pt 0; }
blockquote { margin: 6pt 0; padding: 4pt 10pt; border-left: 3px solid #bbb; color: #444; background: #f6f6f6; }
table { border-collapse: collapse; margin: 6pt 0 10pt; font-size: 8.8pt; width: 100%; page-break-inside: auto; }
th, td { border: 1px solid #bbb; padding: 3pt 5pt; vertical-align: top; text-align: left; }
th { background: #eee; }
tr { page-break-inside: avoid; }
code { font-family: Menlo, monospace; font-size: 8.8pt; background: #f2f2f2; padding: 0 2pt; }
img { max-width: 100%; height: auto; display: block; margin: 8pt auto 2pt; page-break-inside: avoid; }
hr { border: 0; border-top: 1px solid #bbb; margin: 10pt 0; }
"""


def vault_root(path: Path) -> Path:
    for p in [path] + list(path.parents):
        if (p / "INDEX.md").exists() and (p / "experiments").is_dir():
            return p
    return path.parent


def preprocess(text: str, figures: Path) -> str:
    def embed(m):
        name = m.group(1).split("|")[0].strip()
        f = figures / name
        return f'<img src="file://{f}" alt="{name}">' if f.exists() else f"*(missing figure {name})*"
    text = re.sub(r"!\[\[([^\]]+)\]\]", embed, text)
    text = re.sub(r"\[\[([^\]|]+)\|([^\]]+)\]\]", r"\2", text)          # [[note|alias]] → alias
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)                     # [[note]] → note
    return text


def to_html(md_text: str, title: str) -> str:
    body = markdown.markdown(md_text, extensions=["tables", "fenced_code", "sane_lists"])
    return f"<!doctype html><html><head><meta charset='utf-8'><title>{title}</title><style>{CSS}</style></head><body>{body}</body></html>"


def print_pdf(html_path: Path, pdf_path: Path) -> None:
    browser = next((b for b in BROWSERS if Path(b).exists()), None)
    if browser is None:
        raise SystemExit("no Edge/Chrome/Chromium found for PDF printing")
    subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf_path}", f"file://{html_path}"],
                   check=True, capture_output=True, timeout=180)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+")
    args = p.parse_args()
    for f in args.files:
        md = Path(f).expanduser().resolve()
        figures = vault_root(md) / "experiments" / "figures"
        html = to_html(preprocess(md.read_text(), figures), md.stem)
        with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False, encoding="utf-8") as t:
            t.write(html)
            html_path = Path(t.name)
        pdf = md.with_suffix(".pdf")
        print_pdf(html_path, pdf)
        html_path.unlink(missing_ok=True)
        print(f"{md.name} → {pdf} ({pdf.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
