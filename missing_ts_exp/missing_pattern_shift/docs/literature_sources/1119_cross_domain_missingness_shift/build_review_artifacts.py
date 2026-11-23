#!/usr/bin/env python3
"""Rebuild references, a synthetic mask diagram, and the reading PDF.
Requires reportlab and optionally pdftoppm. Does not access experiment data.
"""
from pathlib import Path
import csv
import html
import json
import random
import re
import shutil
import subprocess
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image

ROOT = Path(__file__).resolve().parent
REPORT = ROOT.parent.parent / "1119-文献调研-跨领域缺失模式变化.md"
pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
FONT = "STSong-Light"
BLUE = colors.HexColor("#315C82")
ORANGE = colors.HexColor("#E6A24A")
INK = colors.HexColor("#203348")

def bibliography():
    refs = json.loads((ROOT / "references.json").read_text())
    with (ROOT / "references.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(refs[0]))
        writer.writeheader()
        writer.writerows(refs)
    entries = []
    for r in refs:
        preprint = "preprint" in r["venue"]
        journal = r["id"] in {"R12", "R13", "R19"}
        kind = "misc" if preprint else "article" if journal else "inproceedings"
        d = {"title": "{"+r["title"]+"}", "author": " and ".join(r["authors"].split("; ")),
             "year": str(r["year"]), "url": r["url"]}
        if journal:
            d["journal"] = r["venue"]
        elif not preprint:
            d["booktitle"] = r["venue"]
        if r["doi"]:
            d["doi"] = r["doi"]
        if "arxiv.org/abs/" in r["url"] and preprint:
            d["eprint"] = r["url"].rsplit("/", 1)[-1]
            d["archivePrefix"] = "arXiv"
        d["note"] = "Checked 2026-11-19. " + r["read_depth"]
        entries.append("@"+kind+"{"+r["id"]+",\n"+",\n".join(
            "  "+k+" = {"+v+"}" for k, v in d.items())+"\n}\n")
    (ROOT / "references.bib").write_text("\n".join(entries), encoding="utf-8")

def mask_figure():
    p = ROOT / "mask_geometry.pdf"
    c = canvas.Canvas(str(p), pagesize=(800, 440))
    c.setTitle("Equal missing rates, different observation geometry")
    c.setFillColor(INK); c.setFont(FONT, 20)
    c.drawString(35, 405, "相同缺失率，不同观测结构")
    c.setFont(FONT, 11)
    c.drawString(35, 382, "合成示意：每通道 24 个位置，12 个缺失；不表示真实实验结果")
    for x, color, label in [(595, BLUE, "观测"), (665, ORANGE, "缺失")]:
        c.setFillColor(color); c.rect(x, 400, 12, 12, fill=1, stroke=0)
        c.setFillColor(INK); c.drawString(x+18, 401, label)
    rng = random.Random(20261119)
    pairs = [(a,b) for a in range(12) for b in range(a+7, 19)]
    masks = [
        [set(rng.sample(range(24), 12)) for _ in range(6)],
        [set(range(a,a+6)) | set(range(b,b+6)) for a,b in
         [rng.choice(pairs) for _ in range(6)]],
        [set(range(2,8)) | set(range(15,21)) for _ in range(6)]
    ]
    labels = [
        ("P：分散点缺失", "通道内分散；各通道独立"),
        ("B-ind：独立块缺失", "每通道两个块；位置不同"),
        ("B-sync：同步块缺失", "所有通道共享缺口位置"),
    ]
    for idx, (mask, label) in enumerate(zip(masks, labels)):
        top = 337 - idx*105
        c.setFillColor(INK); c.setFont(FONT, 13); c.drawString(35, top-9, label[0])
        c.setFont(FONT, 10); c.drawString(35, top-28, label[1])
        for channel, missing in enumerate(mask):
            assert len(missing) == 12
            y = top-channel*11-9
            c.setFillColor(INK); c.setFont(FONT, 7)
            c.drawRightString(269, y+1, "c"+str(channel+1))
            for t in range(24):
                c.setFillColor(ORANGE if t in missing else BLUE)
                c.rect(278+t*19, y, 17.5, 9, fill=1, stroke=0)
        c.setFillColor(INK); c.setFont(FONT, 8)
        c.drawString(278, top-76, "过去")
        c.drawRightString(734, top-76, "预测起点前")
    c.setFont(FONT, 10)
    c.drawString(35, 20, "时间连续性与通道同步性是两个不同的变化轴；缺失率相同不代表信息量相同。")
    c.save()
    if shutil.which("pdftoppm"):
        subprocess.run(["pdftoppm", "-singlefile", "-scale-to", "1600", "-png",
                        str(p), str(ROOT / "mask_geometry")], check=True)

def inline(text):
    text = text.replace("↔", " <-> ").replace("•", "-").replace("·", " / ")
    text = html.escape(text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", lambda m:
        '<link href="'+m.group(2)+'" color="#245B87">'+m.group(1)+"</link>", text)
    return re.sub(r"\*\*(.+?)\*\*", r'<font color="#173D60">\1</font>', text)

def formula_text(text):
    for a,b in [
        (r"\mathbb R", "R"), (r"\mathsf T", "T"), (r"\rm ", ""),
        (r"\qquad", "  "), (r"\mid", " | "), (r"\in", " in "),
        (r"\times", " x "), (r"\odot", " * "), (r"\le", " <= "),
        (r"\sum", " sum "), (r"\int", " integral "), (r"\sigma", "sigma"),
        (r"\epsilon", "epsilon"), (r"\Delta", "Delta"), (r"\to", " -> "),
        (r"\,", " "), (r"\;", " "), (r"\left", ""), (r"\right", "")
    ]:
        text = text.replace(a,b)
    return text.replace("{", "").replace("}", "").strip()

def reading_pdf():
    width = 487
    normal = ParagraphStyle("body", fontName=FONT, fontSize=9.3, leading=15,
                            wordWrap="CJK", spaceAfter=6, textColor=INK)
    small = ParagraphStyle("table", parent=normal, fontSize=7.6, leading=11, spaceAfter=0)
    title = ParagraphStyle("title", parent=normal, fontSize=20, leading=28, spaceAfter=15)
    h2 = ParagraphStyle("h2", parent=normal, fontSize=14, leading=21,
                        spaceBefore=12, spaceAfter=7, keepWithNext=True, textColor=BLUE)
    h3 = ParagraphStyle("h3", parent=h2, fontSize=11.5, leading=18, spaceBefore=8)
    formula = ParagraphStyle("formula", parent=normal, fontSize=9, leading=15,
                             backColor=colors.HexColor("#F1F5F8"), borderPadding=7)
    story = []
    lines = REPORT.read_text().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1; continue
        if line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                parts = [v.strip() for v in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r"[-:\s]+", v) for v in parts):
                    rows.append([Paragraph(inline(v), small) for v in parts])
                i += 1
            if rows:
                n = len(rows[0])
                weights = {3:[0.25,0.34,0.41], 4:[0.12,0.27,0.32,0.29],
                           5:[0.07,0.25,0.09,0.17,0.42]}.get(n, [1/n]*n)
                table = Table(rows, colWidths=[width*x for x in weights], repeatRows=1, hAlign="LEFT")
                table.setStyle(TableStyle([
                    ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#E6EEF5")),
                    ("VALIGN",(0,0),(-1,-1),"TOP"),
                    ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#D5DEE6")),
                    ("LEFTPADDING",(0,0),(-1,-1),5),("RIGHTPADDING",(0,0),(-1,-1),5),
                    ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5)
                ]))
                story.extend([table, Spacer(1,8)])
            continue
        if line == r"\[":
            block = []
            i += 1
            while i < len(lines) and lines[i].strip() != r"\]":
                block.append(lines[i].strip()); i += 1
            story.append(Paragraph(html.escape(formula_text(" ".join(block))), formula))
            i += 1; continue
        if line.startswith("!["):
            m = re.search(r"\]\((.+)\)", line)
            if m:
                p = REPORT.parent / m.group(1)
                if p.exists():
                    iw, ih = ImageReader(str(p)).getSize()
                    story.append(Image(str(p), width=width, height=width*ih/iw))
            i += 1; continue
        if line.startswith("# "):
            style, text = title, line[2:]
        elif line.startswith("## "):
            style, text = h2, line[3:]
        elif line.startswith("### "):
            style, text = h3, line[4:]
        else:
            style, text = normal, line
            if text.startswith("- "):
                text = "• " + text[2:]
            elif text.startswith("> "):
                text = text[2:]
            text = re.sub(r"\\\((.+?)\\\)", lambda m:formula_text(m.group(1)), text)
        story.append(Paragraph(inline(text), style))
        i += 1

    def page_footer(c, doc):
        c.saveState(); c.setFillColor(colors.HexColor("#64778A")); c.setFont(FONT,8)
        c.drawString(54,28,"跨领域缺失模式变化调研 / 2026-11-19")
        c.drawRightString(541,28,str(doc.page)); c.restoreState()
    doc = SimpleDocTemplate(str(REPORT.with_suffix(".pdf")), pagesize=(595,842),
        rightMargin=54,leftMargin=54,topMargin=45,bottomMargin=45,
        title="跨领域缺失模式变化文献调研", author="Research notes")
    doc.build(story, onFirstPage=page_footer, onLaterPages=page_footer)

if __name__ == "__main__":
    bibliography()
    mask_figure()
    reading_pdf()
    print("Built references.csv, references.bib, mask_geometry.pdf/png, review PDF")

