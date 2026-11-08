"""
Complete Technical & Research Report Generator for IEEE Paper Preparation.
Project: Digital Persona-Based Synthetic Survey Data Generation for Technology Adoption Research: An SSR-Based Validation Framework
Target: research_documentation/Digital_Persona_Complete_Technical_and_Research_Report.docx
"""

import os
import sys
from pathlib import Path
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "research_documentation"
DOCX_PATH = OUTPUT_DIR / "Digital_Persona_Complete_Technical_and_Research_Report.docx"
PDF_PATH = OUTPUT_DIR / "Digital_Persona_Complete_Technical_and_Research_Report.pdf"
FIG_DIR = OUTPUT_DIR / "figures"

# Color Palette: Professional Academic Navy & Slate
COLOR_PRIMARY_NAVY = RGBColor(27, 54, 93)      # #1B365D - Deep Navy
COLOR_SECONDARY_BLUE = RGBColor(43, 87, 154)   # #2B579A - Academic Blue
COLOR_SLATE_DARK = RGBColor(51, 51, 51)        # #333333 - Dark Slate Charcoal
COLOR_MUTED_GRAY = RGBColor(100, 100, 100)     # #646464 - Muted Caption Gray
HEX_HEADER_NAVY = "1B365D"
HEX_ROW_ALT = "F8F9FA"
HEX_CALLOUT_BG = "F0F4F8"
HEX_CALLOUT_BORDER = "2B579A"
HEX_CODE_BG = "F5F5F5"

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    tcPr.append(parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>'))

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def add_callout(doc, text, title="METHODOLOGICAL HIGHLIGHT / CRITICAL AUDIT FINDING", fill_hex=HEX_CALLOUT_BG, border_hex=HEX_CALLOUT_BORDER):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    cell = tbl.cell(0, 0)
    cell.width = Inches(6.8)
    set_cell_background(cell, fill_hex)
    set_cell_margins(cell, top=140, bottom=140, left=200, right=200)
    
    tcPr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(f'''
        <w:tcBorders {nsdecls("w")}>
            <w:top w:val="none"/>
            <w:left w:val="single" w:sz="24" w:space="0" w:color="{border_hex}"/>
            <w:bottom w:val="none"/>
            <w:right w:val="none"/>
        </w:tcBorders>
    ''')
    tcPr.append(borders)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.15
    run_title = p.add_run(f"[{title}]\n")
    run_title.bold = True
    run_title.font.name = 'Arial'
    run_title.font.size = Pt(10)
    run_title.font.color.rgb = COLOR_SECONDARY_BLUE
    
    run_text = p.add_run(text)
    run_text.font.name = 'Calibri'
    run_text.font.size = Pt(9.5)
    run_text.font.color.rgb = COLOR_SLATE_DARK
    
    p_after = doc.add_paragraph()
    p_after.paragraph_format.space_before = Pt(0)
    p_after.paragraph_format.space_after = Pt(4)

def add_code_block(doc, code_text):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    cell = tbl.cell(0, 0)
    cell.width = Inches(6.8)
    set_cell_background(cell, HEX_CODE_BG)
    set_cell_margins(cell, top=120, bottom=120, left=180, right=180)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.1
    run = p.add_run(code_text)
    run.font.name = 'Consolas'
    run.font.size = Pt(8.5)
    run.font.color.rgb = RGBColor(34, 34, 34)
    
    p_after = doc.add_paragraph()
    p_after.paragraph_format.space_before = Pt(0)
    p_after.paragraph_format.space_after = Pt(4)

def add_table_custom(doc, col_widths, headers, data, header_bg=HEX_HEADER_NAVY, alt_bg=HEX_ROW_ALT):
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    
    # Headers
    hdr_cells = table.rows[0].cells
    for i, title in enumerate(headers):
        hdr_cells[i].text = title
        hdr_cells[i].width = Inches(col_widths[i])
        set_cell_background(hdr_cells[i], header_bg)
        set_cell_margins(hdr_cells[i], top=120, bottom=120, left=140, right=140)
        p = hdr_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        for run in p.runs:
            run.font.bold = True
            run.font.color.rgb = RGBColor(255, 255, 255)
            run.font.size = Pt(9)
            run.font.name = 'Arial'
            
    # Data rows
    for r_idx, row_data in enumerate(data):
        row_cells = table.add_row().cells
        bg_color = alt_bg if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, val in enumerate(row_data):
            row_cells[c_idx].text = str(val)
            row_cells[c_idx].width = Inches(col_widths[c_idx])
            set_cell_background(row_cells[c_idx], bg_color)
            set_cell_margins(row_cells[c_idx], top=80, bottom=80, left=120, right=120)
            p = row_cells[c_idx].paragraphs[0]
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.15
            for run in p.runs:
                run.font.size = Pt(8.5)
                run.font.name = 'Calibri'
                run.font.color.rgb = COLOR_SLATE_DARK
                
    p_after = doc.add_paragraph()
    p_after.paragraph_format.space_before = Pt(0)
    p_after.paragraph_format.space_after = Pt(6)
    return table

def add_heading_1(doc, text):
    h = doc.add_heading(text, level=1)
    h.paragraph_format.space_before = Pt(16)
    h.paragraph_format.space_after = Pt(6)
    h.paragraph_format.keep_with_next = True
    for r in h.runs:
        r.font.name = 'Arial'
        r.font.size = Pt(14)
        r.bold = True
        r.font.color.rgb = COLOR_PRIMARY_NAVY
    return h

def add_heading_2(doc, text):
    h = doc.add_heading(text, level=2)
    h.paragraph_format.space_before = Pt(12)
    h.paragraph_format.space_after = Pt(4)
    h.paragraph_format.keep_with_next = True
    for r in h.runs:
        r.font.name = 'Arial'
        r.font.size = Pt(11.5)
        r.bold = True
        r.font.color.rgb = COLOR_SECONDARY_BLUE
    return h

def add_heading_3(doc, text):
    h = doc.add_heading(text, level=3)
    h.paragraph_format.space_before = Pt(8)
    h.paragraph_format.space_after = Pt(2)
    h.paragraph_format.keep_with_next = True
    for r in h.runs:
        r.font.name = 'Arial'
        r.font.size = Pt(10)
        r.bold = True
        r.font.italic = True
        r.font.color.rgb = COLOR_PRIMARY_NAVY
    return h

def add_body_p(doc, text, bold_prefix=None, space_after=6):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.15
    if bold_prefix:
        r_pre = p.add_run(bold_prefix)
        r_pre.bold = True
        r_pre.font.name = 'Calibri'
        r_pre.font.size = Pt(10)
        r_pre.font.color.rgb = COLOR_SLATE_DARK
    r_body = p.add_run(text)
    r_body.font.name = 'Calibri'
    r_body.font.size = Pt(10)
    r_body.font.color.rgb = COLOR_SLATE_DARK
    return p

def add_figure(doc, img_path, caption, fig_num):
    if os.path.exists(img_path):
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.paragraph_format.space_before = Pt(8)
        p_img.paragraph_format.space_after = Pt(2)
        run_img = p_img.add_run()
        run_img.add_picture(str(img_path), width=Inches(6.2))
        
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.paragraph_format.space_before = Pt(0)
        p_cap.paragraph_format.space_after = Pt(8)
        r_num = p_cap.add_run(f"Fig. {fig_num}. ")
        r_num.bold = True
        r_num.font.name = 'Arial'
        r_num.font.size = Pt(8.5)
        r_num.font.color.rgb = COLOR_PRIMARY_NAVY
        r_text = p_cap.add_run(caption)
        r_text.font.name = 'Calibri'
        r_text.font.size = Pt(8.5)
        r_text.font.italic = True
        r_text.font.color.rgb = COLOR_MUTED_GRAY

def add_page_number_to_section(section):
    # Header
    hdr = section.header
    p_hdr = hdr.paragraphs[0]
    p_hdr.text = "Digital Persona-Based Synthetic Survey Generation | Technical Report & IEEE Foundation"
    p_hdr.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    for r in p_hdr.runs:
        r.font.name = 'Calibri'
        r.font.size = Pt(8)
        r.font.color.rgb = COLOR_MUTED_GRAY

    # Footer
    ftr = section.footer
    p_ftr = ftr.paragraphs[0]
    p_ftr.text = "Confidential & Proprietary Research Documentation | Prepared for IEEE Publication"
    p_ftr.alignment = WD_ALIGN_PARAGRAPH.LEFT
    for r in p_ftr.runs:
        r.font.name = 'Calibri'
        r.font.size = Pt(8)
        r.font.color.rgb = COLOR_MUTED_GRAY

print("Report generator library initialized successfully.")

