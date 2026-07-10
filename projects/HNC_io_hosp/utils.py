"""
utils.py
Shared helpers for HNC IO Hospice project.
"""
import os
import re


def export_xlsx_to_png(xlsx_path, figures_dir):
    """
    Export every sheet in an Excel workbook to a PNG file.
    Uses Excel COM to render to PDF (reliable), then pypdfium2 to convert to PNG.
    Requires Excel and pypdfium2.
    """
    import win32com.client
    import pypdfium2 as pdfium

    xlsx_path   = os.path.abspath(xlsx_path)
    figures_dir = os.path.abspath(figures_dir)
    os.makedirs(figures_dir, exist_ok=True)

    excel = win32com.client.Dispatch("Excel.Application")
    excel.Visible       = False
    excel.DisplayAlerts = False

    try:
        wb = excel.Workbooks.Open(xlsx_path)
        for ws in wb.Sheets:
            safe_name = re.sub(r'[^\w\s-]', '', ws.Name).strip().replace(' ', '_')
            pdf_path  = os.path.join(figures_dir, f"{safe_name}.pdf")
            png_path  = os.path.join(figures_dir, f"{safe_name}.png")

            # Fit used range to a single PDF page
            used = ws.UsedRange
            ws.PageSetup.PrintArea       = used.Address
            ws.PageSetup.Zoom            = False
            ws.PageSetup.FitToPagesWide  = 1
            ws.PageSetup.FitToPagesTall  = 1
            ws.PageSetup.LeftMargin      = 0
            ws.PageSetup.RightMargin     = 0
            ws.PageSetup.TopMargin       = 0
            ws.PageSetup.BottomMargin    = 0
            ws.PageSetup.HeaderMargin    = 0
            ws.PageSetup.FooterMargin    = 0

            ws.ExportAsFixedFormat(
                Type=0,                    # xlTypePDF
                Filename=pdf_path,
                Quality=0,                 # xlQualityStandard
                IncludeDocProperties=False,
                IgnorePrintAreas=False,
            )

            # Convert first PDF page to PNG at 200 DPI
            doc    = pdfium.PdfDocument(pdf_path)
            page   = doc[0]
            bitmap = page.render(scale=600 / 72)   # 600 DPI — publication quality
            img    = bitmap.to_pil()
            img.save(png_path, 'PNG')
            doc.close()
            os.remove(pdf_path)

            print(f"  PNG: {png_path}")
        wb.Close(False)
    finally:
        excel.Quit()
