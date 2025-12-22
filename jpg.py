import fitz  # PyMuPDF
from pathlib import Path

def pdf_to_png(pdf_path, output_path=None, dpi=300):
    """
    Convert PDF to PNG
    
    Args:
        pdf_path: Path to PDF file
        output_path: Output PNG path (if None, will use same name as PDF)
        dpi: Resolution for output image
    """
    pdf_path = Path(pdf_path)
    
    if output_path is None:
        output_path = pdf_path.with_suffix('.png')
    else:
        output_path = Path(output_path)
    
    # Open PDF
    pdf_document = fitz.open(pdf_path)
    
    # Get first page
    page = pdf_document[0]
    
    # Set zoom factor for DPI
    zoom = dpi / 72  # 72 is default DPI
    mat = fitz.Matrix(zoom, zoom)
    
    # Render page to image
    pix = page.get_pixmap(matrix=mat)
    
    # Save as PNG
    pix.save(output_path)
    
    pdf_document.close()
    
    print(f"Converted {pdf_path} to {output_path}")
    return output_path

if __name__ == "__main__":
    # Convert teaser.pdf to PNG
    pdf_to_png("figures/teaser.pdf")

