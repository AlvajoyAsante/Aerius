"""
PDF report generation module.

Purpose: Generate a one-page PDF summary including:
- Metadata (analysis date, scores)
- Image with crack overlay
- Score breakdown and recommendations
"""

import tempfile
import os
from io import BytesIO
from datetime import datetime
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor


def generate_pdf_report(
    image: Image.Image,
    severity_score: int,
    coverage_percent: float,
    defect_count: int,
    recommendation: str
) -> BytesIO:
    """
    Generate a professional 1-page PDF report with image and metrics.
    
    Args:
        image: PIL Image with crack overlay
        severity_score: Severity score (0-100)
        coverage_percent: Coverage area percentage
        defect_count: Number of defects detected
        recommendation: Recommendation text
    
    Returns:
        BytesIO object containing the PDF data
    """
    
    # Create PDF in memory
    pdf_buffer = BytesIO()
    page_width, page_height = letter
    c = canvas.Canvas(pdf_buffer, pagesize=letter)
    
    # Set up colors
    dark_blue = HexColor("#0B1220")
    light_blue = HexColor("#60A5FA")
    
    # Top section: Title
    c.setFont("Helvetica-Bold", 24)
    c.setFillColor(dark_blue)
    c.drawString(0.5 * inch, page_height - 0.6 * inch, "Aerius - Defect Report")
    
    # Date and time (below title)
    c.setFont("Helvetica", 9)
    c.setFillColor(HexColor("#666666"))
    analysis_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.drawString(0.5 * inch, page_height - 0.9 * inch, f"Analysis Date: {analysis_date}")
    
    # Horizontal line (below date)
    c.setStrokeColor(light_blue)
    c.setLineWidth(2)
    c.line(0.5 * inch, page_height - 1.05 * inch, page_width - 0.5 * inch, page_height - 1.05 * inch)
    
    # Metrics boxes (3 columns) - positioned below the line with more spacing
    metrics_y = page_height - 2.0 * inch
    box_width = (page_width - 1.5 * inch) / 3 - 0.1 * inch
    box_height = 0.75 * inch
    
    # Helper function to draw metric box
    def draw_metric_box(x, y, label, value, unit=""):
        c.setStrokeColor(light_blue)
        c.setLineWidth(1)
        c.rect(x, y, box_width, box_height, fill=0)
        
        c.setFont("Helvetica", 8)
        c.setFillColor(HexColor("#999999"))
        c.drawString(x + 0.08 * inch, y + box_height - 0.2 * inch, label)
        
        c.setFont("Helvetica-Bold", 14)
        c.setFillColor(light_blue)
        value_text = f"{value}{unit}"
        c.drawString(x + 0.08 * inch, y + 0.2 * inch, value_text)
    
    draw_metric_box(0.5 * inch, metrics_y, "Defects Detected", defect_count)
    draw_metric_box(0.5 * inch + box_width + 0.2 * inch, metrics_y, "Coverage Area", f"{coverage_percent:.1f}", "%")
    draw_metric_box(0.5 * inch + 2 * (box_width + 0.2 * inch), metrics_y, "Severity Score", f"{severity_score}", "/100")
    
    # Image section - positioned well below metrics
    image_y = metrics_y - 3.0 * inch
    image_width = page_width - 1.0 * inch
    image_height = 2.2 * inch
    
    # Resize image to fit
    aspect_ratio = image.width / image.height
    if image_width / image_height > aspect_ratio:
        image_width = image_height * aspect_ratio
    else:
        image_height = image_width / aspect_ratio
    
    # Center image horizontally
    image_x = (page_width - image_width) / 2
    
    # Save PIL image to temporary file (reportlab needs a file path)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        image.save(tmp.name, format="PNG")
        temp_image_path = tmp.name
    
    try:
        # Draw image on PDF
        c.drawImage(temp_image_path, image_x, image_y - image_height, width=image_width, height=image_height)
    finally:
        # Clean up temp file
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
    
    # Recommendation section
    rec_y = image_y - image_height - 0.3 * inch
    
    c.setFont("Helvetica-Bold", 12)
    c.setFillColor(dark_blue)
    c.drawString(0.5 * inch, rec_y, "Recommendation:")
    
    c.setFont("Helvetica", 11)
    c.setFillColor(HexColor("#333333"))
    
    # Wrap recommendation text
    rec_lines = wrap_text(recommendation, max_width=90)
    for i, line in enumerate(rec_lines[:3]):  # Max 3 lines
        c.drawString(0.7 * inch, rec_y - 0.25 * inch - (i * 0.2 * inch), line)
    
    # Footer
    c.setFont("Helvetica", 8)
    c.setFillColor(HexColor("#999999"))
    c.drawString(0.5 * inch, 0.3 * inch, "Aerius Defect Detection System")
    c.drawString(page_width - 1.5 * inch, 0.3 * inch, "Page 1 of 1")
    
    c.save()
    pdf_buffer.seek(0)
    
    return pdf_buffer


def wrap_text(text: str, max_width: int = 90) -> list:
    """Simple text wrapping for PDF."""
    words = text.split()
    lines = []
    current_line = ""
    
    for word in words:
        if len(current_line) + len(word) + 1 <= max_width:
            current_line += word + " "
        else:
            if current_line:
                lines.append(current_line.strip())
            current_line = word + " "
    
    if current_line:
        lines.append(current_line.strip())
    
    return lines
