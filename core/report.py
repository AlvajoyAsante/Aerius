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
    recommendation: str,
    metrics: dict = None
) -> BytesIO:
    """
    Generate a professional surveying-style PDF report with image and metrics.
    
    Args:
        image: PIL Image with crack overlay
        severity_score: Severity score (0-100)
        coverage_percent: Coverage area percentage
        defect_count: Number of defects detected
        recommendation: Recommendation text
        metrics: Dictionary with detailed metrics (cracks, water_coverage, etc.)
    
    Returns:
        BytesIO object containing the PDF data
    """
    if metrics is None:
        metrics = {}
    
    # Create PDF in memory
    pdf_buffer = BytesIO()
    page_width, page_height = letter
    c = canvas.Canvas(pdf_buffer, pagesize=letter)
    
    # Set up colors
    dark_blue = HexColor("#0B1220")
    light_blue = HexColor("#60A5FA")
    gray = HexColor("#666666")
    light_gray = HexColor("#F3F4F6")
    
    # Helper function to draw centered text
    def draw_centered_text(x, y, text, font_name="Helvetica", font_size=11, color=dark_blue):
        """Draw text centered horizontally"""
        c.setFont(font_name, font_size)
        c.setFillColor(color)
        text_width = c.stringWidth(text, font_name, font_size)
        centered_x = x + (page_width - 2 * x - text_width) / 2
        c.drawString(centered_x, y, text)
    
    # =====================================================================
    # HEADER SECTION
    # =====================================================================
    draw_centered_text(0.5 * inch, page_height - 0.6 * inch, "STRUCTURAL INSPECTION REPORT", "Helvetica-Bold", 28, dark_blue)
    
    # Subheader
    analysis_date = datetime.now().strftime("%B %d, %Y at %H:%M")
    draw_centered_text(0.5 * inch, page_height - 0.9 * inch, f"Generated: {analysis_date}", "Helvetica", 10, gray)
    
    # Horizontal divider
    c.setStrokeColor(light_blue)
    c.setLineWidth(2)
    c.line(0.5 * inch, page_height - 1.0 * inch, page_width - 0.5 * inch, page_height - 1.0 * inch)
    
    # =====================================================================
    # OVERVIEW METRICS SECTION (4 columns)
    # =====================================================================
    metrics_y = page_height - 1.6 * inch
    
    def draw_summary_metric(x, y, label, value, unit="", color=light_blue):
        """Draw a summary metric in a compact box"""
        # Draw colored background
        c.setFillColor(HexColor("#E8F0FF"))
        c.rect(x, y, 1.1 * inch, 0.55 * inch, fill=1, stroke=0)
        
        # Draw border
        c.setStrokeColor(color)
        c.setLineWidth(2)
        c.rect(x, y, 1.1 * inch, 0.55 * inch, fill=0)
        
        # Label
        c.setFont("Helvetica", 7)
        c.setFillColor(gray)
        c.drawString(x + 0.07 * inch, y + 0.38 * inch, label)
        
        # Value
        c.setFont("Helvetica-Bold", 18)
        c.setFillColor(color)
        value_text = f"{value}{unit}"
        c.drawString(x + 0.07 * inch, y + 0.08 * inch, value_text)
    
    severity_color = HexColor("#DC2626") if severity_score > 70 else HexColor("#F59E0B") if severity_score > 40 else HexColor("#10B981")
    
    draw_summary_metric(0.5 * inch, metrics_y, "Severity", f"{severity_score}", "/100", severity_color)
    draw_summary_metric(1.7 * inch, metrics_y, "Defects Found", f"{defect_count}", "", light_blue)
    draw_summary_metric(2.9 * inch, metrics_y, "Coverage", f"{coverage_percent:.1f}", "%", light_blue)
    
    # Add specific metrics if available
    if metrics.get("cracks"):
        draw_summary_metric(4.1 * inch, metrics_y, "Cracks", f"{metrics['cracks']}", "", light_blue)
    
    # =====================================================================
    # INSPECTION IMAGE SECTION
    # =====================================================================
    image_y = metrics_y - 2.5 * inch
    
    # Image label
    draw_centered_text(0.5 * inch, image_y + 0.15 * inch, "Annotated Inspection Image", "Helvetica-Bold", 11, dark_blue)
    
    image_display_y = image_y - 0.2 * inch
    image_width = page_width - 1.0 * inch
    image_height = 2.0 * inch
    
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
        # Draw image border
        c.setStrokeColor(light_blue)
        c.setLineWidth(1)
        c.rect(image_x - 0.05 * inch, image_display_y - image_height - 0.05 * inch, 
               image_width + 0.1 * inch, image_height + 0.1 * inch, fill=0)
        
        # Draw image
        c.drawImage(temp_image_path, image_x, image_display_y - image_height, 
                   width=image_width, height=image_height)
    finally:
        # Clean up temp file
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
    
    # =====================================================================
    # DETAILED FINDINGS SECTION
    # =====================================================================
    findings_y = image_display_y - image_height - 0.3 * inch
    
    draw_centered_text(0.5 * inch, findings_y, "Detailed Findings:", "Helvetica-Bold", 11, dark_blue)
    
    findings_list_y = findings_y - 0.2 * inch
    findings_text = []
    
    if metrics.get("cracks"):
        findings_text.append(f"• Cracks: {metrics['cracks']} detected at {metrics.get('crack_coverage', 0):.1f}% coverage (Severity: {metrics.get('crack_severity', 0)}/100)")
    
    if metrics.get("water_coverage", 0) > 0:
        findings_text.append(f"• Water Damage: {metrics.get('water_coverage', 0):.1f}% coverage (Severity: {metrics.get('water_severity', 0)}/100)")
    
    if not findings_text:
        findings_text.append("• No significant defects detected during inspection")
    
    c.setFont("Helvetica", 9)
    c.setFillColor(HexColor("#333333"))
    for i, finding in enumerate(findings_text):
        finding_width = c.stringWidth(finding, "Helvetica", 9)
        centered_finding_x = 0.5 * inch + (page_width - 1.0 * inch - finding_width) / 2
        c.drawString(centered_finding_x, findings_list_y - (i * 0.18 * inch), finding)
    
    # =====================================================================
    # RECOMMENDATIONS SECTION
    # =====================================================================
    rec_y = findings_list_y - (len(findings_text) * 0.18 * inch) - 0.25 * inch
    
    draw_centered_text(0.5 * inch, rec_y, "Recommendations:", "Helvetica-Bold", 11, dark_blue)
    
    # Generate recommendations text
    rec_lines = []
    
    crack_severity = metrics.get("crack_severity", 0)
    if crack_severity > 70:
        rec_lines.append("URGENT: Structural cracks detected. Consult a structural engineer immediately.")
    elif crack_severity > 40:
        rec_lines.append("Schedule professional inspection. Plan repair work for existing cracks.")
    elif crack_severity > 0:
        rec_lines.append("Monitor cracks and apply protective sealant to prevent water infiltration.")
    
    water_severity = metrics.get("water_severity", 0)
    if water_severity > 70:
        rec_lines.append("CRITICAL: Significant water damage detected. Locate and repair source immediately.")
    elif water_severity > 40:
        rec_lines.append("Identify source of water damage and implement remediation measures.")
    elif water_severity > 0:
        rec_lines.append("Monitor affected areas for further water damage or deterioration.")
    
    if not rec_lines:
        rec_lines.append("Continue routine maintenance and periodic inspections.")
    
    rec_list_y = rec_y - 0.2 * inch
    c.setFont("Helvetica", 9)
    c.setFillColor(HexColor("#333333"))
    for i, rec in enumerate(rec_lines):
        # Calculate centered position for each recommendation
        rec_width = c.stringWidth(rec, "Helvetica", 9)
        centered_rec_x = 0.5 * inch + (page_width - 1.0 * inch - rec_width) / 2
        c.drawString(centered_rec_x, rec_list_y - (i * 0.18 * inch), rec)
    
    # =====================================================================
    # FOOTER
    # =====================================================================
    c.setFont("Helvetica", 8)
    c.setFillColor(gray)
    c.drawString(0.5 * inch, 0.35 * inch, "Aerius - AI-Powered Structural Inspection System")
    c.drawString(page_width - 1.5 * inch, 0.35 * inch, "Page 1 of 1")
    
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
