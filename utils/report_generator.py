from fpdf import FPDF
from datetime import datetime
import os


def generate_mission_report(title, content, output_path="reports/helios_mission_report.pdf"):
    os.makedirs("reports", exist_ok=True)

    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, title, ln=True)

    pdf.set_font("Arial", "", 10)
    pdf.cell(
        0,
        8,
        f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ln=True
    )

    pdf.ln(5)

    safe_content = content.encode("latin-1", "ignore").decode("latin-1")

    pdf.set_font("Arial", "", 11)

    for line in safe_content.split("\n"):
        pdf.multi_cell(0, 7, line)

    pdf.output(output_path)

    return output_path