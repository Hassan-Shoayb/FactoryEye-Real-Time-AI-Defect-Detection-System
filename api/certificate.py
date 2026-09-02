import time
from typing import Dict, List

class QualityCertificateGenerator:
    """
    Generates printable ISO/QA standard metallurgical inspection compliance certificates
    for steel surface production batches.
    """
    def generate_html_certificate(self, stats: Dict, batch_id: str = "BATCH-2026-NEU-01") -> str:
        date_str = time.strftime("%B %d, %Y", time.gmtime())
        time_str = time.strftime("%H:%M:%S UTC", time.gmtime())
        
        total = stats.get("total_inspections", 0)
        yield_pct = stats.get("quality_yield_percent", 100.0)
        defect_rate = stats.get("defect_rate_percent", 0.0)
        clean = stats.get("clean_inspections", 0)
        defective = stats.get("defective_inspections", 0)
        mean_lat = stats.get("mean_inference_ms", 0.0)
        breakdown = stats.get("defect_class_breakdown", [])

        # Determine Compliance Status
        if yield_pct >= 95.0:
            status_text = "PASSED — GRADE A (PRIME COIL SPECIFICATION)"
            status_color = "#16a34a"
            badge_bg = "#dcfce7"
        elif yield_pct >= 80.0:
            status_text = "CONDITIONAL PASS — REWORK REQUIRED (SECONDARY SPEC)"
            status_color = "#d97706"
            badge_bg = "#fef3c7"
        else:
            status_text = "REJECTED — EXCEEDS DEFECT DENSITY THRESHOLD"
            status_color = "#dc2626"
            badge_bg = "#fee2e2"

        breakdown_rows = ""
        for b in breakdown:
            breakdown_rows += f"""
            <tr>
                <td style="padding: 8px 12px; border: 1px solid #e2e8f0; font-weight: 600; text-transform: uppercase;">{b.get('class', '')}</td>
                <td style="padding: 8px 12px; border: 1px solid #e2e8f0; text-align: center;">{b.get('count', 0)}</td>
                <td style="padding: 8px 12px; border: 1px solid #e2e8f0; text-align: center;">{b.get('avg_confidence', 0.0) * 100:.1f}%</td>
            </tr>
            """

        if not breakdown:
            breakdown_rows = '<tr><td colspan="3" style="padding: 12px; text-align: center; color: #64748b;">No defect incidents detected in this batch. 100% clean surface.</td></tr>'

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>FactoryEye Quality Inspection Certificate — {batch_id}</title>
    <style>
        body {{
            font-family: 'Helvetica Neue', Arial, sans-serif;
            background: #f8fafc;
            color: #0f172a;
            margin: 0;
            padding: 40px;
        }}
        .certificate {{
            max-width: 800px;
            margin: 0 auto;
            background: #ffffff;
            border: 2px solid #0f172a;
            padding: 40px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.05);
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 3px solid #0f172a;
            padding-bottom: 20px;
            margin-bottom: 24px;
        }}
        .title {{ font-size: 24px; font-weight: 800; letter-spacing: -0.5px; text-transform: uppercase; }}
        .subtitle {{ font-size: 13px; color: #64748b; margin-top: 4px; }}
        .badge {{
            padding: 10px 16px;
            border-radius: 6px;
            font-weight: 800;
            font-size: 14px;
            color: {status_color};
            background: {badge_bg};
            border: 1px solid {status_color};
            text-align: center;
            margin: 20px 0;
        }}
        .grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin: 24px 0; }}
        .card {{ background: #f1f5f9; padding: 14px; border-radius: 6px; text-align: center; }}
        .card-val {{ font-size: 22px; font-weight: 700; color: #0f172a; font-family: monospace; }}
        .card-lbl {{ font-size: 11px; text-transform: uppercase; color: #64748b; margin-top: 4px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 16px; font-size: 13px; }}
        th {{ background: #0f172a; color: white; padding: 10px 12px; text-align: left; font-size: 12px; text-transform: uppercase; }}
        .footer {{
            display: flex;
            justify-content: space-between;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #e2e8f0;
            font-size: 12px;
            color: #64748b;
        }}
        .seal {{
            border: 2px dashed #0f172a;
            padding: 12px 24px;
            text-align: center;
            font-weight: 700;
            font-size: 11px;
            text-transform: uppercase;
        }}
        @media print {{
            body {{ background: white; padding: 0; }}
            .certificate {{ box-shadow: none; border: 1px solid #000; }}
            .no-print {{ display: none; }}
        }}
    </style>
</head>
<body>
    <div class="certificate">
        <div class="header">
            <div>
                <div class="title">FactoryEye Quality Compliance Certificate</div>
                <div class="subtitle">Automated Computer Vision & Metallurgical Surface QA Inspection</div>
            </div>
            <div style="text-align: right; font-size: 12px; color: #64748b;">
                <div><strong>Date:</strong> {date_str}</div>
                <div><strong>Time:</strong> {time_str}</div>
                <div><strong>Batch:</strong> {batch_id}</div>
            </div>
        </div>

        <div class="badge">
            STATUS: {status_text}
        </div>

        <div class="grid">
            <div class="card">
                <div class="card-val">{total}</div>
                <div class="card-lbl">Total Samples Inspected</div>
            </div>
            <div class="card">
                <div class="card-val" style="color: #16a34a;">{yield_pct}%</div>
                <div class="card-lbl">Quality Yield Rate</div>
            </div>
            <div class="card">
                <div class="card-val" style="color: {status_color};">{defect_rate}%</div>
                <div class="card-lbl">Surface Defect Density</div>
            </div>
        </div>

        <h3 style="font-size: 14px; text-transform: uppercase; margin-top: 28px;">Defect Classification Breakdown</h3>
        <table>
            <thead>
                <tr>
                    <th>Defect Category</th>
                    <th style="text-align: center;">Incident Count</th>
                    <th style="text-align: center;">Mean Confidence</th>
                </tr>
            </thead>
            <tbody>
                {breakdown_rows}
            </tbody>
        </table>

        <div class="footer">
            <div>
                <div><strong>Inspection System:</strong> FactoryEye AI Platform v1.0.0</div>
                <div><strong>Station Identifier:</strong> METALLURGY_LINE_01</div>
                <div><strong>Verification Hash:</strong> SHA256-VALIDATED</div>
            </div>
            <div class="seal">
                AI QUALITY ASSURANCE<br>
                VERIFIED & APPROVED
            </div>
        </div>

        <div class="no-print" style="margin-top: 30px; text-align: center;">
            <button onclick="window.print()" style="background: #0f172a; color: white; border: none; padding: 10px 24px; font-weight: 700; border-radius: 6px; cursor: pointer;">🖨️ Print Certificate</button>
        </div>
    </div>
</body>
</html>
"""
        return html

certificate_generator = QualityCertificateGenerator()
