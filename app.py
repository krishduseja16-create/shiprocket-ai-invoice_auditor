"""
AI Invoice Auditor for Shiprocket Logistics Invoices
=====================================================
A Streamlit application that audits shipping invoices using OCR + LLM.

Install dependencies:
    pip install streamlit pytesseract pillow pdf2image anthropic

Run:
    streamlit run app.py
"""

import streamlit as st
import json
import re
import io
import os
import hashlib
from datetime import datetime
from typing import Optional

# ── PDF / OCR ────────────────────────────────────────────────────────────────
try:
    import pytesseract
    from PIL import Image
    from pdf2image import convert_from_bytes
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# ── Anthropic ─────────────────────────────────────────────────────────────────
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# 1.  RATE CARD  (configurable benchmark — edit to match your actual contract)
# ─────────────────────────────────────────────────────────────────────────────
RATE_CARD = {
    "BlueDart": {
        "slabs": [
            (0.5,  45),
            (1.0,  65),
            (2.0,  85),
            (5.0, 120),
            (10.0, 180),
            (float("inf"), 220),
        ],
        "cod_charge_pct": 1.5,   # % of order value, min ₹30
        "excess_weight_per_kg": 18,
    },
    "Delhivery": {
        "slabs": [
            (0.5,  35),
            (1.0,  55),
            (2.0,  75),
            (5.0, 105),
            (10.0, 155),
            (float("inf"), 195),
        ],
        "cod_charge_pct": 1.5,
        "excess_weight_per_kg": 16,
    },
    "Ekart": {
        "slabs": [
            (0.5,  30),
            (1.0,  50),
            (2.0,  70),
            (5.0,  95),
            (10.0, 140),
            (float("inf"), 175),
        ],
        "cod_charge_pct": 1.5,
        "excess_weight_per_kg": 14,
    },
    "DTDC": {
        "slabs": [
            (0.5,  40),
            (1.0,  60),
            (2.0,  80),
            (5.0, 110),
            (10.0, 165),
            (float("inf"), 205),
        ],
        "cod_charge_pct": 1.5,
        "excess_weight_per_kg": 15,
    },
    "Default": {
        "slabs": [
            (0.5,  40),
            (1.0,  60),
            (2.0,  80),
            (5.0, 110),
            (10.0, 160),
            (float("inf"), 200),
        ],
        "cod_charge_pct": 1.5,
        "excess_weight_per_kg": 15,
    },
}

GST_RATE = 0.18   # 18 % standard
COD_MIN_CHARGE = 30.0

# ─────────────────────────────────────────────────────────────────────────────
# 2.  OCR  ─  PDF → images → text
# ─────────────────────────────────────────────────────────────────────────────
def pdf_to_images(pdf_bytes: bytes) -> list:
    """Convert every page of a PDF to a PIL Image."""
    return convert_from_bytes(pdf_bytes, dpi=200)


def ocr_image(image: Image.Image) -> str:
    """Run Tesseract OCR on a single PIL image."""
    config = "--psm 6 --oem 3"
    return pytesseract.image_to_string(image, config=config)


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """OCR all pages and concatenate results."""
    images = pdf_to_images(pdf_bytes)
    pages = [ocr_image(img) for img in images]
    return "\n\n--- PAGE BREAK ---\n\n".join(pages)


def extract_text_from_image(image_bytes: bytes) -> str:
    """OCR a single image (PNG/JPG label)."""
    img = Image.open(io.BytesIO(image_bytes))
    return ocr_image(img)

# ─────────────────────────────────────────────────────────────────────────────
# 3.  LLM PARSING  ─  raw OCR text → structured JSON
# ─────────────────────────────────────────────────────────────────────────────
EXTRACTION_PROMPT = """
You are a logistics invoice data-extraction assistant.
Extract the following fields from the raw OCR text below and return ONLY a valid JSON object.
If a field is not found, use null.

Fields required:
- invoice_number      (string)
- invoice_date        (string, ISO format YYYY-MM-DD if possible)
- awb_number          (string)
- courier_partner     (string, e.g. BlueDart, Delhivery, Ekart, DTDC)
- shipment_weight     (number, in kg)
- payment_mode        (string: "COD" or "Prepaid")
- shipping_charge     (number, in INR, excluding GST)
- gst_amount          (number, in INR)
- total_amount        (number, in INR)
- order_value         (number, in INR — for COD charge validation; null if not found)

Return ONLY the JSON object, no explanation, no markdown fences.

--- RAW OCR TEXT ---
{ocr_text}
"""


def parse_invoice_with_llm(ocr_text: str, api_key: str) -> dict:
    """Call Claude to extract structured fields from OCR text."""
    client = anthropic.Anthropic(api_key=api_key)
    prompt = EXTRACTION_PROMPT.format(ocr_text=ocr_text[:8000])  # token guard

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = message.content[0].text.strip()

    # Strip markdown code fences if present
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Best-effort: extract JSON object via regex
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError(f"LLM returned non-JSON output:\n{raw}")


def parse_invoice_regex_fallback(ocr_text: str) -> dict:
    """Regex-based fallback parser (no LLM required)."""

    def find(patterns, text, group=1, cast=None):
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                val = m.group(group).strip().replace(",", "")
                return cast(val) if cast else val
        return None

    text = ocr_text

    invoice_number = find([
        r"invoice\s*(?:no|number|#)[:\s#]*([A-Z0-9\-/]+)",
        r"inv[.\s]*no[:\s]*([A-Z0-9\-/]+)",
    ], text)

    invoice_date = find([
        r"invoice\s*date[:\s]*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",
        r"date[:\s]*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",
    ], text)

    awb_number = find([
        r"awb\s*(?:no|number|#)[:\s#]*([A-Z0-9\-]+)",
        r"airway\s*bill[:\s]*([A-Z0-9\-]+)",
        r"tracking\s*(?:no|id)[:\s]*([A-Z0-9\-]+)",
    ], text)

    courier = find([
        r"(bluedart|delhivery|ekart|dtdc|ecom express|xpressbees|shadowfax)",
    ], text)

    weight = find([
        r"(?:chargeable|charged|actual)\s*weight[:\s]*([\d.]+)\s*kg",
        r"weight[:\s]*([\d.]+)\s*kg",
    ], text, cast=float)

    payment_mode = None
    if re.search(r"\bcod\b|\bcash\s*on\s*delivery\b", text, re.I):
        payment_mode = "COD"
    elif re.search(r"\bprepaid\b|\bonline\b|\bpaid\b", text, re.I):
        payment_mode = "Prepaid"

    shipping_charge = find([
        r"(?:shipping|freight|delivery)\s*charge[:\s]*([\d,.]+)",
        r"base\s*(?:price|charge)[:\s]*([\d,.]+)",
    ], text, cast=float)

    gst_amount = find([
        r"(?:gst|igst|cgst\s*\+\s*sgst)[:\s]*([\d,.]+)",
        r"tax[:\s]*([\d,.]+)",
    ], text, cast=float)

    total_amount = find([
        r"(?:grand\s*total|total\s*amount|net\s*payable)[:\s]*([\d,.]+)",
        r"total[:\s]*([\d,.]+)",
    ], text, cast=float)

    order_value = find([
        r"(?:order|declared|invoice)\s*value[:\s]*([\d,.]+)",
        r"cod\s*amount[:\s]*([\d,.]+)",
    ], text, cast=float)

    return {
        "invoice_number": invoice_number,
        "invoice_date": invoice_date,
        "awb_number": awb_number,
        "courier_partner": (courier or "").title() or None,
        "shipment_weight": weight,
        "payment_mode": payment_mode,
        "shipping_charge": shipping_charge,
        "gst_amount": gst_amount,
        "total_amount": total_amount,
        "order_value": order_value,
    }

# ─────────────────────────────────────────────────────────────────────────────
# 4.  AUDIT ENGINE
# ─────────────────────────────────────────────────────────────────────────────
class AuditFinding:
    SEVERITY = {"ERROR": "🔴", "WARNING": "🟡", "INFO": "🔵"}

    def __init__(self, rule: str, severity: str, message: str,
                 expected=None, actual=None, difference=None):
        self.rule = rule
        self.severity = severity
        self.message = message
        self.expected = expected
        self.actual = actual
        self.difference = difference

    def to_dict(self):
        return {
            "rule": self.rule,
            "severity": self.severity,
            "icon": self.SEVERITY.get(self.severity, "⚪"),
            "message": self.message,
            "expected": self.expected,
            "actual": self.actual,
            "difference": self.difference,
        }


def get_rate_card(courier: Optional[str]) -> dict:
    if courier:
        for key in RATE_CARD:
            if key.lower() in (courier or "").lower():
                return RATE_CARD[key]
    return RATE_CARD["Default"]


def expected_shipping_charge(weight_kg: float, card: dict) -> float:
    for slab_max, charge in card["slabs"]:
        if weight_kg <= slab_max:
            return float(charge)
    return float(card["slabs"][-1][1])


def audit_weight_slab(data: dict, card: dict) -> Optional[AuditFinding]:
    """Rule 1 — Weight slab mismatch."""
    weight = data.get("shipment_weight")
    charged = data.get("shipping_charge")
    if weight is None or charged is None:
        return AuditFinding("Weight Slab", "INFO",
                            "Cannot verify weight slab: weight or charge missing.")
    expected = expected_shipping_charge(weight, card)
    tolerance = 5.0   # ₹5 tolerance for rounding differences
    diff = charged - expected
    if abs(diff) > tolerance:
        return AuditFinding(
            "Weight Slab Mismatch", "ERROR",
            f"Shipping charge ₹{charged:.2f} doesn't match rate card for "
            f"{weight} kg (expected ₹{expected:.2f}).",
            expected=f"₹{expected:.2f}", actual=f"₹{charged:.2f}",
            difference=f"₹{diff:+.2f}",
        )
    return None


def audit_cod_on_prepaid(data: dict, card: dict) -> Optional[AuditFinding]:
    """Rule 2 — COD charge applied on a Prepaid shipment."""
    mode = (data.get("payment_mode") or "").upper()
    total = data.get("total_amount") or 0
    shipping = data.get("shipping_charge") or 0
    gst = data.get("gst_amount") or 0

    if mode != "PREPAID":
        return None  # only applies to prepaid

    # Estimate what COD charge looks like
    implied_extras = total - shipping - gst
    order_val = data.get("order_value") or 1000
    cod_estimate = max(order_val * card["cod_charge_pct"] / 100, COD_MIN_CHARGE)

    if implied_extras > (cod_estimate * 0.5):   # if extras > 50% of estimated COD
        return AuditFinding(
            "COD on Prepaid", "ERROR",
            f"Potential COD charge (₹{implied_extras:.2f}) detected on a "
            f"Prepaid shipment.",
            expected="₹0.00", actual=f"₹{implied_extras:.2f}",
            difference=f"₹{implied_extras:+.2f}",
        )
    return None


def audit_duplicate_awb(data: dict, seen_awbs: set) -> Optional[AuditFinding]:
    """Rule 3 — Duplicate AWB in session."""
    awb = data.get("awb_number")
    if not awb:
        return AuditFinding("Duplicate AWB", "INFO",
                            "AWB number not found; cannot check duplicates.")
    if awb in seen_awbs:
        return AuditFinding(
            "Duplicate AWB", "ERROR",
            f"AWB {awb} has already been billed in this audit session.",
            expected="Unique AWB", actual=awb,
        )
    seen_awbs.add(awb)
    return None


def audit_gst(data: dict) -> Optional[AuditFinding]:
    """Rule 4 — GST miscalculation."""
    shipping = data.get("shipping_charge")
    gst = data.get("gst_amount")
    if shipping is None or gst is None:
        return AuditFinding("GST Calculation", "INFO",
                            "Cannot verify GST: shipping charge or GST amount missing.")
    expected_gst = round(shipping * GST_RATE, 2)
    diff = gst - expected_gst
    if abs(diff) > 2.0:   # ₹2 tolerance
        return AuditFinding(
            "GST Miscalculation", "ERROR",
            f"GST ₹{gst:.2f} doesn't match 18% of shipping charge ₹{shipping:.2f}.",
            expected=f"₹{expected_gst:.2f}", actual=f"₹{gst:.2f}",
            difference=f"₹{diff:+.2f}",
        )
    return None


def audit_excess_weight_charge(data: dict, card: dict) -> Optional[AuditFinding]:
    """Rule 5 — Invalid excess weight charge."""
    weight = data.get("shipment_weight")
    total = data.get("total_amount")
    shipping = data.get("shipping_charge")
    gst = data.get("gst_amount")
    if None in (weight, total, shipping, gst):
        return None

    # Check if total > shipping+gst by more than 1 extra-kg charge
    surplus = total - shipping - gst
    per_kg = card["excess_weight_per_kg"]
    # One extra kg is allowed; beyond that is suspicious
    if surplus > (per_kg * 1.5):
        return AuditFinding(
            "Excess Weight Charge", "WARNING",
            f"Extra charges ₹{surplus:.2f} exceed expected excess-weight rate "
            f"(₹{per_kg}/kg).",
            expected=f"≤ ₹{per_kg * 1.5:.2f}", actual=f"₹{surplus:.2f}",
            difference=f"₹{surplus - per_kg * 1.5:+.2f}",
        )
    return None


def run_audit(data: dict, seen_awbs: set) -> list[dict]:
    """Run all audit rules and return list of findings as dicts."""
    card = get_rate_card(data.get("courier_partner"))
    findings = []

    checks = [
        audit_weight_slab(data, card),
        audit_cod_on_prepaid(data, card),
        audit_duplicate_awb(data, seen_awbs),
        audit_gst(data),
        audit_excess_weight_charge(data, card),
    ]

    for f in checks:
        if f:
            findings.append(f.to_dict())

    if not findings:
        findings.append({
            "rule": "All Clear",
            "severity": "INFO",
            "icon": "✅",
            "message": "No billing anomalies detected.",
            "expected": None,
            "actual": None,
            "difference": None,
        })

    return findings

# ─────────────────────────────────────────────────────────────────────────────
# 5.  STREAMLIT UI
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Invoice Auditor",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg: #0d0f14;
    --surface: #161922;
    --surface2: #1e2330;
    --border: #2a3040;
    --accent: #4f8ef7;
    --accent2: #7c3aed;
    --success: #22c55e;
    --warning: #f59e0b;
    --error: #ef4444;
    --text: #e2e8f0;
    --muted: #64748b;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}

h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; }

.block-container { padding-top: 1.5rem !important; }

.stFileUploader > div { background: var(--surface2) !important; border: 1.5px dashed var(--border) !important; border-radius: 12px !important; }

.stButton > button {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.4rem !important;
    transition: opacity .2s;
}
.stButton > button:hover { opacity: 0.88 !important; }

.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
}

.finding-error   { border-left: 4px solid var(--error) !important; }
.finding-warning { border-left: 4px solid var(--warning) !important; }
.finding-info    { border-left: 4px solid var(--accent) !important; }
.finding-ok      { border-left: 4px solid var(--success) !important; }

.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: .05em;
}
.badge-error   { background: #7f1d1d; color: #fca5a5; }
.badge-warning { background: #78350f; color: #fcd34d; }
.badge-info    { background: #1e3a5f; color: #93c5fd; }
.badge-ok      { background: #14532d; color: #86efac; }

.metric-box {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: .9rem 1rem;
    text-align: center;
}
.metric-value { font-size: 1.6rem; font-weight: 700; color: var(--accent); font-family: 'JetBrains Mono', monospace; }
.metric-label { font-size: .75rem; color: var(--muted); text-transform: uppercase; letter-spacing: .05em; margin-top: 2px; }

.ocr-box {
    background: #0a0c10;
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: .75rem;
    color: #94a3b8;
    max-height: 260px;
    overflow-y: auto;
    white-space: pre-wrap;
}

.json-box {
    background: #0a0c10;
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: .78rem;
    color: #7dd3fc;
    max-height: 340px;
    overflow-y: auto;
    white-space: pre;
}

input[type="text"], input[type="password"] {
    background: var(--surface2) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)


# ── Session state ──────────────────────────────────────────────────────────
if "seen_awbs" not in st.session_state:
    st.session_state.seen_awbs = set()
if "audit_history" not in st.session_state:
    st.session_state.audit_history = []


# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        placeholder="sk-ant-...",
        help="Required for LLM-based field extraction. Leave blank to use regex fallback.",
    )

    st.markdown("---")
    st.markdown("### 📋 Rate Card Preview")
    courier_preview = st.selectbox("Courier", list(RATE_CARD.keys()))
    card = RATE_CARD[courier_preview]
    slab_rows = "".join(
        f"<tr><td>≤ {s[0]} kg</td><td>₹{s[1]}</td></tr>"
        for s in card["slabs"]
    )
    st.markdown(
        f"""<table style='width:100%;font-size:.8rem;color:#94a3b8'>
        <thead><tr><th style='text-align:left'>Slab</th><th style='text-align:left'>Rate</th></tr></thead>
        <tbody>{slab_rows}</tbody></table>""",
        unsafe_allow_html=True,
    )
    st.markdown(f"COD %: **{card['cod_charge_pct']}%** | Excess/kg: **₹{card['excess_weight_per_kg']}**")

    st.markdown("---")
    if st.button("🗑️ Reset Session"):
        st.session_state.seen_awbs = set()
        st.session_state.audit_history = []
        st.success("Session cleared.")


# ── Header ────────────────────────────────────────────────────────────────
st.markdown("""
<div style='display:flex;align-items:center;gap:1rem;margin-bottom:1.5rem'>
  <div style='font-size:2.8rem'>🧾</div>
  <div>
    <h1 style='margin:0;font-size:2rem;background:linear-gradient(90deg,#4f8ef7,#7c3aed);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent'>AI Invoice Auditor</h1>
    <p style='margin:0;color:#64748b;font-size:.9rem'>Shiprocket Logistics · Automated Billing Anomaly Detection</p>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Upload section ────────────────────────────────────────────────────────
col_up1, col_up2 = st.columns(2)
with col_up1:
    invoice_file = st.file_uploader(
        "📄 Upload Invoice PDF",
        type=["pdf"],
        help="Shiprocket logistics invoice in PDF format",
    )
with col_up2:
    label_file = st.file_uploader(
        "🏷️ Upload Shipment Label (optional)",
        type=["pdf", "png", "jpg", "jpeg"],
        help="Shipping label to cross-verify AWB / weight",
    )

if not OCR_AVAILABLE:
    st.error("OCR libraries not available. Install: `pip install pytesseract pillow pdf2image`")

run_audit_btn = st.button("🔍 Run Audit", disabled=(invoice_file is None or not OCR_AVAILABLE))


# ── Pipeline execution ────────────────────────────────────────────────────
if run_audit_btn and invoice_file:

    with st.spinner("Running audit pipeline…"):

        # Step 1 – Upload acknowledged
        st.markdown("---")
        st.markdown("### 🔬 Audit Pipeline")
        progress = st.progress(0, text="Step 1 / 5 — Reading file…")

        invoice_bytes = invoice_file.read()
        file_hash = hashlib.md5(invoice_bytes).hexdigest()[:8].upper()

        # Step 2 – OCR
        progress.progress(20, text="Step 2 / 5 — Extracting text via OCR…")
        try:
            if invoice_file.type == "application/pdf":
                ocr_text = extract_text_from_pdf(invoice_bytes)
            else:
                ocr_text = extract_text_from_image(invoice_bytes)
        except Exception as e:
            st.error(f"OCR failed: {e}")
            st.stop()

        # Optional label
        label_ocr = ""
        if label_file:
            label_bytes = label_file.read()
            try:
                if label_file.type == "application/pdf":
                    label_ocr = extract_text_from_pdf(label_bytes)
                else:
                    label_ocr = extract_text_from_image(label_bytes)
            except Exception:
                pass
            if label_ocr:
                ocr_text += "\n\n--- LABEL ---\n\n" + label_ocr

        # Step 3 – LLM / regex parse
        progress.progress(40, text="Step 3 / 5 — Parsing invoice fields…")
        use_llm = bool(api_key and ANTHROPIC_AVAILABLE)
        parse_method = "Claude (LLM)" if use_llm else "Regex Fallback"
        try:
            if use_llm:
                invoice_data = parse_invoice_with_llm(ocr_text, api_key)
            else:
                invoice_data = parse_invoice_regex_fallback(ocr_text)
        except Exception as e:
            st.warning(f"LLM parse failed ({e}), falling back to regex.")
            invoice_data = parse_invoice_regex_fallback(ocr_text)
            parse_method = "Regex Fallback (LLM error)"

        # Step 4 – Convert to structured JSON (already a dict)
        progress.progress(60, text="Step 4 / 5 — Structuring data…")
        invoice_data["_meta"] = {
            "file_name": invoice_file.name,
            "file_hash": file_hash,
            "parse_method": parse_method,
            "audited_at": datetime.now().isoformat(timespec="seconds"),
        }

        # Step 5 – Audit
        progress.progress(80, text="Step 5 / 5 — Running audit rules…")
        findings = run_audit(invoice_data, st.session_state.seen_awbs)

        progress.progress(100, text="✅ Audit complete.")
        st.session_state.audit_history.append({
            "data": invoice_data,
            "findings": findings,
        })

    # ── Results layout ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 📊 Audit Report")

    # Metrics row
errors = sum(1 for f in findings if f.get("severity") == "ERROR") if "findings" in locals() else 0
warnings = sum(1 for f in findings if f.get("severity") == "WARNING") if "findings" in locals() else 0
infos = sum(1 for f in findings if f.get("severity") == "INFO") if "findings" in locals() else 0
status   = "FAIL" if errors else ("REVIEW" if warnings else "PASS")
status_color = "#ef4444" if errors else ("#f59e0b" if warnings else "#22c55e")

# Calculate total overcharge detected
total_overcharge = 0
for f in findings:
    if f.get("difference"):
        try:
            diff = float(str(f["difference"]).replace("₹", "").replace("+", ""))
            if diff > 0:
                total_overcharge += diff
        except:
            pass

m1, m2, m3, m4, m5, m6 = st.columns(6)

metrics = [
    (status, "Audit Status", status_color),
    (str(errors), "Errors", "#ef4444"),
    (str(warnings), "Warnings", "#f59e0b"),
    (f"₹{invoice_data.get('total_amount') or 0:,.2f}", "Total Billed", "#4f8ef7"),
    (f"₹{total_overcharge:,.2f}", "Overcharge Detected", "#ef4444"),
    (invoice_data.get("awb_number") or "—", "AWB Number", "#7c3aed"),
]

for col, (val, label, color) in zip([m1, m2, m3, m4, m5, m6], metrics):
    with col:
        st.markdown(
            f"""<div class='metric-box'>
              <div class='metric-value' style='color:{color}'>{val}</div>
              <div class='metric-label'>{label}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    st.markdown("<br>", unsafe_allow_html=True)

    # Two-column: extracted fields + findings
    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        st.markdown("### 🗂️ Extracted Invoice Fields")
        display_fields = {
            k: v for k, v in invoice_data.items()
            if not k.startswith("_") and v is not None
        }
        field_icons = {
            "invoice_number": "🔢",
            "invoice_date": "📅",
            "awb_number": "📦",
            "courier_partner": "🚚",
            "shipment_weight": "⚖️",
            "payment_mode": "💳",
            "shipping_charge": "💰",
            "gst_amount": "🧾",
            "total_amount": "💵",
            "order_value": "🛍️",
        }
        for field, value in display_fields.items():
            icon = field_icons.get(field, "•")
            label = field.replace("_", " ").title()
            if isinstance(value, float):
                disp = f"₹{value:,.2f}" if "amount" in field or "charge" in field or "value" in field else f"{value} kg"
            else:
                disp = str(value)
            st.markdown(
                f"""<div class='card' style='padding:.7rem 1rem;margin-bottom:.5rem;display:flex;justify-content:space-between;align-items:center'>
                  <span style='color:#94a3b8;font-size:.85rem'>{icon} {label}</span>
                  <span style='font-weight:600;font-family:JetBrains Mono,monospace;color:#e2e8f0'>{disp}</span>
                </div>""",
                unsafe_allow_html=True,
            )

    with col_r:
        st.markdown("### 🚨 Audit Findings")
        for f in findings:
            sev = f["severity"].lower()
            badge_class = f"badge-{sev}" if sev in ("error", "warning", "info") else "badge-ok"
            finding_class = f"finding-{sev}" if sev in ("error", "warning") else (
                "finding-ok" if f["rule"] == "All Clear" else "finding-info"
            )
            extras = ""
            if f.get("expected"):
                extras += f"<br><small style='color:#94a3b8'>Expected: <b>{f['expected']}</b> · Actual: <b>{f['actual']}</b>"
                if f.get("difference"):
                    extras += f" · Diff: <b>{f['difference']}</b>"
                extras += "</small>"

            st.markdown(
                f"""<div class='card {finding_class}'>
                  <div style='display:flex;justify-content:space-between;align-items:flex-start'>
                    <span style='font-weight:600'>{f['icon']} {f['rule']}</span>
                    <span class='badge {badge_class}'>{f['severity']}</span>
                  </div>
                  <p style='margin:.4rem 0 0;font-size:.87rem;color:#cbd5e1'>{f['message']}{extras}</p>
                </div>""",
                unsafe_allow_html=True,
            )

    # Expandable raw data
    with st.expander("🔍 OCR Text", expanded=False):
        st.markdown(f"<div class='ocr-box'>{ocr_text[:4000]}</div>", unsafe_allow_html=True)

    with st.expander("📦 Structured JSON", expanded=False):
        st.markdown(
            f"<div class='json-box'>{json.dumps(invoice_data, indent=2)}</div>",
            unsafe_allow_html=True,
        )

    # Download report
    report = {
        "invoice": invoice_data,
        "findings": findings,
        "summary": {
            "status": status,
            "errors": errors,
            "warnings": warnings,
        },
    }
    st.download_button(
        "⬇️ Download Audit Report (JSON)",
        data=json.dumps(report, indent=2),
        file_name=f"audit_{file_hash}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
    )


# ── Session history ────────────────────────────────────────────────────────
if st.session_state.audit_history:
    st.markdown("---")
    st.markdown("## 📜 Session Audit History")
    for i, entry in enumerate(reversed(st.session_state.audit_history), 1):
        d = entry["data"]
        fs = entry["findings"]
        errs = sum(1 for f in fs if f["severity"] == "ERROR")
        warns = sum(1 for f in fs if f["severity"] == "WARNING")
        status_icon = "🔴" if errs else ("🟡" if warns else "✅")
        with st.expander(
            f"{status_icon} #{i} — {d.get('invoice_number','N/A')} "
            f"| AWB: {d.get('awb_number','N/A')} "
            f"| {d['_meta']['audited_at']}"
        ):
            c1, c2 = st.columns(2)
            with c1:
                st.json({k: v for k, v in d.items() if not k.startswith("_")})
            with c2:
                for f in fs:
                    st.markdown(f"{f['icon']} **{f['rule']}** — {f['message']}")
