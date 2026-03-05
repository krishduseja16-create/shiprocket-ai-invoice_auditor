"""
AI Invoice Auditor for Shiprocket Logistics Invoices
"""

import streamlit as st
import json
import re
import io
import hashlib
from datetime import datetime
from typing import Optional

try:
    import pytesseract
    from PIL import Image
    from pdf2image import convert_from_bytes
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

RATE_CARD = {
    "BlueDart": {
        "slabs": [(0.5,45),(1.0,65),(2.0,85),(5.0,120),(10.0,180),(float("inf"),220)],
        "cod_charge_pct": 1.5, "excess_weight_per_kg": 18,
    },
    "Delhivery": {
        "slabs": [(0.5,35),(1.0,55),(2.0,75),(5.0,105),(10.0,155),(float("inf"),195)],
        "cod_charge_pct": 1.5, "excess_weight_per_kg": 16,
    },
    "Ekart": {
        "slabs": [(0.5,30),(1.0,50),(2.0,70),(5.0,95),(10.0,140),(float("inf"),175)],
        "cod_charge_pct": 1.5, "excess_weight_per_kg": 14,
    },
    "DTDC": {
        "slabs": [(0.5,40),(1.0,60),(2.0,80),(5.0,110),(10.0,165),(float("inf"),205)],
        "cod_charge_pct": 1.5, "excess_weight_per_kg": 15,
    },
    "Default": {
        "slabs": [(0.5,40),(1.0,60),(2.0,80),(5.0,110),(10.0,160),(float("inf"),200)],
        "cod_charge_pct": 1.5, "excess_weight_per_kg": 15,
    },
}
GST_RATE = 0.18
COD_MIN_CHARGE = 30.0

# ── OCR ───────────────────────────────────────────────────────────────────────
def pdf_to_images(pdf_bytes):
    return convert_from_bytes(pdf_bytes, dpi=200)

def ocr_image(image):
    return pytesseract.image_to_string(image, config="--psm 6 --oem 3")

def extract_text_from_pdf(pdf_bytes):
    return "\n\n--- PAGE BREAK ---\n\n".join(ocr_image(img) for img in pdf_to_images(pdf_bytes))

def extract_text_from_image(image_bytes):
    return ocr_image(Image.open(io.BytesIO(image_bytes)))

# ── LLM PARSING ───────────────────────────────────────────────────────────────
EXTRACTION_PROMPT = """
You are a logistics invoice data-extraction assistant.
Extract the following fields from the raw OCR text and return ONLY a valid JSON object.
If a field is not found, use null.

Fields: invoice_number, invoice_date (YYYY-MM-DD), awb_number, courier_partner,
shipment_weight (kg, number), payment_mode ("COD" or "Prepaid"),
shipping_charge (INR excl GST, number), gst_amount (INR, number),
total_amount (INR, number), order_value (INR, number or null).

Return ONLY the JSON object, no explanation, no markdown fences.

--- RAW OCR TEXT ---
{ocr_text}
"""

def parse_invoice_with_llm(ocr_text, api_key):
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-sonnet-4-20250514", max_tokens=1000,
        messages=[{"role": "user", "content": EXTRACTION_PROMPT.format(ocr_text=ocr_text[:8000])}],
    )
    raw = re.sub(r"^```[a-z]*\n?", "", message.content[0].text.strip())
    raw = re.sub(r"\n?```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            return json.loads(m.group())
        raise ValueError(f"LLM returned non-JSON:\n{raw}")

def parse_invoice_regex_fallback(ocr_text):
    def find(patterns, text, cast=None):
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                val = m.group(1).strip().replace(",", "")
                return cast(val) if cast else val
        return None

    t = ocr_text
    payment_mode = None
    if re.search(r"\bcod\b|\bcash\s*on\s*delivery\b", t, re.I):
        payment_mode = "COD"
    elif re.search(r"\bprepaid\b|\bonline\b|\bpaid\b", t, re.I):
        payment_mode = "Prepaid"

    courier = find([r"(bluedart|delhivery|ekart|dtdc|ecom express|xpressbees|shadowfax)"], t)

    return {
        "invoice_number": find([r"invoice\s*(?:no|number|#)[:\s#]*([A-Z0-9\-/]+)", r"inv[.\s]*no[:\s]*([A-Z0-9\-/]+)"], t),
        "invoice_date":   find([r"invoice\s*date[:\s]*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})", r"date[:\s]*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})"], t),
        "awb_number":     find([r"awb\s*(?:no|number|#)[:\s#]*([A-Z0-9\-]+)", r"airway\s*bill[:\s]*([A-Z0-9\-]+)", r"tracking\s*(?:no|id)[:\s]*([A-Z0-9\-]+)"], t),
        "courier_partner": (courier or "").title() or None,
        "shipment_weight": find([r"(?:chargeable|charged|actual)\s*weight[:\s]*([\d.]+)\s*kg", r"weight[:\s]*([\d.]+)\s*kg"], t, cast=float),
        "payment_mode":   payment_mode,
        "shipping_charge": find([r"(?:shipping|freight|delivery)\s*charge[:\s]*([\d,.]+)", r"base\s*(?:price|charge)[:\s]*([\d,.]+)"], t, cast=float),
        "gst_amount":     find([r"(?:gst|igst|cgst\s*\+\s*sgst)[:\s]*([\d,.]+)", r"tax[:\s]*([\d,.]+)"], t, cast=float),
        "total_amount":   find([r"(?:grand\s*total|total\s*amount|net\s*payable)[:\s]*([\d,.]+)", r"total[:\s]*([\d,.]+)"], t, cast=float),
        "order_value":    find([r"(?:order|declared|invoice)\s*value[:\s]*([\d,.]+)", r"cod\s*amount[:\s]*([\d,.]+)"], t, cast=float),
    }

# ── AUDIT ENGINE ──────────────────────────────────────────────────────────────
class AuditFinding:
    ICONS = {"ERROR": "🔴", "WARNING": "🟡", "INFO": "🔵"}
    def __init__(self, rule, severity, message, expected=None, actual=None, difference=None):
        self.rule, self.severity, self.message = rule, severity, message
        self.expected, self.actual, self.difference = expected, actual, difference
    def to_dict(self):
        return {"rule": self.rule, "severity": self.severity,
                "icon": self.ICONS.get(self.severity, "⚪"), "message": self.message,
                "expected": self.expected, "actual": self.actual, "difference": self.difference}

def get_rate_card(courier):
    if courier:
        for key in RATE_CARD:
            if key.lower() in (courier or "").lower():
                return RATE_CARD[key]
    return RATE_CARD["Default"]

def expected_shipping_charge(weight_kg, card):
    for slab_max, charge in card["slabs"]:
        if weight_kg <= slab_max:
            return float(charge)
    return float(card["slabs"][-1][1])

def audit_weight_slab(data, card):
    w, c = data.get("shipment_weight"), data.get("shipping_charge")
    if w is None or c is None:
        return AuditFinding("Weight Slab", "INFO", "Cannot verify weight slab: weight or charge missing.")
    exp = expected_shipping_charge(w, card)
    diff = c - exp
    if abs(diff) > 5.0:
        return AuditFinding("Weight Slab Mismatch", "ERROR",
            f"Shipping charge ₹{c:.2f} doesn't match rate card for {w} kg (expected ₹{exp:.2f}).",
            expected=f"₹{exp:.2f}", actual=f"₹{c:.2f}", difference=f"₹{diff:+.2f}")
    return None

def audit_cod_on_prepaid(data, card):
    if (data.get("payment_mode") or "").upper() != "PREPAID":
        return None
    total = data.get("total_amount") or 0
    shipping = data.get("shipping_charge") or 0
    gst = data.get("gst_amount") or 0
    extras = total - shipping - gst
    cod_est = max((data.get("order_value") or 1000) * card["cod_charge_pct"] / 100, COD_MIN_CHARGE)
    if extras > cod_est * 0.5:
        return AuditFinding("COD on Prepaid", "ERROR",
            f"Potential COD charge (₹{extras:.2f}) detected on a Prepaid shipment.",
            expected="₹0.00", actual=f"₹{extras:.2f}", difference=f"₹{extras:+.2f}")
    return None

def audit_duplicate_awb(data, seen_awbs):
    awb = data.get("awb_number")
    if not awb:
        return AuditFinding("Duplicate AWB", "INFO", "AWB number not found; cannot check duplicates.")
    if awb in seen_awbs:
        return AuditFinding("Duplicate AWB", "ERROR",
            f"AWB {awb} has already been billed in this audit session.",
            expected="Unique AWB", actual=awb)
    seen_awbs.add(awb)
    return None

def audit_gst(data):
    s, g = data.get("shipping_charge"), data.get("gst_amount")
    if s is None or g is None:
        return AuditFinding("GST Calculation", "INFO", "Cannot verify GST: shipping charge or GST amount missing.")
    exp = round(s * GST_RATE, 2)
    diff = g - exp
    if abs(diff) > 2.0:
        return AuditFinding("GST Miscalculation", "ERROR",
            f"GST ₹{g:.2f} doesn't match 18% of shipping charge ₹{s:.2f}.",
            expected=f"₹{exp:.2f}", actual=f"₹{g:.2f}", difference=f"₹{diff:+.2f}")
    return None

def audit_excess_weight_charge(data, card):
    w, tot, s, g = data.get("shipment_weight"), data.get("total_amount"), data.get("shipping_charge"), data.get("gst_amount")
    if None in (w, tot, s, g):
        return None
    surplus = tot - s - g
    per_kg = card["excess_weight_per_kg"]
    if surplus > per_kg * 1.5:
        return AuditFinding("Excess Weight Charge", "WARNING",
            f"Extra charges ₹{surplus:.2f} exceed expected excess-weight rate (₹{per_kg}/kg).",
            expected=f"≤ ₹{per_kg*1.5:.2f}", actual=f"₹{surplus:.2f}",
            difference=f"₹{surplus - per_kg*1.5:+.2f}")
    return None

def run_audit(data, seen_awbs):
    card = get_rate_card(data.get("courier_partner"))
    findings = [f.to_dict() for f in [
        audit_weight_slab(data, card),
        audit_cod_on_prepaid(data, card),
        audit_duplicate_awb(data, seen_awbs),
        audit_gst(data),
        audit_excess_weight_charge(data, card),
    ] if f]
    if not findings:
        findings.append({"rule":"All Clear","severity":"INFO","icon":"✅",
                         "message":"No billing anomalies detected.",
                         "expected":None,"actual":None,"difference":None})
    return findings

# ── UI ────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Invoice Auditor", page_icon="🧾", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
:root{--bg:#0d0f14;--surface:#161922;--surface2:#1e2330;--border:#2a3040;--accent:#4f8ef7;--accent2:#7c3aed;--success:#22c55e;--warning:#f59e0b;--error:#ef4444;--text:#e2e8f0;--muted:#64748b;}
html,body,[data-testid="stAppViewContainer"]{background:var(--bg)!important;color:var(--text)!important;font-family:'Space Grotesk',sans-serif!important;}
[data-testid="stSidebar"]{background:var(--surface)!important;border-right:1px solid var(--border);}
.block-container{padding-top:1.5rem!important;}
.stFileUploader>div{background:var(--surface2)!important;border:1.5px dashed var(--border)!important;border-radius:12px!important;}
.stButton>button{background:linear-gradient(135deg,var(--accent),var(--accent2))!important;color:#fff!important;border:none!important;border-radius:8px!important;font-weight:600!important;padding:.5rem 1.4rem!important;}
.card{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:1.2rem 1.4rem;margin-bottom:1rem;}
.finding-error{border-left:4px solid var(--error)!important;}
.finding-warning{border-left:4px solid var(--warning)!important;}
.finding-info{border-left:4px solid var(--accent)!important;}
.finding-ok{border-left:4px solid var(--success)!important;}
.badge{display:inline-block;padding:2px 10px;border-radius:999px;font-size:.72rem;font-weight:600;text-transform:uppercase;}
.badge-error{background:#7f1d1d;color:#fca5a5;}
.badge-warning{background:#78350f;color:#fcd34d;}
.badge-info{background:#1e3a5f;color:#93c5fd;}
.badge-ok{background:#14532d;color:#86efac;}
.metric-box{background:var(--surface2);border:1px solid var(--border);border-radius:10px;padding:.9rem 1rem;text-align:center;}
.metric-value{font-size:1.6rem;font-weight:700;font-family:'JetBrains Mono',monospace;}
.metric-label{font-size:.75rem;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;margin-top:2px;}
.ocr-box{background:#0a0c10;border:1px solid var(--border);border-radius:10px;padding:1rem;font-family:'JetBrains Mono',monospace;font-size:.75rem;color:#94a3b8;max-height:260px;overflow-y:auto;white-space:pre-wrap;}
.json-box{background:#0a0c10;border:1px solid var(--border);border-radius:10px;padding:1rem;font-family:'JetBrains Mono',monospace;font-size:.78rem;color:#7dd3fc;max-height:340px;overflow-y:auto;white-space:pre;}
</style>
""", unsafe_allow_html=True)

if "seen_awbs" not in st.session_state:
    st.session_state.seen_awbs = set()
if "audit_history" not in st.session_state:
    st.session_state.audit_history = []

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    api_key = st.text_input("Anthropic API Key", type="password", placeholder="sk-ant-...",
        help="Leave blank to use regex fallback.")
    st.markdown("---")
    st.markdown("### 📋 Rate Card Preview")
    cp = st.selectbox("Courier", list(RATE_CARD.keys()))
    cpcard = RATE_CARD[cp]
    slab_rows = "".join(f"<tr><td>≤ {s[0]} kg</td><td>₹{s[1]}</td></tr>" for s in cpcard["slabs"])
    st.markdown(f"<table style='width:100%;font-size:.8rem;color:#94a3b8'><thead><tr><th>Slab</th><th>Rate</th></tr></thead><tbody>{slab_rows}</tbody></table>", unsafe_allow_html=True)
    st.markdown(f"COD %: **{cpcard['cod_charge_pct']}%** | Excess/kg: **₹{cpcard['excess_weight_per_kg']}**")
    st.markdown("---")
    if st.button("🗑️ Reset Session"):
        st.session_state.seen_awbs = set()
        st.session_state.audit_history = []
        st.success("Session cleared.")

# Header
st.markdown("""
<div style='display:flex;align-items:center;gap:1rem;margin-bottom:1.5rem'>
  <div style='font-size:2.8rem'>🧾</div>
  <div>
    <h1 style='margin:0;font-size:2rem;background:linear-gradient(90deg,#4f8ef7,#7c3aed);-webkit-background-clip:text;-webkit-text-fill-color:transparent'>AI Invoice Auditor</h1>
    <p style='margin:0;color:#64748b;font-size:.9rem'>Shiprocket Logistics · Automated Billing Anomaly Detection</p>
  </div>
</div>""", unsafe_allow_html=True)

# Upload
col_up1, col_up2 = st.columns(2)
with col_up1:
    invoice_file = st.file_uploader("📄 Upload Invoice PDF", type=["pdf"])
with col_up2:
    label_file = st.file_uploader("🏷️ Upload Shipment Label (optional)", type=["pdf","png","jpg","jpeg"])

if not OCR_AVAILABLE:
    st.error("OCR libraries not available. Install: `pip install pytesseract pillow pdf2image`")

run_audit_btn = st.button("🔍 Run Audit", disabled=(invoice_file is None or not OCR_AVAILABLE))

# ═══════════════════════════════════════════════════════════════════════════════
# ALL pipeline + results logic lives INSIDE this single if-block.
# This is the fix: nothing below references `findings` or `invoice_data`
# outside of this guarded scope.
# ═══════════════════════════════════════════════════════════════════════════════
if run_audit_btn and invoice_file:

    with st.spinner("Running audit pipeline…"):
        st.markdown("---")
        st.markdown("### 🔬 Audit Pipeline")
        progress = st.progress(0, text="Step 1 / 5 — Reading file…")

        invoice_bytes = invoice_file.read()
        file_hash = hashlib.md5(invoice_bytes).hexdigest()[:8].upper()

        progress.progress(20, text="Step 2 / 5 — Extracting text via OCR…")
        try:
            ocr_text = (extract_text_from_pdf(invoice_bytes)
                        if invoice_file.type == "application/pdf"
                        else extract_text_from_image(invoice_bytes))
        except Exception as e:
            st.error(f"OCR failed: {e}")
            st.stop()

        if label_file:
            label_bytes = label_file.read()
            try:
                label_ocr = (extract_text_from_pdf(label_bytes)
                             if label_file.type == "application/pdf"
                             else extract_text_from_image(label_bytes))
                if label_ocr:
                    ocr_text += "\n\n--- LABEL ---\n\n" + label_ocr
            except Exception:
                pass

        progress.progress(40, text="Step 3 / 5 — Parsing invoice fields…")
        use_llm = bool(api_key and ANTHROPIC_AVAILABLE)
        parse_method = "Claude (LLM)" if use_llm else "Regex Fallback"
        try:
            invoice_data = (parse_invoice_with_llm(ocr_text, api_key)
                            if use_llm else parse_invoice_regex_fallback(ocr_text))
        except Exception as e:
            st.warning(f"LLM parse failed ({e}), falling back to regex.")
            invoice_data = parse_invoice_regex_fallback(ocr_text)
            parse_method = "Regex Fallback (LLM error)"

        progress.progress(60, text="Step 4 / 5 — Structuring data…")
        invoice_data["_meta"] = {
            "file_name": invoice_file.name, "file_hash": file_hash,
            "parse_method": parse_method,
            "audited_at": datetime.now().isoformat(timespec="seconds"),
        }

        progress.progress(80, text="Step 5 / 5 — Running audit rules…")
        findings = run_audit(invoice_data, st.session_state.seen_awbs)

        progress.progress(100, text="✅ Audit complete.")
        st.session_state.audit_history.append({"data": invoice_data, "findings": findings})

    # ── Report ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 📊 Audit Report")

    errors   = sum(1 for f in findings if f["severity"] == "ERROR")
    warnings = sum(1 for f in findings if f["severity"] == "WARNING")
    status   = "FAIL" if errors else ("REVIEW" if warnings else "PASS")
    status_color = "#ef4444" if errors else ("#f59e0b" if warnings else "#22c55e")

    total_overcharge = 0.0
    for f in findings:
        if f.get("difference"):
            try:
                v = float(str(f["difference"]).replace("₹","").replace("+",""))
                if v > 0:
                    total_overcharge += v
            except (ValueError, TypeError):
                pass

    # Metrics
    cols = st.columns(6)
    for col, (val, label, color) in zip(cols, [
        (status,                                            "Audit Status",        status_color),
        (str(errors),                                       "Errors",              "#ef4444"),
        (str(warnings),                                     "Warnings",            "#f59e0b"),
        (f"₹{invoice_data.get('total_amount') or 0:,.2f}", "Total Billed",        "#4f8ef7"),
        (f"₹{total_overcharge:,.2f}",                      "Overcharge Detected", "#ef4444"),
        (invoice_data.get("awb_number") or "—",            "AWB Number",          "#7c3aed"),
    ]):
        with col:
            st.markdown(
                f"<div class='metric-box'>"
                f"<div class='metric-value' style='color:{color}'>{val}</div>"
                f"<div class='metric-label'>{label}</div></div>",
                unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns(2, gap="large")

    # Extracted fields
    with col_l:
        st.markdown("### 🗂️ Extracted Invoice Fields")
        field_icons = {
            "invoice_number":"🔢","invoice_date":"📅","awb_number":"📦",
            "courier_partner":"🚚","shipment_weight":"⚖️","payment_mode":"💳",
            "shipping_charge":"💰","gst_amount":"🧾","total_amount":"💵","order_value":"🛍️",
        }
        for field, value in {k:v for k,v in invoice_data.items() if not k.startswith("_") and v is not None}.items():
            icon = field_icons.get(field, "•")
            label = field.replace("_"," ").title()
            if isinstance(value, float):
                disp = (f"₹{value:,.2f}" if any(x in field for x in ("amount","charge","value")) else f"{value} kg")
            else:
                disp = str(value)
            st.markdown(
                f"<div class='card' style='padding:.7rem 1rem;margin-bottom:.5rem;"
                f"display:flex;justify-content:space-between;align-items:center'>"
                f"<span style='color:#94a3b8;font-size:.85rem'>{icon} {label}</span>"
                f"<span style='font-weight:600;font-family:JetBrains Mono,monospace;color:#e2e8f0'>{disp}</span></div>",
                unsafe_allow_html=True)

    # Audit findings
    with col_r:
        st.markdown("### 🚨 Audit Findings")
        for f in findings:
            sev = f["severity"].lower()
            badge_cls = f"badge-{sev}" if sev in ("error","warning","info") else "badge-ok"
            find_cls  = (f"finding-{sev}" if sev in ("error","warning")
                         else ("finding-ok" if f["rule"]=="All Clear" else "finding-info"))
            extras = ""
            if f.get("expected"):
                extras = (f"<br><small style='color:#94a3b8'>Expected: <b>{f['expected']}</b>"
                          f" · Actual: <b>{f['actual']}</b>")
                if f.get("difference"):
                    extras += f" · Diff: <b>{f['difference']}</b>"
                extras += "</small>"
            st.markdown(
                f"<div class='card {find_cls}'>"
                f"<div style='display:flex;justify-content:space-between;align-items:flex-start'>"
                f"<span style='font-weight:600'>{f['icon']} {f['rule']}</span>"
                f"<span class='badge {badge_cls}'>{f['severity']}</span></div>"
                f"<p style='margin:.4rem 0 0;font-size:.87rem;color:#cbd5e1'>{f['message']}{extras}</p></div>",
                unsafe_allow_html=True)

    with st.expander("🔍 OCR Text", expanded=False):
        st.markdown(f"<div class='ocr-box'>{ocr_text[:4000]}</div>", unsafe_allow_html=True)

    with st.expander("📦 Structured JSON", expanded=False):
        st.markdown(f"<div class='json-box'>{json.dumps(invoice_data, indent=2)}</div>", unsafe_allow_html=True)

    report = {"invoice": invoice_data, "findings": findings,
              "summary": {"status": status, "errors": errors,
                          "warnings": warnings, "total_overcharge": total_overcharge}}
    st.download_button("⬇️ Download Audit Report (JSON)",
        data=json.dumps(report, indent=2),
        file_name=f"audit_{file_hash}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json")

# Session history — safe: only shown when history exists, no unguarded variables
if st.session_state.audit_history:
    st.markdown("---")
    st.markdown("## 📜 Session Audit History")
    for i, entry in enumerate(reversed(st.session_state.audit_history), 1):
        d, fs = entry["data"], entry["findings"]
        errs  = sum(1 for f in fs if f["severity"] == "ERROR")
        warns = sum(1 for f in fs if f["severity"] == "WARNING")
        icon  = "🔴" if errs else ("🟡" if warns else "✅")
        with st.expander(f"{icon} #{i} — {d.get('invoice_number','N/A')} | AWB: {d.get('awb_number','N/A')} | {d['_meta']['audited_at']}"):
            c1, c2 = st.columns(2)
            with c1:
                st.json({k: v for k, v in d.items() if not k.startswith("_")})
            with c2:
                for f in fs:
                    st.markdown(f"{f['icon']} **{f['rule']}** — {f['message']}")
