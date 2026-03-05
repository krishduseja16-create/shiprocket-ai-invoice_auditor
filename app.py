"""
AI Invoice Auditor
Automated Logistics Invoice Audit Tool
"""

import streamlit as st
import json
import re
import io
import hashlib
from datetime import datetime

import pytesseract
from PIL import Image
from pdf2image import convert_from_bytes

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

GST_RATE = 0.18
COD_MIN = 50  # Real market minimum: ₹30-50, using ₹50 as conservative

# Real approximate market rates via Shiprocket (per kg slab, surface/standard)
# Source: edesy.in rate comparison + Shiprocket published estimates (2025)
# Note: Actual rates vary by zone (local/regional/national) and volume contract.
# These are indicative national-zone rates for audit reference.
RATE_CARD = {

    "BlueDart": {
        # Premium courier — highest rates, fastest delivery
        # Base ₹72 + ₹26/kg approx from edesy comparison
        "slabs": [
            (0.5,  98),
            (1.0,  130),
            (2.0,  175),
            (5.0,  290),
            (10.0, 520),
            (float("inf"), 620),
        ],
        "cod_charge_pct": 2.0,   # 2% of order value
        "cod_min": 50,
        "excess_weight_per_kg": 52,
    },

    "Delhivery": {
        # ₹59 base + ₹20/kg approx from edesy comparison
        "slabs": [
            (0.5,  78),
            (1.0,  100),
            (2.0,  140),
            (5.0,  230),
            (10.0, 390),
            (float("inf"), 470),
        ],
        "cod_charge_pct": 1.75,
        "cod_min": 50,
        "excess_weight_per_kg": 40,
    },

    "Ekart": {
        # Flipkart's courier — ₹52 base + ₹18/kg from edesy
        "slabs": [
            (0.5,  70),
            (1.0,  90),
            (2.0,  125),
            (5.0,  200),
            (10.0, 340),
            (float("inf"), 410),
        ],
        "cod_charge_pct": 1.5,
        "cod_min": 50,
        "excess_weight_per_kg": 34,
    },

    "DTDC": {
        # Budget courier — ₹52 base + ₹16/kg from edesy
        "slabs": [
            (0.5,  68),
            (1.0,  88),
            (2.0,  120),
            (5.0,  195),
            (10.0, 325),
            (float("inf"), 390),
        ],
        "cod_charge_pct": 1.5,
        "cod_min": 50,
        "excess_weight_per_kg": 32,
    },

    "Xpressbees": {
        # ₹49 base + ₹17/kg from edesy
        "slabs": [
            (0.5,  66),
            (1.0,  85),
            (2.0,  118),
            (5.0,  190),
            (10.0, 315),
            (float("inf"), 380),
        ],
        "cod_charge_pct": 1.5,
        "cod_min": 50,
        "excess_weight_per_kg": 33,
    },

    "Shadowfax": {
        # ₹46 base + ₹14/kg from edesy
        "slabs": [
            (0.5,  60),
            (1.0,  78),
            (2.0,  108),
            (5.0,  175),
            (10.0, 290),
            (float("inf"), 350),
        ],
        "cod_charge_pct": 1.5,
        "cod_min": 50,
        "excess_weight_per_kg": 30,
    },

    "Ecom Express": {
        # ₹55 base + ₹18/kg from edesy
        "slabs": [
            (0.5,  73),
            (1.0,  94),
            (2.0,  130),
            (5.0,  210),
            (10.0, 350),
            (float("inf"), 420),
        ],
        "cod_charge_pct": 1.5,
        "cod_min": 50,
        "excess_weight_per_kg": 35,
    },

    "Default": {
        "slabs": [
            (0.5,  70),
            (1.0,  90),
            (2.0,  125),
            (5.0,  200),
            (10.0, 340),
            (float("inf"), 410),
        ],
        "cod_charge_pct": 1.75,
        "cod_min": 50,
        "excess_weight_per_kg": 35,
    },
}

# --------------------------------------------------
# OCR
# --------------------------------------------------

def pdf_to_images(pdf_bytes):
    return convert_from_bytes(pdf_bytes, dpi=300)

def ocr_image(image):
    return pytesseract.image_to_string(image, config="--oem 3 --psm 4")

def extract_text_from_pdf(pdf_bytes):
    pages = pdf_to_images(pdf_bytes)
    return "\n\n".join(ocr_image(p) for p in pages)

def extract_text_from_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    return ocr_image(img)

# --------------------------------------------------
# COURIER DETECTION
# --------------------------------------------------

def normalize_courier(name):
    if not name:
        return "Default"
    name = name.lower()
    if "blue" in name or "bluedart" in name:
        return "BlueDart"
    if "delhivery" in name:
        return "Delhivery"
    if "ekart" in name:
        return "Ekart"
    if "dtdc" in name:
        return "DTDC"
    if "xpress" in name:
        return "Xpressbees"
    if "shadowfax" in name:
        return "Shadowfax"
    if "ecom" in name:
        return "Ecom Express"
    return "Default"

# --------------------------------------------------
# FIELD EXTRACTION
# --------------------------------------------------

def parse_invoice(text):

    def find(patterns, cast=None):
        for p in patterns:
            m = re.search(p, text, re.I)
            if m:
                val = m.group(1).replace(",", "").strip()
                try:
                    return cast(val) if cast else val
                except (ValueError, TypeError):
                    continue
        return None

    # Detect courier
    courier_raw = find([
        r"(bluedart|blue\s*dart|delhivery|ekart|dtdc|xpressbees|shadowfax|ecom\s*express)"
    ])

    # ✅ FIX: Payment mode — COD check first, then Prepaid (avoids dual-match bug)
    payment_mode = None
    if re.search(r"\bcod\b|cash\s*on\s*delivery", text, re.I):
        payment_mode = "COD"
    elif re.search(r"\bprepaid\b", text, re.I):
        payment_mode = "Prepaid"

    return {
        "invoice_number": find([
            r"invoice\s*(?:no|number|#)[:\s]*([A-Z0-9\-\/]+)",
            r"inv[.\s]*no[:\s]*([A-Z0-9\-\/]+)",
        ]),
        "invoice_date": find([
            r"invoice\s*date[:\s]*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",
            r"date[:\s]*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",
        ]),
        "awb_number": find([
            r"awb\s*(?:no|number|id)?[:\s]*([A-Z0-9\-]{8,})",
            r"tracking\s*(?:no|id)?[:\s]*([A-Z0-9\-]{8,})",
            r"\b([0-9]{10,})\b",
        ]),
        "courier_partner": courier_raw,
        "shipment_weight": find([
            r"(?:chargeable|charged|actual)\s*weight[:\s]*([\d.]+)",
            r"weight[:\s]*([\d.]+)",
        ], float),
        "payment_mode": payment_mode,
        "shipping_charge": find([
            r"(?:shipping|freight|delivery)\s*charge[:\s]*([\d.]+)",
            r"base\s*(?:price|charge)[:\s]*([\d.]+)",
        ], float),
        "gst_amount": find([
            r"(?:igst|cgst|sgst|gst)[:\s]*([\d.]+)",
            r"tax[:\s]*([\d.]+)",
        ], float),
        "total_amount": find([
            r"(?:grand\s*total|total\s*amount|net\s*payable)[:\s]*([\d.]+)",
            r"total[:\s]*([\d.]+)",
        ], float),
        "order_value": find([
            r"(?:order|declared|invoice)\s*value[:\s]*([\d.]+)",
            r"cod\s*amount[:\s]*([\d.]+)",
        ], float),
    }

# --------------------------------------------------
# AUDIT ENGINE
# --------------------------------------------------

def get_rate_card(courier):
    key = normalize_courier(courier)
    return RATE_CARD.get(key, RATE_CARD["Default"])

def expected_shipping(weight, card):
    for slab, price in card["slabs"]:
        if weight <= slab:
            return price
    return card["slabs"][-1][1]

def run_audit(data, seen_awb):
    findings = []
    card = get_rate_card(data.get("courier_partner"))

    weight = data.get("shipment_weight")
    charge = data.get("shipping_charge")
    gst    = data.get("gst_amount")
    total  = data.get("total_amount")
    mode   = (data.get("payment_mode") or "").upper()
    awb    = data.get("awb_number")
    order_val = data.get("order_value")

    # ── Rule 1: Weight slab mismatch ──────────────────────────────────────
    if weight and charge:
        expected = expected_shipping(weight, card)
        diff = charge - expected
        if abs(diff) > 10:  # ₹10 tolerance
            findings.append({
                "rule": "Weight Slab Mismatch",
                "severity": "ERROR",
                "message": f"Charged ₹{charge:.2f} but rate card says ₹{expected:.2f} for {weight} kg. Difference: ₹{diff:+.2f}",
            })
    else:
        findings.append({
            "rule": "Weight Slab",
            "severity": "INFO",
            "message": "Weight or shipping charge not found — cannot verify slab.",
        })

    # ── Rule 2: COD charge on Prepaid shipment ────────────────────────────
    if mode == "PREPAID" and total and charge and gst:
        implied_extras = total - charge - gst
        cod_estimate = max(
            (order_val or 1000) * card["cod_charge_pct"] / 100,
            card["cod_min"]
        )
        if implied_extras > (cod_estimate * 0.5):
            findings.append({
                "rule": "COD Charge on Prepaid",
                "severity": "ERROR",
                "message": f"Shipment is Prepaid but extra charges of ₹{implied_extras:.2f} detected — possible incorrect COD fee.",
            })

    # ── Rule 3: Duplicate AWB ─────────────────────────────────────────────
    if awb:
        if awb in seen_awb:
            findings.append({
                "rule": "Duplicate AWB",
                "severity": "ERROR",
                "message": f"AWB {awb} has already been billed in this session — possible duplicate invoice.",
            })
        seen_awb.add(awb)
    else:
        findings.append({
            "rule": "Duplicate AWB",
            "severity": "INFO",
            "message": "AWB number not found — cannot check for duplicates.",
        })

    # ── Rule 4: GST miscalculation ────────────────────────────────────────
    if charge and gst:
        expected_gst = round(charge * GST_RATE, 2)
        diff = gst - expected_gst
        if abs(diff) > 2:
            findings.append({
                "rule": "GST Miscalculation",
                "severity": "ERROR",
                "message": f"GST should be ₹{expected_gst:.2f} (18% of ₹{charge:.2f}) but ₹{gst:.2f} was charged. Diff: ₹{diff:+.2f}",
            })
    else:
        findings.append({
            "rule": "GST",
            "severity": "INFO",
            "message": "Shipping charge or GST amount missing — cannot verify.",
        })

    # ── Rule 5: Excess weight charge ──────────────────────────────────────
    if total and charge and gst:
        surplus = total - charge - gst
        per_kg  = card["excess_weight_per_kg"]
        if surplus > per_kg * 2:
            findings.append({
                "rule": "Excess Weight Charge",
                "severity": "WARNING",
                "message": f"Extra charges of ₹{surplus:.2f} exceed expected excess-weight rate of ₹{per_kg}/kg — verify with courier.",
            })

    return findings

# --------------------------------------------------
# UI
# --------------------------------------------------

st.set_page_config(page_title="AI Invoice Auditor", page_icon="🧾", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono&display=swap');
:root{--bg:#0d0f14;--surface:#161922;--surface2:#1e2330;--border:#2a3040;
      --accent:#4f8ef7;--accent2:#7c3aed;--success:#22c55e;
      --warning:#f59e0b;--error:#ef4444;--text:#e2e8f0;--muted:#64748b;}
html,body,[data-testid="stAppViewContainer"]{background:var(--bg)!important;color:var(--text)!important;font-family:'Space Grotesk',sans-serif!important;}
[data-testid="stSidebar"]{background:var(--surface)!important;border-right:1px solid var(--border);}
.block-container{padding-top:1.5rem!important;}
.stFileUploader>div{background:var(--surface2)!important;border:1.5px dashed var(--border)!important;border-radius:12px!important;}
.stButton>button{background:linear-gradient(135deg,var(--accent),var(--accent2))!important;color:#fff!important;border:none!important;border-radius:8px!important;font-weight:600!important;padding:.5rem 1.4rem!important;}
.card{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:1.1rem 1.3rem;margin-bottom:.8rem;}
.fe{border-left:4px solid var(--error)!important;}
.fw{border-left:4px solid var(--warning)!important;}
.fi{border-left:4px solid var(--accent)!important;}
.fok{border-left:4px solid var(--success)!important;}
.badge{display:inline-block;padding:2px 10px;border-radius:999px;font-size:.72rem;font-weight:600;text-transform:uppercase;}
.be{background:#7f1d1d;color:#fca5a5;}
.bw{background:#78350f;color:#fcd34d;}
.bi{background:#1e3a5f;color:#93c5fd;}
.bok{background:#14532d;color:#86efac;}
.mbox{background:var(--surface2);border:1px solid var(--border);border-radius:10px;padding:.9rem 1rem;text-align:center;}
.mv{font-size:1.5rem;font-weight:700;font-family:'JetBrains Mono',monospace;}
.ml{font-size:.72rem;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;margin-top:3px;}
.note{background:#1a1f2e;border:1px solid #2a3040;border-radius:8px;padding:.7rem 1rem;font-size:.78rem;color:#64748b;margin-top:1rem;}
</style>
""", unsafe_allow_html=True)

# Session state
if "seen_awb" not in st.session_state:
    st.session_state.seen_awb = set()
if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")
    st.markdown("### 📋 Rate Card Preview")
    st.markdown(
        "<div class='note'>ℹ️ Rates shown are indicative national-zone estimates "
        "based on 2025 Shiprocket market data. Actual rates depend on your "
        "contract, zone, and volume.</div>",
        unsafe_allow_html=True
    )
    cp = st.selectbox("Courier", list(RATE_CARD.keys()))
    cpcard = RATE_CARD[cp]
    rows = "".join(
        f"<tr><td style='padding:4px 8px'>≤ {s[0]} kg</td><td style='padding:4px 8px'>₹{s[1]}</td></tr>"
        for s in cpcard["slabs"]
    )
    st.markdown(
        f"<table style='width:100%;font-size:.82rem;color:#94a3b8;border-collapse:collapse'>"
        f"<thead><tr><th style='text-align:left;padding:4px 8px;border-bottom:1px solid #2a3040'>Slab</th>"
        f"<th style='text-align:left;padding:4px 8px;border-bottom:1px solid #2a3040'>Rate</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"COD: **{cpcard['cod_charge_pct']}%** (min ₹{cpcard['cod_min']}) | "
        f"Excess/kg: **₹{cpcard['excess_weight_per_kg']}**"
    )
    st.markdown("---")
    if st.button("🗑️ Reset Session"):
        st.session_state.seen_awb = set()
        st.session_state.history  = []
        st.success("Session cleared.")

# Header
st.markdown("""
<div style='display:flex;align-items:center;gap:1rem;margin-bottom:1.5rem'>
  <div style='font-size:2.8rem'>🧾</div>
  <div>
    <h1 style='margin:0;font-size:2rem;background:linear-gradient(90deg,#4f8ef7,#7c3aed);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent'>AI Invoice Auditor</h1>
    <p style='margin:0;color:#64748b;font-size:.9rem'>Shiprocket Logistics · Automated Billing Anomaly Detection</p>
  </div>
</div>""", unsafe_allow_html=True)

# Upload
c1, c2 = st.columns(2)
with c1:
    invoice_file = st.file_uploader("📄 Upload Invoice PDF", type=["pdf"])
with c2:
    label_file = st.file_uploader("🏷️ Upload Shipment Label (optional)", type=["pdf","png","jpg","jpeg"])

run_btn = st.button("🔍 Run Audit", disabled=invoice_file is None)

# ── ALL pipeline + results inside this block ──────────────────────────────────
if run_btn and invoice_file:

    with st.spinner("Running OCR and audit…"):

        # OCR
        pdf_bytes = invoice_file.read()
        file_hash = hashlib.md5(pdf_bytes).hexdigest()[:8].upper()

        try:
            ocr_text = extract_text_from_pdf(pdf_bytes)
        except Exception as e:
            st.error(f"OCR failed: {e}")
            st.stop()

        # Optional label
        if label_file:
            try:
                lb = label_file.read()
                label_ocr = (extract_text_from_pdf(lb)
                             if label_file.type == "application/pdf"
                             else extract_text_from_image(lb))
                if label_ocr:
                    ocr_text += "\n\n--- LABEL ---\n\n" + label_ocr
            except Exception:
                pass

        # Parse + audit
        data     = parse_invoice(ocr_text)
        findings = run_audit(data, st.session_state.seen_awb)

        data["_meta"] = {
            "file": invoice_file.name,
            "hash": file_hash,
            "audited_at": datetime.now().isoformat(timespec="seconds"),
        }
        st.session_state.history.append({"data": data, "findings": findings})

    # ── Report ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 📊 Audit Report")

    errors   = sum(1 for f in findings if f["severity"] == "ERROR")
    warnings = sum(1 for f in findings if f["severity"] == "WARNING")
    status   = "FAIL" if errors else ("REVIEW" if warnings else "PASS")
    s_color  = "#ef4444" if errors else ("#f59e0b" if warnings else "#22c55e")

    overcharge = 0.0
    for f in findings:
        m = re.search(r"Diff(?:erence)?:\s*₹([+\-]?[\d.]+)", f["message"])
        if m:
            try:
                v = float(m.group(1).replace("+",""))
                if v > 0:
                    overcharge += v
            except ValueError:
                pass

    cols = st.columns(6)
    for col, (val, label, color) in zip(cols, [
        (status,                                             "Audit Status",        s_color),
        (str(errors),                                        "Errors",              "#ef4444"),
        (str(warnings),                                      "Warnings",            "#f59e0b"),
        (f"₹{data.get('total_amount') or 0:,.2f}",          "Total Billed",        "#4f8ef7"),
        (f"₹{overcharge:,.2f}",                             "Overcharge Detected", "#ef4444"),
        (data.get("awb_number") or "—",                     "AWB Number",          "#7c3aed"),
    ]):
        with col:
            st.markdown(
                f"<div class='mbox'><div class='mv' style='color:{color}'>{val}</div>"
                f"<div class='ml'>{label}</div></div>",
                unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns(2, gap="large")

    # Extracted fields
    with col_l:
        st.markdown("### 🗂️ Extracted Invoice Fields")
        icons = {
            "invoice_number":"🔢","invoice_date":"📅","awb_number":"📦",
            "courier_partner":"🚚","shipment_weight":"⚖️","payment_mode":"💳",
            "shipping_charge":"💰","gst_amount":"🧾","total_amount":"💵","order_value":"🛍️",
        }
        for field, value in {k: v for k, v in data.items() if not k.startswith("_") and v is not None}.items():
            icon  = icons.get(field, "•")
            label = field.replace("_", " ").title()
            if isinstance(value, float):
                disp = (f"₹{value:,.2f}"
                        if any(x in field for x in ("amount","charge","value"))
                        else f"{value} kg")
            else:
                disp = str(value)
            st.markdown(
                f"<div class='card' style='padding:.6rem 1rem;margin-bottom:.5rem;"
                f"display:flex;justify-content:space-between;align-items:center'>"
                f"<span style='color:#94a3b8;font-size:.85rem'>{icon} {label}</span>"
                f"<span style='font-weight:600;font-family:JetBrains Mono,monospace;"
                f"color:#e2e8f0'>{disp}</span></div>",
                unsafe_allow_html=True)

    # Findings
    with col_r:
        st.markdown("### 🚨 Audit Findings")
        sev_icon = {"ERROR":"🔴","WARNING":"🟡","INFO":"🔵"}
        for f in findings:
            sev = f["severity"]
            fc  = {"ERROR":"fe","WARNING":"fw","INFO":"fi"}.get(sev,"fok")
            bc  = {"ERROR":"be","WARNING":"bw","INFO":"bi"}.get(sev,"bok")
            st.markdown(
                f"<div class='card {fc}'>"
                f"<div style='display:flex;justify-content:space-between;align-items:center'>"
                f"<span style='font-weight:600'>{sev_icon.get(sev,'✅')} {f['rule']}</span>"
                f"<span class='badge {bc}'>{sev}</span></div>"
                f"<p style='margin:.4rem 0 0;font-size:.85rem;color:#cbd5e1'>{f['message']}</p>"
                f"</div>",
                unsafe_allow_html=True)

    # Rate card note
    st.markdown(
        "<div class='note'>⚠️ <b>Rate Card Note:</b> Rates used for audit are indicative "
        "2025 national-zone estimates via Shiprocket. Your actual contracted rates may differ. "
        "Always cross-check findings against your courier service agreement.</div>",
        unsafe_allow_html=True)

    # Expandables
    with st.expander("🔍 Raw OCR Text"):
        st.text(ocr_text[:4000])

    with st.expander("📦 Structured JSON"):
        st.json({k: v for k, v in data.items() if not k.startswith("_")})

    # Download
    report = {
        "invoice": {k: v for k, v in data.items() if not k.startswith("_")},
        "findings": findings,
        "summary": {"status": status, "errors": errors,
                    "warnings": warnings, "overcharge_detected": overcharge},
    }
    st.download_button(
        "⬇️ Download Audit Report (JSON)",
        data=json.dumps(report, indent=2),
        file_name=f"audit_{file_hash}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
    )

# Session history
if st.session_state.history:
    st.markdown("---")
    st.markdown("## 📜 Session History")
    for i, entry in enumerate(reversed(st.session_state.history), 1):
        d, fs = entry["data"], entry["findings"]
        errs  = sum(1 for f in fs if f["severity"] == "ERROR")
        warns = sum(1 for f in fs if f["severity"] == "WARNING")
        icon  = "🔴" if errs else ("🟡" if warns else "✅")
        with st.expander(
            f"{icon} #{i} — {d.get('invoice_number','N/A')} | "
            f"AWB: {d.get('awb_number','N/A')} | {d['_meta']['audited_at']}"
        ):
            c1, c2 = st.columns(2)
            with c1:
                st.json({k: v for k, v in d.items() if not k.startswith("_")})
            with c2:
                sev_icon = {"ERROR":"🔴","WARNING":"🟡","INFO":"🔵"}
                for f in fs:
                    st.markdown(f"{sev_icon.get(f['severity'],'✅')} **{f['rule']}** — {f['message']}")
