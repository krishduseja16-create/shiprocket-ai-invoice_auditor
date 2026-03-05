# 🧾 AI Invoice Auditor — Shiprocket Logistics

An end-to-end Streamlit application that audits logistics invoices using OCR + LLM.

## Quick Start

```bash
# 1. Install system dependency (Tesseract OCR)
# macOS:
brew install tesseract poppler

# Ubuntu/Debian:
sudo apt-get install tesseract-ocr poppler-utils

# 2. Install Python packages
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

## Architecture

```
PDF Upload
    │
    ▼
pdf2image          ← converts PDF pages to PIL images
    │
    ▼
pytesseract OCR    ← extracts raw text from each page
    │
    ▼
Claude LLM         ← parses invoice fields into JSON
(or Regex fallback if no API key)
    │
    ▼
Audit Engine       ← runs 5 billing anomaly checks
    │
    ▼
Streamlit Report   ← displays findings + download
```

## Extracted Fields

| Field | Description |
|---|---|
| `invoice_number` | Invoice ID |
| `invoice_date` | Date of invoice |
| `awb_number` | Air Waybill / tracking number |
| `courier_partner` | BlueDart, Delhivery, Ekart, DTDC… |
| `shipment_weight` | Chargeable weight in kg |
| `payment_mode` | COD or Prepaid |
| `shipping_charge` | Base freight (excl. GST) |
| `gst_amount` | GST billed |
| `total_amount` | Grand total |

## Audit Rules

| # | Rule | Severity |
|---|---|---|
| 1 | Weight slab mismatch vs rate card | 🔴 ERROR |
| 2 | COD charge on prepaid shipment | 🔴 ERROR |
| 3 | Duplicate AWB number in session | 🔴 ERROR |
| 4 | GST miscalculation (≠18%) | 🔴 ERROR |
| 5 | Invalid excess weight charge | 🟡 WARNING |
| 6 | Hidden charges anomaly (total >> expected) | 🟡 WARNING |

## Configuration

- **API Key**: Enter your Anthropic key in the sidebar for LLM extraction.  
  Without it, the app falls back to regex-based parsing.
- **Rate Card**: Edit the `RATE_CARD` dict in `app.py` to match your contract rates.
- **GST Rate**: Change `GST_RATE = 0.18` for non-standard rates.
