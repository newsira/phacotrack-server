import os
import json
import re
import unicodedata
from typing import Dict, Any, List, Tuple, Optional

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


app = FastAPI()


# -----------------------------
# Helpers
# -----------------------------
def must_env(name: str) -> str:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        raise RuntimeError(f"Missing environment variable: {name}")
    return v


def normalize_endpoint(raw: str) -> str:
    v = (raw or "").strip()
    # Azure endpoints work with or without trailing slash, but normalize anyway
    return v[:-1] if v.endswith("/") else v


def normalize_secret(raw: str) -> str:
    """
    Remove ALL whitespace (including unicode separators/newlines) from secrets.
    Prevents weird header encoding issues.
    """
    if not raw:
        return ""
    raw = unicodedata.normalize("NFC", raw)
    return re.sub(r"\s+", "", raw).strip()


def sanitize_text(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFC", s)
    s = s.replace("\u2028", "\n").replace("\u2029", "\n")  # line/para separators
    s = s.replace("\ufeff", "").replace("\u200b", "")      # BOM / zero-width
    # Strip control chars except \n
    s = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", " ", s)
    # Normalize whitespace
    s = re.sub(r"[ \t\r\f\v]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def polygon_min_max_x(polygon) -> Optional[Tuple[float, float]]:
    """
    Azure polygon is usually a list of Point-like objects with .x/.y,
    sometimes dicts. We only need x.
    """
    if not polygon:
        return None
    xs: List[float] = []
    for p in polygon:
        if hasattr(p, "x"):
            xs.append(float(p.x))
        elif isinstance(p, dict) and "x" in p:
            xs.append(float(p["x"]))
    if not xs:
        return None
    return (min(xs), max(xs))


# -----------------------------
# Azure: Layout OCR + Tables
# -----------------------------
def analyze_with_layout(file_bytes: bytes) -> Dict[str, Any]:
    endpoint = normalize_endpoint(must_env("AZURE_DI_ENDPOINT"))
    key = normalize_secret(must_env("AZURE_DI_KEY"))

    client = DocumentAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(key))

    # prebuilt-layout gives tables + reading lines
    poller = client.begin_analyze_document(model_id="prebuilt-layout", document=file_bytes)
    result = poller.result()

    # Pages: width/height used for left/right split
    pages = getattr(result, "pages", None) or []
    page_by_number = {}  # 1-based page number in bounding regions
    for p in pages:
        # In SDK, p.page_number exists
        pn = getattr(p, "page_number", None)
        if pn is not None:
            page_by_number[int(pn)] = p

    # Full OCR lines
    all_lines: List[str] = []
    for p in pages:
        for line in (getattr(p, "lines", None) or []):
            content = getattr(line, "content", None)
            if content:
                all_lines.append(content)

    # Tables -> assign to OD/OS using bounding region polygon x-mid
    od_tables: List[str] = []
    os_tables: List[str] = []
    unknown_tables: List[str] = []

    tables = getattr(result, "tables", None) or []
    for t_index, table in enumerate(tables, start=1):
        # Find bounding region polygon (take first region)
        regions = getattr(table, "bounding_regions", None) or []
        side = "unknown"
        page_width = None

        if regions:
            r0 = regions[0]
            page_number = getattr(r0, "page_number", None)
            polygon = getattr(r0, "polygon", None)

            page_obj = page_by_number.get(int(page_number)) if page_number else None
            if page_obj is not None:
                page_width = float(getattr(page_obj, "width", 0.0) or 0.0)

            mm = polygon_min_max_x(polygon)
            if mm and page_width and page_width > 0:
                min_x, max_x = mm
                mid_x = (min_x + max_x) / 2.0
                side = "OD" if mid_x < (page_width / 2.0) else "OS"

        # Build table text: preserve row order/col order
        row_count = int(getattr(table, "row_count", 0) or 0)
        col_count = int(getattr(table, "column_count", 0) or 0)
        grid: List[List[str]] = [["" for _ in range(col_count)] for _ in range(row_count)]

        for cell in (getattr(table, "cells", None) or []):
            r = int(getattr(cell, "row_index", 0) or 0)
            c = int(getattr(cell, "column_index", 0) or 0)
            txt = getattr(cell, "content", None) or ""
            txt = sanitize_text(txt)
            # Some tables have repeated cells; keep the longest
            if 0 <= r < row_count and 0 <= c < col_count:
                if len(txt) > len(grid[r][c]):
                    grid[r][c] = txt

        # Render table to text (simple, stable)
        # Important: this keeps columns aligned for OpenAI parsing.
        lines: List[str] = []
        lines.append(f"[TABLE {t_index}] side={side} rows={row_count} cols={col_count}")
        for r in range(row_count):
            # join with " | " so it’s obvious column breaks exist
            lines.append(" | ".join(grid[r]).strip())
        table_text = sanitize_text("\n".join(lines))

        if side == "OD":
            od_tables.append(table_text)
        elif side == "OS":
            os_tables.append(table_text)
        else:
            unknown_tables.append(table_text)

    return {
        "ocrText": sanitize_text("\n".join(all_lines)),
        "odTablesText": sanitize_text("\n\n".join(od_tables)),
        "osTablesText": sanitize_text("\n\n".join(os_tables)),
        "unknownTablesText": sanitize_text("\n\n".join(unknown_tables)),
    }


# -----------------------------
# OpenAI: Parse to your Schema A
# -----------------------------
def call_openai_to_json(layout: Dict[str, Any]) -> Dict[str, Any]:
    api_key = normalize_secret(os.getenv("OPENAI_API_KEY", ""))
    if not api_key or OpenAI is None:
        return {
            "success": True,
            "documentType": "unknown",
            "fields": {"global": {}, "OD": {}, "OS": {}, "efx": None, "ust": None, "avg": None},
            "warnings": ["OPENAI_API_KEY not set (returning OCR/layout only)"],
            "raw": layout,
        }

    model = (os.getenv("OPENAI_MODEL", "gpt-4.1-mini") or "gpt-4.1-mini").strip()
    client = OpenAI(api_key=api_key)

    # Hard rule: OD blocks ONLY from OD_TABLES, OS blocks ONLY from OS_TABLES
    prompt = f"""
You are a medical document parser for ophthalmology documents.

Input is:
- OCR_TEXT: all extracted text lines
- OD_TABLES: tables from the LEFT half of the page (OD/right eye area)
- OS_TABLES: tables from the RIGHT half of the page (OS/left eye area)
- UNKNOWN_TABLES: tables that could not be placed

IMPORTANT RULES (must follow):
1) OD.blocks MUST be parsed ONLY from OD_TABLES.
2) OS.blocks MUST be parsed ONLY from OS_TABLES.
3) Do NOT move any table from one eye to the other.
4) If a table is missing, leave that eye’s blocks incomplete rather than guessing.
5) Output STRICT JSON only (no extra text). Use null when unknown.

documentType must be ONE of:
- opticalBiometry
- immersionBiometry
- ecc
- autokeratometry
- phacoSummary
- clinicNote
- unknown

Output JSON shape (Schema A):

{{
  "success": true,
  "documentType": "...",
  "fields": {{
    "global": {{
      "patientName": null|string,
      "hospitalNumber": null|string,
      "examDate": null|string,
      "scanDate": null|string,
      "biometryMethod": null|string,
      "pd": null|number
    }},
    "OD": {{
      "AL": null|number, "K1": null|number, "K2": null|number, "ACD": null|number,
      "LT": null|number, "WTW": null|number, "CCT": null|number,
      "ecc": null|number, "cv": null|number, "hex": null|number,
      "avgCellSize": null|number, "maxCellSize": null|number, "minCellSize": null|number,
      "sd": null|number, "numCells": null|number, "pachy": null|number,
      "blocks": [
        {{
          "IOLModel": null|string,
          "Aconst": null|number,
          "IOLrefs": [{{"IOL(D)": number, "REF(D)": number}}],
          "EmmeIOL": null|number
        }}
      ]
    }},
    "OS": {{
      "AL": null|number, "K1": null|number, "K2": null|number, "ACD": null|number,
      "LT": null|number, "WTW": null|number, "CCT": null|number,
      "ecc": null|number, "cv": null|number, "hex": null|number,
      "avgCellSize": null|number, "maxCellSize": null|number, "minCellSize": null|number,
      "sd": null|number, "numCells": null|number, "pachy": null|number,
      "blocks": [
        {{
          "IOLModel": null|string,
          "Aconst": null|number,
          "IOLrefs": [{{"IOL(D)": number, "REF(D)": number}}],
          "EmmeIOL": null|number
        }}
      ]
    }},
    "efx": null|string,
    "ust": null|string,
    "avg": null|string
  }},
  "warnings": []
}}

Extra notes:
- For ZEISS IOLMaster sheet like the example: each eye usually has 4 IOL tables (blocks).
- Table headers often contain IOL model name and A const.
- REF(D) can be negative.
- Keep IOLrefs in the same order they appear in the table.

OCR_TEXT:
\"\"\"
{layout.get("ocrText","")}
\"\"\"

OD_TABLES:
\"\"\"
{layout.get("odTablesText","")}
\"\"\"

OS_TABLES:
\"\"\"
{layout.get("osTablesText","")}
\"\"\"

UNKNOWN_TABLES (ignore unless absolutely necessary):
\"\"\"
{layout.get("unknownTablesText","")}
\"\"\"
""".strip()

    # Prefer Responses API with json_object; fallback to chat.completions
    try:
        resp = client.responses.create(
            model=model,
            input=prompt,
            response_format={"type": "json_object"},
        )
        return json.loads(resp.output_text)
    except Exception:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content)


# -----------------------------
# FastAPI endpoints
# -----------------------------
@app.get("/")
def root():
    return {"status": "server running"}


@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        if not file_bytes:
            return JSONResponse(status_code=400, content={"success": False, "error": "Empty file"})

        layout = analyze_with_layout(file_bytes)

        # If OCR totally empty, return minimal
        if not layout.get("ocrText"):
            return {
                "success": True,
                "documentType": "unknown",
                "fields": {"global": {}, "OD": {}, "OS": {}, "efx": None, "ust": None, "avg": None},
                "warnings": ["No OCR text found"],
            }

        parsed = call_openai_to_json(layout)
        return parsed

    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})
