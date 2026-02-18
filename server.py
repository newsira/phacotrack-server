# server.py
import os
import json
import re
import unicodedata
from typing import Dict, Any, List, Optional, Tuple

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


app = FastAPI()


# -------------------------
# Utilities
# -------------------------
def must_env(name: str) -> str:
    v = os.getenv(name)
    if v is None or v == "":
        raise RuntimeError(f"Missing environment variable: {name}")
    return v


def normalize_endpoint(raw: str) -> str:
    v = (raw or "").strip()
    if v.endswith("/"):
        v = v[:-1]
    return v


def normalize_key(raw: str) -> str:
    """
    Remove ALL whitespace (including unicode separators/newlines) from secrets.
    Prevents header encoding crashes.
    """
    if not raw:
        return ""
    raw = unicodedata.normalize("NFC", raw)
    return re.sub(r"\s+", "", raw).strip()


def sanitize_text(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFC", s)
    s = s.replace("\u2028", "\n").replace("\u2029", "\n")
    s = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", " ", s)
    s = s.replace("\ufeff", "").replace("\u200b", "")
    s = re.sub(r"[ \t\r\f\v]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def polygon_center_xy(polygon: Optional[List[Any]]) -> Tuple[Optional[float], Optional[float]]:
    """
    polygon comes from azure form recognizer: list of Points (x,y)
    Return center (avg x, avg y).
    """
    if not polygon:
        return (None, None)
    xs = []
    ys = []
    for p in polygon:
        x = getattr(p, "x", None)
        y = getattr(p, "y", None)
        if x is None or y is None:
            continue
        xs.append(float(x))
        ys.append(float(y))
    if not xs or not ys:
        return (None, None)
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def polygon_bbox(polygon: Optional[List[Any]]) -> Optional[Dict[str, float]]:
    if not polygon:
        return None
    xs = []
    ys = []
    for p in polygon:
        x = getattr(p, "x", None)
        y = getattr(p, "y", None)
        if x is None or y is None:
            continue
        xs.append(float(x))
        ys.append(float(y))
    if not xs or not ys:
        return None
    return {"minX": min(xs), "minY": min(ys), "maxX": max(xs), "maxY": max(ys)}


# -------------------------
# Health
# -------------------------
@app.get("/")
def root():
    return {"status": "server running"}


# -------------------------
# Azure DI (prebuilt-layout)
# -------------------------
def analyze_layout_with_azure_di(file_bytes: bytes) -> Dict[str, Any]:
    """
    Returns a compact, geometry-aware payload:
    - pageWidth/pageHeight (page 1)
    - midX (pageWidth / 2)
    - headerLines: line text + (cx, cy)
    - tables: each table with side (OD/OS), bbox, center, and all cells (row, col, text)
    """
    endpoint = normalize_endpoint(must_env("AZURE_DI_ENDPOINT"))
    key = normalize_key(must_env("AZURE_DI_KEY"))

    client = DocumentAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(key))

    poller = client.begin_analyze_document(model_id="prebuilt-layout", document=file_bytes)
    result = poller.result()

    # We assume single-page images for your use case (phone photo of one sheet).
    # If multi-page ever appears, we still label items by pageNumber.
    pages = getattr(result, "pages", None) or []
    first_page = pages[0] if pages else None

    page_width = float(getattr(first_page, "width", 0.0) or 0.0)
    page_height = float(getattr(first_page, "height", 0.0) or 0.0)
    mid_x = page_width / 2.0 if page_width else None

    header_lines: List[Dict[str, Any]] = []
    if first_page and getattr(first_page, "lines", None):
        for ln in first_page.lines:
            txt = sanitize_text(getattr(ln, "content", "") or "")
            if not txt:
                continue
            poly = getattr(ln, "polygon", None)
            cx, cy = polygon_center_xy(poly)
            header_lines.append(
                {
                    "text": txt,
                    "cx": cx,
                    "cy": cy,
                }
            )

    tables_out: List[Dict[str, Any]] = []
    tables = getattr(result, "tables", None) or []
    for ti, t in enumerate(tables):
        # Determine table page + polygon
        # table.bounding_regions is a list, each has page_number + polygon
        brs = getattr(t, "bounding_regions", None) or []
        if not brs:
            continue

        # If multiple regions, use the first for side calculation
        br0 = brs[0]
        page_num = int(getattr(br0, "page_number", 1) or 1)
        poly = getattr(br0, "polygon", None)
        bbox = polygon_bbox(poly)
        cx, cy = polygon_center_xy(poly)

        side = "unknown"
        if mid_x is not None and cx is not None:
            side = "OD" if cx < mid_x else "OS"

        cells_out: List[Dict[str, Any]] = []
        cells = getattr(t, "cells", None) or []
        for c in cells:
            text = sanitize_text(getattr(c, "content", "") or "")
            if text == "":
                continue
            row = int(getattr(c, "row_index", 0) or 0)
            col = int(getattr(c, "column_index", 0) or 0)
            kind = getattr(c, "kind", None)  # may be "columnHeader"/"rowHeader"/"content"
            cells_out.append(
                {
                    "row": row,
                    "col": col,
                    "text": text,
                    "kind": kind,
                }
            )

        # Skip tables that have no useful text
        if not cells_out:
            continue

        tables_out.append(
            {
                "tableId": ti + 1,
                "pageNumber": page_num,
                "side": side,
                "center": {"x": cx, "y": cy},
                "bbox": bbox,
                "rowCount": int(getattr(t, "row_count", 0) or 0),
                "colCount": int(getattr(t, "column_count", 0) or 0),
                "cells": cells_out,
            }
        )

    # Sort tables top-to-bottom within each side (helps OpenAI)
    def sort_key(tbl: Dict[str, Any]) -> float:
        c = tbl.get("center") or {}
        y = c.get("y")
        return float(y) if y is not None else 1e9

    tables_out.sort(key=sort_key)

    payload = {
        "page": {
            "width": page_width,
            "height": page_height,
            "midX": mid_x,
        },
        "headerLines": header_lines,
        "tables": tables_out,
    }

    return payload


# -------------------------
# OpenAI parsing (AI reads structured payload, but OD/OS is already decided)
# -------------------------
def call_openai_to_json(layout_payload: Dict[str, Any]) -> Dict[str, Any]:
    api_key = normalize_key(os.getenv("OPENAI_API_KEY", ""))
    if not api_key or OpenAI is None:
        return {
            "success": True,
            "documentType": "unknown",
            "fields": {"global": {}, "OD": {}, "OS": {}, "efx": None, "ust": None, "avg": None},
            "warnings": ["OPENAI_API_KEY not set (returning layout payload only)"],
            "layoutPayload": layout_payload,
        }

    model = (os.getenv("OPENAI_MODEL", "gpt-4.1-mini") or "gpt-4.1-mini").strip()
    client = OpenAI(api_key=api_key)

    # Keep payload smaller/safer
    payload_json = json.dumps(layout_payload, ensure_ascii=False)

    prompt = f"""
You are a medical document parser for ophthalmology clinic documents.

You will receive a STRUCTURED payload extracted by Azure Document Intelligence prebuilt-layout.
IMPORTANT: OD/OS assignment is already done deterministically by the backend:
- tables[].side == "OD" means RIGHT EYE tables (left half of page)
- tables[].side == "OS" means LEFT EYE tables (right half of page)

Your job:
1) Decide documentType as ONE of:
   - opticalBiometry
   - immersionBiometry
   - ecc
   - autokeratometry
   - phacoSummary
   - clinicNote
   - unknown

2) Output STRICT JSON only (no extra text) with this exact shape:

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
      "blocks": []
    }},
    "OS": {{
      "AL": null|number, "K1": null|number, "K2": null|number, "ACD": null|number,
      "LT": null|number, "WTW": null|number, "CCT": null|number,
      "ecc": null|number, "cv": null|number, "hex": null|number,
      "avgCellSize": null|number, "maxCellSize": null|number, "minCellSize": null|number,
      "sd": null|number, "numCells": null|number, "pachy": null|number,
      "blocks": []
    }},
    "efx": null|string,
    "ust": null|string,
    "avg": null|string
  }},
  "warnings": []
}}

Rules (strict):
- Use null when unknown. Do NOT guess.
- Numbers must be real numbers (no commas, no stray characters).
- Use headerLines for patient demographics and global info (name/HN/date/machine/formula if global).
- Use tables for IOL blocks and any tabular values.
- IOL blocks: each block should contain:
  - blockIndex: "1"/"2"/"3"/"4" in top-to-bottom order WITHIN EACH SIDE (OD separately from OS)
  - AConstant: extract from table text like "A const: 119.00"
  - Formula: if clearly found as global formula (e.g. "Formula: SRK/T" in headerLines), apply it to all blocks.
            If formula is clearly inside the table, use that.
            If unclear, set null.
  - rows: list of {{ "iolPower": number, "targetRefraction": number }} from the table columns (IOL(D) and REF(D)).
- Do not merge OD and OS blocks. OD blocks come only from tables where side=="OD". OS blocks only from side=="OS".
- If you cannot confidently interpret a table as an IOL block, ignore it and add a warning.

INPUT PAYLOAD (JSON):
{payload_json}
""".strip()

    try:
        resp = client.responses.create(
            model=model,
            input=prompt,
            response_format={"type": "json_object"},
        )
        return json.loads(resp.output_text)
    except Exception:
        # Fallback for older SDK surfaces
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content)


# -------------------------
# API
# -------------------------
@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        if not file_bytes:
            return JSONResponse(status_code=400, content={"success": False, "error": "Empty file"})

        layout_payload = analyze_layout_with_azure_di(file_bytes)

        # If Azure returned nothing useful, return unknown
        if not layout_payload.get("headerLines") and not layout_payload.get("tables"):
            return {
                "success": True,
                "documentType": "unknown",
                "fields": {"global": {}, "OD": {}, "OS": {}, "efx": None, "ust": None, "avg": None},
                "warnings": ["No text/tables found by Azure layout"],
            }

        return call_openai_to_json(layout_payload)

    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})
