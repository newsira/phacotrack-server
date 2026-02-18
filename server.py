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
    Prevents weird header / ascii crashes if key got pasted with hidden chars.
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


@app.get("/")
def root():
    return {"status": "server running"}


def _polygon_center_x(polygon: List[Dict[str, float]]) -> Optional[float]:
    """
    polygon looks like: [{"x":..., "y":...}, ...]
    Return average x.
    """
    if not polygon:
        return None
    xs = [p.get("x") for p in polygon if isinstance(p, dict) and p.get("x") is not None]
    if not xs:
        return None
    return sum(xs) / float(len(xs))


def _page_width(result, page_number_1based: int) -> Optional[float]:
    pages = getattr(result, "pages", None) or []
    for p in pages:
        if getattr(p, "page_number", None) == page_number_1based:
            return getattr(p, "width", None)
    return None


def _table_to_matrix(table) -> List[List[str]]:
    """
    Convert Azure table cells into a 2D matrix [row][col] of strings.
    """
    row_count = int(getattr(table, "row_count", 0) or 0)
    col_count = int(getattr(table, "column_count", 0) or 0)

    if row_count <= 0 or col_count <= 0:
        return []

    grid: List[List[str]] = [["" for _ in range(col_count)] for _ in range(row_count)]

    cells = getattr(table, "cells", None) or []
    for c in cells:
        r = getattr(c, "row_index", None)
        k = getattr(c, "column_index", None)
        txt = getattr(c, "content", None) or ""
        if r is None or k is None:
            continue
        if 0 <= r < row_count and 0 <= k < col_count:
            # Sometimes same cell appears twice; keep the longer content
            if len(txt.strip()) > len(grid[r][k].strip()):
                grid[r][k] = txt.strip()

    # Trim trailing empty columns per row (but keep at least 2 cols if present)
    out: List[List[str]] = []
    for row in grid:
        # remove trailing empty
        while len(row) > 0 and row[-1].strip() == "":
            row.pop()
        out.append(row)

    return out


def extract_text_and_tables_by_eye(file_bytes: bytes) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Use Azure DI (prebuilt-layout) to get:
    - text lines
    - tables + coordinates
    Then split tables into OD (left) vs OS (right) using page width midpoint.
    """
    endpoint = normalize_endpoint(must_env("AZURE_DI_ENDPOINT"))
    key = normalize_key(must_env("AZURE_DI_KEY"))

    client = DocumentAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(key))

    poller = client.begin_analyze_document(
        model_id="prebuilt-layout",
        document=file_bytes,
    )
    result = poller.result()

    # ---- Text (for headers like patient name, HN, exam date, etc.)
    lines_out: List[str] = []
    pages = getattr(result, "pages", None) or []
    for page in pages:
        page_lines = getattr(page, "lines", None) or []
        for line in page_lines:
            content = getattr(line, "content", None)
            if content:
                lines_out.append(content)

    full_text = sanitize_text("\n".join(lines_out))

    # ---- Tables split by left/right
    od_tables: List[Dict[str, Any]] = []
    os_tables: List[Dict[str, Any]] = []

    tables = getattr(result, "tables", None) or []
    for idx, t in enumerate(tables, start=1):
        # bounding_regions: list of regions, each has page_number + polygon
        regions = getattr(t, "bounding_regions", None) or []
        if not regions:
            continue

        region0 = regions[0]
        page_num = getattr(region0, "page_number", None)  # 1-based
        polygon = getattr(region0, "polygon", None) or []
        center_x = _polygon_center_x(polygon)

        width = _page_width(result, page_num) if page_num else None
        if width is None or center_x is None:
            # If no geometry, don't guess: put into "unassigned" bucket by treating as OD for now? better: skip.
            continue

        side = "left" if center_x < (float(width) / 2.0) else "right"
        matrix = _table_to_matrix(t)

        table_obj = {
            "index": idx,
            "page": page_num,
            "side": side,
            "centerX": center_x,
            "pageWidth": width,
            "matrix": matrix
        }

        if side == "left":
            od_tables.append(table_obj)
        else:
            os_tables.append(table_obj)

    return full_text, od_tables, os_tables


def call_openai_to_json(ocr_text: str, od_tables: List[Dict[str, Any]], os_tables: List[Dict[str, Any]]) -> Dict[str, Any]:
    api_key = normalize_key(os.getenv("OPENAI_API_KEY", "") or "")
    if not api_key or OpenAI is None:
        return {
            "success": True,
            "documentType": "unknown",
            "fields": {"global": {}, "OD": {}, "OS": {}, "efx": None, "ust": None, "avg": None},
            "warnings": ["OPENAI_API_KEY not set (returning OCR only)"],
            "rawText": ocr_text,
            "tables": {"OD": od_tables, "OS": os_tables},
        }

    model = (os.getenv("OPENAI_MODEL", "gpt-4.1-mini") or "gpt-4.1-mini").strip()
    client = OpenAI(api_key=api_key)

    ocr_text = sanitize_text(ocr_text)

    # We feed OpenAI BOTH text + tables, already grouped by eye using Azure coordinates.
    # Rule: DO NOT move tables between eyes.
    payload = {
        "ocrText": ocr_text,
        "tablesByEye": {
            "OD": od_tables,
            "OS": os_tables
        },
        "importantRule": "OD tables come from LEFT side of page, OS tables come from RIGHT side of page. Do NOT swap them."
    }

    prompt = f"""
You are a careful ophthalmology document parser.

You will receive:
1) OCR text (headers + general text)
2) Tables already grouped by eye using Azure table coordinates:
   - tablesByEye.OD = LEFT side of page (OD)
   - tablesByEye.OS = RIGHT side of page (OS)

CRITICAL RULE:
- DO NOT move any table from OD to OS or OS to OD.
- If a table cannot be interpreted, leave it out (do not guess).

Your job:
A) Choose documentType as ONE of:
   - opticalBiometry
   - immersionBiometry
   - ecc
   - autokeratometry
   - phacoSummary
   - clinicNote
   - unknown

B) Output STRICT JSON only (no extra text) in this exact shape:

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
          "blockIndex": "string",          // e.g. lens name or '1','2','3','4'
          "AConstant": "string|null",
          "Formula": "string|null",
          "rows": [{{"iolPower": number, "targetRefraction": number}}]
        }}
      ]
    }},
    "OS": {{ same keys as OD }},
    "efx": null|string,
    "ust": null|string,
    "avg": null|string
  }},
  "warnings": []
}}

Rules:
- Use null when unknown.
- Numbers must be numbers (not strings).
- For optical/immersion biometry:
  - Extract AL, K1, K2, ACD (and others if present) for OD and OS.
  - Extract ALL IOL target tables you can find from the provided tablesByEye matrices.
  - Each eye may have multiple blocks.
- For ECC:
  - Put ECC values into OD/OS ecc/cv/hex etc.
- For phacoSummary:
  - Fill efx/ust/avg only; keep OD/OS mostly null.
- Be conservative. No guessing.

INPUT (JSON):
{json.dumps(payload, ensure_ascii=False)}
""".strip()

    # Prefer Responses API with json_object formatting
    try:
        resp = client.responses.create(
            model=model,
            input=prompt,
            response_format={"type": "json_object"},
        )
        return json.loads(resp.output_text)
    except Exception:
        # fallback
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content)


@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        if not file_bytes:
            return JSONResponse(status_code=400, content={"success": False, "error": "Empty file"})

        ocr_text, od_tables, os_tables = extract_text_and_tables_by_eye(file_bytes)

        # If absolutely no text and no tables, return unknown
        if not ocr_text and not od_tables and not os_tables:
            return {
                "success": True,
                "documentType": "unknown",
                "fields": {"global": {}, "OD": {}, "OS": {}, "efx": None, "ust": None, "avg": None},
                "warnings": ["No OCR text or tables found"],
            }

        return call_openai_to_json(ocr_text, od_tables, os_tables)

    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})
