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


def env_bool(name: str) -> bool:
    v = (os.getenv(name, "") or "").strip().lower()
    return v in ("1", "true", "yes", "on")


@app.get("/")
def root():
    return {"status": "server running"}


def _polygon_center_x(polygon) -> Optional[float]:
    if not polygon:
        return None
    xs = []
    for p in polygon:
        if hasattr(p, "x") and p.x is not None:
            xs.append(float(p.x))
        elif isinstance(p, dict) and p.get("x") is not None:
            xs.append(float(p["x"]))
    if not xs:
        return None
    return sum(xs) / float(len(xs))


def _table_to_matrix(table) -> List[List[str]]:
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
            t = txt.strip()
            if len(t) > len(grid[r][k].strip()):
                grid[r][k] = t

    out: List[List[str]] = []
    for row in grid:
        while len(row) > 0 and row[-1].strip() == "":
            row.pop()
        out.append(row)

    return out


def extract_text_and_tables_half_split(file_bytes: bytes) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Uses Azure prebuilt-layout.
    Splits page into halves using page.width:
      - LEFT half -> OD (right eye)
      - RIGHT half -> OS (left eye)
    """
    endpoint = normalize_endpoint(must_env("AZURE_DI_ENDPOINT"))
    key = normalize_key(must_env("AZURE_DI_KEY"))

    client = DocumentAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(key))

    poller = client.begin_analyze_document(
        model_id="prebuilt-layout",
        document=file_bytes,
    )
    result = poller.result()

    # OCR text (useful for patient name, dates, etc.)
    lines_out: List[str] = []
    pages = getattr(result, "pages", None) or []
    for page in pages:
        for line in (getattr(page, "lines", None) or []):
            content = getattr(line, "content", None)
            if content:
                lines_out.append(content)
    full_text = sanitize_text("\n".join(lines_out))

    # Map page_number -> page_width (same unit as polygons)
    page_width: Dict[int, float] = {}
    page_unit: Dict[int, str] = {}
    for p in pages:
        pn = int(getattr(p, "page_number", 0) or 0)
        w = getattr(p, "width", None)
        u = getattr(p, "unit", None)
        if pn and w is not None:
            page_width[pn] = float(w)
            page_unit[pn] = str(u) if u is not None else ""

    od_tables: List[Dict[str, Any]] = []
    os_tables: List[Dict[str, Any]] = []

    tables = getattr(result, "tables", None) or []
    for idx, t in enumerate(tables, start=1):
        regions = getattr(t, "bounding_regions", None) or []
        if not regions:
            continue

        r0 = regions[0]
        pn = int(getattr(r0, "page_number", 0) or 0)
        polygon = getattr(r0, "polygon", None) or []
        cx = _polygon_center_x(polygon)
        if pn <= 0 or cx is None:
            continue

        w = page_width.get(pn)
        if w is None:
            # If we can't get page width, we cannot do half split safely.
            continue

        half = w / 2.0
        side = "left" if cx < half else "right"

        table_obj = {
            "index": idx,
            "page": pn,
            "pageWidth": w,
            "unit": page_unit.get(pn, ""),
            "centerX": float(cx),
            "side": side,
            "matrix": _table_to_matrix(t),
        }

        # Print debug in Railway logs
        print(f"[TABLE] idx={idx} page={pn} width={w} cx={cx:.4f} half={half:.4f} side={side}")

        if side == "left":
            od_tables.append(table_obj)
        else:
            os_tables.append(table_obj)

    debug_info = {
        "tablesTotal": len(tables),
        "tablesCaptured": len(od_tables) + len(os_tables),
        "odTables": len(od_tables),
        "osTables": len(os_tables),
        "pagesWithWidth": list(page_width.keys()),
    }

    return full_text, od_tables, os_tables, debug_info


def call_openai_to_json(ocr_text: str, od_tables: List[Dict[str, Any]], os_tables: List[Dict[str, Any]], debug_info: Dict[str, Any]) -> Dict[str, Any]:
    api_key = normalize_key(os.getenv("OPENAI_API_KEY", "") or "")
    if not api_key or OpenAI is None:
        out = {
            "success": True,
            "documentType": "unknown",
            "fields": {"global": {}, "OD": {}, "OS": {}, "efx": None, "ust": None, "avg": None},
            "warnings": ["OPENAI_API_KEY not set (returning OCR only)"],
            "rawText": ocr_text,
        }
        if env_bool("DEBUG"):
            out["debug"] = debug_info
            out["debug"]["note"] = "No OpenAI key; tables were not parsed to JSON."
        return out

    model = (os.getenv("OPENAI_MODEL", "gpt-4.1-mini") or "gpt-4.1-mini").strip()
    client = OpenAI(api_key=api_key)

    payload = {
        "ocrText": sanitize_text(ocr_text),
        "tablesByEye": {
            "OD": od_tables,  # LEFT half of page
            "OS": os_tables,  # RIGHT half of page
        },
        "rule": "DO NOT swap tables. OD is LEFT half. OS is RIGHT half.",
    }

    prompt = f"""
You are a careful ophthalmology document parser.

You receive:
- OCR text
- Tables grouped by eye using page halves:
  - OD (right eye) = LEFT half of page
  - OS (left eye)  = RIGHT half of page

CRITICAL:
- Do NOT move a table from OD to OS or OS to OD.
- If you cannot read a table clearly, SKIP it (do not guess).
- If there are no tables, blocks must be [].

Return STRICT JSON only with this shape:

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
          "blockIndex": "string",
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

documentType must be one of:
- opticalBiometry
- immersionBiometry
- ecc
- autokeratometry
- phacoSummary
- clinicNote
- unknown

INPUT JSON:
{json.dumps(payload, ensure_ascii=False)}
""".strip()

    try:
        resp = client.responses.create(
            model=model,
            input=prompt,
            response_format={"type": "json_object"},
        )
        out = json.loads(resp.output_text)
    except Exception:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        out = json.loads(resp.choices[0].message.content)

    if env_bool("DEBUG"):
        out["debug"] = debug_info
        out["debug"]["odTablesSent"] = len(od_tables)
        out["debug"]["osTablesSent"] = len(os_tables)

        # Small preview so you can see if Azure even detected the tables
        def _preview(tables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            pv = []
            for t in tables[:2]:
                m = t.get("matrix") or []
                pv.append({
                    "index": t.get("index"),
                    "page": t.get("page"),
                    "side": t.get("side"),
                    "centerX": t.get("centerX"),
                    "matrixFirst2Rows": m[:2],
                })
            return pv

        out["debug"]["odPreview"] = _preview(od_tables)
        out["debug"]["osPreview"] = _preview(os_tables)

    return out


@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        if not file_bytes:
            return JSONResponse(status_code=400, content={"success": False, "error": "Empty file"})

        ocr_text, od_tables, os_tables, debug_info = extract_text_and_tables_half_split(file_bytes)

        # If Azure detected no tables, this will explain why blocks are empty.
        if not ocr_text and not od_tables and not os_tables:
            out = {
                "success": True,
                "documentType": "unknown",
                "fields": {"global": {}, "OD": {}, "OS": {}, "efx": None, "ust": None, "avg": None},
                "warnings": ["No OCR text or tables found"],
            }
            if env_bool("DEBUG"):
                out["debug"] = debug_info
            return out

        return call_openai_to_json(ocr_text, od_tables, os_tables, debug_info)

    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})
