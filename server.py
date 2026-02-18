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


@app.get("/")
def root():
    return {"status": "server running"}


def _polygon_center_x(polygon: List[Dict[str, float]]) -> Optional[float]:
    if not polygon:
        return None
    xs = [p.get("x") for p in polygon if isinstance(p, dict) and p.get("x") is not None]
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
            if len(txt.strip()) > len(grid[r][k].strip()):
                grid[r][k] = txt.strip()

    out: List[List[str]] = []
    for row in grid:
        while len(row) > 0 and row[-1].strip() == "":
            row.pop()
        out.append(row)

    return out


def _compute_split_x(table_centers: List[float]) -> Optional[float]:
    """
    Robust split without relying on page.width.
    If we have multiple tables, take median and split around it.
    """
    if not table_centers:
        return None
    if len(table_centers) == 1:
        return None

    xs = sorted(table_centers)
    mid = len(xs) // 2
    if len(xs) % 2 == 1:
        median = xs[mid]
    else:
        median = (xs[mid - 1] + xs[mid]) / 2.0

    # We want a divider between left-cluster and right-cluster.
    # Use midpoint between max(left) and min(right) around the median.
    left = [x for x in xs if x <= median]
    right = [x for x in xs if x > median]

    if not right or not left:
        # fallback: just use median as divider
        return median

    return (max(left) + min(right)) / 2.0


def extract_text_and_tables_split_by_eye(file_bytes: bytes) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    endpoint = normalize_endpoint(must_env("AZURE_DI_ENDPOINT"))
    key = normalize_key(must_env("AZURE_DI_KEY"))

    client = DocumentAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(key))

    poller = client.begin_analyze_document(
        model_id="prebuilt-layout",
        document=file_bytes,
    )
    result = poller.result()

    # ---- OCR text lines (headers)
    lines_out: List[str] = []
    pages = getattr(result, "pages", None) or []
    for page in pages:
        for line in (getattr(page, "lines", None) or []):
            content = getattr(line, "content", None)
            if content:
                lines_out.append(content)
    full_text = sanitize_text("\n".join(lines_out))

    # ---- Gather tables with centerX + page number first
    raw_tables: List[Dict[str, Any]] = []
    tables = getattr(result, "tables", None) or []
    for idx, t in enumerate(tables, start=1):
        regions = getattr(t, "bounding_regions", None) or []
        if not regions:
            continue
        r0 = regions[0]
        page_num = getattr(r0, "page_number", None)  # 1-based
        polygon = getattr(r0, "polygon", None) or []
        cx = _polygon_center_x(polygon)
        if page_num is None or cx is None:
            continue

        raw_tables.append(
            {
                "index": idx,
                "page": int(page_num),
                "centerX": float(cx),
                "matrix": _table_to_matrix(t),
            }
        )

    # ---- Compute split per page using table centerXs
    page_to_centers: Dict[int, List[float]] = {}
    for t in raw_tables:
        page_to_centers.setdefault(t["page"], []).append(t["centerX"])

    page_to_split: Dict[int, Optional[float]] = {}
    for page, centers in page_to_centers.items():
        page_to_split[page] = _compute_split_x(centers)

    # ---- Assign OD/OS based on split
    od_tables: List[Dict[str, Any]] = []
    os_tables: List[Dict[str, Any]] = []

    for t in raw_tables:
        split_x = page_to_split.get(t["page"])
        if split_x is None:
            # If only 1 table on page (rare), don't guess -> treat as unassigned by putting in OD? better: keep in OD AND warn via logs.
            side = "left"
        else:
            side = "left" if t["centerX"] < split_x else "right"

        table_obj = {
            "index": t["index"],
            "page": t["page"],
            "side": side,
            "centerX": t["centerX"],
            "splitX": split_x,
            "matrix": t["matrix"],
        }

        # Helpful debug in Railway logs (safe)
        print(f"[TABLE] idx={t['index']} page={t['page']} centerX={t['centerX']:.4f} splitX={split_x} side={side}")

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
        }

    model = (os.getenv("OPENAI_MODEL", "gpt-4.1-mini") or "gpt-4.1-mini").strip()
    client = OpenAI(api_key=api_key)

    payload = {
        "ocrText": sanitize_text(ocr_text),
        "tablesByEye": {"OD": od_tables, "OS": os_tables},
        "rule": "OD tables are LEFT side. OS tables are RIGHT side. Do not swap.",
    }

    prompt = f"""
You are a careful ophthalmology document parser.

You receive:
- OCR text (headers)
- Tables grouped by eye using table coordinates:
  - OD = LEFT side
  - OS = RIGHT side

CRITICAL:
- Do NOT move a table from OD to OS or OS to OD.
- If you cannot confidently interpret a table, skip it (do not guess).

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

DocumentType must be one of:
- opticalBiometry
- immersionBiometry
- ecc
- autokeratometry
- phacoSummary
- clinicNote
- unknown

Rules:
- Use null when unknown.
- Numbers must be numbers.
- If optical/immersion biometry and you see IOL target tables, extract them into blocks.
- If you see Formula (e.g., SRK/T, SRK®/T), fill it; otherwise null.
- If phacoSummary, fill efx/ust/avg and leave blocks empty.

INPUT JSON:
{json.dumps(payload, ensure_ascii=False)}
""".strip()

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


@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        if not file_bytes:
            return JSONResponse(status_code=400, content={"success": False, "error": "Empty file"})

        ocr_text, od_tables, os_tables = extract_text_and_tables_split_by_eye(file_bytes)

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
