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


# ----------------------------
# Helpers
# ----------------------------

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


def to_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        s = s.replace(",", "")
        m = re.search(r"[-+]?\d+(\.\d+)?", s)
        if not m:
            return None
        try:
            return float(m.group(0))
        except Exception:
            return None
    return None


def clean_str(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, str):
        s = sanitize_text(v)
        return s if s else None
    return None


# ----------------------------
# Canonical schema shaping
# ----------------------------

CANON_GLOBAL_KEYS = ["patientName", "hospitalNumber", "examDate", "scanDate", "biometryMethod", "pd"]

CANON_EYE_KEYS = [
    "AL", "K1", "K2", "ACD", "LT", "WTW", "CCT",
    "ecc", "cv", "hex",
    "avgCellSize", "maxCellSize", "minCellSize",
    "sd", "numCells", "pachy",
    "blocks",
]

def empty_eye() -> Dict[str, Any]:
    return {
        "AL": None, "K1": None, "K2": None, "ACD": None,
        "LT": None, "WTW": None, "CCT": None,
        "ecc": None, "cv": None, "hex": None,
        "avgCellSize": None, "maxCellSize": None, "minCellSize": None,
        "sd": None, "numCells": None, "pachy": None,
        "blocks": [],
    }


def coerce_global(g: Any) -> Dict[str, Any]:
    g = g if isinstance(g, dict) else {}
    out: Dict[str, Any] = {}
    for k in CANON_GLOBAL_KEYS:
        if k == "pd":
            out[k] = to_float(g.get(k))
        else:
            out[k] = clean_str(g.get(k))
    return out


def normalize_rows(rows_any: Any) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not isinstance(rows_any, list):
        return rows

    for r in rows_any:
        if not isinstance(r, dict):
            continue

        iol_candidates = [
            r.get("iolPower"), r.get("IOL"), r.get("IOLPower"), r.get("IOL(D)"),
            r.get("IOL_D"), r.get("IOLd"), r.get("IOLpower")
        ]
        ref_candidates = [
            r.get("targetRefraction"), r.get("REF"), r.get("Ref"), r.get("REF(D)"),
            r.get("REF_D"), r.get("Target"), r.get("target")
        ]

        iol = None
        for c in iol_candidates:
            iol = to_float(c)
            if iol is not None:
                break

        ref = None
        for c in ref_candidates:
            ref = to_float(c)
            if ref is not None:
                break

        if iol is None and ref is None:
            continue

        rows.append({"iolPower": iol, "targetRefraction": ref})

    seen: set[Tuple[Optional[float], Optional[float]]] = set()
    deduped: List[Dict[str, Any]] = []
    for r in rows:
        key = (
            round(r["iolPower"], 3) if r["iolPower"] is not None else None,
            round(r["targetRefraction"], 3) if r["targetRefraction"] is not None else None,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)

    def sort_key(x: Dict[str, Any]) -> Tuple[int, float]:
        if x["iolPower"] is None:
            return (1, 0.0)
        return (0, -x["iolPower"])

    deduped.sort(key=sort_key)
    return deduped


def normalize_blocks(blocks_any: Any) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    if not isinstance(blocks_any, list):
        return blocks

    for b in blocks_any:
        if not isinstance(b, dict):
            continue

        block_index = clean_str(b.get("blockIndex")) or clean_str(b.get("IOLModel")) or clean_str(b.get("lensModel"))

        aconst = b.get("AConstant")
        if aconst is None:
            aconst = b.get("Aconst") or b.get("aconst")
        aconst_str = None
        if aconst is not None:
            fv = to_float(aconst)
            if fv is not None:
                # keep 2 decimals if it looks like A-const from IOLMaster
                aconst_str = f"{fv:.2f}"
            else:
                aconst_str = clean_str(aconst)

        formula = clean_str(b.get("Formula")) or clean_str(b.get("formula"))

        rows_any = b.get("rows") or b.get("IOLrefs") or b.get("table")
        rows = normalize_rows(rows_any)

        blocks.append({
            "blockIndex": block_index,
            "AConstant": aconst_str,
            "Formula": formula,
            "rows": rows,
        })

    return blocks


def coerce_eye(e: Any) -> Dict[str, Any]:
    e = e if isinstance(e, dict) else {}
    out = empty_eye()

    for k in CANON_EYE_KEYS:
        if k == "blocks":
            out["blocks"] = normalize_blocks(e.get("blocks"))
        else:
            out[k] = to_float(e.get(k))

    if not out["blocks"]:
        out["blocks"] = normalize_blocks(e.get("Blocks") or e.get("iolBlocks") or e.get("IOLBlocks"))

    return out


def enforce_schema(parsed: Any) -> Dict[str, Any]:
    if not isinstance(parsed, dict):
        parsed = {}

    document_type = parsed.get("documentType")
    if not isinstance(document_type, str) or not document_type:
        document_type = "unknown"

    fields_any = parsed.get("fields") if isinstance(parsed.get("fields"), dict) else {}

    global_obj = coerce_global(fields_any.get("global"))
    od_obj = coerce_eye(fields_any.get("OD"))
    os_obj = coerce_eye(fields_any.get("OS"))

    efx = clean_str(fields_any.get("efx"))
    ust = clean_str(fields_any.get("ust"))
    avg = clean_str(fields_any.get("avg"))

    warnings: List[str] = []
    if isinstance(parsed.get("warnings"), list):
        for w in parsed["warnings"]:
            if isinstance(w, str) and w.strip():
                warnings.append(sanitize_text(w))

    return {
        "success": True,
        "documentType": document_type,
        "fields": {
            "global": global_obj,
            "OD": od_obj,
            "OS": os_obj,
            "efx": efx,
            "ust": ust,
            "avg": avg,
        },
        "warnings": warnings,
    }


# ----------------------------
# OCR with TABLES (critical fix)
# ----------------------------

def extract_tables_text(result: Any) -> str:
    """
    Turn DI tables into a readable grid string so the LLM can separate the 2x2 blocks.
    """
    tables = getattr(result, "tables", None)
    if not tables:
        return ""

    out_lines: List[str] = []
    for ti, t in enumerate(tables, start=1):
        row_count = getattr(t, "row_count", 0) or 0
        col_count = getattr(t, "column_count", 0) or 0
        cells = getattr(t, "cells", None) or []

        # Build empty grid
        grid: List[List[str]] = [["" for _ in range(col_count)] for _ in range(row_count)]

        for c in cells:
            r = getattr(c, "row_index", None)
            cc = getattr(c, "column_index", None)
            txt = getattr(c, "content", "") or ""
            if r is None or cc is None:
                continue
            if 0 <= r < row_count and 0 <= cc < col_count:
                grid[r][cc] = sanitize_text(txt)

        out_lines.append(f"[TABLE {ti}] rows={row_count} cols={col_count}")
        for r in range(row_count):
            # join with a strong delimiter so the model “sees” columns
            row_txt = " | ".join((grid[r][c] or "").strip() for c in range(col_count))
            out_lines.append(row_txt)

        out_lines.append("")  # spacer

    return sanitize_text("\n".join(out_lines))


def ocr_with_azure_di_layout(file_bytes: bytes) -> Dict[str, str]:
    endpoint = normalize_endpoint(must_env("AZURE_DI_ENDPOINT"))
    key = normalize_key(must_env("AZURE_DI_KEY"))

    client = DocumentAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(key))

    # KEY CHANGE: prebuilt-layout gives tables
    poller = client.begin_analyze_document(model_id="prebuilt-layout", document=file_bytes)
    result = poller.result()

    # Plain text lines
    lines: List[str] = []
    if getattr(result, "pages", None):
        for page in result.pages:
            if getattr(page, "lines", None):
                for line in page.lines:
                    if getattr(line, "content", None):
                        lines.append(line.content)

    plain_text = sanitize_text("\n".join(lines))
    tables_text = extract_tables_text(result)

    return {"plainText": plain_text, "tablesText": tables_text}


# ----------------------------
# OpenAI parse
# ----------------------------

@app.get("/")
def root():
    return {"status": "server running"}


def call_openai_to_json(plain_text: str, tables_text: str) -> Dict[str, Any]:
    api_key = normalize_key(os.getenv("OPENAI_API_KEY", ""))

    if not api_key or OpenAI is None:
        return {
            "success": True,
            "documentType": "unknown",
            "fields": {
                "global": {k: None for k in CANON_GLOBAL_KEYS},
                "OD": empty_eye(),
                "OS": empty_eye(),
                "efx": None,
                "ust": None,
                "avg": None,
            },
            "warnings": ["OPENAI_API_KEY not set (OCR ran but AI parse skipped)"],
        }

    model = (os.getenv("OPENAI_MODEL", "gpt-4.1-mini") or "gpt-4.1-mini").strip()
    client = OpenAI(api_key=api_key)

    plain_text = sanitize_text(plain_text)
    tables_text = sanitize_text(tables_text)

    prompt = f"""
You are a STRICT JSON generator for ophthalmology documents.

You MUST output ONLY a JSON object. No extra text.

Pick documentType EXACTLY as one of:
- opticalBiometry
- immersionBiometry
- ecc
- autokeratometry
- phacoSummary
- clinicNote
- unknown

IMPORTANT FOR IOLMASTER SHEETS:
- There are OFTEN 4 IOL tables for OD and 4 IOL tables for OS (8 total).
- Each mini-table has an IOL model name + "A const:" + a 2-column table "IOL (D)" and "REF (D)".
- Use blockIndex = the IOL model name (e.g., "Alcon SA60WF", "ALCON MA60AC", "RX Rohto RE06F", "JOHNSON OPTIBLUE (ZCB00V)").
- If you can see 4 tables per eye, output 4 blocks per eye. Do NOT merge different IOL models into one block.

Schema (use EXACT keys, do not add new keys):

{{
  "success": true,
  "documentType": "one_of_the_values_above",
  "fields": {{
    "global": {{
      "patientName": null,
      "hospitalNumber": null,
      "examDate": null,
      "scanDate": null,
      "biometryMethod": null,
      "pd": null
    }},
    "OD": {{
      "AL": null, "K1": null, "K2": null, "ACD": null,
      "LT": null, "WTW": null, "CCT": null,
      "ecc": null, "cv": null, "hex": null,
      "avgCellSize": null, "maxCellSize": null, "minCellSize": null,
      "sd": null, "numCells": null, "pachy": null,
      "blocks": [
        {{
          "blockIndex": null,
          "AConstant": null,
          "Formula": null,
          "rows": [
            {{ "iolPower": null, "targetRefraction": null }}
          ]
        }}
      ]
    }},
    "OS": {{
      "AL": null, "K1": null, "K2": null, "ACD": null,
      "LT": null, "WTW": null, "CCT": null,
      "ecc": null, "cv": null, "hex": null,
      "avgCellSize": null, "maxCellSize": null, "minCellSize": null,
      "sd": null, "numCells": null, "pachy": null,
      "blocks": [
        {{
          "blockIndex": null,
          "AConstant": null,
          "Formula": null,
          "rows": [
            {{ "iolPower": null, "targetRefraction": null }}
          ]
        }}
      ]
    }},
    "efx": null,
    "ust": null,
    "avg": null
  }},
  "warnings": []
}}

Rules:
- Numbers must be real numbers (no commas).
- If unsure, use null. Do NOT guess.
- Use TABLES below to separate each IOL model correctly.

PLAIN OCR TEXT:
\"\"\"
{plain_text}
\"\"\"

TABLES (from Azure layout):
\"\"\"
{tables_text}
\"\"\"
""".strip()

    parsed_any: Any

    try:
        resp = client.responses.create(
            model=model,
            input=prompt,
            response_format={"type": "json_object"},
        )
        parsed_any = json.loads(resp.output_text)
    except Exception:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        parsed_any = json.loads(resp.choices[0].message.content)

    return enforce_schema(parsed_any)


@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        if not file_bytes:
            return JSONResponse(status_code=400, content={"success": False, "error": "Empty file"})

        ocr = ocr_with_azure_di_layout(file_bytes)
        plain_text = ocr.get("plainText", "")
        tables_text = ocr.get("tablesText", "")

        if not plain_text and not tables_text:
            return {
                "success": True,
                "documentType": "unknown",
                "fields": {
                    "global": {k: None for k in CANON_GLOBAL_KEYS},
                    "OD": empty_eye(),
                    "OS": empty_eye(),
                    "efx": None,
                    "ust": None,
                    "avg": None,
                },
                "warnings": ["No OCR text found"],
            }

        return call_openai_to_json(plain_text, tables_text)

    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})
