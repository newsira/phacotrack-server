# server.py
import os
import json
import re
import unicodedata
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient

# OpenAI (only used if OPENAI_API_KEY is set)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None


app = FastAPI()


# -----------------------------
# Small helpers (safe + simple)
# -----------------------------
def must_env(name: str) -> str:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        raise RuntimeError(f"Missing environment variable: {name}")
    return v.strip()


def normalize_endpoint(raw: str) -> str:
    v = (raw or "").strip()
    if v.endswith("/"):
        v = v[:-1]
    return v


def normalize_key(raw: str) -> str:
    """
    Remove ALL whitespace (including unicode newlines like \u2028) from secrets.
    Prevents weird header encoding crashes.
    """
    if not raw:
        return ""
    raw = unicodedata.normalize("NFC", raw)
    return re.sub(r"\s+", "", raw).strip()


def sanitize_text(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFC", s)
    # Normalize odd unicode line separators
    s = s.replace("\u2028", "\n").replace("\u2029", "\n")
    # Remove control chars
    s = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", " ", s)
    s = s.replace("\ufeff", "").replace("\u200b", "")
    # Compact whitespace
    s = re.sub(r"[ \t\r\f\v]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip()
        if s == "":
            return None
        s = s.replace(",", "")
        return float(s)
    except Exception:
        return None


# -----------------------------
# Health
# -----------------------------
@app.get("/")
def root():
    return {"status": "server running"}


# -----------------------------
# Azure DI: PREBUILT-LAYOUT
# Returns: full text + tables/cells
# -----------------------------
def layout_with_azure_di(file_bytes: bytes) -> Dict[str, Any]:
    endpoint = normalize_endpoint(must_env("AZURE_DI_ENDPOINT"))
    key = normalize_key(must_env("AZURE_DI_KEY"))

    client = DocumentAnalysisClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(key),
    )

    # IMPORTANT:
    # - Use model_id="prebuilt-layout"
    # - Pass document=... (NOT body=...)
    # - Do NOT pass content_type here (avoids "multiple values for content_type")
    poller = client.begin_analyze_document(
        model_id="prebuilt-layout",
        document=file_bytes,
    )
    result = poller.result()

    # Collect lines (reading order)
    lines: List[str] = []
    if getattr(result, "pages", None):
        for page in result.pages:
            if getattr(page, "lines", None):
                for line in page.lines:
                    if getattr(line, "content", None):
                        lines.append(line.content)

    full_text = sanitize_text("\n".join(lines))

    # Collect tables/cells
    tables_out: List[Dict[str, Any]] = []
    if getattr(result, "tables", None):
        for t_index, table in enumerate(result.tables):
            t = {
                "tableIndex": t_index,
                "rowCount": int(getattr(table, "row_count", 0) or 0),
                "columnCount": int(getattr(table, "column_count", 0) or 0),
                "cells": [],
            }

            # Cells have row_index, column_index, content
            for cell in getattr(table, "cells", []) or []:
                t["cells"].append(
                    {
                        "rowIndex": int(getattr(cell, "row_index", 0) or 0),
                        "columnIndex": int(getattr(cell, "column_index", 0) or 0),
                        "content": sanitize_text(getattr(cell, "content", "") or ""),
                    }
                )

            tables_out.append(t)

    return {
        "fullText": full_text,
        "tables": tables_out,
    }


# -----------------------------
# OpenAI: use TABLES (not flat OCR)
# -----------------------------
def call_openai_to_json(layout_payload: Dict[str, Any]) -> Dict[str, Any]:
    api_key = normalize_key(os.getenv("OPENAI_API_KEY", ""))
    if not api_key or OpenAI is None:
        # No OpenAI available: return what we have (so you can debug tables)
        return {
            "success": True,
            "documentType": "unknown",
            "fields": {"global": {}, "OD": {}, "OS": {}, "efx": None, "ust": None, "avg": None},
            "warnings": ["OPENAI_API_KEY not set (returning Azure layout payload only)"],
            "azureLayout": layout_payload,
        }

    model = (os.getenv("OPENAI_MODEL", "gpt-4.1-mini") or "gpt-4.1-mini").strip()
    client = OpenAI(api_key=api_key)

    # Keep payload small but structured
    payload_for_ai = {
        "fullText": layout_payload.get("fullText", ""),
        "tables": layout_payload.get("tables", []),
    }

    # Clear, strict rules: OpenAI should NOT guess.
    # It must only use evidence from tables/text.
    prompt = f"""
You convert ophthalmology document content into STRICT JSON for an app.

Input contains:
1) fullText (all text lines)
2) tables: array of tables, each with rowCount/columnCount and cells (rowIndex, columnIndex, content)

IMPORTANT:
- DO NOT guess.
- If a value is not clearly present, set it to null.
- For biometry sheets, you MUST extract ALL IOL tables you can see.
- For ZEISS/IOLMaster Advanced sheets: there are usually 4 tables per eye (8 total). Do not stop early if more tables exist.
- Only place an IOL table into OD or OS when it is clearly labeled or grouped near that eye in the content. If you cannot be sure, put a warning and still include the table under a special group "unassignedBlocks" (see schema below).

Allowed documentType (choose ONE):
- opticalBiometry
- immersionBiometry
- ecc
- autokeratometry
- phacoSummary
- clinicNote
- unknown

OUTPUT: return ONLY JSON with this exact shape:

{{
  "success": true,
  "documentType": "opticalBiometry|immersionBiometry|ecc|autokeratometry|phacoSummary|clinicNote|unknown",
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
          "blockIndex": string,              // lens model name OR "1"/"2"/"3"/"4"
          "AConstant": string|null,
          "Formula": string|null,
          "rows": [{{"iolPower": number, "targetRefraction": number}}]
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
          "blockIndex": string,
          "AConstant": string|null,
          "Formula": string|null,
          "rows": [{{"iolPower": number, "targetRefraction": number}}]
        }}
      ]
    }},
    "efx": null|string,
    "ust": null|string,
    "avg": null|string,
    "unassignedBlocks": [
      {{
        "blockIndex": string,
        "AConstant": string|null,
        "Formula": string|null,
        "rows": [{{"iolPower": number, "targetRefraction": number}}],
        "reason": string
      }}
    ]
  }},
  "warnings": []
}}

Parsing rules for TABLES:
- A table row is valid only if it contains BOTH an IOL power and a target refraction number.
- Ignore headers.
- Convert numbers correctly (e.g., "19.5" -> 19.5).
- If a table contains duplicated rows, keep them only once.
- If you see labels like lens model (e.g. "Alcon SA60WF") or A-constant near the table, attach them to that block.
- If you cannot confidently assign to OD or OS, put it in unassignedBlocks with a short reason.

INPUT:
{json.dumps(payload_for_ai, ensure_ascii=False)}
""".strip()

    try:
        resp = client.responses.create(
            model=model,
            input=prompt,
            response_format={"type": "json_object"},
        )
        out = json.loads(resp.output_text)
    except Exception:
        # Fallback to chat.completions if needed
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        out = json.loads(resp.choices[0].message.content)

    # Minimal safety normalization (keep app stable)
    if not isinstance(out, dict):
        return {"success": False, "error": "OpenAI returned non-object JSON"}

    if out.get("success") is not True:
        out["success"] = True  # keep shape consistent; warnings will explain
        out.setdefault("warnings", []).append("OpenAI did not set success=true; forced by server")

    out.setdefault("documentType", "unknown")
    out.setdefault("fields", {})
    out["fields"].setdefault("global", {})
    out["fields"].setdefault("OD", {})
    out["fields"].setdefault("OS", {})
    out["fields"].setdefault("efx", None)
    out["fields"].setdefault("ust", None)
    out["fields"].setdefault("avg", None)
    out["fields"].setdefault("unassignedBlocks", [])
    out.setdefault("warnings", [])

    return out


# -----------------------------
# Main endpoint
# -----------------------------
@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        if not file_bytes:
            return JSONResponse(status_code=400, content={"success": False, "error": "Empty file"})

        layout = layout_with_azure_di(file_bytes)

        # If Azure returned no text and no tables, stop early
        has_text = bool((layout.get("fullText") or "").strip())
        has_tables = bool(layout.get("tables") or [])
        if not has_text and not has_tables:
            return {
                "success": True,
                "documentType": "unknown",
                "fields": {"global": {}, "OD": {}, "OS": {}, "efx": None, "ust": None, "avg": None, "unassignedBlocks": []},
                "warnings": ["Azure returned no text and no tables"],
            }

        return call_openai_to_json(layout)

    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})
