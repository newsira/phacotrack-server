import os
import json
import re
import unicodedata
from typing import Dict, Any

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient

# OpenAI (optional – only used if OPENAI_API_KEY is set)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None


app = FastAPI()


def env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing environment variable: {name}")
    return v


@app.get("/")
def root():
    return {"status": "server running"}


def normalize_endpoint(raw: str) -> str:
    v = (raw or "").strip()
    if v.endswith("/"):
        v = v[:-1]
    return v


def sanitize_text(s: str) -> str:
    """
    Remove problematic unicode/control characters that can crash some HTTP stacks
    when they try to encode text as ASCII (e.g. U+2028 / U+2029).
    """
    if not s:
        return ""
    s = unicodedata.normalize("NFC", s)

    # Replace common “line separator” / “paragraph separator” with normal newline
    s = s.replace("\u2028", "\n").replace("\u2029", "\n")

    # Remove ASCII control chars + C1 controls
    s = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", " ", s)

    # Remove BOM/zero-width stuff
    s = s.replace("\ufeff", "").replace("\u200b", "")

    # Collapse weird whitespace
    s = re.sub(r"[ \t\r\f\v]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)

    return s.strip()


def ocr_with_azure_di(file_bytes: bytes) -> str:
    endpoint = normalize_endpoint(env("AZURE_DI_ENDPOINT"))
    key = env("AZURE_DI_KEY").strip()

    client = DocumentAnalysisClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(key),
    )

    poller = client.begin_analyze_document(
        model_id="prebuilt-read",
        document=file_bytes,
    )
    result = poller.result()

    lines: list[str] = []
    if getattr(result, "pages", None):
        for page in result.pages:
            if getattr(page, "lines", None):
                for line in page.lines:
                    if getattr(line, "content", None):
                        lines.append(line.content)

    return sanitize_text("\n".join(lines))


def call_openai_to_json(ocr_text: str) -> Dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return {
            "success": True,
            "documentType": "unknown",
            "fields": {"global": {}, "OD": {}, "OS": {}, "efx": None, "ust": None, "avg": None},
            "warnings": ["OPENAI_API_KEY not set (returning OCR only)"],
            "rawText": ocr_text,
        }

    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    client = OpenAI(api_key=api_key)

    # extra safety: sanitize again before prompting
    ocr_text = sanitize_text(ocr_text)

    prompt = f"""
You are a medical document parser for ophthalmology clinic documents.

Input is OCR text from ONE uploaded image.

Your job:
1) Decide documentType as ONE of:
   - opticalBiometry
   - immersionBiometry
   - ecc
   - autokeratometry
   - phacoSummary
   - clinicNote
   - unknown

2) Output STRICT JSON only (no extra text) with this shape:

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
    "OS": {{ "AL": null|number, "K1": null|number, "K2": null|number, "ACD": null|number,
             "LT": null|number, "WTW": null|number, "CCT": null|number,
             "ecc": null|number, "cv": null|number, "hex": null|number,
             "avgCellSize": null|number, "maxCellSize": null|number, "minCellSize": null|number,
             "sd": null|number, "numCells": null|number, "pachy": null|number,
             "blocks": [] }},
    "efx": null|string,
    "ust": null|string,
    "avg": null|string
  }},
  "warnings": []
}}

Rules:
- Use null when unknown.
- Numbers must be real numbers (no commas).
- If phacoSummary, fill efx/ust/avg in fields.
- If biometry, blocks can be included if you see IOL tables. Otherwise empty array.
- Be conservative: don't guess.

OCR TEXT:
\"\"\"
{ocr_text}
\"\"\"
""".strip()

    # Responses API (preferred)
    try:
        resp = client.responses.create(
            model=model,
            input=prompt,
            response_format={"type": "json_object"},
        )
        return json.loads(resp.output_text)
    except Exception:
        # Fallback: chat.completions
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

        ocr_text = ocr_with_azure_di(file_bytes)

        if not ocr_text:
            return {
                "success": True,
                "documentType": "unknown",
                "fields": {"global": {}, "OD": {}, "OS": {}, "efx": None, "ust": None, "avg": None},
                "warnings": ["No OCR text found"],
            }

        return call_openai_to_json(ocr_text)

    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})
