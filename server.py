import os
from typing import Dict, Any

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient

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


def ocr_with_azure_di(file_bytes: bytes, content_type: str) -> str:
    endpoint = env("AZURE_DI_ENDPOINT")
    key = env("AZURE_DI_KEY")

    client = DocumentIntelligenceClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(key),
    )

    # Use positional body argument to avoid SDK keyword mismatch issues.
    poller = client.begin_analyze_document(
        "prebuilt-read",
        file_bytes,
        content_type=content_type or "application/octet-stream",
    )
    result = poller.result()

    lines = []
    # Extract all text lines in reading order
    if getattr(result, "pages", None):
        for page in result.pages:
            if getattr(page, "lines", None):
                for line in page.lines:
                    content = getattr(line, "content", None)
                    if content:
                        lines.append(content)

    return "\n".join(lines).strip()


def call_openai_to_json(ocr_text: str) -> Dict[str, Any]:
    # If OpenAI is not installed or key missing, return a simple stub
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return {
            "success": True,
            "documentType": "unknown",
            "fields": {"global": {}, "OD": {}, "OS": {}},
            "warnings": ["OPENAI_API_KEY not set (returning OCR only)"],
            "rawText": ocr_text,
        }

    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")  # set in Railway Variables
    client = OpenAI(api_key=api_key)

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
    "OS": {{ same keys as OD }},
    "efx": null|string,
    "ust": null|string,
    "avg": null|string
  }},
  "warnings": []
}}

Rules:
- Use null when unknown.
- Numbers must be real numbers (no commas).
- If phacoSummary, fill efx/ust/avg in fields (global/OD/OS can stay mostly null).
- If biometry, blocks can be included if you see IOL tables. Otherwise empty array.
- Be conservative: don't guess.

OCR TEXT:
\"\"\"
{ocr_text}
\"\"\"
""".strip()

    resp = client.responses.create(
        model=model,
        input=prompt,
        response_format={"type": "json_object"},
    )

    out_text = resp.output_text

    import json
    return json.loads(out_text)


@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        if not file_bytes:
            return JSONResponse(status_code=400, content={"success": False, "error": "Empty file"})

        ocr_text = ocr_with_azure_di(file_bytes, file.content_type or "application/octet-stream")
        if not ocr_text:
            return {
                "success": True,
                "documentType": "unknown",
                "fields": {"global": {}, "OD": {}, "OS": {}},
                "warnings": ["No OCR text found"],
            }

        parsed = call_openai_to_json(ocr_text)
        return parsed

    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})
