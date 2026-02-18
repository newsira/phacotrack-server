import os
import json
import re
from typing import Any, Dict, Optional

import requests
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient


app = FastAPI()


def _env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing environment variable: {name}")
    return v


def _make_di_client() -> DocumentIntelligenceClient:
    endpoint = _env("AZURE_DI_ENDPOINT")
    key = _env("AZURE_DI_KEY")
    return DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))


def _ocr_with_azure(image_bytes: bytes) -> str:
    """
    OCR using Azure Document Intelligence "prebuilt-read".
    Returns plain text.
    """
    client = _make_di_client()
    poller = client.begin_analyze_document(
        model_id="prebuilt-read",
        analyze_request=image_bytes,
        content_type="application/octet-stream",
    )
    result = poller.result()

    # Safest: use full content if present
    text = getattr(result, "content", None)
    if text:
        return text

    # Fallback: try to reconstruct from pages/lines
    parts = []
    if getattr(result, "pages", None):
        for p in result.pages:
            if getattr(p, "lines", None):
                for line in p.lines:
                    if getattr(line, "content", None):
                        parts.append(line.content)
    return "\n".join(parts).strip()


def _call_openai_json(text: str) -> Dict[str, Any]:
    """
    Sends OCR text to OpenAI and expects STRICT JSON back.
    """
    api_key = _env("OPENAI_API_KEY")

    # Keep prompt simple + strict
    system = (
        "You extract ophthalmology document data and output STRICT JSON only. "
        "No explanations, no markdown."
    )

    schema_hint = {
        "success": True,
        "documentType": "opticalBiometry|immersionBiometry|ecc|autokeratometry|phacoSummary|clinicNote|unknown",
        "fields": {
            "global": {
                "patientName": None,
                "hospitalNumber": None,
                "scanDate": None,
                "biometryMethod": None,
                "examDate": None,
                "pd": None
            },
            "OD": {
                "AL": None, "K1": None, "K2": None, "ACD": None, "LT": None, "WTW": None, "CCT": None,
                "ecc": None, "cv": None, "hex": None, "avgCellSize": None, "maxCellSize": None,
                "minCellSize": None, "sd": None, "numCells": None, "pachy": None,
                "blocks": []
            },
            "OS": {
                "AL": None, "K1": None, "K2": None, "ACD": None, "LT": None, "WTW": None, "CCT": None,
                "ecc": None, "cv": None, "hex": None, "avgCellSize": None, "maxCellSize": None,
                "minCellSize": None, "sd": None, "numCells": None, "pachy": None,
                "blocks": []
            },
            # phacoSummary extra keys (optional)
            "efx": None,
            "ust": None,
            "avg": None
        },
        "warnings": []
    }

    user = (
        "Return JSON in exactly this shape (same keys). Use null if missing.\n\n"
        f"JSON SHAPE EXAMPLE:\n{json.dumps(schema_hint, ensure_ascii=False)}\n\n"
        "OCR TEXT:\n"
        f"{text}"
    )

    # OpenAI Responses API call
    url = "https://api.openai.com/v1/responses"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "gpt-5",
        "input": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        # Keep it bounded
        "max_output_tokens": 1200,
    }

    r = requests.post(url, headers=headers, json=payload, timeout=60)
    if r.status_code >= 400:
        raise RuntimeError(f"OpenAI error {r.status_code}: {r.text}")

    data = r.json()

    # Responses API can return text in different places; this is a robust grab:
    output_text = ""
    for item in data.get("output", []):
        for c in item.get("content", []):
            if c.get("type") == "output_text" and c.get("text"):
                output_text += c["text"]

    output_text = output_text.strip()

    # Sometimes models add stray whitespace/newlines; enforce JSON by extracting first {...} block
    if not output_text.startswith("{"):
        m = re.search(r"\{.*\}", output_text, flags=re.DOTALL)
        if m:
            output_text = m.group(0).strip()

    try:
        return json.loads(output_text)
    except Exception:
        # Return a helpful debug payload (still JSON)
        return {
            "success": False,
            "documentType": "unknown",
            "fields": schema_hint["fields"],
            "warnings": ["Model did not return valid JSON."],
            "debug": {
                "model_output": output_text[:4000]
            }
        }


@app.get("/")
def root():
    return {"status": "server running"}


@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty file.")

        ocr_text = _ocr_with_azure(image_bytes)

        if not ocr_text.strip():
            return JSONResponse(
                status_code=200,
                content={
                    "success": False,
                    "documentType": "unknown",
                    "fields": {
                        "global": {"patientName": None, "hospitalNumber": None, "scanDate": None, "biometryMethod": None, "examDate": None, "pd": None},
                        "OD": {"AL": None, "K1": None, "K2": None, "ACD": None, "LT": None, "WTW": None, "CCT": None, "ecc": None, "cv": None, "hex": None, "avgCellSize": None, "maxCellSize": None, "minCellSize": None, "sd": None, "numCells": None, "pachy": None, "blocks": []},
                        "OS": {"AL": None, "K1": None, "K2": None, "ACD": None, "LT": None, "WTW": None, "CCT": None, "ecc": None, "cv": None, "hex": None, "avgCellSize": None, "maxCellSize": None, "minCellSize": None, "sd": None, "numCells": None, "pachy": None, "blocks": []},
                    },
                    "warnings": ["OCR returned no text."]
                }
            )

        parsed = _call_openai_json(ocr_text)

        # Always include OCR text (helps you debug in Postman)
        if isinstance(parsed, dict):
            parsed.setdefault("debug", {})
            parsed["debug"]["ocr_text_preview"] = ocr_text[:2000]

        return JSONResponse(status_code=200, content=parsed)

    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )
