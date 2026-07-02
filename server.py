"""
FastAPI wrapper around SGLang Unlimited-OCR for PDF parsing.

Accepts a PDF file, converts each page to PNG, then calls the local
SGLang model with multi-image inference to produce markdown output.

Start SGLang first:
    python start_sglang.py --gpu 0

Then run this service:
    python server.py
"""

import base64
import json
import os

import fitz  # PyMuPDF
import requests
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse

app = FastAPI(title="Unlimited-OCR PDF Parser")

SGLANG_BASE_URL = os.getenv("SGLANG_BASE_URL", "http://localhost:10000")
MODEL_NAME = "Unlimited-OCR"
PDF_DPI = 300
NGRAM_SIZE = 35
NGRAM_WINDOW_SINGLE = 128
NGRAM_WINDOW_MULTI = 1024
REQUEST_TIMEOUT = 1200
NO_REPEAT_NGRAM_PROCESSOR_STR = None


def get_ngram_processor_str():
    global NO_REPEAT_NGRAM_PROCESSOR_STR
    if NO_REPEAT_NGRAM_PROCESSOR_STR is None:
        from sglang.srt.sampling.custom_logit_processor import (
            DeepseekOCRNoRepeatNGramLogitProcessor,
        )
        NO_REPEAT_NGRAM_PROCESSOR_STR = DeepseekOCRNoRepeatNGramLogitProcessor.to_str()
    return NO_REPEAT_NGRAM_PROCESSOR_STR


def pdf_to_pngs(pdf_bytes: bytes, dpi: int = PDF_DPI) -> list[bytes]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pages = []
    for page in doc:
        pix = page.get_pixmap(matrix=mat)
        pages.append(pix.tobytes("png"))
    doc.close()
    return pages


def encode_png_bytes(png_data: bytes) -> dict:
    b64 = base64.b64encode(png_data).decode("utf-8")
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}


def build_payload(
    page_images: list[bytes],
    prompt: str = "document parsing.",
    temperature: float = 0,
) -> dict:
    is_multi = len(page_images) > 1
    if is_multi:
        prompt = "Multi page parsing."

    content = [{"type": "text", "text": prompt}]
    content.extend(encode_png_bytes(img) for img in page_images)

    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": content}],
        "temperature": temperature,
        "max_tokens": 32768,
        "skip_special_tokens": False,
        "stream": True,
    }
    if NGRAM_SIZE > 0:
        payload["custom_logit_processor"] = get_ngram_processor_str()
        payload["custom_params"] = {
            "ngram_size": NGRAM_SIZE,
            "window_size": NGRAM_WINDOW_MULTI if is_multi else NGRAM_WINDOW_SINGLE,
        }
    return payload


def stream_from_sglang(payload: dict):
    """Yield text chunks from SGLang streaming response."""
    resp = requests.post(
        f"{SGLANG_BASE_URL}/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=REQUEST_TIMEOUT,
        stream=True,
    )
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    for raw_line in resp.iter_lines():
        if not raw_line:
            continue
        line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
        if not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
            delta = chunk["choices"][0]["delta"].get("content", "")
        except (json.JSONDecodeError, KeyError):
            continue
        if delta:
            yield delta


def collect_full_response(payload: dict) -> str:
    return "".join(stream_from_sglang(payload))


@app.get("/health")
def health():
    try:
        r = requests.get(f"{SGLANG_BASE_URL}/health", timeout=5)
        return {"status": "ok", "sglang": r.status_code == 200}
    except Exception:
        return {"status": "ok", "sglang": False}


@app.post("/parse-pdf")
async def parse_pdf(
    file: UploadFile = File(...),
    dpi: int = Query(default=300, ge=72, le=600),
    stream: bool = Query(default=False),
):
    """
    Upload a PDF file and get back the OCR-parsed markdown.

    - Multi-page PDFs are sent as a single multi-image request (base mode).
    - Set stream=true to receive Server-Sent Events.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    page_images = pdf_to_pngs(pdf_bytes, dpi=dpi)
    if not page_images:
        raise HTTPException(status_code=400, detail="PDF has no pages")

    payload = build_payload(page_images)

    if stream:
        def event_stream():
            for chunk in stream_from_sglang(payload):
                yield f"data: {json.dumps({'content': chunk})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    result = collect_full_response(payload)
    return {"pages": len(page_images), "markdown": result}


@app.post("/parse-images")
async def parse_images(
    files: list[UploadFile] = File(...),
    stream: bool = Query(default=False),
):
    """
    Upload one or more image files (PNG/JPG) for OCR parsing.
    Multiple images are treated as multi-page input.
    """
    page_images = []
    for f in files:
        data = await f.read()
        if data:
            page_images.append(data)

    if not page_images:
        raise HTTPException(status_code=400, detail="No valid images provided")

    payload = build_payload(page_images)

    if stream:
        def event_stream():
            for chunk in stream_from_sglang(payload):
                yield f"data: {json.dumps({'content': chunk})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    result = collect_full_response(payload)
    return {"pages": len(page_images), "markdown": result}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
