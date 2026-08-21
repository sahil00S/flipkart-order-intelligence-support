"""
Thin FastAPI bridge for the Flipkart Order Intelligence frontend.

This file does NOT duplicate any ML/RAG/LangGraph logic.
It imports and calls the existing implementations directly.
"""

import json
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

# ============================================================
# Existing backend imports (DO NOT duplicate their logic)
# ============================================================

from part3.tools.order_risk_tool import check_return_risk
from part3.tools.image_classifier_tool import classify_product_image
from part3.langgraph_assistant import run_assistant


# ============================================================
# FastAPI app
# ============================================================

app = FastAPI(
    title="Flipkart Order Intelligence API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Request / Response models
# ============================================================


class PolicyRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None


class ReturnRiskRequest(BaseModel):
    product_category: str
    price_inr: float
    discount_pct: float
    payment_method: str
    customer_tenure_days: int
    num_previous_orders: int
    num_previous_returns: int
    delivery_distance_km: float
    delivery_days: int
    is_weekend_order: int
    rating_given: Optional[float] = None


# ============================================================
# In-memory conversation store
# ============================================================

conversations: dict[str, list] = {}


# ============================================================
# Endpoints
# ============================================================


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "flipkart-order-intelligence"}


@app.post("/api/policy")
def policy(request: PolicyRequest):
    conversation_id = request.conversation_id or str(uuid.uuid4())
    previous_messages = conversations.get(conversation_id, [])

    try:
        response, updated_messages = run_assistant(
            user_input=request.message,
            conversation_id=conversation_id,
            previous_messages=previous_messages,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    conversations[conversation_id] = updated_messages

    return {
        "response": response,
        "conversation_id": conversation_id,
    }


@app.post("/api/return-risk")
def return_risk(request: ReturnRiskRequest):
    order_features = {
        "product_category": request.product_category,
        "price_inr": request.price_inr,
        "discount_pct": request.discount_pct,
        "payment_method": request.payment_method,
        "customer_tenure_days": request.customer_tenure_days,
        "num_previous_orders": request.num_previous_orders,
        "num_previous_returns": request.num_previous_returns,
        "delivery_distance_km": request.delivery_distance_km,
        "delivery_days": request.delivery_days,
        "is_weekend_order": request.is_weekend_order,
        "rating_given": request.rating_given,
    }

    try:
        result = check_return_risk(order_features)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return result


SAMPLE_IMAGES_DIR = Path(__file__).parent / "data" / "sample_images"
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


@app.get("/api/sample-images")
def list_sample_images():
    """Return the list of available sample image filenames."""
    if not SAMPLE_IMAGES_DIR.exists():
        return {"images": []}
    files = sorted(f.name for f in SAMPLE_IMAGES_DIR.iterdir() if f.suffix.lower() in ALLOWED_EXTENSIONS)
    return {"images": files}


@app.get("/api/sample-images/{filename}")
def get_sample_image(filename: str):
    """Serve a sample image by filename."""
    if "/" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported file type")
    path = SAMPLE_IMAGES_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Sample image not found")
    return FileResponse(path, media_type="image/png", filename=filename)


@app.post("/api/classify-image")
async def classify_image(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: PNG, JPG, JPEG",
        )

    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large (max 10 MB)")

    with tempfile.NamedTemporaryFile(
        suffix=ext, delete=False
    ) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        result = classify_product_image(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return result


# ============================================================
# Serve frontend static files (production)
# ============================================================

DIST_DIR = Path(__file__).parent / "frontend" / "dist"

if DIST_DIR.exists():
    app.mount(
        "/",
        StaticFiles(directory=str(DIST_DIR), html=True),
        name="frontend",
    )
