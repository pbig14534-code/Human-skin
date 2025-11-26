"""
Bio-Vision Phase-1 — Backend API (Simplified)

FastAPI + PyTorch demo for single-image medical (skin / wound) analysis.

Endpoints:
- POST /infer   : run inference on a single uploaded image
- POST /cleanup : optionally delete temporary heatmap files
- GET  /health  : simple health check
"""

import os
import time
import uuid
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ===================== Paths & Constants =====================

APP_ROOT = Path(__file__).resolve().parent
MODEL_DIR = APP_ROOT / "models"
MODEL_PATH = MODEL_DIR / "mobilenetv3_phase1_best.pt"

STATIC_DIR = APP_ROOT / "static"
HEATMAP_DIR = STATIC_DIR / "heatmaps"
HEATMAP_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES: List[str] = [
    "01_normal",
    "02_irritation_rash",
    "03_erythema",
    "04_dry_cracks",
    "05_mild_infection",
    "06_ulcer_general",
    "07_dfu",
    "08_severe_infection_pus",
    "09_burns",
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(False)

# ===================== Model =====================


def build_model(num_classes: int) -> nn.Module:
    """Create a MobileNetV3-Large classification model with a custom head."""
    model = models.mobilenet_v3_large(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


def load_checkpoint(model: nn.Module, ckpt_path: Path) -> None:
    """Load model weights from checkpoint file if available."""
    if not ckpt_path.exists():
        print(f"[WARN] Checkpoint not found at {ckpt_path}. Using random weights.")
        return

    print(f"[INFO] Loading checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)

    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "model_state" in ckpt:
            state_dict = ckpt["model_state"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = {k: v for k, v in ckpt.items() if isinstance(v, torch.Tensor)}
    else:
        state_dict = ckpt

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WARN] Missing keys in checkpoint: {missing[:5]} ...")
    if unexpected:
        print(f"[WARN] Unexpected keys in checkpoint: {unexpected[:5]} ...")


_model = build_model(num_classes=len(CLASS_NAMES)).to(DEVICE)
load_checkpoint(_model, MODEL_PATH)
_model.eval()

_preprocess = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

# ===================== Schemas =====================


class TopKItem(BaseModel):
    label: str
    prob: float


class PredMetrics(BaseModel):
    redness: float
    cyanosis: float
    area_cm2: float


class PredResult(BaseModel):
    label: str
    prob: float
    topk: List[TopKItem]
    metrics: PredMetrics
    uncertainty: float


class ExplainResult(BaseModel):
    heatmap_uri: Optional[str] = None


class InferenceResponse(BaseModel):
    version: str
    inference_id: str
    pred: PredResult
    explain: ExplainResult
    timing_ms: float


class CleanupRequest(BaseModel):
    inference_id: str


class CleanupResponse(BaseModel):
    ok: bool


class SessionData(BaseModel):
    heatmap_path: Optional[str] = None


SESSIONS: Dict[str, SessionData] = {}

# ===================== FastAPI App =====================

app = FastAPI(
    title="Bio-Vision Phase-1 API",
    description="Single-image skin / wound analysis with MobileNetV3 (demo).",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ===================== Helpers =====================


def _check_file_type(upload: UploadFile) -> None:
    if upload.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload a PNG or JPG image.",
        )


def _load_image_bytes(upload: UploadFile) -> Image.Image:
    data = upload.file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Received empty file.")
    try:
        img = Image.open(BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read image: {exc}")
    return img


def _simple_color_metrics(img: Image.Image) -> PredMetrics:
    arr = np.array(img.resize((256, 256)), dtype=np.float32)
    r = arr[..., 0]
    g = arr[..., 1]
    b = arr[..., 2]
    denom = r + g + b + 1e-6

    redness = float(np.clip(np.mean(r / denom), 0.0, 1.0))
    cyanosis = float(np.clip(np.mean(b / denom), 0.0, 1.0))

    gray = (r + g + b) / 3.0
    non_bright_fraction = float(np.mean(gray < 220.0))
    area_cm2 = float(5.0 * non_bright_fraction)

    return PredMetrics(
        redness=round(redness, 3),
        cyanosis=round(cyanosis, 3),
        area_cm2=round(area_cm2, 3),
    )


def _make_fake_heatmap(img: Image.Image, inference_id: str) -> str:
    img = img.convert("RGBA").resize((512, 512))
    red_overlay = Image.new("RGBA", img.size, (255, 0, 0, 120))
    heatmap_img = Image.alpha_composite(img, red_overlay)

    heatmap_path = HEATMAP_DIR / f"{inference_id}_heatmap.png"
    heatmap_img.save(heatmap_path)
    return f"/static/heatmaps/{heatmap_path.name}"


# ===================== Endpoints =====================


@app.post("/infer", response_model=InferenceResponse)
async def infer(image: UploadFile = File(...)) -> JSONResponse:
    start_time = time.perf_counter()

    _check_file_type(image)
    img = _load_image_bytes(image)

    with torch.no_grad():
        tensor = _preprocess(img).unsqueeze(0).to(DEVICE)
        logits = _model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    topk_indices = probs.argsort()[::-1][:3]
    topk_items = [
        TopKItem(label=CLASS_NAMES[idx], prob=float(probs[idx]))
        for idx in topk_indices
    ]

    best_idx = int(topk_indices[0])
    best_label = CLASS_NAMES[best_idx]
    best_prob = float(probs[best_idx])

    uncertainty = float(np.clip(1.0 - best_prob, 0.0, 1.0))
    metrics = _simple_color_metrics(img)

    inference_id = str(uuid.uuid4())
    heatmap_uri = _make_fake_heatmap(img, inference_id)

    SESSIONS[inference_id] = SessionData(heatmap_path=heatmap_uri)

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    response = InferenceResponse(
        version="1.0",
        inference_id=inference_id,
        pred=PredResult(
            label=best_label,
            prob=best_prob,
            topk=topk_items,
            metrics=metrics,
            uncertainty=uncertainty,
        ),
        explain=ExplainResult(heatmap_uri=heatmap_uri),
        timing_ms=elapsed_ms,
    )
    return JSONResponse(content=response.dict())


@app.post("/cleanup", response_model=CleanupResponse)
async def cleanup(req: CleanupRequest) -> CleanupResponse:
    data = SESSIONS.pop(req.inference_id, None)
    if data and data.heatmap_path:
        rel = data.heatmap_path.replace("/static/", "")
        local_path = STATIC_DIR / rel
        if local_path.exists():
            try:
                local_path.unlink()
            except Exception as exc:
                print(f"[WARN] Failed to delete file {local_path}: {exc}")

    return CleanupResponse(ok=True)


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok", "device": DEVICE}


# ===================== Local Run (for Render) =====================

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
