"""FastAPI service: upload a leaf photo, get diagnosis + severity + overlay."""

from __future__ import annotations

import base64
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from phytolabs.io import decode_image_bytes
from phytolabs.logreg import LogisticRegressionSGD
from phytolabs.pipeline import predict_bgr
from phytolabs.segmentation import LeafGMM
from phytolabs.viz import overlay_lesions

DEFAULT_GMM = Path(os.environ.get("PHYTOLABS_GMM", "artifacts/gmm.joblib"))
DEFAULT_LOGREG = Path(os.environ.get("PHYTOLABS_LOGREG", "artifacts/logreg.joblib"))
DEFAULT_MAX_SIZE = int(os.environ.get("PHYTOLABS_MAX_SIZE", "512"))
DEFAULT_BAND = (
    float(os.environ.get("PHYTOLABS_BAND_LOW", "0.45")),
    float(os.environ.get("PHYTOLABS_BAND_HIGH", "0.55")),
)
CORS_ORIGINS = os.environ.get("PHYTOLABS_CORS_ORIGINS", "*").split(",")


class FeatureResponse(BaseModel):
    lesion_area_fraction: float
    blob_count: float
    blob_size_mean: float
    blob_size_std: float
    blob_size_max: float
    blob_density: float


class AnalyzeResponse(BaseModel):
    label: str = Field(description="healthy | suspicious | diseased")
    probability: float = Field(description="P(rust), 0–1")
    severity: str = Field(description="none | mild | moderate | severe")
    severity_percent: float = Field(description="Percent of leaf area with rust")
    features: FeatureResponse
    overlay_image_base64: str = Field(description="PNG lesion overlay, base64-encoded")
    message: str = Field(description="Human-readable summary for the UI")


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    gmm_path: str
    logreg_path: str


def _result_message(label: str, severity: str, severity_percent: float) -> str:
    if label == "healthy" and severity != "none":
        return (
            "Likely healthy, but minor spots detected. "
            "Consider re-scanning or a closer look."
        )
    if label == "suspicious":
        return "Borderline result — consider re-scanning or a closer look."
    if label == "diseased":
        return (
            f"Rust detected with {severity} severity "
            f"({severity_percent:.1f}% leaf area). "
            "Results are estimates based on visible lesions."
        )
    return "No significant rust detected."


def _encode_overlay_png(bgr, rust_mask: Any) -> str:
    overlay = overlay_lesions(bgr, rust_mask)
    ok, buf = cv2.imencode(".png", overlay)
    if not ok:
        raise RuntimeError("Failed to encode overlay image")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _load_models(gmm_path: Path, logreg_path: Path) -> Tuple[Optional[LeafGMM], Optional[LogisticRegressionSGD]]:
    if not gmm_path.exists() or not logreg_path.exists():
        return None, None
    return LeafGMM.load(gmm_path), LogisticRegressionSGD.load(logreg_path)


def create_app(
    gmm_path: Path = DEFAULT_GMM,
    logreg_path: Path = DEFAULT_LOGREG,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.gmm_path = gmm_path
        app.state.logreg_path = logreg_path
        app.state.leaf_gmm, app.state.model = _load_models(gmm_path, logreg_path)
        yield

    application = FastAPI(
        title="PhytoLabs API",
        description="Wheat leaf rust detection: GMM segmentation + logistic regression + severity grading.",
        version="0.1.0",
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in CORS_ORIGINS if o.strip()],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @application.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        loaded = application.state.leaf_gmm is not None and application.state.model is not None
        return HealthResponse(
            status="ok" if loaded else "degraded",
            models_loaded=loaded,
            gmm_path=str(application.state.gmm_path),
            logreg_path=str(application.state.logreg_path),
        )

    @application.post("/analyze", response_model=AnalyzeResponse)
    async def analyze(image: UploadFile = File(...)) -> AnalyzeResponse:
        if application.state.leaf_gmm is None or application.state.model is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Model artifacts not loaded. Place gmm.joblib and logreg.joblib "
                    f"at {application.state.gmm_path} and {application.state.logreg_path}, "
                    "then restart the server. Train in Colab and save with "
                    "leaf_gmm.save(...) and model.save(...)."
                ),
            )

        data = await image.read()
        if not data:
            raise HTTPException(status_code=400, detail="Empty upload")

        try:
            bgr = decode_image_bytes(data, max_size=DEFAULT_MAX_SIZE)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        result, seg, bgr = predict_bgr(
            bgr,
            application.state.leaf_gmm,
            application.state.model,
            band=DEFAULT_BAND,
        )
        feats: Dict[str, float] = result["features"]  # type: ignore[assignment]
        label = str(result["label"])
        severity = str(result["severity"])
        severity_percent = float(result["severity_percent"])

        return AnalyzeResponse(
            label=label,
            probability=float(result["probability"]),
            severity=severity,
            severity_percent=severity_percent,
            features=FeatureResponse(**feats),
            overlay_image_base64=_encode_overlay_png(bgr, seg["rust"]),
            message=_result_message(label, severity, severity_percent),
        )

    return application


app = create_app()


def main() -> None:
    import uvicorn

    host = os.environ.get("PHYTOLABS_HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", os.environ.get("PHYTOLABS_PORT", "8000")))
    uvicorn.run("phytolabs.api.main:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
