"""Hugging Face Space entrypoint for TruthLens."""

import io
import os
import sys
from pathlib import Path

# Ensure application directories are discoverable on PYTHONPATH
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

# Default production configurations for Hugging Face Space
os.environ.setdefault("CELERY_TASK_ALWAYS_EAGER", "True")
os.environ.setdefault("DATABASE_URL", "sqlite:///./truthlens.db")
os.environ.setdefault("MODELS_YAML_PATH", str(BASE_DIR / "backend" / "models.yaml"))
os.environ.setdefault("STORAGE_LOCAL_FALLBACK_DIR", str(BASE_DIR / "data" / "storage"))

import gradio as gr
import spaces
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

# Import backend routers and pipelines
from backend.app.api.admin import router as admin_router
from backend.app.api.analyses import router as analyses_router
from backend.app.api.analyze import router as analyze_router
from backend.app.api.api_keys import router as api_keys_router
from backend.app.api.auth import router as auth_router
from backend.app.api.health import router as health_router
from backend.app.api.models import router as models_router
from backend.app.api.reports import router as reports_router
from backend.app.api.websocket import router as websocket_router
from backend.app.core.config import settings
from backend.app.db.session import create_tables
from backend.app.inference.image import run_image_inference
from backend.app.inference.text import run_text_inference

# Ensure storage directory and database tables exist
storage_dir = BASE_DIR / "data" / "storage"
storage_dir.mkdir(parents=True, exist_ok=True)
create_tables()


# ZeroGPU inference functions
@spaces.GPU
def predict_image(input_img):
    if input_img is None:
        return "Please upload an image to analyze.", None, None
    try:
        # Score the uploaded file as is: re-encoding it (e.g. to JPEG) erases the artifacts the model looks for.
        with open(input_img, "rb") as file:
            image_bytes = file.read()
        result, artefacts = run_image_inference(image_bytes)

        # The validated threshold can be far below 1%, so show enough digits to compare against it.
        prob = result.get("probability", 0.0) * 100
        threshold = result.get("decision_threshold", 0.5) * 100
        verdict = result.get("verdict_label", "Unknown")
        summary = (
            f"Verdict: {verdict}\nAI probability: {prob:.3g}% (flagged as AI at {threshold:.3g}% or above)"
            f"\nModel: {result.get('model_name', 'unknown')}"
        )

        heatmap_img = None
        if "heatmap.png" in artefacts:
            heatmap_img = Image.open(io.BytesIO(artefacts["heatmap.png"]))

        spectrum_img = None
        if "spectrum.png" in artefacts:
            spectrum_img = Image.open(io.BytesIO(artefacts["spectrum.png"]))

        return summary, heatmap_img, spectrum_img
    except Exception as e:
        return f"Inference error: {e!s}", None, None


@spaces.GPU
def predict_text(input_text: str):
    if not input_text or not input_text.strip():
        return "Please enter text to analyze."
    try:
        result, _ = run_text_inference(input_text)
        prob = round(result.get("probability", 0.0) * 100, 1)
        verdict = result.get("verdict_label", "Unknown")
        explanation = result.get("explanation", "")
        return f"Verdict: {verdict}\nAI Probability: {prob}%\n\nExplanation:\n{explanation}"
    except Exception as e:
        return f"Inference error: {e!s}"


# Build Gradio UI with ZeroGPU support
with gr.Blocks(title="TruthLens - Multi-Modal Deepfake Detection") as demo:
    gr.Markdown("# 🛡️ TruthLens: Multi-Modal Deepfake & Forensic Attribution Platform")
    gr.Markdown("TruthLens provides tamper-evident multi-modal forensic inspection across visual, acoustic, and textual media.")

    with gr.Tabs():
        with gr.TabItem("🖼️ Image Forensics (ZeroGPU)"):
            with gr.Row():
                with gr.Column():
                    img_input = gr.Image(type="filepath", format="png", label="Upload Image (Face, Portrait, or Scene)")
                    img_btn = gr.Button("Analyze Image with ZeroGPU", variant="primary")
                with gr.Column():
                    img_output_text = gr.Textbox(label="Forensic Verdict & Probability", lines=4)
                    with gr.Row():
                        img_heatmap = gr.Image(label="Explainability Heatmap (Grad-CAM)")
                        img_spectrum = gr.Image(label="2D DCT Radial Spectrum")
            img_btn.click(fn=predict_image, inputs=img_input, outputs=[img_output_text, img_heatmap, img_spectrum])

        with gr.TabItem("📝 Text Forensics (ZeroGPU)"):
            with gr.Row():
                with gr.Column():
                    txt_input = gr.Textbox(label="Input Text (Articles, Essays, Transcripts)", lines=8, placeholder="Paste text here...")
                    txt_btn = gr.Button("Analyze Text with ZeroGPU", variant="primary")
                with gr.Column():
                    txt_output = gr.Textbox(label="Forensic Linguistic Verdict", lines=8)
            txt_btn.click(fn=predict_text, inputs=txt_input, outputs=txt_output)

        with gr.TabItem("🌐 Full React Platform & REST API"):
            gr.Markdown("""
            ### Additional TruthLens Capabilities:
            * **Interactive React Application:** Open the [Full React Web Platform](/app)
            * **Interactive Swagger REST API:** Explore the [OpenAPI / Swagger Documentation](/api/v1/docs)
            * **Tamper-Evident HMAC Signed Reports:** Generated via `/api/v1/reports` with QR verification
            """)


from backend.app.main import app as fastapi_app

# Mount all FastAPI routes onto demo.app
demo.app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

prefix = "/api/v1"
demo.app.include_router(health_router, prefix=prefix)
demo.app.include_router(auth_router, prefix=prefix)
demo.app.include_router(analyze_router, prefix=prefix)
demo.app.include_router(analyses_router, prefix=prefix)
demo.app.include_router(reports_router, prefix=prefix)
demo.app.include_router(api_keys_router, prefix=prefix)
demo.app.include_router(models_router, prefix=prefix)
demo.app.include_router(admin_router, prefix=prefix)
demo.app.include_router(websocket_router, prefix=prefix)

# Mount React static assets onto demo.app
dist_dir = BASE_DIR / "dist"
assets_dir = dist_dir / "assets"
if assets_dir.exists():
    demo.app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")


@demo.app.get("/app", include_in_schema=False)
@demo.app.get("/app/{full_path:path}", include_in_schema=False)
async def serve_react_app(request: Request, full_path: str = ""):
    index_file = dist_dir / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return HTMLResponse("<h1>TruthLens Full Platform</h1>")


# Launch with demo.launch() and mount FastAPI application
if __name__ == "__main__":
    server, _, _ = demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        strict_cors=False,
        ssr_mode=False,
        prevent_thread_lock=True,
    )
    server.include_router(fastapi_app.router)

    if assets_dir.exists():
        server.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @server.get("/app", include_in_schema=False)
    @server.get("/app/{full_path:path}", include_in_schema=False)
    async def serve_server_react_app(request: Request, full_path: str = ""):
        index_file = dist_dir / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return HTMLResponse("<h1>TruthLens Full Platform</h1>")

    print("[TruthLens] FastAPI routes successfully mounted onto ZeroGPU server.", flush=True)
    demo.block_thread()
