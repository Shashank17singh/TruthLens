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
os.environ["CORS_ORIGINS"] = '["*"]'

import gradio as gr
import spaces
from PIL import Image
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

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
        buf = io.BytesIO()
        input_img.save(buf, format="JPEG")
        image_bytes = buf.getvalue()
        result, artefacts = run_image_inference(image_bytes)

        prob = round(result.get("probability", 0.0) * 100, 1)
        verdict = result.get("verdict_label", "Unknown")
        summary = f"Verdict: {verdict}\nAI Probability: {prob}%\nModel: {result.get('model_name', 'convnext-tiny')}"

        heatmap_img = None
        if "heatmap.png" in artefacts:
            heatmap_img = Image.open(io.BytesIO(artefacts["heatmap.png"]))

        spectrum_img = None
        if "spectrum.png" in artefacts:
            spectrum_img = Image.open(io.BytesIO(artefacts["spectrum.png"]))

        return summary, heatmap_img, spectrum_img
    except Exception as e:
        return f"Inference error: {str(e)}", None, None


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
        return f"Inference error: {str(e)}"


# Build Gradio UI with ZeroGPU support
with gr.Blocks(title="TruthLens - Multi-Modal Deepfake Detection") as demo:
    gr.Markdown("# 🛡️ TruthLens: Multi-Modal Deepfake & Forensic Attribution Platform")
    gr.Markdown("TruthLens provides tamper-evident multi-modal forensic inspection across visual, acoustic, and textual media.")

    with gr.Tabs():
        with gr.TabItem("🖼️ Image Forensics (ZeroGPU)"):
            with gr.Row():
                with gr.Column():
                    img_input = gr.Image(type="pil", label="Upload Image (Face, Portrait, or Scene)")
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


# Mount all FastAPI routes onto demo.app
demo.app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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


# Launch with demo.launch() so ZeroGPU hooks trigger naturally
if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
