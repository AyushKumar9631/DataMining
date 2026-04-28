import numpy as np
import torch
import torchvision.transforms as T
import gradio as gr
from PIL import Image, ImageDraw, ImageFont
import os

from model import load_model

# ── device & model ────────────────────────────────────────────────
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_PATH = "crowd_net_v2_final.pth"
model     = load_model(CKPT_PATH, DEVICE)

# ── preprocessing ─────────────────────────────────────────────────
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
])

# ── density thresholds ────────────────────────────────────────────
def density_label(count: int) -> tuple[str, str]:
    if count < 20:
        return "Sparse",   "#3B6D11"   # green
    elif count < 75:
        return "Moderate", "#BA7517"   # amber
    elif count < 200:
        return "Dense",    "#993C1D"   # coral
    else:
        return "Very Dense","#A32D2D"  # red


def annotate_image(img: Image.Image, count: int) -> Image.Image:
    """Overlay count badge on the image."""
    img   = img.copy().resize((480, 360))
    draw  = ImageDraw.Draw(img)
    label, color = density_label(count)
    text  = f"Count: {count}  |  {label}"
    # semi-transparent banner at bottom
    draw.rectangle([0, 320, 480, 360], fill=(0, 0, 0, 180))
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except Exception:
        font = ImageFont.load_default()
    draw.text((12, 328), text, fill=color, font=font)
    return img


def predict(image: Image.Image):
    if image is None:
        return None, "Please upload an image.", ""

    tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        log_pred = model(tensor).item()
    count = max(0, int(round(np.expm1(log_pred))))

    label, color = density_label(count)
    annotated    = annotate_image(image, count)

    summary = f"### Predicted crowd count: **{count}**"
    detail  = (
        f"**Density level:** {label}\n\n"
        f"**Model:** CrowdNetV2 (MobileNetV2 backbone)\n\n"
        f"**Dataset:** ShanghaiTech Part B\n\n"
        f"*Trained with log₁p target transform + Huber loss*"
    )
    return annotated, summary, detail


# ── examples ──────────────────────────────────────────────────────
example_dir = "examples"
examples = []
if os.path.isdir(example_dir):
    examples = [[os.path.join(example_dir, f)]
                for f in os.listdir(example_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))]

# ── UI ────────────────────────────────────────────────────────────
with gr.Blocks(
    title="Crowd Counting",
    theme=gr.themes.Soft(primary_hue="violet"),
    css="""
        #title  { text-align: center; margin-bottom: 0.5rem; }
        #sub    { text-align: center; color: #888; margin-bottom: 1.5rem; font-size: 14px; }
        .count  { font-size: 1.6rem; font-weight: 600; text-align: center; padding: 1rem 0; }
        footer  { display: none !important; }
    """
) as demo:

    gr.Markdown("# 👥 Crowd Counting", elem_id="title")
    gr.Markdown(
        "Upload a crowd image — the model estimates how many people are present.",
        elem_id="sub"
    )

    with gr.Row():
        with gr.Column(scale=1):
            inp     = gr.Image(type="pil", label="Upload image")
            btn     = gr.Button("Count people", variant="primary")

        with gr.Column(scale=1):
            out_img = gr.Image(label="Result", type="pil")
            out_sum = gr.Markdown(elem_classes=["count"])
            out_det = gr.Markdown()

    btn.click(fn=predict, inputs=inp, outputs=[out_img, out_sum, out_det])
    inp.change(fn=predict, inputs=inp, outputs=[out_img, out_sum, out_det])

    if examples:
        gr.Examples(examples=examples, inputs=inp)

    gr.Markdown(
        "---\n"
        "Model: **CrowdNetV2** · Backbone: MobileNetV2 (ImageNet) · "
        "Dataset: ShanghaiTech Part B · Built with PyTorch + Gradio"
    )

if __name__ == "__main__":
    demo.launch()
