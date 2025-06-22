from diffusers import DiffusionPipeline, StableDiffusionInpaintPipeline
import torch
import sys
from PIL import Image
import asyncio
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import os
from datetime import datetime

# Setup event loop policy on Windows
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Prepare save/load directory
SAVE_DIR = "saved_images"
os.makedirs(SAVE_DIR, exist_ok=True)

# Torch workaround
if hasattr(torch, 'classes'):
    sys.modules['torch.classes'].__path__ = []

# Load models
@st.cache_resource
def load_all_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    txt2img_pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16"
    ).to(device)
    txt2img_pipe.load_lora_weights("DGM_project/Kream-model-lora-finetune", weight_name="pytorch_lora_weights.safetensors")

    inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float16
    ).to(device)

    return txt2img_pipe, inpaint_pipe

# UI
st.set_page_config(page_title="Diffusion Playground", layout="wide")
st.title("🎨 Diffusion model Playground: Generate | image-to-image | Inpaint")

# Prompt inputs
prompt = st.text_input("Prompt", value="A fashion-forward portrait of a model wearing a Nike tracksuit in an urban setting.")
negative_prompt = st.text_input("Negative Prompt", value="low quality, blurry, distorted, watermark, logo cut-off, bad composition")

# Guidance sliders
guidance_scale = st.slider("Guidance Scale", 1.0, 20.0, 7.5)
num_inference_steps = st.slider("Number of Inference Steps", 10, 100, 50)

# Upload image section
uploaded_img = st.file_uploader("Upload image for image-to-image or inpainting", type=["png", "jpg", "jpeg"])
if uploaded_img:
    user_image = Image.open(uploaded_img).convert("RGB")
    st.image(user_image, caption="Uploaded Image", use_column_width=True)
    st.session_state.generated_image = user_image.resize((512, 512))

# Buttons
generate_button = st.button("Generate Image")
upscale_button = st.button("image-to-image")
inpaint_button = st.button("Start Inpainting")

if "generated_image" not in st.session_state:
    st.session_state.generated_image = None

txt2img_pipe, inpaint_pipe = load_all_models()

# GENERATE
if generate_button:
    with st.spinner("Generating..."):
        image = txt2img_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps
        ).images[0]
        st.session_state.generated_image = image
        st.image(image, caption="Generated", use_column_width=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{SAVE_DIR}/generated_{timestamp}.png"
        image.save(filename)
        st.success(f"Saved to {filename}")

# UPSCALE
if upscale_button and st.session_state.generated_image:
    with st.spinner("Upscaling..."):
        resized = st.session_state.generated_image.resize((512, 512), Image.BICUBIC)
        upscaled = txt2img_pipe(
            prompt=prompt,
            image=resized,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps
        ).images[0]
        st.image(upscaled, caption="Upscaled", use_column_width=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{SAVE_DIR}/upscaled_{timestamp}.png"
        upscaled.save(filename)
        st.success(f"Saved to {filename}")

# INPAINTING
if inpaint_button:
    st.session_state.show_canvas = True

if st.session_state.get("show_canvas", False) and st.session_state.generated_image:
    st.subheader("🎭 Draw your mask for inpainting")

    image_pil = st.session_state.generated_image.convert("RGB")

    with st.sidebar:
        stroke_width = st.slider("Stroke width:", 1, 100, 30)
        stroke_color = st.color_picker("Stroke color:", "#FFFFFF")

    with st.container():
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 1)",
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_image=image_pil,
            update_streamlit=True,
            height=image_pil.height,
            width=image_pil.width,
            drawing_mode="freedraw",
            key="canvas"
        )

    if canvas_result.image_data is not None:
        if st.button("✅ Done Painting Mask"):
            st.session_state.mask_ready = True
            st.session_state.canvas_mask = canvas_result.image_data

# INPAINT APPLY
if st.session_state.get("mask_ready", False):
    alpha = st.session_state.canvas_mask[:, :, 3]
    mask_image = Image.fromarray((alpha > 0).astype(np.uint8) * 255)
    mask_image = mask_image.resize((512, 512))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mask_path = f"{SAVE_DIR}/masked_{timestamp}.png"
    mask_image.save(mask_path)
    st.success(f"Mask saved to {mask_path}")

    if st.button("🎨 Apply Inpainting"):
        with st.spinner("Inpainting..."):
            result = inpaint_pipe(
                prompt=prompt,
                image=st.session_state.generated_image.convert("RGB").resize((512, 512)),
                mask_image=mask_image,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=torch.manual_seed(42)
            )
            inpainted_img = result.images[0]
            st.image(inpainted_img, caption="Inpainted Result", use_column_width=True)

            filename = f"{SAVE_DIR}/inpainting_{timestamp}.png"
            inpainted_img.save(filename)
            st.success(f"Saved to {filename}")
