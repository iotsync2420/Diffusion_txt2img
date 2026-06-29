
import requests
import io
from PIL import Image

MODELS = {
    "Stable Diffusion 1.5": "runwayml/stable-diffusion-v1-5",
    "SDXL": "stabilityai/stable-diffusion-xl-base-1.0",
}

STYLE_PRESETS = {
    "None": "",
    "Photorealistic": "photorealistic, DSLR, 8K, sharp focus, detailed",
    "Anime": "anime style, Studio Ghibli, vibrant colors, illustrated",
    "Oil Painting": "oil painting, brush strokes, classical art, canvas texture",
    "Watercolor": "watercolor painting, soft edges, pastel colors, artistic",
    "Cyberpunk": "cyberpunk, neon lights, futuristic, dark city, rain",
}

def generate_text2img(prompt, negative_prompt, hf_token, model_key="Stable Diffusion 1.5", style_key="None"):
    model_id = MODELS[model_key]
    API_URL = f"https://api-inference.huggingface.co/models/{model_id}" # Corrected API URL
    headers = {
    "Authorization": f"Bearer {hf_token}",
    "Host": "api-inference.huggingface.co"
}

    # Apply style preset if selected
    style_suffix = STYLE_PRESETS.get(style_key, "")
    final_prompt = f"{prompt}, {style_suffix}" if style_suffix else prompt

    payload = {
        "inputs": final_prompt,
        "parameters": {"negative_prompt": negative_prompt}
    }

    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return Image.open(io.BytesIO(response.content)), None
    return None, f"Error {response.status_code}: {response.text}"

def generate_img2img(image, prompt, hf_token, model_key="Stable Diffusion 1.5"):
    """
    Hugging Face Inference API img2img requires sending the raw image bytes
    as the main data payload, and secondary inputs (like the prompt) via headers
    or special parameter mapping depending on the endpoint backend.
    """
    model_id = MODELS[model_key]
    API_URL = f"https://api-inference.huggingface.co/models/{model_id}" # Corrected API URL for consistency

    headers = {
        "Authorization": f"Bearer {hf_token}",
    }

    # Convert PIL Image to raw bytes
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_bytes = buffered.getvalue()

    # Form data payload structure for HF image-to-image
    files = {
        "image": ("image.jpeg", img_bytes, "image/jpeg")
    }
    data = {
        "prompt": prompt
    }

    response = requests.post(API_URL, headers=headers, files=files, data=data)

    if response.status_code == 200:
        return Image.open(io.BytesIO(response.content)), None
    return None, f"Error {response.status_code}: {response.text}"

import importlib
import generate
# This forces Colab to read the brand new version of generate.py on disk
importlib.reload(generate)

from generate import generate_text2img
from google.colab import userdata

try:
    HF_TOKEN = userdata.get('HF_TOKEN')
except Exception:
    print("Make sure to add 'HF_TOKEN' to your Colab Secrets!")
    HF_TOKEN = ""

if HF_TOKEN:
    image, error = generate_text2img(
        prompt="a futuristic city at sunset, cinematic 8K",
        negative_prompt="blurry, low quality",
        hf_token=HF_TOKEN,
        model_key="Stable Diffusion 1.5",
        style_key="None"
    )

    if error:
        print("❌ Error:", error)
    else:
        display(image)
        image.save("test_output.png")
        print("✅ Generation works perfectly!")
