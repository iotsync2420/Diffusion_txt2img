import streamlit as st
from generate import generate_text2img, generate_img2img, STYLE_PRESETS, MODELS
from PIL import Image
import datetime
import io

st.set_page_config(page_title="DreamForge AI", page_icon="🎨", layout="wide")
st.title("🎨 DreamForge — AI Image Generator")
st.caption("Powered by Stable Diffusion via Hugging Face")

# Initialize Session States
if "history" not in st.session_state:
    st.session_state.history = []

# Initialize the main prompt tracking variable explicitly
if "main_prompt_input" not in st.session_state:
    st.session_state.main_prompt_input = ""

# ── Sidebar ──────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    hf_token = st.text_input("🔑 HF Token", type="password")
    st.markdown("[Get free token →](https://huggingface.co/settings/tokens)")

    model_choice = st.selectbox("Model", list(MODELS.keys()))
    mode = st.radio("Mode", ["Text → Image", "Image → Image"])

    st.divider()
    st.header("🕓 Prompt History")

    if st.session_state.history:
        # Loop over a static copied slice to avoid index shifting mutation bugs
        for i, item in enumerate(reversed(st.session_state.history[-5:])):
            st.caption(f"🕐 {item['time']}")
            st.write(f"_{item['prompt'][:60]}..._" if len(item['prompt']) > 60 else f"_{item['prompt']}_")

            # Directly mutate the text area key to instantly force update the widget state
            if st.button("Reuse", key=f"reuse_{i}"):
                st.session_state.main_prompt_input = item['prompt']
                st.rerun()
    else:
        st.caption("No history yet.")

# ── Main Area ─────────────────────────────────────────
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Prompt")

    # Using the state-tied key here directly solves the reuse sync glitch
    prompt = st.text_area("Prompt", key="main_prompt_input", height=100)
    neg_prompt = st.text_input("Negative Prompt", "blurry, low quality, deformed, ugly")
    style = st.selectbox("🎨 Style Preset", list(STYLE_PRESETS.keys()))

    uploaded = None
    if mode == "Image → Image":
        uploaded = st.file_uploader("Upload Image to Transform", type=["png", "jpg", "jpeg"])

    generate_btn = st.button("✨ Generate", use_container_width=True, type="primary")

with col2:
    st.subheader("🖼️ Output")
    output_placeholder = st.empty()

    if generate_btn:
        if not hf_token:
            st.error("Please enter your HF Token in the sidebar!")
        elif not prompt:
            st.error("Please enter a prompt!")
        else:
            with st.spinner("Generating your image... ✨"):
                if mode == "Text → Image":
                    # Let generate_text2img handle the style parsing implicitly
                    image, error = generate_text2img(
                        prompt, neg_prompt, hf_token, model_choice, style_key=style
                    )
                else:
                    if not uploaded:
                        st.error("Please upload an image!")
                        image, error = None, None
                    else:
                        input_img = Image.open(uploaded)
                        # Let generate_img2img take the raw prompt input
                        image, error = generate_img2img(input_img, prompt, hf_token, model_choice)

                if error:
                    st.error(error)
                elif image:
                    output_placeholder.image(image, use_container_width=True)

                    # Save to history
                    st.session_state.history.append({
                        "prompt": prompt,
                        "time": datetime.datetime.now().strftime("%H:%M:%S"),
                        "style": style
                    })

                    # Streamlined buffer management
                    buf = io.BytesIO()
                    image.save(buf, format="PNG")
                    st.download_button(
                        "⬇️ Download Image",
                        buf.getvalue(),
                        "dreamforge_output.png",
                        "image/png",
                        use_container_width=True
                    )
