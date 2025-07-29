import streamlit as st
from deepface.DeepFace import analyze  # ← FIXED import
import pandas as pd
import numpy as np
from PIL import Image
import base64
import io
import plotly.express as px

# ====== Set Background ======
def set_background(image_path):
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
    """, unsafe_allow_html=True)

# ====== Setup ======
st.set_page_config(page_title="Emotion Scope", layout="wide")
set_background("assets/background.png")

st.markdown("""
    <h1 style='text-align: center; color: #ffc8dd; font-size: 4em;'>Emotion Scope</h1>
    <p style='text-align: center; color: #f1f1f1; font-size: 1.2em;'>Facial expression surveillance mapped into data-driven visuals</p>
""", unsafe_allow_html=True)

st.markdown("---")

# ====== Upload Image ======
uploaded_image = st.file_uploader("📷 Upload a facial image to analyze", type=["jpg", "jpeg", "png"])

if uploaded_image:
    # Display uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Analyze emotion
    with st.spinner("Analyzing emotion..."):
        result = analyze(img_path=np.array(image), actions=['emotion'], enforce_detection=False)[0]  # ← FIXED usage

    dominant_emotion = result['dominant_emotion']
    emotion_scores = result['emotion']

    st.markdown(f"<h2 style='text-align: center; color: #00ffcc;'>Detected Emotion: <span style='color:white'>{dominant_emotion.upper()}</span></h2>", unsafe_allow_html=True)
    st.markdown("---")

    # ====== Prepare DataFrame ======
    emotion_df = pd.DataFrame(emotion_scores.items(), columns=["emotion", "score"])

    # ====== Pie Chart ======
    st.subheader("Emotion Distribution (Pie Chart)")
    fig_pie = px.pie(emotion_df, names="emotion", values="score", hole=0.3, color_discrete_sequence=px.colors.sequential.Rainbow)
    fig_pie.update_layout(template="plotly_dark")
    st.plotly_chart(fig_pie, use_container_width=True)

    # ====== Bar Chart ======
    st.subheader("Emotion Scores (Bar Chart)")
    fig_bar = px.bar(emotion_df, x="emotion", y="score", color="emotion", template="plotly_dark", color_discrete_sequence=px.colors.sequential.Magma)
    fig_bar.update_traces(marker_line_color="white", marker_line_width=1.5)
    st.plotly_chart(fig_bar, use_container_width=True)

    # ====== Radar Chart ======
    st.subheader("Emotion Profile (Radar Chart)")
    fig_radar = px.line_polar(emotion_df, r="score", theta="emotion", line_close=True, template="plotly_dark", color_discrete_sequence=["#00f0ff"])
    fig_radar.update_traces(fill='toself')
    st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("---")

# ====== Footer ======
st.markdown("<p style='text-align: center; color: gray;'>Built by Blake Murray · Powered by DeepFace & Streamlit</p>", unsafe_allow_html=True)
