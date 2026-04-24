import streamlit as st
from PIL import Image
from model_inference import KidneyStoneDetectionModel
from fpdf import FPDF
import tempfile
import datetime

# --- CONFIG ---
st.set_page_config(page_title="StoneVision AI", layout="wide")

# --- CSS (IMPORTANT FOR UI) ---
st.markdown("""
<style>
body { background-color: #0E1117; }

.main-title {
    text-align: center;
    color: white;
    font-size: 38px;
    font-weight: bold;
}

.sub-text {
    text-align: center;
    color: #9CA3AF;
    margin-bottom: 30px;
}

.box {
    background-color: #1C2029;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #2D3139;
}

.stButton>button {
    width: 100%;
    background-color: #22c55e;
    color: white;
    font-size: 16px;
    border-radius: 10px;
    height: 45px;
}

.success-bar {
    background-color: #14532d;
    padding: 10px;
    border-radius: 8px;
    color: #86efac;
    text-align: center;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown('<div class="main-title">🧠 StoneVision AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Automated Kidney Stone Detection from Medical Imaging</div>', unsafe_allow_html=True)

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    return KidneyStoneDetectionModel()

model = load_model()

st.markdown('<div class="success-bar">✅ Model ks_detection.pt loaded</div>', unsafe_allow_html=True)

# --- PDF FUNCTION ---
def create_pdf(img, out_img, label, conf):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "StoneVision AI Report", ln=True)

    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Result: {label}", ln=True)
    pdf.cell(0, 10, f"Confidence: {conf:.2%}", ln=True)

    temp1 = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    temp2 = tempfile.NamedTemporaryFile(delete=False, suffix=".png")

    img.save(temp1.name)
    Image.fromarray(out_img).save(temp2.name)

    pdf.image(temp1.name, x=10, w=90)
    pdf.image(temp2.name, x=110, w=90)

    return pdf.output(dest="S").encode("latin-1")

# --- LAYOUT ---
col1, col2 = st.columns(2)

# LEFT SIDE
with col1:
    st.markdown('<div class="box">', unsafe_allow_html=True)
    st.subheader("1. Upload X-ray Image")

    uploaded_file = st.file_uploader(
        "Drag & drop or browse",
        type=["png", "jpg", "jpeg", "tif"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)
    else:
        st.info("Upload image to start")

    st.markdown('</div>', unsafe_allow_html=True)

# RIGHT SIDE
with col2:
    st.markdown('<div class="box">', unsafe_allow_html=True)
    st.subheader("2. AI Analysis Result")

    if uploaded_file:
        if st.button("🔍 Analyze Image"):
            label, conf, out_img = model.predict(image)

            st.image(out_img, use_column_width=True)

            if "Detected" in label:
                st.error(f"{label}")
            else:
                st.success(f"{label}")

            st.metric("Confidence", f"{conf:.2%}")

            st.session_state["result"] = (image, out_img, label, conf)

    # DOWNLOAD
    if "result" in st.session_state:
        img, out_img, lbl, conf = st.session_state["result"]

        pdf = create_pdf(img, out_img, lbl, conf)

        st.download_button(
            "📄 Download Report",
            pdf,
            file_name="StoneVision_Report.pdf"
        )

    st.markdown('</div>', unsafe_allow_html=True)