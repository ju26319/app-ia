import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import anthropic
import base64
import io
import os
import tempfile
import time

st.set_page_config(
    page_title="Detector de Placas · Talento Tech 2026",
    page_icon="🚗",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; }

.stApp { background: #0a0a0f; color: #e8e6f0; }

.hero {
    background: linear-gradient(135deg, #1a0533 0%, #0d1a3a 50%, #001a1a 100%);
    border: 1px solid #2a1a4a;
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
}
.hero h1 {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #38bdf8, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.4rem 0;
}
.hero p { color: #94a3b8; font-size: 1rem; margin: 0; }

.plate-box {
    background: linear-gradient(135deg, #0f172a, #1e1b4b);
    border: 2px solid #7c3aed;
    border-radius: 12px;
    padding: 1.8rem;
    text-align: center;
}
.plate-text {
    font-family: 'DM Mono', monospace;
    font-size: 3rem;
    font-weight: 500;
    letter-spacing: 0.18em;
    color: #f1f5f9;
    background: #0f172a;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 0.6rem 2rem;
    display: inline-block;
    margin: 0.8rem 0;
}
.conf-badge {
    display: inline-block;
    background: #064e3b;
    color: #34d399;
    font-size: 0.85rem;
    font-family: 'DM Mono', monospace;
    padding: 0.25rem 0.9rem;
    border-radius: 99px;
    border: 1px solid #34d39944;
}
.low-conf { background: #451a03; color: #fb923c; border-color: #fb923c44; }

.metric-row { display: flex; gap: 1rem; margin: 1rem 0; flex-wrap: wrap; }
.metric-tile {
    flex: 1; min-width: 130px;
    background: #0f0f1a;
    border: 1px solid #1e2035;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-tile .val { font-size: 1.6rem; font-weight: 700; color: #a78bfa; font-family: 'DM Mono', monospace; }
.metric-tile .lbl { font-size: 0.78rem; color: #64748b; margin-top: 2px; }

section[data-testid="stSidebar"] { background: #08080e; border-right: 1px solid #1a1a2e; }

.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #4f46e5);
    color: white; border: none; border-radius: 8px;
    font-weight: 700; padding: 0.6rem 2rem;
}

.history-item {
    background: #0d0d18; border: 1px solid #1a1a2e; border-radius: 8px;
    padding: 0.8rem 1rem; margin-bottom: 0.5rem;
    font-family: 'DM Mono', monospace; font-size: 0.9rem;
    display: flex; justify-content: space-between; align-items: center;
}
.history-plate { color: #e2e8f0; font-weight: 500; }
.history-conf { color: #64748b; font-size: 0.8rem; }
</style>
""", unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []

@st.cache_resource
def load_yolo():
    return YOLO("best.pt")

@st.cache_resource
def get_claude():
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        try:
            api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
        except Exception:
            api_key = ""
    if not api_key:
        return None
    return anthropic.Anthropic(api_key=api_key)

def pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return base64.standard_b64encode(buf.getvalue()).decode()

def detect_plates(model, img_np, conf_thr=0.35):
    results = model(img_np, conf=conf_thr, verbose=False)
    plates = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            pad = 6
            x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
            x2, y2 = min(img_np.shape[1], x2 + pad), min(img_np.shape[0], y2 + pad)
            crop = img_np[y1:y2, x1:x2]
            crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            plates.append((crop_pil, (x1, y1, x2, y2), float(box.conf[0])))
    return sorted(plates, key=lambda x: -x[2])

def upscale_for_ocr(img: Image.Image, min_width=320) -> Image.Image:
    w, h = img.size
    if w < min_width:
        scale = min_width / w
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img

def extract_text_claude(client, plate_img):
    import json, re
    plate_img = upscale_for_ocr(plate_img)
    b64 = pil_to_base64(plate_img)
    prompt = """You are an expert license plate OCR system specialized in Latin American plates.
Look at this vehicle license plate image carefully.
Return ONLY a JSON object with these fields (no markdown, no extra text):
{"plate": "ABC123", "confidence": 0.97, "country_hint": "Colombia", "notes": ""}
Rules:
- plate: exact alphanumeric characters, uppercase, no spaces or dashes
- confidence: 0.0 to 1.0
- If unreadable, return confidence < 0.5 and plate = "ILEGIBLE"
- NO markdown or text outside the JSON"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=256,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                {"type": "text", "text": prompt}
            ]
        }]
    )
    raw = response.content[0].text.strip()
    raw = re.sub(r"```[a-z]*", "", raw).strip().strip("`")
    data = json.loads(raw)
    return data.get("plate", "ERROR"), float(data.get("confidence", 0.0))

def draw_boxes(img_np, detections, texts):
    out = img_np.copy()
    for (crop, (x1, y1, x2, y2), yconf), (text, lconf) in zip(detections, texts):
        color = (124, 58, 237)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
        label = f"{text}  {lconf*100:.0f}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)
        cv2.rectangle(out, (x1, y1 - th - 14), (x1 + tw + 12, y1), color, -1)
        cv2.putText(out, label, (x1 + 6, y1 - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
    return out

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Configuración")
    api_key_input = st.text_input("API Key de Anthropic", type="password", placeholder="sk-ant-...")
    if api_key_input:
        os.environ["ANTHROPIC_API_KEY"] = api_key_input
    st.markdown("---")
    conf_threshold = st.slider("Confianza YOLO mínima", 0.1, 0.9, 0.35, 0.05)
    show_crops = st.checkbox("Mostrar recortes de placa", value=True)
    st.markdown("---")
    st.markdown("**Historial de placas**")
    if st.session_state.history:
        for entry in reversed(st.session_state.history[-10:]):
            conf_class = "" if entry["conf"] >= 0.85 else "low-conf"
            st.markdown(
                f'<div class="history-item"><span class="history-plate">{entry["plate"]}</span>'
                f'<span class="history-conf {conf_class}">{entry["conf"]*100:.0f}%</span></div>',
                unsafe_allow_html=True)
        if st.button("Limpiar historial"):
            st.session_state.history = []
            st.rerun()
    else:
        st.caption("Sin placas aún")

# Hero
st.markdown("""
<div class="hero">
  <h1>🚗 Detector de Placas</h1>
  <p>YOLO v8 + Claude Vision · Talento Tech Bootcamp IA 2026</p>
</div>
""", unsafe_allow_html=True)

with st.spinner("Cargando modelo YOLO…"):
    yolo_model = load_yolo()
claude_client = get_claude()

if not claude_client:
    st.warning("⚠️ Ingresa tu API Key de Anthropic en el panel izquierdo para activar el OCR con Claude Vision.", icon="🔑")

tab_img, tab_cam, tab_video = st.tabs(["📷 Imagen", "📸 Cámara", "🎥 Video"])

def process_image(img_pil):
    img_np = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    with st.spinner("🔍 Detectando placas con YOLO…"):
        t0 = time.time()
        detections = detect_plates(yolo_model, img_np, conf_thr=conf_threshold)
        yolo_time = time.time() - t0

    if not detections:
        st.error("❌ No se detectaron placas. Intenta con otra foto o ajusta la confianza mínima.")
        return

    texts = []
    llm_time = 0
    if claude_client:
        with st.spinner(f"🤖 Extrayendo texto con Claude Vision ({len(detections)} placa(s))…"):
            t1 = time.time()
            for crop_pil, box, yconf in detections:
                text, lconf = extract_text_claude(claude_client, crop_pil)
                texts.append((text, lconf))
                st.session_state.history.append({"plate": text, "conf": lconf})
            llm_time = time.time() - t1
    else:
        texts = [("(OCR desactivado)", 0.0)] * len(detections)

    annotated = draw_boxes(img_np, detections, texts)
    annotated_pil = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

    col_img, col_res = st.columns([1.3, 1])
    with col_img:
        st.image(annotated_pil, use_column_width=True, caption="Imagen anotada")
    with col_res:
        avg_conf = np.mean([c for _, c in texts]) if texts else 0
        st.markdown(f"""
        <div class="metric-row">
          <div class="metric-tile"><div class="val">{len(detections)}</div><div class="lbl">Placas detectadas</div></div>
          <div class="metric-tile"><div class="val">{avg_conf*100:.0f}%</div><div class="lbl">Confianza OCR</div></div>
          <div class="metric-tile"><div class="val">{yolo_time*1000:.0f}ms</div><div class="lbl">YOLO</div></div>
          <div class="metric-tile"><div class="val">{llm_time*1000:.0f}ms</div><div class="lbl">Claude Vision</div></div>
        </div>
        """, unsafe_allow_html=True)

        for i, ((crop_pil, box, yconf), (text, lconf)) in enumerate(zip(detections, texts)):
            badge_class = "" if lconf >= 0.85 else "low-conf"
            st.markdown(f"""
            <div class="plate-box" style="margin-bottom:1rem">
              <div style="color:#94a3b8;font-size:0.8rem;margin-bottom:0.4rem">PLACA {i+1}</div>
              <div class="plate-text">{text}</div><br>
              <span class="conf-badge {badge_class}">Claude: {lconf*100:.1f}% · YOLO: {yconf*100:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)
            if show_crops:
                st.image(crop_pil, caption=f"Recorte placa {i+1}", width=260)

with tab_img:
    uploaded = st.file_uploader("Sube una imagen del vehículo",
        type=["jpg", "jpeg", "png", "webp", "bmp"], label_visibility="collapsed")
    if uploaded:
        process_image(Image.open(uploaded).convert("RGB"))

with tab_cam:
    cam_img = st.camera_input("Toma una foto del vehículo")
    if cam_img:
        process_image(Image.open(cam_img).convert("RGB"))

with tab_video:
    video_file = st.file_uploader("Sube un video corto (MP4, AVI, MOV)",
        type=["mp4", "avi", "mov", "mkv"], label_visibility="collapsed", key="vid")
    frame_step = st.slider("Analizar cada N fotogramas", 10, 60, 30)

    if video_file and st.button("▶ Analizar video"):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_file.read())
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        found_plates = {}
        progress = st.progress(0, text="Analizando video…")
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_step == 0:
                dets = detect_plates(yolo_model, frame, conf_thr=conf_threshold)
                if dets and claude_client:
                    for crop_pil, _, yconf in dets:
                        text, lconf = extract_text_claude(claude_client, crop_pil)
                        if text not in ("ILEGIBLE", "ERROR") and lconf > 0.6:
                            found_plates[text] = max(found_plates.get(text, 0), lconf)
                            st.session_state.history.append({"plate": text, "conf": lconf})
            frame_idx += 1
            progress.progress(min(frame_idx / max(total_frames, 1), 1.0),
                              text=f"Fotograma {frame_idx}/{total_frames}")

        cap.release()
        os.unlink(tmp_path)
        progress.empty()

        if found_plates:
            st.success(f"✅ {len(found_plates)} placa(s) únicas encontradas")
            for plate, conf in sorted(found_plates.items(), key=lambda x: -x[1]):
                badge = "" if conf >= 0.85 else "low-conf"
                st.markdown(f"""
                <div class="plate-box" style="margin-bottom:0.8rem">
                  <div class="plate-text">{plate}</div><br>
                  <span class="conf-badge {badge}">Confianza: {conf*100:.1f}%</span>
                </div>""", unsafe_allow_html=True)
        else:
            st.warning("No se detectaron placas legibles en el video.")

st.markdown("""
<div style="text-align:center;color:#334155;font-size:0.78rem;margin-top:3rem;
padding-top:1rem;border-top:1px solid #1a1a2e">
  Talento Tech · Bootcamp IA Innovador 2026 · YOLO v8 + Claude Vision
</div>
""", unsafe_allow_html=True)
