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
import json
import re

st.set_page_config(
    page_title="Detector de Placas - Talento Tech 2026",
    page_icon="🚗",
    layout="wide",
)

CSS = (
    "<style>"
    ".stApp { background: #0a0a0f; color: #e8e6f0; }"
    "section[data-testid='stSidebar'] { background: #08080e; border-right: 1px solid #1a1a2e; }"
    ".hero { background: #0d1a3a; border: 1px solid #2a1a4a; border-radius: 16px; padding: 2rem 2.5rem; margin-bottom: 2rem; }"
    ".hero h1 { font-size: 2rem; font-weight: 800; color: #a78bfa; margin: 0 0 0.3rem 0; }"
    ".hero p { color: #94a3b8; margin: 0; }"
    ".plate-box { background: #0f172a; border: 2px solid #7c3aed; border-radius: 12px; padding: 1.5rem; text-align: center; margin-bottom: 1rem; }"
    ".plate-text { font-size: 2.2rem; font-weight: 700; letter-spacing: 0.15em; color: #f1f5f9; background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 0.4rem 1.2rem; display: inline-block; margin: 0.5rem 0; font-family: monospace; }"
    ".badge-ok { display: inline-block; background: #064e3b; color: #34d399; font-size: 0.82rem; padding: 0.2rem 0.8rem; border-radius: 99px; }"
    ".badge-low { display: inline-block; background: #451a03; color: #fb923c; font-size: 0.82rem; padding: 0.2rem 0.8rem; border-radius: 99px; }"
    ".metrics { display: flex; gap: 0.8rem; margin: 1rem 0; flex-wrap: wrap; }"
    ".met { flex: 1; min-width: 100px; background: #0f0f1a; border: 1px solid #1e2035; border-radius: 10px; padding: 0.8rem; text-align: center; }"
    ".met-val { font-size: 1.3rem; font-weight: 700; color: #a78bfa; font-family: monospace; }"
    ".met-lbl { font-size: 0.7rem; color: #64748b; }"
    ".hist-item { background: #0d0d18; border: 1px solid #1a1a2e; border-radius: 6px; padding: 0.5rem 0.8rem; margin-bottom: 0.4rem; font-family: monospace; font-size: 0.85rem; display: flex; justify-content: space-between; }"
    ".hist-plate { color: #e2e8f0; }"
    ".hist-conf { color: #64748b; }"
    ".footer { text-align: center; color: #334155; font-size: 0.75rem; margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #1a1a2e; }"
    "</style>"
)
st.markdown(CSS, unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []

MODELS = [
    "claude-haiku-4-5-20251001",
    "claude-sonnet-4-5-20250929",
    "claude-sonnet-4-6",
]

@st.cache_resource
def load_yolo():
    return YOLO("best.pt")

def get_api_key():
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if key:
        return key
    try:
        key = st.secrets["ANTHROPIC_API_KEY"].strip()
        if key:
            return key
    except Exception:
        pass
    return ""

def make_client(api_key):
    if not api_key:
        return None
    return anthropic.Anthropic(api_key=api_key)

def pil_to_b64(img):
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
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(img_np.shape[1], x2 + pad)
            y2 = min(img_np.shape[0], y2 + pad)
            crop = img_np[y1:y2, x1:x2]
            crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            plates.append((crop_pil, (x1, y1, x2, y2), float(box.conf[0])))
    return sorted(plates, key=lambda x: -x[2])

def upscale(img, min_w=320):
    w, h = img.size
    if w < min_w:
        s = min_w / w
        img = img.resize((int(w * s), int(h * s)), Image.LANCZOS)
    return img

def ocr_claude(client, plate_img):
    plate_img = upscale(plate_img)
    b64 = pil_to_b64(plate_img)
    prompt = (
        "Look at this license plate image. "
        "Return ONLY a JSON object, no markdown, no explanation: "
        "{\"plate\":\"ABC123\",\"confidence\":0.97} "
        "Rules: plate must be uppercase alphanumeric only, no spaces or dashes. "
        "confidence is 0.0 to 1.0. "
        "If unreadable return: {\"plate\":\"ILEGIBLE\",\"confidence\":0.2}"
    )
    for model_name in MODELS:
        try:
            response = client.messages.create(
                model=model_name,
                max_tokens=100,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": b64
                            }
                        },
                        {"type": "text", "text": prompt}
                    ]
                }]
            )
            raw = response.content[0].text.strip()
            raw = re.sub(r"```[a-z]*", "", raw).strip().strip("`")
            data = json.loads(raw)
            plate = data.get("plate", "ERROR")
            conf = float(data.get("confidence", 0.0))
            return plate, conf, model_name
        except anthropic.NotFoundError:
            continue
        except json.JSONDecodeError:
            return "ERROR_JSON", 0.0, model_name
        except Exception as e:
            return "ERROR", 0.0, str(e)
    return "SIN_MODELO", 0.0, "ninguno"

def draw_boxes(img_np, detections, texts):
    out = img_np.copy()
    for (_, (x1, y1, x2, y2), _), (text, lconf, _) in zip(detections, texts):
        color = (124, 58, 237)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
        label = text + "  " + str(round(lconf * 100)) + "%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)
        cv2.rectangle(out, (x1, y1 - th - 14), (x1 + tw + 12, y1), color, -1)
        cv2.putText(out, label, (x1 + 6, y1 - 6),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
    return out

with st.sidebar:
    st.markdown("### Configuracion")
    api_key_input = st.text_input(
        "API Key de Anthropic",
        type="password",
        placeholder="sk-ant-api03-..."
    )
    if api_key_input:
        os.environ["ANTHROPIC_API_KEY"] = api_key_input.strip()
    st.markdown("---")
    conf_threshold = st.slider("Confianza YOLO minima", 0.1, 0.9, 0.35, 0.05)
    show_crops = st.checkbox("Mostrar recortes", value=True)
    st.markdown("---")
    st.markdown("**Historial**")
    if st.session_state.history:
        for e in reversed(st.session_state.history[-10:]):
            pct = str(round(e["conf"] * 100))
            st.markdown(
                '<div class="hist-item">'
                '<span class="hist-plate">' + e["plate"] + '</span>'
                '<span class="hist-conf">' + pct + '%</span>'
                '</div>',
                unsafe_allow_html=True)
        if st.button("Limpiar"):
            st.session_state.history = []
            st.rerun()
    else:
        st.caption("Sin placas aun")

st.markdown(
    '<div class="hero">'
    '<h1>Detector de Placas</h1>'
    '<p>YOLO v8 + Claude Vision &middot; Talento Tech 2026</p>'
    '</div>',
    unsafe_allow_html=True)

with st.spinner("Cargando modelo YOLO..."):
    yolo_model = load_yolo()

api_key = get_api_key()
claude_client = make_client(api_key)

if not api_key:
    st.warning("Ingresa tu API Key de Anthropic en el panel izquierdo.", icon="🔑")
else:
    st.success("API Key detectada correctamente.", icon="✅")

tab_img, tab_cam, tab_video = st.tabs(["Imagen", "Camara", "Video"])

def process_image(img_pil):
    img_np = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    with st.spinner("Detectando placas con YOLO..."):
        t0 = time.time()
        detections = detect_plates(yolo_model, img_np, conf_thr=conf_threshold)
        yolo_ms = int((time.time() - t0) * 1000)

    if not detections:
        st.error("No se detectaron placas. Baja la confianza minima o usa otra foto.")
        return

    texts = []
    llm_ms = 0
    model_used = "N/A"

    if claude_client:
        with st.spinner("Extrayendo texto con Claude Vision..."):
            t1 = time.time()
            for crop_pil, _, _ in detections:
                text, lconf, model_used = ocr_claude(claude_client, crop_pil)
                texts.append((text, lconf, model_used))
                st.session_state.history.append({"plate": text, "conf": lconf})
            llm_ms = int((time.time() - t1) * 1000)
    else:
        texts = [("OCR desactivado", 0.0, "N/A")] * len(detections)

    annotated = draw_boxes(img_np, detections, texts)
    annotated_pil = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

    col_img, col_res = st.columns([1.3, 1])
    with col_img:
        st.image(annotated_pil, use_column_width=True, caption="Resultado")

    with col_res:
        avg = int(np.mean([c for _, c, _ in texts]) * 100) if texts else 0
        n = str(len(detections))
        st.markdown(
            '<div class="metrics">'
            '<div class="met"><div class="met-val">' + n + '</div><div class="met-lbl">Placas</div></div>'
            '<div class="met"><div class="met-val">' + str(avg) + '%</div><div class="met-lbl">Confianza</div></div>'
            '<div class="met"><div class="met-val">' + str(yolo_ms) + 'ms</div><div class="met-lbl">YOLO</div></div>'
            '<div class="met"><div class="met-val">' + str(llm_ms) + 'ms</div><div class="met-lbl">Claude</div></div>'
            '</div>',
            unsafe_allow_html=True)

        st.caption("Modelo: " + model_used)

        for i, ((crop_pil, _, yconf), (text, lconf, _)) in enumerate(zip(detections, texts)):
            badge = "badge-ok" if lconf >= 0.85 else "badge-low"
            yp = str(round(yconf * 100, 1))
            lp = str(round(lconf * 100, 1))
            num = str(i + 1)
            st.markdown(
                '<div class="plate-box">'
                '<div style="color:#94a3b8;font-size:0.75rem;margin-bottom:0.3rem">PLACA ' + num + '</div>'
                '<div class="plate-text">' + text + '</div><br>'
                '<span class="' + badge + '">Claude ' + lp + '%  YOLO ' + yp + '%</span>'
                '</div>',
                unsafe_allow_html=True)
            if show_crops:
                st.image(crop_pil, caption="Recorte " + num, width=240)

with tab_img:
    uploaded = st.file_uploader(
        "Sube una imagen del vehiculo",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        label_visibility="collapsed")
    if uploaded:
        process_image(Image.open(uploaded).convert("RGB"))

with tab_cam:
    cam = st.camera_input("Toma una foto")
    if cam:
        process_image(Image.open(cam).convert("RGB"))

with tab_video:
    vfile = st.file_uploader(
        "Sube un video",
        type=["mp4", "avi", "mov", "mkv"],
        label_visibility="collapsed",
        key="vid")
    fstep = st.slider("Analizar cada N fotogramas", 10, 60, 30)

    if vfile and st.button("Analizar video"):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(vfile.read())
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        found = {}
        prog = st.progress(0, text="Analizando...")
        idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if idx % fstep == 0:
                dets = detect_plates(yolo_model, frame, conf_thr=conf_threshold)
                if dets and claude_client:
                    for crop_pil, _, _ in dets:
                        text, lconf, _ = ocr_claude(claude_client, crop_pil)
                        if text not in ("ILEGIBLE", "ERROR", "SIN_MODELO", "ERROR_JSON") and lconf > 0.6:
                            found[text] = max(found.get(text, 0), lconf)
                            st.session_state.history.append({"plate": text, "conf": lconf})
            idx += 1
            prog.progress(min(idx / max(total, 1), 1.0),
                          text="Fotograma " + str(idx) + "/" + str(total))

        cap.release()
        os.unlink(tmp_path)
        prog.empty()

        if found:
            st.success(str(len(found)) + " placa(s) encontradas")
            for plate, conf in sorted(found.items(), key=lambda x: -x[1]):
                badge = "badge-ok" if conf >= 0.85 else "badge-low"
                cp = str(round(conf * 100, 1))
                st.markdown(
                    '<div class="plate-box">'
                    '<div class="plate-text">' + plate + '</div><br>'
                    '<span class="' + badge + '">Confianza: ' + cp + '%</span>'
                    '</div>',
                    unsafe_allow_html=True)
        else:
            st.warning("No se detectaron placas legibles en el video.")

st.markdown(
    '<div class="footer">Talento Tech &middot; Bootcamp IA Innovador 2026 &middot; YOLO v8 + Claude Vision</div>',
    unsafe_allow_html=True).hist-plate { color: #e2e8f0; }
.hist-conf { color: #64748b; }
.footer { text-align: center; color: #334155; font-size: 0.75rem; margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #1a1a2e; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []

# Modelos en orden de prioridad - el primero disponible se usa
MODELS = [
    "claude-haiku-4-5-20251001",
    "claude-sonnet-4-5-20250929",
    "claude-sonnet-4-6",
]

@st.cache_resource
def load_yolo():
    return YOLO("best.pt")

def get_api_key():
    # 1. Variable de entorno (ingresada en sidebar)
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if key:
        return key
    # 2. Streamlit secrets (configurado en Streamlit Cloud)
    try:
        key = st.secrets["ANTHROPIC_API_KEY"].strip()
        if key:
            return key
    except Exception:
        pass
    return ""

def make_client(api_key):
    if not api_key:
        return None
    return anthropic.Anthropic(api_key=api_key)

def pil_to_b64(img):
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
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(img_np.shape[1], x2 + pad)
            y2 = min(img_np.shape[0], y2 + pad)
            crop = img_np[y1:y2, x1:x2]
            crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            plates.append((crop_pil, (x1, y1, x2, y2), float(box.conf[0])))
    return sorted(plates, key=lambda x: -x[2])

def upscale(img, min_w=320):
    w, h = img.size
    if w < min_w:
        s = min_w / w
        img = img.resize((int(w * s), int(h * s)), Image.LANCZOS)
    return img

def ocr_claude(client, plate_img):
    plate_img = upscale(plate_img)
    b64 = pil_to_b64(plate_img)
    prompt = (
        "Look at this license plate image. "
        "Return ONLY a JSON object, no markdown, no explanation:\n"
        "{\"plate\":\"ABC123\",\"confidence\":0.97}\n"
        "Rules: plate must be uppercase alphanumeric only, no spaces or dashes. "
        "confidence is 0.0 to 1.0. "
        "If unreadable return: {\"plate\":\"ILEGIBLE\",\"confidence\":0.2}"
    )
    for model_name in MODELS:
        try:
            response = client.messages.create(
                model=model_name,
                max_tokens=100,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": b64
                            }
                        },
                        {"type": "text", "text": prompt}
                    ]
                }]
            )
            raw = response.content[0].text.strip()
            raw = re.sub(r"```[a-z]*", "", raw).strip().strip("`")
            data = json.loads(raw)
            plate = data.get("plate", "ERROR")
            conf = float(data.get("confidence", 0.0))
            return plate, conf, model_name
        except anthropic.NotFoundError:
            continue
        except json.JSONDecodeError:
            return "ERROR_JSON", 0.0, model_name
        except Exception as e:
            return "ERROR", 0.0, str(e)
    return "SIN_MODELO", 0.0, "ninguno"

def draw_boxes(img_np, detections, texts):
    out = img_np.copy()
    for (_, (x1, y1, x2, y2), _), (text, lconf, _) in zip(detections, texts):
        color = (124, 58, 237)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
        label = text + "  " + str(round(lconf * 100)) + "%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)
        cv2.rectangle(out, (x1, y1 - th - 14), (x1 + tw + 12, y1), color, -1)
        cv2.putText(out, label, (x1 + 6, y1 - 6),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
    return out

# ── Sidebar ──
with st.sidebar:
    st.markdown("### Configuracion")
    api_key_input = st.text_input(
        "API Key de Anthropic",
        type="password",
        placeholder="sk-ant-api03-...",
        help="Pega aqui tu API key de platform.claude.com"
    )
    if api_key_input:
        os.environ["ANTHROPIC_API_KEY"] = api_key_input.strip()

    st.markdown("---")
    conf_threshold = st.slider("Confianza YOLO minima", 0.1, 0.9, 0.35, 0.05)
    show_crops = st.checkbox("Mostrar recortes", value=True)
    st.markdown("---")
    st.markdown("**Historial**")
    if st.session_state.history:
        for e in reversed(st.session_state.history[-10:]):
            pct = str(round(e["conf"] * 100))
            st.markdown(
                '<div class="hist-item">'
                '<span class="hist-plate">' + e["plate"] + '</span>'
                '<span class="hist-conf">' + pct + '%</span>'
                '</div>',
                unsafe_allow_html=True)
        if st.button("Limpiar"):
            st.session_state.history = []
            st.rerun()
    else:
        st.caption("Sin placas aun")

# ── Hero ──
st.markdown(
    '<div class="hero">'
    '<h1>Detector de Placas</h1>'
    '<p>YOLO v8 + Claude Vision &middot; Talento Tech 2026</p>'
    '</div>',
    unsafe_allow_html=True)

with st.spinner("Cargando modelo YOLO..."):
    yolo_model = load_yolo()

# Obtener key y crear cliente en cada ejecucion
api_key = get_api_key()
claude_client = make_client(api_key)

if not api_key:
    st.warning("Ingresa tu API Key de Anthropic en el panel izquierdo para activar el OCR.", icon="🔑")
else:
    st.success("API Key detectada correctamente.", icon="✅")

tab_img, tab_cam, tab_video = st.tabs(["Imagen", "Camara", "Video"])

def process_image(img_pil):
    img_np = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    with st.spinner("Detectando placas con YOLO..."):
        t0 = time.time()
        detections = detect_plates(yolo_model, img_np, conf_thr=conf_threshold)
        yolo_ms = int((time.time() - t0) * 1000)

    if not detections:
        st.error("No se detectaron placas. Baja la confianza minima o usa otra foto.")
        return

    texts = []
    llm_ms = 0
    model_used = "N/A"

    if claude_client:
        with st.spinner("Extrayendo texto con Claude Vision..."):
            t1 = time.time()
            for crop_pil, _, _ in detections:
                text, lconf, model_used = ocr_claude(claude_client, crop_pil)
                texts.append((text, lconf, model_used))
                st.session_state.history.append({"plate": text, "conf": lconf})
            llm_ms = int((time.time() - t1) * 1000)
    else:
        texts = [("OCR desactivado", 0.0, "N/A")] * len(detections)

    annotated = draw_boxes(img_np, detections, texts)
    annotated_pil = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

    col_img, col_res = st.columns([1.3, 1])
    with col_img:
        st.image(annotated_pil, use_column_width=True, caption="Resultado")

    with col_res:
        avg = int(np.mean([c for _, c, _ in texts]) * 100) if texts else 0
        n = str(len(detections))
        st.markdown(
            '<div class="metrics">'
            '<div class="met"><div class="met-val">' + n + '</div><div class="met-lbl">Placas</div></div>'
            '<div class="met"><div class="met-val">' + str(avg) + '%</div><div class="met-lbl">Confianza</div></div>'
            '<div class="met"><div class="met-val">' + str(yolo_ms) + 'ms</div><div class="met-lbl">YOLO</div></div>'
            '<div class="met"><div class="met-val">' + str(llm_ms) + 'ms</div><div class="met-lbl">Claude</div></div>'
            '</div>',
            unsafe_allow_html=True)

        st.caption("Modelo usado: " + model_used)

        for i, ((crop_pil, _, yconf), (text, lconf, _)) in enumerate(zip(detections, texts)):
            badge = "badge-ok" if lconf >= 0.85 else "badge-low"
            yp = str(round(yconf * 100, 1))
            lp = str(round(lconf * 100, 1))
            num = str(i + 1)
            st.markdown(
                '<div class="plate-box">'
                '<div style="color:#94a3b8;font-size:0.75rem;margin-bottom:0.3rem">PLACA ' + num + '</div>'
                '<div class="plate-text">' + text + '</div><br>'
                '<span class="' + badge + '">Claude ' + lp + '%  YOLO ' + yp + '%</span>'
                '</div>',
                unsafe_allow_html=True)
            if show_crops:
                st.image(crop_pil, caption="Recorte " + num, width=240)

with tab_img:
    uploaded = st.file_uploader(
        "Sube una imagen del vehiculo",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        label_visibility="collapsed")
    if uploaded:
        process_image(Image.open(uploaded).convert("RGB"))

with tab_cam:
    cam = st.camera_input("Toma una foto")
    if cam:
        process_image(Image.open(cam).convert("RGB"))

with tab_video:
    vfile = st.file_uploader(
        "Sube un video (MP4, AVI, MOV)",
        type=["mp4", "avi", "mov", "mkv"],
        label_visibility="collapsed",
        key="vid")
    fstep = st.slider("Analizar cada N fotogramas", 10, 60, 30)

    if vfile and st.button("Analizar video"):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(vfile.read())
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        found = {}
        prog = st.progress(0, text="Analizando...")
        idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if idx % fstep == 0:
                dets = detect_plates(yolo_model, frame, conf_thr=conf_threshold)
                if dets and claude_client:
                    for crop_pil, _, _ in dets:
                        text, lconf, _ = ocr_claude(claude_client, crop_pil)
                        if text not in ("ILEGIBLE", "ERROR", "SIN_MODELO", "ERROR_JSON") and lconf > 0.6:
                            found[text] = max(found.get(text, 0), lconf)
                            st.session_state.history.append({"plate": text, "conf": lconf})
            idx += 1
            pct = min(idx / max(total, 1), 1.0)
            prog.progress(pct, text="Fotograma " + str(idx) + "/" + str(total))

        cap.release()
        os.unlink(tmp_path)
        prog.empty()

        if found:
            st.success(str(len(found)) + " placa(s) encontradas en el video")
            for plate, conf in sorted(found.items(), key=lambda x: -x[1]):
                badge = "badge-ok" if conf >= 0.85 else "badge-low"
                cp = str(round(conf * 100, 1))
                st.markdown(
                    '<div class="plate-box">'
                    '<div class="plate-text">' + plate + '</div><br>'
                    '<span class="' + badge + '">Confianza: ' + cp + '%</span>'
                    '</div>',
                    unsafe_allow_html=True)
        else:
            st.warning("No se detectaron placas legibles en el video.")

st.markdown(
    '<div class="footer">Talento Tech &middot; Bootcamp IA Innovador 2026 &middot; YOLO v8 + Claude Vision</div>',
    unsafe_allow_html=True).hist-plate { color: #e2e8f0; }
.hist-conf { color: #64748b; }
.footer { text-align: center; color: #334155; font-size: 0.75rem; margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #1a1a2e; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []

# Modelos en orden de prioridad - el primero disponible se usa
MODELS = [
    "claude-3-haiku-20240307",
    "claude-3-5-haiku-20241022",
    "claude-3-5-sonnet-20241022",
]

@st.cache_resource
def load_yolo():
    return YOLO("best.pt")

def get_api_key():
    # 1. Variable de entorno (ingresada en sidebar)
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if key:
        return key
    # 2. Streamlit secrets (configurado en Streamlit Cloud)
    try:
        key = st.secrets["ANTHROPIC_API_KEY"].strip()
        if key:
            return key
    except Exception:
        pass
    return ""

def make_client(api_key):
    if not api_key:
        return None
    return anthropic.Anthropic(api_key=api_key)

def pil_to_b64(img):
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
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(img_np.shape[1], x2 + pad)
            y2 = min(img_np.shape[0], y2 + pad)
            crop = img_np[y1:y2, x1:x2]
            crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            plates.append((crop_pil, (x1, y1, x2, y2), float(box.conf[0])))
    return sorted(plates, key=lambda x: -x[2])

def upscale(img, min_w=320):
    w, h = img.size
    if w < min_w:
        s = min_w / w
        img = img.resize((int(w * s), int(h * s)), Image.LANCZOS)
    return img

def ocr_claude(client, plate_img):
    plate_img = upscale(plate_img)
    b64 = pil_to_b64(plate_img)
    prompt = (
        "Look at this license plate image. "
        "Return ONLY a JSON object, no markdown, no explanation:\n"
        "{\"plate\":\"ABC123\",\"confidence\":0.97}\n"
        "Rules: plate must be uppercase alphanumeric only, no spaces or dashes. "
        "confidence is 0.0 to 1.0. "
        "If unreadable return: {\"plate\":\"ILEGIBLE\",\"confidence\":0.2}"
    )
    for model_name in MODELS:
        try:
            response = client.messages.create(
                model=model_name,
                max_tokens=100,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": b64
                            }
                        },
                        {"type": "text", "text": prompt}
                    ]
                }]
            )
            raw = response.content[0].text.strip()
            raw = re.sub(r"```[a-z]*", "", raw).strip().strip("`")
            data = json.loads(raw)
            plate = data.get("plate", "ERROR")
            conf = float(data.get("confidence", 0.0))
            return plate, conf, model_name
        except anthropic.NotFoundError:
            continue
        except json.JSONDecodeError:
            return "ERROR_JSON", 0.0, model_name
        except Exception as e:
            return "ERROR", 0.0, str(e)
    return "SIN_MODELO", 0.0, "ninguno"

def draw_boxes(img_np, detections, texts):
    out = img_np.copy()
    for (_, (x1, y1, x2, y2), _), (text, lconf, _) in zip(detections, texts):
        color = (124, 58, 237)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
        label = text + "  " + str(round(lconf * 100)) + "%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)
        cv2.rectangle(out, (x1, y1 - th - 14), (x1 + tw + 12, y1), color, -1)
        cv2.putText(out, label, (x1 + 6, y1 - 6),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
    return out

# ── Sidebar ──
with st.sidebar:
    st.markdown("### Configuracion")
    api_key_input = st.text_input(
        "API Key de Anthropic",
        type="password",
        placeholder="sk-ant-api03-...",
        help="Pega aqui tu API key de platform.claude.com"
    )
    if api_key_input:
        os.environ["ANTHROPIC_API_KEY"] = api_key_input.strip()

    st.markdown("---")
    conf_threshold = st.slider("Confianza YOLO minima", 0.1, 0.9, 0.35, 0.05)
    show_crops = st.checkbox("Mostrar recortes", value=True)
    st.markdown("---")
    st.markdown("**Historial**")
    if st.session_state.history:
        for e in reversed(st.session_state.history[-10:]):
            pct = str(round(e["conf"] * 100))
            st.markdown(
                '<div class="hist-item">'
                '<span class="hist-plate">' + e["plate"] + '</span>'
                '<span class="hist-conf">' + pct + '%</span>'
                '</div>',
                unsafe_allow_html=True)
        if st.button("Limpiar"):
            st.session_state.history = []
            st.rerun()
    else:
        st.caption("Sin placas aun")

# ── Hero ──
st.markdown(
    '<div class="hero">'
    '<h1>Detector de Placas</h1>'
    '<p>YOLO v8 + Claude Vision &middot; Talento Tech 2026</p>'
    '</div>',
    unsafe_allow_html=True)

with st.spinner("Cargando modelo YOLO..."):
    yolo_model = load_yolo()

# Obtener key y crear cliente en cada ejecucion
api_key = get_api_key()
claude_client = make_client(api_key)

if not api_key:
    st.warning("Ingresa tu API Key de Anthropic en el panel izquierdo para activar el OCR.", icon="🔑")
else:
    st.success("API Key detectada correctamente.", icon="✅")

tab_img, tab_cam, tab_video = st.tabs(["Imagen", "Camara", "Video"])

def process_image(img_pil):
    img_np = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    with st.spinner("Detectando placas con YOLO..."):
        t0 = time.time()
        detections = detect_plates(yolo_model, img_np, conf_thr=conf_threshold)
        yolo_ms = int((time.time() - t0) * 1000)

    if not detections:
        st.error("No se detectaron placas. Baja la confianza minima o usa otra foto.")
        return

    texts = []
    llm_ms = 0
    model_used = "N/A"

    if claude_client:
        with st.spinner("Extrayendo texto con Claude Vision..."):
            t1 = time.time()
            for crop_pil, _, _ in detections:
                text, lconf, model_used = ocr_claude(claude_client, crop_pil)
                texts.append((text, lconf, model_used))
                st.session_state.history.append({"plate": text, "conf": lconf})
            llm_ms = int((time.time() - t1) * 1000)
    else:
        texts = [("OCR desactivado", 0.0, "N/A")] * len(detections)

    annotated = draw_boxes(img_np, detections, texts)
    annotated_pil = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

    col_img, col_res = st.columns([1.3, 1])
    with col_img:
        st.image(annotated_pil, use_column_width=True, caption="Resultado")

    with col_res:
        avg = int(np.mean([c for _, c, _ in texts]) * 100) if texts else 0
        n = str(len(detections))
        st.markdown(
            '<div class="metrics">'
            '<div class="met"><div class="met-val">' + n + '</div><div class="met-lbl">Placas</div></div>'
            '<div class="met"><div class="met-val">' + str(avg) + '%</div><div class="met-lbl">Confianza</div></div>'
            '<div class="met"><div class="met-val">' + str(yolo_ms) + 'ms</div><div class="met-lbl">YOLO</div></div>'
            '<div class="met"><div class="met-val">' + str(llm_ms) + 'ms</div><div class="met-lbl">Claude</div></div>'
            '</div>',
            unsafe_allow_html=True)

        st.caption("Modelo usado: " + model_used)

        for i, ((crop_pil, _, yconf), (text, lconf, _)) in enumerate(zip(detections, texts)):
            badge = "badge-ok" if lconf >= 0.85 else "badge-low"
            yp = str(round(yconf * 100, 1))
            lp = str(round(lconf * 100, 1))
            num = str(i + 1)
            st.markdown(
                '<div class="plate-box">'
                '<div style="color:#94a3b8;font-size:0.75rem;margin-bottom:0.3rem">PLACA ' + num + '</div>'
                '<div class="plate-text">' + text + '</div><br>'
                '<span class="' + badge + '">Claude ' + lp + '%  YOLO ' + yp + '%</span>'
                '</div>',
                unsafe_allow_html=True)
            if show_crops:
                st.image(crop_pil, caption="Recorte " + num, width=240)

with tab_img:
    uploaded = st.file_uploader(
        "Sube una imagen del vehiculo",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        label_visibility="collapsed")
    if uploaded:
        process_image(Image.open(uploaded).convert("RGB"))

with tab_cam:
    cam = st.camera_input("Toma una foto")
    if cam:
        process_image(Image.open(cam).convert("RGB"))

with tab_video:
    vfile = st.file_uploader(
        "Sube un video (MP4, AVI, MOV)",
        type=["mp4", "avi", "mov", "mkv"],
        label_visibility="collapsed",
        key="vid")
    fstep = st.slider("Analizar cada N fotogramas", 10, 60, 30)

    if vfile and st.button("Analizar video"):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(vfile.read())
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        found = {}
        prog = st.progress(0, text="Analizando...")
        idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if idx % fstep == 0:
                dets = detect_plates(yolo_model, frame, conf_thr=conf_threshold)
                if dets and claude_client:
                    for crop_pil, _, _ in dets:
                        text, lconf, _ = ocr_claude(claude_client, crop_pil)
                        if text not in ("ILEGIBLE", "ERROR", "SIN_MODELO", "ERROR_JSON") and lconf > 0.6:
                            found[text] = max(found.get(text, 0), lconf)
                            st.session_state.history.append({"plate": text, "conf": lconf})
            idx += 1
            pct = min(idx / max(total, 1), 1.0)
            prog.progress(pct, text="Fotograma " + str(idx) + "/" + str(total))

        cap.release()
        os.unlink(tmp_path)
        prog.empty()

        if found:
            st.success(str(len(found)) + " placa(s) encontradas en el video")
            for plate, conf in sorted(found.items(), key=lambda x: -x[1]):
                badge = "badge-ok" if conf >= 0.85 else "badge-low"
                cp = str(round(conf * 100, 1))
                st.markdown(
                    '<div class="plate-box">'
                    '<div class="plate-text">' + plate + '</div><br>'
                    '<span class="' + badge + '">Confianza: ' + cp + '%</span>'
                    '</div>',
                    unsafe_allow_html=True)
        else:
            st.warning("No se detectaron placas legibles en el video.")

st.markdown(
    '<div class="footer">Talento Tech &middot; Bootcamp IA Innovador 2026 &middot; YOLO v8 + Claude Vision</div>',
    unsafe_allow_html=True)
