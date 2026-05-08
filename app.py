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
from collections import Counter

st.set_page_config(
    page_title="Detector de Placas - Talento Tech 2026",
    page_icon="🚗",
    layout="wide",
)

css_parts = [
    "<style>",
    ".stApp{background:rgb(10,10,15);color:rgb(232,230,240);}",
    "section[data-testid='stSidebar']{background:rgb(8,8,14);}",
    ".hero{background:rgb(13,26,58);border:1px solid rgb(42,26,74);border-radius:16px;padding:2rem;margin-bottom:2rem;}",
    ".hero h1{font-size:2rem;font-weight:800;color:rgb(167,139,250);margin:0;}",
    ".hero p{color:rgb(148,163,184);margin:0;}",
    ".plate-box{background:rgb(15,23,42);border:2px solid rgb(124,58,237);border-radius:12px;padding:1rem;text-align:center;margin-bottom:1rem;}",
    ".plate-text{font-size:2rem;font-weight:700;letter-spacing:2px;color:rgb(241,245,249);background:rgb(30,41,59);border:1px solid rgb(51,65,85);border-radius:8px;padding:8px 20px;display:inline-block;margin:8px 0;font-family:monospace;}",
    ".badge-ok{display:inline-block;background:rgb(6,78,59);color:rgb(52,211,153);font-size:13px;padding:3px 12px;border-radius:99px;}",
    ".badge-low{display:inline-block;background:rgb(69,26,3);color:rgb(251,146,60);font-size:13px;padding:3px 12px;border-radius:99px;}",
    ".metrics{display:flex;gap:12px;margin:12px 0;flex-wrap:wrap;}",
    ".met{flex:1;min-width:100px;background:rgb(15,15,26);border:1px solid rgb(30,32,53);border-radius:10px;padding:12px;text-align:center;}",
    ".met-val{font-size:22px;font-weight:700;color:rgb(167,139,250);font-family:monospace;}",
    ".met-lbl{font-size:11px;color:rgb(100,116,139);}",
    ".hist-item{background:rgb(13,13,24);border:1px solid rgb(26,26,46);border-radius:6px;padding:8px;margin-bottom:6px;font-family:monospace;font-size:13px;display:flex;justify-content:space-between;}",
    ".hist-plate{color:rgb(226,232,240);}",
    ".hist-conf{color:rgb(100,116,139);}",
    ".footer{text-align:center;color:rgb(51,65,85);font-size:12px;margin-top:3rem;padding-top:1rem;border-top:1px solid rgb(26,26,46);}",
    "</style>",
]
st.markdown("".join(css_parts), unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []

MODELS = [
    "claude-sonnet-4-5-20250929",
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
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

def upscale(img, min_w=480):
    from PIL import ImageEnhance
    w, h = img.size
    if w < min_w:
        s = min_w / w
        img = img.resize((int(w * s), int(h * s)), Image.LANCZOS)
    img = ImageEnhance.Contrast(img).enhance(1.5)
    img = ImageEnhance.Sharpness(img).enhance(2.0)
    return img

def fix_plate(plate):
    if not plate or plate in ("ILEGIBLE", "ERROR", "ERROR_JSON", "SIN_MODELO"):
        return plate
    plate = plate.upper().strip().replace(" ", "").replace("-", "")
    if len(plate) != 6:
        return plate
    digit_to_letter = {"0": "O", "1": "I", "2": "Z", "5": "S", "6": "G", "8": "B"}
    letter_to_digit = {"O": "0", "I": "1", "Z": "2", "S": "5", "G": "6", "B": "8", "Q": "0"}
    result = list(plate)
    for i in range(3):
        if result[i].isdigit():
            result[i] = digit_to_letter.get(result[i], result[i])
    for i in range(3, 6):
        if result[i].isalpha():
            result[i] = letter_to_digit.get(result[i], result[i])
    return "".join(result)

def build_prompt():
    lines = [
        "You are a specialized OCR system for Colombian vehicle license plates.",
        "Analyze this plate image with extreme attention to detail.",
        "Colombian plates format: exactly 3 uppercase LETTERS then 3 DIGITS.",
        "RULE: positions 1-2-3 are ALWAYS letters, positions 4-5-6 are ALWAYS digits.",
        "CRITICAL character distinctions:",
        "- Q vs O: Q has a small diagonal tail cutting through the bottom-right of the oval.",
        "  O is a perfectly clean oval with NO tail and NO extra marks.",
        "  Q is very common in Colombian plates. If the oval has ANY mark at bottom-right it is Q.",
        "- M vs N: M has FOUR vertical strokes with two inner diagonals forming a V shape in the center.",
        "  N has THREE strokes with only one diagonal. Count the strokes carefully.",
        "- B vs 8: B has flat left side with two bumps right, 8 is symmetrical.",
        "- A vs 4: A has pointed top and crossbar, 4 has open top.",
        "Ignore city or department text below the main characters (CALI, BOGOTA, PASTO, ENVIGADO, FUNZA, MEDELLIN).",
        "Remove any spaces or dashes between the 6 characters.",
        'Return ONLY JSON, no markdown: {"plate":"KDQ742","confidence":0.97}',
        "confidence 0.0 to 1.0. ILEGIBLE only if completely unreadable.",
    ]
    return " ".join(lines)

def parse_response(raw):
    raw = re.sub(r"```[a-z]*", "", raw).strip().strip("`")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{[^}]+\}", raw)
        if match:
            try:
                data = json.loads(match.group())
            except Exception:
                return None, 0.0
        else:
            return None, 0.0
    plate = fix_plate(data.get("plate", "ERROR"))
    conf = float(data.get("confidence", 0.0))
    return plate, conf

def ocr_claude(client, plate_img):
    plate_img = upscale(plate_img)
    b64 = pil_to_b64(plate_img)
    prompt = build_prompt()

    def call_model(model_name):
        response = client.messages.create(
            model=model_name,
            max_tokens=100,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        return parse_response(response.content[0].text.strip())

    for model_name in MODELS:
        try:
            plate1, conf1 = call_model(model_name)
            if plate1 is None:
                return "ERROR_JSON", 0.0, model_name

            plate2, conf2 = call_model(model_name)
            if plate2 is None:
                return plate1, conf1, model_name

            if plate1 == plate2:
                return plate1, max(conf1, conf2), model_name

            # Discrepancia: tercera lectura de desempate
            plate3, conf3 = call_model(model_name)
            if plate3 is None:
                return plate1, conf1, model_name

            winner = Counter([plate1, plate2, plate3]).most_common(1)[0][0]
            avg_conf = (conf1 + conf2 + conf3) / 3
            return winner, avg_conf, model_name + "(3x)"

        except anthropic.NotFoundError:
            continue
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
    api_key_input = st.text_input("API Key de Anthropic", type="password", placeholder="sk-ant-api03-...")
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
                '<div class="hist-item"><span class="hist-plate">' + e["plate"] +
                '</span><span class="hist-conf">' + pct + '%</span></div>',
                unsafe_allow_html=True,
            )
        if st.button("Limpiar"):
            st.session_state.history = []
            st.rerun()
    else:
        st.caption("Sin placas aun")

st.markdown(
    '<div class="hero"><h1>Detector de Placas</h1>'
    '<p>YOLO v8 + Claude Vision - Talento Tech 2026</p></div>',
    unsafe_allow_html=True,
)

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
        with st.spinner("Extrayendo texto con Claude Vision (hasta 3 lecturas por placa)..."):
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
        st.markdown(
            '<div class="metrics">'
            '<div class="met"><div class="met-val">' + str(len(detections)) + '</div><div class="met-lbl">Placas</div></div>'
            '<div class="met"><div class="met-val">' + str(avg) + '%</div><div class="met-lbl">Confianza</div></div>'
            '<div class="met"><div class="met-val">' + str(yolo_ms) + 'ms</div><div class="met-lbl">YOLO</div></div>'
            '<div class="met"><div class="met-val">' + str(llm_ms) + 'ms</div><div class="met-lbl">Claude</div></div>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.caption("Modelo: " + model_used)
        for i, ((crop_pil, _, yconf), (text, lconf, _)) in enumerate(zip(detections, texts)):
            badge = "badge-ok" if lconf >= 0.85 else "badge-low"
            st.markdown(
                '<div class="plate-box">'
                '<div style="color:rgb(148,163,184);font-size:12px;margin-bottom:6px">PLACA ' + str(i + 1) + '</div>'
                '<div class="plate-text">' + text + '</div><br>'
                '<span class="' + badge + '">Claude ' + str(round(lconf * 100, 1)) + '%  YOLO ' + str(round(yconf * 100, 1)) + '%</span>'
                '</div>',
                unsafe_allow_html=True,
            )
            if show_crops:
                st.image(crop_pil, caption="Recorte " + str(i + 1), width=240)

with tab_img:
    uploaded = st.file_uploader(
        "Sube una imagen del vehiculo",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        label_visibility="collapsed",
    )
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
        key="vid",
    )
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
            prog.progress(min(idx / max(total, 1), 1.0), text="Fotograma " + str(idx) + "/" + str(total))

        cap.release()
        os.unlink(tmp_path)
        prog.empty()

        if found:
            st.success(str(len(found)) + " placa(s) encontradas")
            for plate, conf in sorted(found.items(), key=lambda x: -x[1]):
                badge = "badge-ok" if conf >= 0.85 else "badge-low"
                st.markdown(
                    '<div class="plate-box"><div class="plate-text">' + plate + '</div><br>'
                    '<span class="' + badge + '">Confianza: ' + str(round(conf * 100, 1)) + '%</span></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.warning("No se detectaron placas legibles en el video.")

st.markdown(
    '<div class="footer">Talento Tech - Bootcamp IA Innovador 2026 - YOLO v8 + Claude Vision</div>',
    unsafe_allow_html=True,
)
