# 🚗 Detector de Placas · Talento Tech 2026

Detección y extracción de texto de placas vehiculares usando **YOLO v8 + Claude Vision**.

## Pipeline de precisión

```
Imagen → YOLO (detecta y recorta la placa) → Claude Vision (lee el texto exacto)
```

Este enfoque dual supera al OCR tradicional (~90%+ de precisión).

## Configuración local

1. Clona el repositorio
2. Instala dependencias:
   ```bash
   pip install -r requirements.txt
   ```
3. Asegúrate de tener `best.pt` en la raíz del proyecto
4. Exporta tu API Key de Anthropic:
   ```bash
   export ANTHROPIC_API_KEY="sk-ant-..."
   ```
5. Ejecuta:
   ```bash
   streamlit run app.py
   ```
6. Abre http://localhost:8501

## Deploy en Streamlit Cloud

1. Sube el proyecto a GitHub (con `best.pt`, `app.py`, `requirements.txt`, `runtime.txt`)
2. Crea una app en [share.streamlit.io](https://share.streamlit.io)
3. En **Secrets**, agrega:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```
4. Selecciona Python 3.10 en configuración avanzada
5. Deploy

## Funcionalidades

- 📷 **Imagen**: Sube JPG/PNG y extrae placas al instante
- 📸 **Cámara**: Toma foto directamente desde el navegador
- 🎥 **Video**: Analiza MP4/AVI fotograma a fotograma
- 📋 **Historial**: Panel lateral con placas detectadas en la sesión

## Variables de entorno / Secrets

| Variable | Descripción |
|---|---|
| `ANTHROPIC_API_KEY` | API Key de Anthropic (obligatoria para OCR) |

## Notas técnicas

- El modelo YOLO detecta la región de la placa
- Claude Vision (`claude-opus-4-5`) extrae el texto con contexto semántico
- Los recortes se upscalea a mínimo 320px antes de enviarse a Claude
- Se retorna JSON estructurado: `{ plate, confidence, country_hint }`
# app-ia
