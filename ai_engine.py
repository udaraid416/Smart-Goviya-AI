"""
ai_engine.py — Smart Goviya AI
Handles all Google Gemini API calls: chat, vision, hydroponics analysis.
"""

import io
import base64
import streamlit as st
import google.generativeai as genai
from PIL import Image

# ─────────────────────────────────────────────
# Initialise Gemini client (key from st.secrets)
# ─────────────────────────────────────────────
def _get_model(vision: bool = False):
    """Return a configured GenerativeModel."""
    api_key = st.secrets.get("GEMINI_API_KEY", "")
    if not api_key:
        st.error("⚠️  GEMINI_API_KEY not found in Streamlit secrets.")
        st.stop()
    genai.configure(api_key=api_key)
    model_name = "gemini-1.5-flash" if not vision else "gemini-1.5-flash"
    return genai.GenerativeModel(model_name)


# ─────────────────────────────────────────────
# SYSTEM PROMPTS
# ─────────────────────────────────────────────
SINHALA_AGRO_PROMPT = """
ඔබ "ස්මාර්ට් ගොවියා AI" — ශ්‍රී ලාංකීය ගොවීන් සඳහා නිර්මිත, ඉතා හිතකාමී
කෘෂිකාර්මික විශේෂඥ AI සහකාරයෙකි.

නිතර නිතර කළ යුතු දේ:
• සෑම විටම ස්වාභාවික, සරල සිංහලෙන් පිළිතුරු දෙන්න.
• ශ්‍රී ලාංකීය ජලාශ ගොවිතැන, NFT/DFT හයිඩ්‍රොපොනික්ස්, වී, එළවළු සහ
  පලතුරු ගැන ගොඩාක් දැනුම ඇති කෙනෙකු ලෙස ක්‍රියා කරන්න.
• ප්‍රයෝගික, ක්‍රියාත්මක කළ හැකි උපදෙස් ලබා දෙන්න.
• ශ්‍රී ලාංකාවේ දේශගුණ කලාප (නිල් හා රතු කලාප), කාලගුණය, ජල ආශ්‍රිත
  ගැටලු ගැන දැනුවත් ලෙස කතා කරන්න.
• ගොවියා "ගොවි මිතුරා" ලෙස ආමන්ත්‍රණය කරන්න.
• දිග පිළිතුරු වෙනුවට කෙටි, පිරිසිදු ලිෝකෝ-ලිස්ට් ආකාරයෙන් ලිවිය හැකිය.

කිසිදා නොකළ යුතු:
• ඉංග්‍රීසි පිළිතුරු ලබා නොදෙන්න (දෘෂ්‍ය/රූප නාම හෝ ශාස්ත්‍රීය යෙදුම්
  ඉංග්‍රීසියෙන් ලිවිය හැකි, සිංහල ඇතුළෙන්).
• අනවශ්‍ය ඉංජිනේරු කතා කිරීම.
"""

DISEASE_VISION_PROMPT = """
You are an expert plant pathologist and agronomist specializing in Sri Lankan crops.
Analyze the provided crop image thoroughly and respond STRICTLY in Sinhala.

Your response must include:
1. **රෝගය / ගැටලුව** (Disease/Problem): Name (Sinhala + scientific name in brackets)
2. **රෝග ලක්ෂණ** (Symptoms): What you observe in the image
3. **හේතු** (Causes): Pathogen, pest, or abiotic cause
4. **ප්‍රතිකාර** (Treatment):
   - ජෛව ප්‍රතිකාර (Bio-treatment)
   - රසායනික ප්‍රතිකාර (Chemical, with Sri Lanka-available products)
5. **වැළැක්වීම** (Prevention): Future steps
6. **ඉතා වැදගත් ඇඟවීම** (Urgency): Low / Medium / High in Sinhala

If the image is not a plant/crop, politely say so in Sinhala.
"""

YIELD_VISION_PROMPT = """
You are an expert agricultural computer vision analyst.
Analyze this crop image (likely lettuce or leafy vegetable) for yield prediction.
Respond STRICTLY in Sinhala.

Your analysis must cover:
1. **බෝගය හඳුනාගැනීම** (Crop identification)
2. **කැනොපි ආවරණය** (Canopy coverage estimate as %)
3. **ශාක සෞඛ්‍ය ලකුණු** (Health indicators — color, density, leaf count estimate)
4. **අස්වනු පුරෝකථනය** (Yield prediction):
   - Estimated fresh weight per plant (grams)
   - Estimated days to harvest
   - Yield confidence: Low/Medium/High
5. **නිර්දේශ** (Recommendations to improve yield)

Use observable visual cues: canopy spread, leaf color (SPAD estimate), leaf tip burn, compactness.
Be scientifically rigorous yet farmer-friendly in Sinhala.
"""


# ─────────────────────────────────────────────
# CHAT  (Sinhala Conversational AI)
# ─────────────────────────────────────────────
def get_sinhala_response(user_message: str, history: list[dict]) -> str:
    """
    Send a user message + conversation history to Gemini and get a Sinhala reply.
    history is a list of {"role": "user"|"model", "parts": [str]} dicts.
    """
    try:
        model = _get_model()
        chat = model.start_chat(history=history)
        # Prepend system context on the first call via a user turn
        if not history:
            full_msg = SINHALA_AGRO_PROMPT + "\n\nගොවි මිතුරාගේ ප්‍රශ්නය:\n" + user_message
        else:
            full_msg = user_message
        response = chat.send_message(full_msg)
        return response.text
    except Exception as e:
        return f"⚠️ AI ප්‍රතිචාරය ලබාගැනීමේ දෝෂයක් ඇතිවිය: {e}"


# ─────────────────────────────────────────────
# VISION — Disease Detection
# ─────────────────────────────────────────────
def analyze_crop_disease(image: Image.Image) -> str:
    """Run Gemini Vision on a PIL Image to detect crop disease."""
    try:
        model = _get_model(vision=True)
        response = model.generate_content([DISEASE_VISION_PROMPT, image])
        return response.text
    except Exception as e:
        return f"⚠️ රෝග විශ්ලේෂණයේ දෝෂයක් ඇතිවිය: {e}"


# ─────────────────────────────────────────────
# VISION — Yield Prediction
# ─────────────────────────────────────────────
def analyze_crop_yield(image: Image.Image) -> str:
    """Run Gemini Vision on a PIL Image to predict crop yield."""
    try:
        model = _get_model(vision=True)
        response = model.generate_content([YIELD_VISION_PROMPT, image])
        return response.text
    except Exception as e:
        return f"⚠️ අස්වනු පුරෝකථනයේ දෝෂයක් ඇතිවිය: {e}"


# ─────────────────────────────────────────────
# HYDROPONICS — Scientific Advisory
# ─────────────────────────────────────────────
def generate_hydro_advisory(params: dict) -> str:
    """
    Generate a scientific corrective-action report for NFT/DFT system.
    params keys: system_type, crop, ph, ec, temp, humidity, pressure,
                 solution_temp, dissolved_oxygen, light_hours
    """
    prompt = f"""
ඔබ ශ්‍රී ලාංකාවේ හයිඩ්‍රොපොනික් ගොවිතැන් විශේෂඥයෙකි.
පහත සෙන්සර් දත්ත අනුව IMMEDIATE corrective actions ලබා දෙන්න.
සිංහලෙන් පිළිතුරු ලිවිය යුතුය.

## පද්ධති දත්ත
- පද්ධති වර්ගය     : {params.get('system_type', 'NFT')}
- බෝගය             : {params.get('crop', 'Lettuce / කොළ සලාද')}
- pH මට්ටම          : {params.get('ph', 6.0)}
- EC (mS/cm)        : {params.get('ec', 1.8)}
- වාතාවරණ උෂ්ණත්වය : {params.get('temp', 28)}°C
- සාපේක්ෂ තෙතමනය  : {params.get('humidity', 70)}%
- වායු පීඩනය        : {params.get('pressure', 1013)} hPa
- පෝෂ්‍ය ද්‍රාවණ       : {params.get('solution_temp', 22)}°C
- දිනකට ආලෝක පැය  : {params.get('light_hours', 16)} h

## ඔබේ වාර්තාව ඇතුළත් විය යුතු:
1. **pH තත්ත්‍ව ඇගයීම** — ලොකු අකුරෙන් "✅ හොඳයි", "⚠️ සකස් කරන්න", "🚨 හදිසි"
2. **EC/Nutrient ඇගයීම** — pH ලෙසම
3. **දේශගුණ ඇගයීම** — කෘෂිකාර්මික කළමනාකරණ උපදෙස්
4. **වහාම ගත යුතු ක්‍රියාමාර්ග** (Numbered steps)
5. **දිගුකාලීන නිර්දේශ** (2–3 points)
6. **ලබාගත හැකි ශ්‍රී ලාංකාවේ නිෂ්පාදන** (නිෂ්පාදන නාම සහිතව)

ශාස්ත්‍රීය නිරවද්‍යතාව + ගොවි-හිතකාමී භාෂාව අවශ්‍යයි.
"""
    try:
        model = _get_model()
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ හයිඩ්‍රොපොනික් විශ්ලේෂණ දෝෂය: {e}"


# ─────────────────────────────────────────────
# QUICK IMAGE PRE-PROCESSING  (OpenCV / Pillow)
# ─────────────────────────────────────────────
def preprocess_image(image: Image.Image, max_dim: int = 1024) -> Image.Image:
    """
    Resize large mobile images, enhance contrast slightly for better AI analysis.
    Returns a processed PIL Image.
    """
    import cv2
    import numpy as np

    # Convert PIL → numpy → OpenCV BGR
    img_np = np.array(image.convert("RGB"))
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Resize keeping aspect ratio
    h, w = img_cv.shape[:2]
    scale = min(max_dim / h, max_dim / w, 1.0)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        img_cv = cv2.resize(img_cv, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # Mild CLAHE for contrast enhancement on L-channel
    lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    img_cv = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Back to PIL RGB
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)


def image_to_base64(image: Image.Image) -> str:
    """Helper: PIL Image → base64 string (for HTML display)."""
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()