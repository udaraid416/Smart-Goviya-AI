"""
ai_engine.py — Smart Goviya AI
Handles all Google Gemini API interactions, Vision processing, and prompt engineering.
"""

import streamlit as st
import google.generativeai as genai
from PIL import Image

# ════════════════════════════════════════════════════
#  GEMINI API SETUP
# ════════════════════════════════════════════════════
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=API_KEY)
except Exception as e:
    st.error("⚠️ API Key එක හමුවූයේ නැත! කරුණාකර Streamlit Secrets වල GEMINI_API_KEY ඇතුලත් කරන්න.")

# Cloud එකට සපෝට් කරන අලුත්ම Model එක
MODEL_NAME = 'gemini-1.5-flash-latest'

# ════════════════════════════════════════════════════
#  1. SINHALA CHATBOT (Text/Voice)
# ════════════════════════════════════════════════════
def get_sinhala_response(user_input, gemini_hist):
    model = genai.GenerativeModel(MODEL_NAME)
    chat = model.start_chat(history=gemini_hist)
    
    # System Prompt 
    prompt = f"""
    You are 'Smart Goviya', an expert agricultural AI assistant in Sri Lanka. 
    Always reply in natural, polite, and friendly Sinhala language. 
    Answer the following question accurately based on agronomy and farming best practices.
    
    User Question: {user_input}
    """
    
    try:
        response = chat.send_message(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ සමාවෙන්න, AI පද්ධතියේ දෝෂයක් ඇත: {str(e)}"

# ════════════════════════════════════════════════════
#  2. IMAGE PREPROCESSING
# ════════════════════════════════════════════════════
def preprocess_image(img):
    # රූපය ඉක්මනින් process වීමට ප්‍රමාණය වෙනස් කිරීම (Resize)
    img.thumbnail((800, 800))
    return img

# ════════════════════════════════════════════════════
#  3. DISEASE DETECTION (Vision)
# ════════════════════════════════════════════════════
def analyze_crop_disease(img):
    model = genai.GenerativeModel(MODEL_NAME)
    prompt = """
    Analyze this crop/plant leaf image carefully. 
    1. Identify any visible diseases, pest attacks, or nutrient deficiencies.
    2. Provide organic/natural treatments suitable for Sri Lanka.
    3. Provide chemical treatments if necessary.
    Format the output nicely with bullet points and emojis. 
    STRICTLY answer in the Sinhala language.
    """
    try:
        response = model.generate_content([prompt, img])
        return response.text
    except Exception as e:
        return f"⚠️ රූප විශ්ලේෂණයේදී දෝෂයක් ඇතිවිය: {str(e)}"

# ════════════════════════════════════════════════════
#  4. YIELD PREDICTION (Vision)
# ════════════════════════════════════════════════════
def analyze_crop_yield(img):
    model = genai.GenerativeModel(MODEL_NAME)
    prompt = """
    Analyze this crop image for a non-destructive yield estimation.
    1. Estimate the canopy coverage and overall health.
    2. Identify the growth stage (e.g., vegetative, flowering, near-harvest).
    3. Give an estimation of the potential yield/harvest quality.
    Format with clear bullet points. 
    STRICTLY answer in the Sinhala language.
    """
    try:
        response = model.generate_content([prompt, img])
        return response.text
    except Exception as e:
        return f"⚠️ අස්වනු විශ්ලේෂණයේදී දෝෂයක් ඇතිවිය: {str(e)}"

# ════════════════════════════════════════════════════
#  5. HYDROPONICS ADVISORY
# ════════════════════════════════════════════════════
def generate_hydro_advisory(params):
    model = genai.GenerativeModel(MODEL_NAME)
    prompt = f"""
    Act as a highly scientific Hydroponics Expert.
    Analyze the following system data for {params['crop']} growing in a {params['system_type']} system in Sri Lanka:
    - pH Level: {params['ph']}
    - EC Level: {params['ec']} mS/cm
    - Ambient Temp: {params['temp']}°C
    - Humidity: {params['humidity']}%
    - Pressure: {params['pressure']} hPa
    - Solution Temp: {params['solution_temp']}°C
    - Light: {params['light_hours']} hours/day

    Provide an expert advisory report. If any value is out of the optimal range for {params['crop']}, 
    provide IMMEDIATE corrective actions (e.g., ventilation adjustments, nutrient balancing). 
    STRICTLY answer in professional yet understandable Sinhala language, using bullet points and emojis.
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ හයිඩ්‍රොපොනික්ස් විශ්ලේෂණයේදී දෝෂයක් ඇතිවිය: {str(e)}"
