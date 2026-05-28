"""
app.py — Smart Goviya AI  (ස්මාර්ට් ගොවියා AI)
Main Streamlit application: routing, UI, module integration.

Author  : Smart Goviya AI Project
Tech    : Streamlit · Google Gemini 1.5 · OpenCV · gTTS · Plotly
Deploy  : Streamlit Community Cloud (free tier)
"""

import io
import os
import json
import time
import base64
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from PIL import Image
from streamlit_lottie import st_lottie

# ─── local modules ───
import ai_engine as ai
import audio_handler as audio

# ════════════════════════════════════════════════════
#  PAGE CONFIG  (must be first Streamlit call)
# ════════════════════════════════════════════════════
st.set_page_config(
    page_title="ස්මාර්ට් ගොවියා AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ════════════════════════════════════════════════════
#  LOAD CSS
# ════════════════════════════════════════════════════
def _load_css():
    css_path = os.path.join(os.path.dirname(__file__), "style.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    # Also inject a small JS snippet to enable smooth anchor scrolling
    st.markdown("""
    <script>
    document.querySelectorAll('a[href^="#"]').forEach(a => {
      a.addEventListener('click', e => {
        e.preventDefault();
        document.querySelector(a.getAttribute('href'))
                .scrollIntoView({ behavior: 'smooth' });
      });
    });
    </script>
    """, unsafe_allow_html=True)


_load_css()


# ════════════════════════════════════════════════════
#  LOTTIE HELPERS
# ════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def _lottie_url(url: str) -> dict | None:
    """Fetch a Lottie JSON animation from URL."""
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


# Free Lottie animation URLs (LottieFiles CDN)
LOTTIE = {
    "farm":    _lottie_url("https://assets2.lottiefiles.com/packages/lf20_iorpbol0.json"),
    "plant":   _lottie_url("https://assets6.lottiefiles.com/packages/lf20_xvmprung.json"),
    "ai":      _lottie_url("https://assets9.lottiefiles.com/packages/lf20_fcfjwiyb.json"),
    "scan":    _lottie_url("https://assets3.lottiefiles.com/packages/lf20_qjpxuqzp.json"),
    "hydro":   _lottie_url("https://assets5.lottiefiles.com/packages/lf20_UJNc2t.json"),
}


def _safe_lottie(key: str, height: int = 200):
    """Display Lottie or fallback emoji if unavailable."""
    anim = LOTTIE.get(key)
    fallback = {"farm": "🌾", "plant": "🌱", "ai": "🤖", "scan": "🔬", "hydro": "💧"}
    if anim:
        st_lottie(anim, height=height, key=f"lottie_{key}_{id(anim)}")
    else:
        st.markdown(
            f"<div style='font-size:{height//3}px;text-align:center;'>{fallback.get(key,'🌿')}</div>",
            unsafe_allow_html=True,
        )


# ════════════════════════════════════════════════════
#  SESSION STATE INITIALISATION
# ════════════════════════════════════════════════════
def _init_state():
    defaults = {
        "chat_history":    [],   # list of {"role": str, "content": str}
        "gemini_history":  [],   # Gemini-format history for API
        "tts_audio":       None,
        "last_page":       "dashboard",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ════════════════════════════════════════════════════
#  SIDEBAR NAVIGATION
# ════════════════════════════════════════════════════
def _sidebar():
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center;padding:1rem 0 0.5rem;'>
          <div style='font-size:3rem;'>🌿</div>
          <div class='hero-title' style='font-size:1.5rem;'>ස්මාර්ට් ගොවියා</div>
          <div style='color:var(--text-muted);font-size:0.8rem;letter-spacing:0.15em;
                      text-transform:uppercase;font-family:DM Sans;'>Smart Farmer AI</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='sg-divider'></div>", unsafe_allow_html=True)

        pages = {
            "🏠  ප්‍රධාන පිටුව":           "dashboard",
            "💬  AI සහකාරයා":              "chat",
            "🔬  රෝග හඳුනාගැනීම":          "disease",
            "🌿  අස්වනු පුරෝකථනය":         "yield",
            "💧  හයිඩ්‍රොපොනික්ස්":         "hydro",
            "ℹ️  මා ගැන":                  "about",
        }

        selected = st.radio(
            "ක්‍රියාකාරී මොඩියුලය",
            list(pages.keys()),
            label_visibility="collapsed",
        )
        page = pages[selected]

        st.markdown("<div class='sg-divider'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div style='font-size:0.72rem;color:var(--text-muted);text-align:center;line-height:1.8;'>
          <div><span class='pulse-dot'></span> Gemini 1.5 Flash</div>
          <div style='margin-top:6px;'>🇱🇰 ශ්‍රී ලාංකා ගොවිතැන</div>
          <div style='margin-top:6px;opacity:0.5;'>v2.0 · 2025</div>
        </div>
        """, unsafe_allow_html=True)

    return page


# ════════════════════════════════════════════════════
#  PAGE: DASHBOARD
# ════════════════════════════════════════════════════
def page_dashboard():
    # Hero section
    col_txt, col_anim = st.columns([3, 2], gap="large")
    with col_txt:
        st.markdown("""
        <div class='welcome-banner'>
          <div class='hero-title'>ස්මාර්ට් ගොවියා AI 🌿</div>
          <div class='hero-subtitle'>ශ්‍රී ලාංකීය ගොවීන් සඳහා AI-ශක්තිය ලත් කෘෂිකාර්මික සහකාරයා</div>
          <div style='margin-top:1rem;font-size:0.85rem;color:var(--text-muted);'>
            Powered by Google Gemini 1.5 · Computer Vision · Sinhala NLP
          </div>
        </div>
        """, unsafe_allow_html=True)
    with col_anim:
        _safe_lottie("farm", height=220)

    st.markdown("<br>", unsafe_allow_html=True)

    # Feature cards
    st.markdown("<div class='sg-section-title'>🧩 ලබා ගත හැකි සේවාවන්</div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4, gap="medium")
    cards = [
        ("💬", "සිංහල AI සහකාරයා",    "හඬ හා පෙළ ආශ්‍රිත සිංහල කෘෂිකාර්මික Q&A",     "chat"),
        ("🔬", "රෝග හඳුනාගැනීම",       "ජංගම කැමරා රූපයෙන් රෝග හඳුනා ගන්න",         "disease"),
        ("📊", "අස්වනු පුරෝකථනය",      "AI Vision ශිල්පය ඔස්සේ අස්වනු ඇස්තමේන්තු",  "yield"),
        ("💧", "හයිඩ්‍රොපොනික්ස්",      "NFT/DFT පද්ධති සෙන්සර් දත්ත විශ්ලේෂණය",    "hydro"),
    ]
    for col, (icon, title, desc, _page) in zip([c1, c2, c3, c4], cards):
        with col:
            st.markdown(f"""
            <div class='glass-card' style='text-align:center;min-height:160px;'>
              <div style='font-size:2.5rem;margin-bottom:8px;'>{icon}</div>
              <div style='font-weight:700;font-family:Noto Sans Sinhala,DM Sans;
                          color:var(--lime);margin-bottom:8px;font-size:0.95rem;'>{title}</div>
              <div style='font-size:0.78rem;color:var(--text-muted);
                          font-family:Noto Sans Sinhala,DM Sans;line-height:1.6;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Quick tips section
    st.markdown("<div class='sg-section-title'>💡 ශ්‍රී ලංකාව සඳහා කෘෂිකාර්මික ඉඟි</div>", unsafe_allow_html=True)

    tips = [
        ("🌧️", "කාලගුණ සැලැස්ම",
         "නොවැම්බර් – ජනවාරි කාලය තුළ ඊසාන මෝසම් ක්‍රියාත්මකයි. "
         "වගාව හා ජලාපවාහනය ශ්‍රේෂ්ඨ ලෙස සකස් කරන්න."),
        ("🌡️", "NFT ද්‍රාවණ ව‍ෙල‍ෙ",
         "ශ්‍රී ලංකාවේ ගිම්හානයේ ද්‍රාවණ උෂ්ණත්වය 28°C ඉක්මවිය හැකිය. "
         "සෙවන සහ මාධ්‍ය ශීතකරණය ගැන සලකා බලන්න."),
        ("🌿", "ජෛව කෘෂිකාර්මිකය",
         "ට්‍රයිකෝඩර්මා, PSB, Rhizobium යන ජෛව-රසායන ද්‍රව්‍ය "
         "කිරිඇල්ල, රත්නපුර ප්‍රදේශවල ඉතා ඵලදායකයි."),
    ]

    t1, t2, t3 = st.columns(3, gap="medium")
    for col, (icon, title, body) in zip([t1, t2, t3], tips):
        with col:
            st.markdown(f"""
            <div class='glass-card' style='height:100%;'>
              <div style='display:flex;align-items:center;gap:10px;margin-bottom:10px;'>
                <span style='font-size:1.5rem;'>{icon}</span>
                <span style='font-weight:700;color:var(--gold);
                             font-family:Noto Sans Sinhala,DM Sans;font-size:0.9rem;'>{title}</span>
              </div>
              <p style='font-size:0.82rem;color:var(--text-muted);
                        font-family:Noto Sans Sinhala,DM Sans;line-height:1.7;margin:0;'>{body}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Sample chart: monthly rainfall Sri Lanka
    st.markdown("<div class='sg-section-title'>🌧️ ශ්‍රී ලංකාවේ මාසික සාමාන්‍ය වර්ෂාපතනය (mm)</div>",
                unsafe_allow_html=True)

    months = ["ජන", "පෙබ", "මාර්", "අප්‍ර", "මැයි", "ජූනි",
              "ජූලි", "අගෝ", "සැප්", "ඔක්", "නොව", "දෙසැ"]
    wet_zone  = [89, 63, 120, 262, 368, 178, 116, 109, 154, 314, 392, 189]
    dry_zone  = [60, 28, 40,  55,  80,  42,  40,  38,  66, 128, 245, 110]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="තෙත් කලාපය", x=months, y=wet_zone,
                         marker_color="rgba(45,155,94,0.75)",
                         marker_line_color="rgba(82,201,122,0.9)",
                         marker_line_width=1))
    fig.add_trace(go.Bar(name="වියළි කලාපය", x=months, y=dry_zone,
                         marker_color="rgba(212,168,67,0.65)",
                         marker_line_color="rgba(240,201,106,0.9)",
                         marker_line_width=1))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#a8c4b0", family="DM Sans"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#e8f5ee")),
        barmode="group",
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        margin=dict(l=20, r=20, t=30, b=20),
        height=320,
    )
    st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════
#  PAGE: CHAT  (Sinhala Conversational AI)
# ════════════════════════════════════════════════════
def page_chat():
    col_h, col_a = st.columns([3, 1], gap="medium")
    with col_h:
        st.markdown("<div class='hero-title' style='font-size:2rem;'>💬 සිංහල AI සහකාරයා</div>",
                    unsafe_allow_html=True)
        st.markdown("<div class='hero-subtitle'>ඔබේ ගොවිතැන් ගැටලු සිංහලෙන් අසන්න</div>",
                    unsafe_allow_html=True)
    with col_a:
        _safe_lottie("ai", height=120)

    st.markdown("<div class='sg-divider'></div>", unsafe_allow_html=True)

    # ── INPUT AREA ──────────────────────────────────
    tabs = st.tabs(["⌨️  පෙළ ආදානය", "🎤  හඬ ආදානය"])

    user_input = ""

    with tabs[0]:
        st.markdown("<br>", unsafe_allow_html=True)
        user_input_text = st.text_area(
            "ඔබේ ප්‍රශ්නය ලියන්න",
            placeholder="නිදසුන: 'කහ කොළ රෝගය සඳහා ස්වාභාවික ප්‍රතිකාරයක් තිබේද?'",
            height=90,
            key="chat_text_input",
            label_visibility="collapsed",
        )
        btn_col, tts_col = st.columns([2, 1])
        with btn_col:
            send_text = st.button("📤  ප්‍රශ්නය යවන්න", use_container_width=True,
                                  key="send_text_btn")
        with tts_col:
            tts_enabled = st.checkbox("🔊 හඬ ප්‍රතිචාරය", value=True, key="tts_toggle")

        if send_text and user_input_text.strip():
            user_input = user_input_text.strip()

    with tabs[1]:
        st.markdown("<br>", unsafe_allow_html=True)
        st.info("📁 WAV / FLAC ගොනුවක් ඇතුළත් කර AI ප්‍රශ්නය කෙරෙහි යවන්න.")
        audio_file = st.file_uploader(
            "ශ්‍රව්‍ය ගොනුව (WAV / FLAC)",
            type=["wav", "flac"],
            key="audio_upload",
            label_visibility="collapsed",
        )
        if audio_file:
            with st.spinner("🎙️ කතාව හඳුනාගනිමින්…"):
                recognised = audio.speech_to_text_sinhala(audio_file)
            if recognised and not recognised.startswith("⚠️"):
                st.success(f"🎤 හඳුනාගත් පෙළ: **{recognised}**")
                user_input = recognised
            else:
                st.warning(recognised)

    # ── PROCESS + DISPLAY ───────────────────────────
    if user_input:
        # Store user message
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Build Gemini-format history (exclude last user msg — it's sent separately)
        gemini_hist = []
        for msg in st.session_state.chat_history[:-1]:
            role = "user" if msg["role"] == "user" else "model"
            gemini_hist.append({"role": role, "parts": [msg["content"]]})

        with st.spinner("🤔 AI සිතාමතා…"):
            ai_reply = ai.get_sinhala_response(user_input, gemini_hist)

        st.session_state.chat_history.append({"role": "assistant", "content": ai_reply})

        # TTS
        if tts_enabled:
            with st.spinner("🔊 හඬ නිෂ්පාදනය කරමින්…"):
                mp3_bytes = audio.text_to_speech_sinhala(ai_reply)
                st.session_state.tts_audio = mp3_bytes

    # ── RENDER CHAT HISTORY ─────────────────────────
    st.markdown("<div class='sg-section-title' style='margin-top:1.5rem;'>💬 සංවාදය</div>",
                unsafe_allow_html=True)

    chat_container = st.container()
    with chat_container:
        if not st.session_state.chat_history:
            st.markdown("""
            <div class='glass-card' style='text-align:center;padding:2rem;'>
              <div style='font-size:3rem;margin-bottom:1rem;'>🌱</div>
              <div style='color:var(--text-muted);font-family:Noto Sans Sinhala,DM Sans;'>
                ඔබේ ප්‍රශ්නය ඉහළ ඇති කොටුවේ ලිවිය හැකිය.<br>
                <small>උදා: "ලෙටිස් වලට හොඳ pH range එක මොකද?"</small>
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.markdown(f"""
                    <div class='chat-bubble-user'>
                      <div class='chat-role-label chat-role-user'>👤 ගොවි මිතුරා</div>
                      <div style='font-family:Noto Sans Sinhala,DM Sans;'>{msg['content']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class='chat-bubble-ai'>
                      <div class='chat-role-label chat-role-ai'>🌿 ස්මාර්ට් ගොවියා AI</div>
                      <div style='font-family:Noto Sans Sinhala,DM Sans;line-height:1.8;'>{msg['content']}</div>
                    </div>
                    """, unsafe_allow_html=True)

    # TTS player
    if st.session_state.tts_audio:
        html_player = audio.get_audio_player_html(st.session_state.tts_audio)
        st.markdown(html_player, unsafe_allow_html=True)

    # Clear button
    if st.session_state.chat_history:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑️  සංවාදය මකන්න", key="clear_chat"):
            st.session_state.chat_history = []
            st.session_state.gemini_history = []
            st.session_state.tts_audio = None
            st.rerun()


# ════════════════════════════════════════════════════
#  PAGE: DISEASE DETECTION
# ════════════════════════════════════════════════════
def page_disease():
    col_h, col_a = st.columns([3, 1])
    with col_h:
        st.markdown("<div class='hero-title' style='font-size:2rem;'>🔬 රෝග හඳුනාගැනීම</div>",
                    unsafe_allow_html=True)
        st.markdown("<div class='hero-subtitle'>ජංගම දුරකථන කැමරාවෙන් ගත් රූපය upload කරන්න</div>",
                    unsafe_allow_html=True)
    with col_a:
        _safe_lottie("scan", height=120)

    st.markdown("<div class='sg-divider'></div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='glass-card' style='margin-bottom:1rem;'>
      <b style='color:var(--gold);font-family:DM Sans;'>📱 ජංගම දුරකථනය භාවිතා කරන ආකාරය</b>
      <ol style='margin:10px 0 0;padding-left:20px;
                 font-size:0.85rem;color:var(--text-muted);
                 font-family:Noto Sans Sinhala,DM Sans;line-height:2;'>
        <li>ආසාදිත ශාකයේ සමීප රූපයක් ජංගම දුරකථනයෙන් ගන්න.</li>
        <li>ජාලයට සම්බන්ධ වී, ඔබේ browser හරහා මෙම URL විවෘත කරන්න.</li>
        <li>'Browse files' ඔබා ජංගම ගැලරිය හෝ කැමරාව භාවිත කරන්න.</li>
        <li>AI විශ්ලේෂණය ස්වයංක්‍රීයව ආරම්භ වේ.</li>
      </ol>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "🌿 බෝග රූපය upload කරන්න (JPG / PNG)",
        type=["jpg", "jpeg", "png", "webp"],
        key="disease_upload",
        label_visibility="collapsed",
    )

    if uploaded:
        img = Image.open(uploaded)

        col_img, col_result = st.columns([1, 1], gap="large")
        with col_img:
            st.markdown("<div class='sg-section-title'>📸 ඔබේ රූපය</div>", unsafe_allow_html=True)
            processed = ai.preprocess_image(img)
            st.image(processed, use_column_width=True,
                     caption="AI preprocessing ✅")

            # Image stats
            arr = np.array(processed)
            st.markdown(f"""
            <div class='glass-card' style='margin-top:1rem;'>
              <div style='display:grid;grid-template-columns:1fr 1fr;gap:10px;'>
                <div class='metric-card'>
                  <span class='metric-value' style='font-size:1.2rem;'>{processed.size[0]}×{processed.size[1]}</span>
                  <span class='metric-label'>රූප විභේදනය</span>
                </div>
                <div class='metric-card'>
                  <span class='metric-value' style='font-size:1.2rem;'>
                    {'RGB' if arr.ndim == 3 else 'Gray'}
                  </span>
                  <span class='metric-label'>වර්ණ ආකාරය</span>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        with col_result:
            st.markdown("<div class='sg-section-title'>🧪 AI රෝග විශ්ලේෂණය</div>",
                        unsafe_allow_html=True)

            with st.spinner("🤖 Gemini Vision AI විශ්ලේෂණය කරමින්… (5-15 s)"):
                result = ai.analyze_crop_disease(processed)

            st.markdown(f"""
            <div class='analysis-result'>
              {result.replace(chr(10), '<br>')}
            </div>
            """, unsafe_allow_html=True)

            # TTS for result
            if st.button("🔊 ප්‍රතිඵලය සිංහලෙන් ඇසෙන්න", key="disease_tts"):
                with st.spinner("🔊 හඬ සකස් කරමින්…"):
                    mp3 = audio.text_to_speech_sinhala(result)
                if mp3:
                    st.markdown(audio.get_audio_player_html(mp3), unsafe_allow_html=True)


# ════════════════════════════════════════════════════
#  PAGE: YIELD PREDICTION
# ════════════════════════════════════════════════════
def page_yield():
    col_h, col_a = st.columns([3, 1])
    with col_h:
        st.markdown("<div class='hero-title' style='font-size:2rem;'>🌿 අස්වනු පුරෝකථනය</div>",
                    unsafe_allow_html=True)
        st.markdown("<div class='hero-subtitle'>Computer Vision ශිල්පය ඔස්සේ ශාක අස්වනු ඇස්තමේන්තු</div>",
                    unsafe_allow_html=True)
    with col_a:
        _safe_lottie("plant", height=120)

    st.markdown("<div class='sg-divider'></div>", unsafe_allow_html=True)

    # Info about methodology
    with st.expander("🔬 AI Yield Estimation ක්‍රමවේදය ගැන දැනගන්න"):
        st.markdown("""
        <div style='font-family:Noto Sans Sinhala,DM Sans;
                    font-size:0.85rem;color:var(--text-muted);line-height:1.9;'>
          <b style='color:var(--gold);'>Non-Destructive Estimation (NDE) ක්‍රමය:</b><br>
          • <b>Canopy Coverage Analysis</b>: OpenCV CLAHE enhancement + Gemini Vision ශාකපෙළ ආවරණය ඇස්කිරීම<br>
          • <b>SPAD Chlorophyll Estimate</b>: කොළ වර්ණ ඉතිහාසයෙන් Chlorophyll අගය ඇස්කිරීම<br>
          • <b>Morphometric Analysis</b>: කොළ ගණන, ව්‍යාපාරය, ශ්‍රේෂ්ඨ රළු ගණනය<br>
          • <b>Growth Stage</b>: ශාක වර්ධන අදියර (Vegetative / Near-Harvest) හඳුනාගැනීම<br><br>
          <i>⚠️ Note: AI-based estimation — calibrate with actual harvests for best accuracy.</i>
        </div>
        """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "🌱 ශාක / බෝග රූපය upload කරන්න",
        type=["jpg", "jpeg", "png", "webp"],
        key="yield_upload",
        label_visibility="collapsed",
    )

    if uploaded:
        img = Image.open(uploaded)
        processed = ai.preprocess_image(img)

        col_img, col_result = st.columns([1, 1], gap="large")

        with col_img:
            st.markdown("<div class='sg-section-title'>📸 ශාකයේ රූපය</div>", unsafe_allow_html=True)
            st.image(processed, use_column_width=True)

            # OpenCV green pixel analysis
            import cv2
            arr = np.array(processed)
            hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
            lower_g = np.array([35, 40, 40])
            upper_g = np.array([85, 255, 255])
            mask = cv2.inRange(hsv, lower_g, upper_g)
            green_pct = (np.sum(mask > 0) / mask.size) * 100

            st.markdown(f"""
            <div class='glass-card' style='margin-top:1rem;'>
              <div style='font-weight:700;color:var(--gold);margin-bottom:12px;
                          font-family:DM Sans;font-size:0.9rem;'>
                📊 OpenCV කෙළින් මැනීම
              </div>
              <div style='font-family:DM Sans;font-size:0.82rem;color:var(--text-muted);'>
                කොළ Pixel ආවරණය
              </div>
              <div class='gauge-bar-wrap'>
                <div class='gauge-bar-fill {"ok" if green_pct > 50 else "warn"}'
                     style='width:{min(green_pct, 100):.1f}%;'></div>
              </div>
              <div style='font-size:1.4rem;font-weight:700;color:var(--lime);'>
                {green_pct:.1f}%
              </div>
              <div style='font-size:0.75rem;color:var(--text-muted);margin-top:4px;'>
                {'✅ හොඳ ශාක ඝනත්වය' if green_pct > 50 else '⚠️ ශාකය තවම කොළ නොගත් ප්‍රදේශ ඇත'}
              </div>
            </div>
            """, unsafe_allow_html=True)

        with col_result:
            st.markdown("<div class='sg-section-title'>📈 AI අස්වනු ඇස්කිරීම</div>",
                        unsafe_allow_html=True)

            with st.spinner("🌿 AI Vision ශාකය විශ්ලේෂණය කරමින්…"):
                result = ai.analyze_crop_yield(processed)

            st.markdown(f"""
            <div class='analysis-result'>
              {result.replace(chr(10), '<br>')}
            </div>
            """, unsafe_allow_html=True)

            if st.button("🔊 ප්‍රතිඵලය ඇසෙන්න", key="yield_tts"):
                with st.spinner("🔊 …"):
                    mp3 = audio.text_to_speech_sinhala(result)
                if mp3:
                    st.markdown(audio.get_audio_player_html(mp3), unsafe_allow_html=True)


# ════════════════════════════════════════════════════
#  PAGE: HYDROPONICS
# ════════════════════════════════════════════════════
def page_hydro():
    col_h, col_a = st.columns([3, 1])
    with col_h:
        st.markdown("<div class='hero-title' style='font-size:2rem;'>💧 හයිඩ්‍රොපොනික්ස් AI</div>",
                    unsafe_allow_html=True)
        st.markdown("<div class='hero-subtitle'>NFT / DFT පද්ධතිය සඳහා ශාස්ත්‍රීය AI උපදෙස්</div>",
                    unsafe_allow_html=True)
    with col_a:
        _safe_lottie("hydro", height=120)

    st.markdown("<div class='sg-divider'></div>", unsafe_allow_html=True)

    col_sys, col_env = st.columns(2, gap="large")

    with col_sys:
        st.markdown("<div class='sg-section-title'>⚙️ පද්ධති සැකසුම්</div>", unsafe_allow_html=True)
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

        system_type = st.selectbox("හයිඩ්‍රොපොනික් පද්ධතිය",
                                   ["NFT (Nutrient Film Technique)",
                                    "DFT (Deep Flow Technique)",
                                    "DWC (Deep Water Culture)",
                                    "Kratky Method"],
                                   key="hydro_system")

        crop = st.selectbox("බෝගය",
                            ["Lettuce (සලාද / ලෙටිස්)",
                             "Spinach (නිවිති)",
                             "Basil (වදකහ)",
                             "Kale (කේල්)",
                             "Pak Choi (පෑෂෝ)",
                             "Tomato (තක්කාලි)",
                             "Cucumber (පිපිඤ්ඤා)"],
                            key="hydro_crop")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**💧 ද්‍රාවණ ගුණාංග**", unsafe_allow_html=False)

        ph_val = st.slider("pH මට්ටම", min_value=4.0, max_value=9.0,
                           value=6.2, step=0.1, key="ph_slider")
        ec_val = st.slider("EC (mS/cm)", min_value=0.1, max_value=5.0,
                           value=1.8, step=0.1, key="ec_slider")
        sol_temp = st.slider("ද්‍රාවණ උෂ්ණත්වය (°C)", min_value=10, max_value=35,
                             value=22, key="sol_temp_slider")

        # Visual pH gauge
        def _ph_color(ph):
            if 5.5 <= ph <= 6.5:   return ("ok",     "✅ ශ්‍රේෂ්ඨ")
            elif 6.5 < ph <= 7.0:  return ("warn",   "⚠️ ඉහළ")
            elif 5.0 <= ph < 5.5:  return ("warn",   "⚠️ පහළ")
            else:                   return ("danger", "🚨 හදිසිය")

        ph_cls, ph_lbl = _ph_color(ph_val)
        ph_pct = ((ph_val - 4) / 5) * 100
        st.markdown(f"""
        <div style='margin-top:10px;'>
          <div style='display:flex;justify-content:space-between;align-items:center;
                      font-family:DM Sans;font-size:0.8rem;margin-bottom:4px;'>
            <span style='color:var(--text-muted);'>pH {ph_val}</span>
            <span class='badge badge-{ph_cls}'>{ph_lbl}</span>
          </div>
          <div class='gauge-bar-wrap'>
            <div class='gauge-bar-fill {ph_cls}' style='width:{ph_pct:.0f}%;'></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with col_env:
        st.markdown("<div class='sg-section-title'>🌡️ BME280 සෙන්සර් දත්ත</div>",
                    unsafe_allow_html=True)
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

        st.info("📡 BME280 / DHT22 සෙන්සර් දත්ත ඇතුළු කරන්න (ශ්‍රී ලාංකාවේ සාමාන්‍ය අගයන් default ලෙස ඇත)")

        temp  = st.number_input("🌡️ උෂ්ණත්වය (°C)",   min_value=10.0, max_value=50.0,
                                value=29.5, step=0.5, key="bme_temp")
        hum   = st.number_input("💧 තෙතමනය (%)",       min_value=10.0, max_value=100.0,
                                value=72.0, step=1.0, key="bme_hum")
        press = st.number_input("🔵 වායු පීඩනය (hPa)", min_value=900.0, max_value=1100.0,
                                value=1010.0, step=0.5, key="bme_press")
        light = st.slider("☀️ දිනකට ආලෝකය (පැය)", min_value=6, max_value=20,
                          value=16, key="light_hours")

        # Mini sensor dashboard
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style='display:grid;grid-template-columns:1fr 1fr;gap:10px;'>
          <div class='metric-card'>
            <span class='metric-value'>🌡️ {temp}°C</span>
            <span class='metric-label'>{'✅ ශ්‍රේෂ්ඨ' if 18 <= temp <= 28 else '⚠️ ඉහළ'}</span>
          </div>
          <div class='metric-card'>
            <span class='metric-value'>💧 {hum:.0f}%</span>
            <span class='metric-label'>{'✅ ශ්‍රේෂ්ඨ' if 50 <= hum <= 80 else '⚠️ සකස් කරන්න'}</span>
          </div>
          <div class='metric-card'>
            <span class='metric-value'>🔵 {press:.0f}</span>
            <span class='metric-label'>hPa · {'✅ සාමාන්‍ය' if 1000 <= press <= 1020 else '⚠️'}</span>
          </div>
          <div class='metric-card'>
            <span class='metric-value'>☀️ {light}h</span>
            <span class='metric-label'>{'✅ ශ්‍රේෂ්ඨ' if 14 <= light <= 18 else '⚠️ සකස් කරන්න'}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Analyse button
    btn_col = st.columns([1, 2, 1])[1]
    with btn_col:
        analyse_btn = st.button("🧪  AI විශ්ලේෂණය ආරම්භ කරන්න",
                                use_container_width=True, key="hydro_analyse")

    if analyse_btn:
        params = {
            "system_type":  system_type,
            "crop":         crop,
            "ph":           ph_val,
            "ec":           ec_val,
            "temp":         temp,
            "humidity":     hum,
            "pressure":     press,
            "solution_temp":sol_temp,
            "light_hours":  light,
        }

        st.markdown("<div class='sg-divider'></div>", unsafe_allow_html=True)
        st.markdown("<div class='sg-section-title'>📋 AI නිර්දේශ වාර්තාව</div>",
                    unsafe_allow_html=True)

        with st.spinner("🤖 Gemini AI විශ්ලේෂණය කරමින්… ශාස්ත්‍රීය වාර්තාව සකස් කරමින්…"):
            report = ai.generate_hydro_advisory(params)

        st.markdown(f"""
        <div class='analysis-result'>
          {report.replace(chr(10), '<br>')}
        </div>
        """, unsafe_allow_html=True)

        # Radar chart: system health
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='sg-section-title'>📊 පද්ධති සෞඛ්‍ය රේඩාර්</div>",
                    unsafe_allow_html=True)

        # Normalise scores 0-100
        ph_score  = max(0, 100 - abs(ph_val - 6.0) * 30)
        ec_score  = max(0, 100 - abs(ec_val - 1.8) * 25)
        tmp_score = max(0, 100 - abs(temp - 23) * 4)
        hum_score = max(0, 100 - abs(hum - 65) * 1.5)
        lgt_score = max(0, 100 - abs(light - 16) * 6)

        radar_fig = go.Figure(go.Scatterpolar(
            r=[ph_score, ec_score, tmp_score, hum_score, lgt_score, ph_score],
            theta=["pH", "EC", "Temperature", "Humidity", "Light", "pH"],
            fill="toself",
            fillcolor="rgba(45,155,94,0.2)",
            line=dict(color="rgba(82,201,122,0.9)", width=2),
        ))
        radar_fig.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0, 100],
                                tickfont=dict(color="#a8c4b0", size=10),
                                gridcolor="rgba(255,255,255,0.08)"),
                angularaxis=dict(tickfont=dict(color="#e8f5ee", size=12),
                                 gridcolor="rgba(255,255,255,0.08)"),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e8f5ee"),
            showlegend=False,
            height=400,
            margin=dict(l=60, r=60, t=40, b=40),
        )
        st.plotly_chart(radar_fig, use_container_width=True)

        if st.button("🔊 වාර්තාව ඇසෙන්න", key="hydro_tts"):
            with st.spinner("🔊 …"):
                mp3 = audio.text_to_speech_sinhala(report)
            if mp3:
                st.markdown(audio.get_audio_player_html(mp3), unsafe_allow_html=True)


# ════════════════════════════════════════════════════
#  PAGE: ABOUT
# ════════════════════════════════════════════════════
def page_about():
    st.markdown("<div class='hero-title' style='font-size:2rem;'>ℹ️ ස්මාර්ට් ගොවියා AI ගැන</div>",
                unsafe_allow_html=True)
    st.markdown("<div class='sg-divider'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1], gap="large")
    with col1:
        st.markdown("""
        <div class='glass-card'>
          <div style='font-family:Noto Sans Sinhala,DM Sans;line-height:2;color:var(--text-main);'>
            <h3 style='color:var(--gold);font-family:Playfair Display;'>ව්‍යාපෘතිය ගැන</h3>
            <p>
              <b>ස්මාර්ට් ගොවියා AI</b> යනු ශ්‍රී ලාංකාවේ ගොවීන් සඳහා
              නිර්මිත, නොමිලේ භාවිතා කළ හැකි AI-ශක්තිය ලත් ස්මාර්ට්
              කෘෂිකාර්මික සහකාරයෙකි.
            </p>
            <h4 style='color:var(--lime);'>🛠️ ශිල්ප (Tech Stack)</h4>
            <ul style='margin-left:1.2rem;'>
              <li>🐍 Python 3.11 + Streamlit 1.35</li>
              <li>🤖 Google Gemini 1.5 Flash (Vision + Text)</li>
              <li>📷 OpenCV + Pillow (Image Processing)</li>
              <li>🔊 gTTS (Sinhala Text-to-Speech)</li>
              <li>🎤 SpeechRecognition (si-LK)</li>
              <li>📊 Plotly (Interactive Charts)</li>
            </ul>
            <h4 style='color:var(--lime);'>✨ ප්‍රධාන විශේෂාංග</h4>
            <ul style='margin-left:1.2rem;'>
              <li>💬 සිංහල Conversational AI</li>
              <li>📸 ජංගම Camera Image Upload</li>
              <li>🔬 AI රෝග හඳුනාගැනීම</li>
              <li>🌿 Non-Destructive Yield Estimation</li>
              <li>💧 NFT/DFT Hydroponics Advisory</li>
              <li>🌡️ BME280 Sensor Data Integration</li>
            </ul>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        _safe_lottie("farm", height=300)
        st.markdown("""
        <div class='glass-card' style='text-align:center;margin-top:1rem;'>
          <div style='font-size:0.8rem;color:var(--text-muted);font-family:DM Sans;line-height:1.8;'>
            <div style='font-size:1.5rem;margin-bottom:8px;'>🌿</div>
            <b style='color:var(--gold);'>Deployment</b><br>
            Streamlit Community Cloud<br>
            <span style='color:var(--lime);'>✅ 100% Free</span><br><br>
            <b style='color:var(--gold);'>API</b><br>
            Google Gemini 1.5 Flash<br>
            Free Tier: 15 req/min<br><br>
            <b style='color:var(--gold);'>License</b><br>
            MIT Open Source
          </div>
        </div>
        """, unsafe_allow_html=True)

    # Secrets setup guide
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='sg-section-title'>🔑 API Key Setup</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='glass-card'>
      <p style='font-family:DM Sans;color:var(--text-muted);font-size:0.85rem;'>
        Streamlit Cloud Secrets (<code>.streamlit/secrets.toml</code>) හෝ
        Settings → Secrets panel හි ඇතුළු කරන්න:
      </p>
    </div>
    """, unsafe_allow_html=True)
    st.code("""# .streamlit/secrets.toml
GEMINI_API_KEY = "your_google_gemini_api_key_here"
""", language="toml")
    st.markdown("""
    <div style='font-size:0.8rem;color:var(--text-muted);font-family:DM Sans;'>
      🔗 API key ලබාගන්න:
      <a href='https://aistudio.google.com/app/apikey' target='_blank'
         style='color:var(--lime);'>Google AI Studio → API Keys</a>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════
#  MAIN ROUTER
# ════════════════════════════════════════════════════
def main():
    page = _sidebar()
    if   page == "dashboard": page_dashboard()
    elif page == "chat":      page_chat()
    elif page == "disease":   page_disease()
    elif page == "yield":     page_yield()
    elif page == "hydro":     page_hydro()
    elif page == "about":     page_about()


if __name__ == "__main__":
    main()