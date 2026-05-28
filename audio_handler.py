"""
audio_handler.py — Smart Goviya AI
Handles Sinhala Text-to-Speech (gTTS) and Speech-to-Text (SpeechRecognition).
"""

import io
import base64
import tempfile
import os
import streamlit as st

# ─────────────────────────────────────────────
# TEXT-TO-SPEECH  (gTTS — Sinhala)
# ─────────────────────────────────────────────
def text_to_speech_sinhala(text: str) -> bytes | None:
    """
    Convert a Sinhala text string to MP3 audio bytes using gTTS.
    Returns raw MP3 bytes on success, None on failure.
    """
    try:
        from gtts import gTTS

        # gTTS language code for Sinhala is 'si'
        tts = gTTS(text=text, lang="si", slow=False)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return buf.read()
    except Exception as e:
        st.warning(f"⚠️ TTS දෝෂය (Text-to-Speech error): {e}")
        return None


def get_audio_player_html(audio_bytes: bytes) -> str:
    """
    Build an HTML <audio> autoplay element from raw MP3 bytes.
    Embeds audio as a base64 data URI so it works on Streamlit Cloud.
    """
    b64 = base64.b64encode(audio_bytes).decode("utf-8")
    return f"""
    <audio controls autoplay style="width:100%; margin-top:8px; border-radius:12px;">
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        ඔබේ browser audio සඳහා සහාය නොදක්වයි.
    </audio>
    """


# ─────────────────────────────────────────────
# SPEECH-TO-TEXT  (Google Speech Recognition)
# ─────────────────────────────────────────────
def speech_to_text_sinhala(audio_file) -> str:
    """
    Convert an uploaded audio file (WAV/MP3) to Sinhala text.
    `audio_file` is a Streamlit UploadedFile object.
    Returns the recognised text or an error string.
    """
    try:
        import speech_recognition as sr

        recognizer = sr.Recognizer()

        # Write the uploaded bytes to a temp file so SpeechRecognition can read it
        suffix = ".wav"
        if hasattr(audio_file, "name"):
            ext = os.path.splitext(audio_file.name)[-1].lower()
            suffix = ext if ext in (".wav", ".flac", ".aiff") else ".wav"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_file.read())
            tmp_path = tmp.name

        try:
            with sr.AudioFile(tmp_path) as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.3)
                audio_data = recognizer.record(source)

            # Google Web Speech API — Sinhala language code: si-LK
            text = recognizer.recognize_google(audio_data, language="si-LK")
            return text

        except sr.UnknownValueError:
            return "⚠️ කතාව හඳුනාගත නොහැකි විය. කරුණාකර නැවත උත්සාහ කරන්න."
        except sr.RequestError as e:
            return f"⚠️ කතා හඳුනාගැනීමේ සේවා දෝෂය: {e}"
        finally:
            os.unlink(tmp_path)

    except Exception as e:
        return f"⚠️ Audio processing error: {e}"


# ─────────────────────────────────────────────
# MP3 → WAV CONVERSION HELPER  (pydub fallback)
# ─────────────────────────────────────────────
def convert_mp3_to_wav(mp3_bytes: bytes) -> bytes | None:
    """
    Convert MP3 bytes to WAV bytes using pydub.
    Returns WAV bytes, or None on failure.
    """
    try:
        from pydub import AudioSegment

        mp3_buf = io.BytesIO(mp3_bytes)
        audio = AudioSegment.from_file(mp3_buf, format="mp3")
        wav_buf = io.BytesIO()
        audio.export(wav_buf, format="wav")
        wav_buf.seek(0)
        return wav_buf.read()
    except Exception as e:
        st.warning(f"Audio conversion error: {e}")
        return None


# ─────────────────────────────────────────────
# LANGUAGE DETECTION HELPER
# ─────────────────────────────────────────────
def detect_sinhala_unicode(text: str) -> bool:
    """
    Returns True if the text contains Sinhala Unicode characters.
    Sinhala Unicode range: U+0D80 – U+0DFF
    """
    return any("\u0D80" <= ch <= "\u0DFF" for ch in text)