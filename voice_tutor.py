from streamlit_mic_recorder import mic_recorder
import speech_recognition as sr
import streamlit as st
import tempfile

def voice_to_text():

    audio = mic_recorder(
        start_prompt="🎤 Start recording",
        stop_prompt="⏹ Stop recording",
        key="recorder"
    )

    if audio:

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(audio["bytes"])
            audio_path = f.name

        r = sr.Recognizer()

        with sr.AudioFile(audio_path) as source:
            audio_data = r.record(source)

        try:
            text = r.recognize_google(audio_data)
            return text
        except:
            st.warning("Could not understand audio")
            return None