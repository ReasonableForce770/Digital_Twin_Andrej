import streamlit as st
import json
import os
import uuid
import io
from gtts import gTTS
import speech_recognition as sr
from langchain_core.messages import HumanMessage
from agent import compiled_agent

st.set_page_config(page_title="Andrej Karpathy | Digital Twin", layout="wide")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "grounding_score" not in st.session_state:
    st.session_state.grounding_score = 0.0
if "last_retrieved" not in st.session_state:
    st.session_state.last_retrieved = []

def load_profile():
    if os.path.exists("user_profile.json"):
        with open("user_profile.json", "r") as f:
            return json.load(f)
    return {"name": "DTU ECE Student", "skill_level": "Intermediate Python", "goals": "Godot Game Dev & AI"}

profile = load_profile()


with st.sidebar:
    st.title("⚙️ Twin Diagnostics")
    st.metric(label="RAG Grounding Confidence", value=f"{st.session_state.grounding_score}%")
    
    st.subheader("🎛️ Tone Calibration")
    ui_temperature = st.slider("Deterministic vs Creative", min_value=0.0, max_value=1.0, value=0.4, step=0.1)
    
    st.divider()
    
    st.subheader("👤 Long-Term Memory")
    with st.form("profile_form"):
        new_name = st.text_input("Name", value=profile.get("name", ""))
        new_skill = st.text_input("Skill Level", value=profile.get("skill_level", ""))
        new_goals = st.text_input("Goals", value=profile.get("goals", ""))
        
        if st.form_submit_button("Update Memory File"):
            with open("user_profile.json", "w") as f:
                json.dump({"name": new_name, "skill_level": new_skill, "goals": new_goals}, f)
            st.success("JSON database rewritten!")
            st.rerun()


st.title("🧠 Andrej Karpathy")
st.caption("Digital Twin Engine | Neural Networks & Code Architecture")

def handle_prompt(user_text):
    st.session_state.messages.append({"role": "user", "content": user_text})
    
    with st.spinner("Processing graph execution logic..."):
        contextual_prompt = f"System Note: The user you are talking to is named {profile['name']}. Their skill level is {profile['skill_level']} and their goals are {profile['goals']}. \n\nUser Query: {user_text}"
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        
        response = compiled_agent.invoke({
            "messages": [HumanMessage(content=contextual_prompt)],
            "temperature": ui_temperature
        }, config=config)
        
        ai_text = response["messages"][-1].content
        st.session_state.grounding_score = response.get("grounding_score", 0.0)
        st.session_state.last_retrieved = response.get("retrieved_docs", [])
        st.session_state.messages.append({"role": "assistant", "content": ai_text})
        
        try:
            tts = gTTS(ai_text, lang='en', tld='us')
            audio_bytes = io.BytesIO()
            tts.write_to_fp(audio_bytes)
            st.session_state.latest_audio = audio_bytes.getvalue()
        except Exception:
            st.session_state.latest_audio = None
            
    st.rerun()


cols = st.columns(3)
if cols[0].button("🚀 1-Year DL Roadmap"):
    handle_prompt("Give me a 1-year roadmap to go from zero to hero in deep learning.")
if cols[1].button("🧩 Explain Backprop"):
    handle_prompt("Explain backpropagation like I am a beginner.")
if cols[2].button("🛠️ Best Coding Habit"):
    handle_prompt("What is the single most important coding habit I should build?")

st.divider()


st.write("🎙️ **Voice Transmission Node**")
if st.button("Click to Speak into Mic"):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        with st.spinner("Listening... Speak now:"):
            try:
                audio = r.listen(source, timeout=4, phrase_time_limit=5)
                text = r.recognize_google(audio)
                st.success(f"Recognized: '{text}'")
                handle_prompt(text)
            except Exception:
                st.error("Audio processing failed or timeout occurred.")

st.divider()


for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        if msg["role"] == "assistant" and i == len(st.session_state.messages) - 1:
            if "latest_audio" in st.session_state and st.session_state.latest_audio:
                st.audio(st.session_state.latest_audio, format='audio/mp3', autoplay=True)
            
            if st.session_state.last_retrieved:
                with st.expander("👀 View Retrieved Vector Context Chunks"):
                    for idx, chunk in enumerate(st.session_state.last_retrieved):
                        st.info(f"**FAISS Database Match #{idx+1}:**\n{chunk}")

if prompt := st.chat_input("Input prompt parameters..."):
    handle_prompt(prompt)