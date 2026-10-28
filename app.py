import streamlit as st
import pandas as pd
import joblib
import gzip

# 1. Page Config
st.set_page_config(page_title="AI Personality Predictor", page_icon="🧠", layout="centered")

# 2. Load the trained compressed model
@st.cache_resource
def load_model():
    with gzip.open("personality_model.pkl.gz", "rb") as f:
        return joblib.load(f)

loaded_model = load_model()

# 3. Explanation Generator
def get_explanation(mbti):
    traits = {
        "I": "Introverted - Gains energy from spending time alone or in small groups.",
        "E": "Extraverted - Gains energy from socializing and being around people.",
        "N": "Intuitive - Focuses on the big picture, future possibilities, and new ideas.",
        "S": "Sensing - Focuses on facts, practical details, and present realities.",
        "T": "Thinking - Makes decisions based on logic, objective analysis, and reason.",
        "F": "Feeling - Makes decisions based on emotions, values, and empathy.",
        "P": "Perceiving - Prefers to stay flexible, spontaneous, and open-minded.",
        "J": "Judging - Prefers structure, planning, and clear organization."
    }
    titles = {
        "INTP": "The Logician", "INTJ": "The Architect", "ENTP": "The Debater", "ENTJ": "The Commander", 
        "INFP": "The Mediator", "INFJ": "The Advocate", "ENFP": "The Campaigner", "ENFJ": "The Protagonist", 
        "ISTJ": "The Logistician", "ISFJ": "The Defender", "ESTJ": "The Executive", "ESFJ": "The Consul", 
        "ISTP": "The Virtuoso", "ISFP": "The Adventurer", "ESTP": "The Entrepreneur", "ESFP": "The Entertainer"
    }
    title = titles.get(mbti, "Personality Type")
    explanation = f"### 🌟 Predicted Type: **{mbti}** - *{title}*\n\n**What this means:**\n"
    for letter in mbti:
        explanation += f"- **{letter}**: {traits.get(letter, '')} \n"
    return explanation

# 4. Streamlit UI
st.markdown("<h1 style='text-align: center;'>🧠 AI Personality Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: gray;'>B.Tech 1st Sem AI/ML Project | By Pranav Kumar Pathak (UID: 26BCS11652)</h3>", unsafe_allow_html=True)
st.write("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Personal Details")
    age = st.number_input("Age", min_value=10, max_value=100, value=20)
    gender = st.radio("Gender", ["Male", "Female"])
    education = st.selectbox("Education", options=[0, 1], format_func=lambda x: "Undergrad (0)" if x==0 else "Grad+ (1)")
    interest = st.text_input("Main Interest", placeholder="e.g. Sports, Technology...")

with col2:
    st.subheader("📊 Personality Scores")
    intro = st.slider("Introversion vs Extraversion", 0.0, 10.0, 5.0, 0.1)
    sensing = st.slider("Sensing vs Intuition", 0.0, 10.0, 5.0, 0.1)
    thinking = st.slider("Thinking vs Feeling", 0.0, 10.0, 5.0, 0.1)
    judging = st.slider("Judging vs Perceiving", 0.0, 10.0, 5.0, 0.1)

st.write("")
if st.button("🔮 Predict Personality", type="primary", use_container_width=True):
    input_data = pd.DataFrame([{
        "Age": age, "Gender": gender, "Education": education, "Interest": interest,
        "Introversion Score": intro, "Sensing Score": sensing,
        "Thinking Score": thinking, "Judging Score": judging
    }])
    prediction = loaded_model.predict(input_data)[0]
    st.success("Prediction Complete!")
    st.markdown(get_explanation(prediction))