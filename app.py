import gradio as gr
import pandas as pd
import joblib
import gzip

# 1. Load the trained compressed model
with gzip.open("personality_model.pkl.gz", "rb") as f:
    loaded_model = joblib.load(f)

# 2. Explanation Generator
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

# 3. Prediction Function
def predict_personality(age, gender, education, interest, intro, sensing, thinking, judging):
    input_data = pd.DataFrame([{
        "Age": age, "Gender": gender, "Education": education, "Interest": interest,
        "Introversion Score": intro, "Sensing Score": sensing,
        "Thinking Score": thinking, "Judging Score": judging
    }])
    prediction = loaded_model.predict(input_data)[0]
    return get_explanation(prediction)

# 4. Create UI using Gradio
custom_theme = gr.themes.Soft(primary_hue="indigo", secondary_hue="blue")
with gr.Blocks(theme=custom_theme) as interface:
    gr.Markdown(
        """
        <div style="text-align: center;">
            <h1>🧠 AI Personality Predictor</h1>
            <h3>B.Tech 1st Sem AI/ML Project | By Pranav Kumar Pathak (UID: 26BCS11652)</h3>
            <p>Enter the details below to predict the Myers-Briggs Type Indicator (MBTI).</p>
        </div>
        """
    )
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 👤 Personal Details")
            age = gr.Number(label="Age", value=20)
            gender = gr.Radio(choices=["Male", "Female"], label="Gender", value="Male")
            education = gr.Radio(choices=[0, 1], label="Education (0=Undergrad, 1=Grad+)", value=0)
            interest = gr.Textbox(label="Main Interest", placeholder="e.g. Sports, Technology...")
        with gr.Column():
            gr.Markdown("### 📊 Personality Scores (0 to 10)")
            intro = gr.Slider(minimum=0, maximum=10, step=0.1, label="Introversion vs Extraversion")
            sensing = gr.Slider(minimum=0, maximum=10, step=0.1, label="Sensing vs Intuition")
            thinking = gr.Slider(minimum=0, maximum=10, step=0.1, label="Thinking vs Feeling")
            judging = gr.Slider(minimum=0, maximum=10, step=0.1, label="Judging vs Perceiving")
            
    submit_btn = gr.Button("🔮 Predict Personality", variant="primary")
    result = gr.Markdown()

    submit_btn.click(
        fn=predict_personality,
        inputs=[age, gender, education, interest, intro, sensing, thinking, judging],
        outputs=result
    )

interface.launch()