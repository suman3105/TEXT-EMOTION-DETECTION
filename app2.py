import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib
import os

# =========================
# Load the trained model safely
# =========================
def load_model():
    # Get directory of current file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "text_emotion.pkl")

    if not os.path.exists(model_path):
        # Try one folder up (project root)
        model_path = os.path.join(os.path.dirname(current_dir), "text_emotion.pkl")

    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found. Expected at: {model_path}")
        raise FileNotFoundError("text_emotion.pkl not found")

    # Removed st.info notification
    return joblib.load(model_path)

pipe_lr = load_model()

# =========================
# Emojis for each emotion
# =========================
emotions_emoji_dict = {
    "anger": "😠",
    "disgust": "🤮",
    "fear": "😨😱",
    "happy": "🤗",
    "joy": "😂",
    "neutral": "😐",
    "sad": "😔",
    "sadness": "😔",
    "shame": "😳",
    "surprise": "😮"
}

# =========================
# Prediction Helpers
# =========================
def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    if hasattr(pipe_lr, "predict_proba"):
        return pipe_lr.predict_proba([docx])
    else:
        # If model does not support probability (like LinearSVC), fallback
        return np.array([[1.0 if label == pipe_lr.predict([docx])[0] else 0.0
                          for label in pipe_lr.classes_]])

# =========================
# Streamlit App
# =========================
def main():
    st.title("✨😊Text Emotion Detection ")   # Changed icon to smile
    st.subheader("Enter text and detect its emotion instantly!")

    with st.form(key='my_form'):
        raw_text = st.text_area("Type Here")
        submit_text = st.form_submit_button(label='Submit')

    if submit_text and raw_text.strip() != "":
        col1, col2 = st.columns(2)

        prediction = predict_emotions(raw_text)
        probability = get_prediction_proba(raw_text)

        with col1:
            st.success("Original Text")
            st.write(raw_text)

            st.success("Prediction")
            emoji_icon = emotions_emoji_dict.get(prediction, "❓")
            st.markdown(f"<h1 style='font-size:80px;'>{emoji_icon}</h1>", unsafe_allow_html=True)
            st.subheader(f"Prediction: {prediction}")

            st.write(f"Confidence: {np.max(probability):.2f}")

        with col2:
            st.success("Prediction Probability")
            proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
            proba_df_clean = proba_df.T.reset_index()
            proba_df_clean.columns = ["emotions", "probability"]

            fig = alt.Chart(proba_df_clean).mark_bar().encode(
                x='emotions',
                y='probability',
                color='emotions'
            )
            st.altair_chart(fig, use_container_width=True)

if __name__ == '__main__':
    main()
