# 📦 Streamlit Email Spam Classifier App (High-Level Polished Version)
import streamlit as st
import joblib
from datetime import datetime

# Load model and vectorizer
model = joblib.load('spam_classifier_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# ---------------- Page Configuration ----------------
st.set_page_config(page_title="Smart Email Spam Detector", page_icon="📧", layout="centered")

# ---------------- Header ----------------
st.markdown("""
    <h1 style='text-align: center; color: #004080;'>📧 Smart Email Spam Detector</h1>
    <p style='text-align: center; color: #555;'>Powered by Machine Learning (Logistic Regression)</p>
    <hr style='border: 1px solid #ccc;'>
""", unsafe_allow_html=True)

# ---------------- Input Section ----------------
st.markdown("""
<div style='background-color: #f9f9f9; padding: 20px; border-radius: 10px;'>
    <h3 style='color:#333;'>📨 Enter the Email Content Below:</h3>
""", unsafe_allow_html=True)

input_msg = st.text_area("", height=180, placeholder="E.g. You've won a free lottery! Click here to claim your reward...")

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Predict Button ----------------
if st.button("🔍 Classify Message"):
    if input_msg.strip() == "":
        st.warning("Please enter a message to classify.")
    else:
        if "http" in input_msg.lower():
            st.warning("⚠️ This message contains a URL. Be cautious of phishing even if classified as ham.")

        # Predict
        input_transformed = vectorizer.transform([input_msg])
        prediction = model.predict(input_transformed)[0]

        # Display result
        result_container = """
        <div style='margin-top: 20px; padding: 25px; border-radius: 12px; 
                    background-color: {}; color: white; font-size: 20px; text-align: center;'>
            {}
        </div>
        """

        if prediction == 1:
            st.markdown(result_container.format("#d32f2f", "🚫 SPAM DETECTED! This message is likely spam."), unsafe_allow_html=True)
        else:
            st.markdown(result_container.format("#388e3c", "✅ HAM! This message seems safe."), unsafe_allow_html=True)

# ---------------- Footer ----------------
st.markdown("""
    <hr style='border: 1px solid #ccc;'>
    <p style='text-align: center; color: #777;'>
        Built with ❤️ by <b>Prince Joshi</b> | Last Updated: {}<br>
        <small>This project is part of an ongoing portfolio in intelligent applications.</small>
    </p>
""".format(datetime.today().strftime("%d %b %Y")), unsafe_allow_html=True)