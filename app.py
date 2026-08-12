import streamlit as st
import joblib
import pandas as pd

# 1. Configure the Web Page
st.set_page_config(page_title="Proactive Intent Firewall", page_icon="🛡️", layout="centered")

st.title("🛡️ Proactive Intent-Driven LLM Firewall")
st.markdown("This middleware intercepts user prompts and classifies their semantic intent in real-time before reaching the core LLM.")

# 2. Load the Saved Model Pipeline
# @st.cache_resource ensures the model is only loaded into memory once when the app starts
@st.cache_resource
def load_model():
    return joblib.load('firewall_model.pkl')

try:
    pipeline = load_model()
except FileNotFoundError:
    st.error("Model file 'firewall_model.pkl' not found. Please ensure it is in the same directory.")
    st.stop()

# 3. Build the User Interface
st.markdown("### Simulate API Request")
user_prompt = st.text_area("User Input:", height=150, placeholder="Type a benign question or a jailbreak attempt...")

# 4. Process the Input & Run Inference
if st.button("Transmit Payload", use_container_width=True):
    if user_prompt.strip() == "":
        st.warning("Please enter a prompt to test.")
    else:
        # Calculate dynamic features required by your model
        word_count = len(user_prompt.split())
        char_count = len(user_prompt)
        
        # Construct the DataFrame exactly as your pipeline expects it
        # (Setting heuristic scores to 0 for this standalone prompt demo)
        input_data = pd.DataFrame([{
            'prompt': user_prompt,
            'roleplay_indicator': 0, 
            'system_prompt_ref': 0, 
            'jailbreak_keyword_score': 0, 
            'word_count': word_count,
            'char_count': char_count
        }])
        
        # Make the prediction
        prediction = pipeline.predict(input_data)[0]
        
        # 5. Display the Firewall Decision
        st.divider()
        st.markdown("### Firewall Diagnostics")
        
        if prediction == 1:
            st.error("🚨 **THREAT DETECTED: Adversarial Intent / Jailbreak**")
            st.write("**Action:** Payload isolated. The request was blocked and will not execute on the expensive backend LLM.")
        else:
            st.success("✅ **BENIGN INTENT**")
            st.write("**Action:** Request validated and safely forwarded to the core LLM.")
            st.info("*LLM Response: I am an AI assistant. How can I help you today?*")