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
    model = joblib.load('firewall_model.pkl')
    meta = None
    try:
        meta = joblib.load('model_meta.pkl')
    except Exception:
        meta = None
    return model, meta

try:
    pipeline, meta = load_model()
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
        # Try to get a probability if available
        prob = None
        try:
            prob = pipeline.predict_proba(input_data)[0]
        except Exception:
            prob = None

        # Semantic similarity to centroids (if metadata available)
        sim_info = None
        if meta is not None and "class_centroids" in meta:
            try:
                vect = pipeline.named_steps["preprocessor"].named_transformers_["text"]
                vec = vect.transform([user_prompt])
                # compute cosine similarity with centroids
                from numpy.linalg import norm
                import numpy as _np

                def cosine(a, b):
                    da = a.ravel()
                    db = b.ravel()
                    if _np.linalg.norm(da) == 0 or _np.linalg.norm(db) == 0:
                        return 0.0
                    return float((da @ db) / (norm(da) * norm(db)))

                sims = {str(k): cosine(vec, _np.asarray(v)) for k, v in meta["class_centroids"].items()}
                sim_info = sims
            except Exception:
                sim_info = None

        # 5. Display the Firewall Decision and diagnostics
        st.divider()
        st.markdown("### Firewall Diagnostics")
        if prob is not None:
            # show probability for positive class if available
            if len(prob) == 2:
                p_pos = prob[1]
                st.write(f"Prediction probability (Adversarial/1): {p_pos:.3f}")
        if sim_info is not None:
            st.write("Semantic similarity to class centroids:")
            st.json(sim_info)

        if prediction == 1:
            st.error("🚨 **THREAT DETECTED: Adversarial Intent / Jailbreak**")
            st.write("**Action:** Payload isolated. The request was blocked and will not execute on the expensive backend LLM.")
        else:
            st.success("✅ **BENIGN INTENT**")
            st.write("**Action:** Request validated and safely forwarded to the core LLM.")
            st.info("*LLM Response: I am an AI assistant. How can I help you today?*")