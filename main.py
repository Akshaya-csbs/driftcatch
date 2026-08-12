from transformers import pipeline

# Load the brain once
classifier = pipeline("text-classification", model="protectai/deberta-v3-base-prompt-injection-v2")

# Session storage
user_sessions = {}

def process_user_prompt(user_id, prompt):
    if user_id not in user_sessions:
        user_sessions[user_id] = []
    
    # 1. Analyze
    result = classifier(prompt)
    is_malicious = result[0]['label'].lower() == 'injection' and result[0]['score'] > 0.8
    
    # 2. Track
    user_sessions[user_id].append(is_malicious)
    recent_history = user_sessions[user_id][-3:]
    
    print(f"DEBUG: Session History for {user_id}: {user_sessions[user_id]}")
    
    # 3. BLOCKING LOGIC (Change the 2 to a 1 for immediate blocking)
    if sum(recent_history) >= 1: 
        return "BLOCK: Suspicious activity detected in session."
    
    return "ALLOW: Prompt is safe."