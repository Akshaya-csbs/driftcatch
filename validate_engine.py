import pandas as pd
from analyze_intent import process_user_prompt, user_sessions # Import user_sessions

# 1. Load your dataset
df = pd.read_csv('jailbreak_data.csv')

# --- ADD THIS TO RESET MEMORY ---
user_sessions.clear() 
# --------------------------------

results = []
print("--- Starting Empirical Validation ---")

for index, row in df.iterrows():
    prompt = row['text']
    expected_val = row['label']

    # Run through your engine
    decision = process_user_prompt("test_user", prompt)
    
    # Determine if predicted as 'Injection' (Block) or 'Safe'
    predicted_val = 1 if "BLOCK" in decision else 0
    results.append({'text': prompt, 'Expected': expected_val, 'Predicted': predicted_val})
    
    # --- ADD THIS TO RESET AFTER EVERY PROMPT ---
    user_sessions["test_user"] = [] 

# 4. Calculate Accuracy
df_results = pd.DataFrame(results)
accuracy = (df_results['Expected'] == df_results['Predicted']).mean()

print(f"\n--- Validation Complete ---")
print(f"Total Prompts Tested: {len(df_results)}")
print(f"System Accuracy: {accuracy * 100:.2f}%")

# Optional: Print first few results to verify
print("\nSample Results:")
print(df_results.head())