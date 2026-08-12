from datasets import load_dataset
import pandas as pd
from analyze_intent import process_user_prompt, user_sessions

print("--- Downloading Dataset from Hugging Face ---")
# Load an open security dataset for validation
dataset = load_dataset("S-Labs/prompt-injection-dataset")

# Convert the validation or test split to a pandas DataFrame for evaluation
df = pd.DataFrame(dataset['validation'])

print(df.head())
print("Unique labels in dataset:", df['label'].unique())

# Save the Hugging Face validation split locally as a CSV
df.to_csv("hf_validation_data.csv", index=False)

# Reset global memory tracking
user_sessions.clear()

results = []
print("--- Starting Large-Scale Empirical Validation ---")

for index, row in df.iterrows():
    prompt = row['text']
    expected_val = row['label']  # 0 for benign, 1 for malicious injection

    # Run Through your security engine
    decision = process_user_prompt("test_user", prompt)
    
    # Print the first 5 prompts to inspect what your engine is returning
    if index < 5:
        print(f"Prompt: {prompt[:50]}... | Expected: {expected_val} | Decision: {decision}")

    # Map decision output to a binary prediction format by checking if "BLOCK" is in the decision text
    predicted_val = 1 if "BLOCK" in str(decision) else 0
    results.append({'text': prompt, 'Expected': expected_val, 'Predicted': predicted_val})

    # Clear session history after each loop to ensure an independent evaluation
    user_sessions["test_user"] = []

# Convert results into a DataFrame to compute metrics
df_results = pd.DataFrame(results)

# Calculate Confusion Matrix parameters
tp = ((df_results['Expected'] == 1) & (df_results['Predicted'] == 1)).sum()
fp = ((df_results['Expected'] == 0) & (df_results['Predicted'] == 1)).sum()
fn = ((df_results['Expected'] == 1) & (df_results['Predicted'] == 0)).sum()
tn = ((df_results['Expected'] == 0) & (df_results['Predicted'] == 0)).sum()

# Compute performance metrics safely avoiding division by zero
accuracy = (tp + tn) / len(df_results) if len(df_results) > 0 else 0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print("\n--- Validation Complete ---")
print(f"Total Prompts Evaluated : {len(df_results)}")
print(f"System Accuracy         : {accuracy * 100:.2f}%")
print(f"Precision               : {precision:.4f}")
print(f"Recall                  : {recall:.4f}")
print(f"F1-Score                : {f1:.4f}")