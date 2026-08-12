import pandas as pd
from datasets import load_dataset

# Load the dataset from the Hugging Face hub
print("Downloading dataset...")
# Updated line 6 - using a widely available dataset for testing
dataset = load_dataset("deepset/prompt-injections", split="train")
# Convert the data into a structured table (DataFrame)
df = pd.DataFrame(dataset[:100])

# Save the table as a CSV file in your project folder
df.to_csv("jailbreak_data.csv", index=False)

print("Success! 'jailbreak_data.csv' has been created.")