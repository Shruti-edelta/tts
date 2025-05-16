import numpy as np
import pandas as pd
import os

# df=pd.read_csv("dataset/LJSpeech/metadata.csv")
df = pd.read_csv("dataset/LJSpeech/metadata.csv", sep='|', usecols=['File_Name', 'Normalize_text'])
df=df[:10]

print(df)
# Create output folder if needed
output_dir = 'mfa/texts'
os.makedirs(output_dir, exist_ok=True)

# Iterate over each row
for _, row in df.iterrows():
    print(row)
    file_id = str(row['File_Name'])  # ID column
    text = str(row['Normalize_text'])   # Text column

    # Define file path using ID
    file_path = os.path.join(output_dir, f"{file_id}.txt")

    # Write text to file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(text)
