import pandas as pd
import numpy as np
import os
import tqdm
'''
print("hello world")

folder_path='dataset/LJSpeech/'

df=pd.read_csv(folder_path+"metadata.csv",sep='|',usecols=['File_Name', 'Normalize_text'])
# print(df)

df.dropna(inplace=True)
aligner_files = [f"{fname}.csv" for fname in df['File_Name']]
print(df)
# print(aligner_files[:5])

def extract_duration():
    # for file in tqdm(aligner_files,desc="Processing audio", unit="audio"):
    for file in aligner_files[:5]:
        align_path = os.path.join(folder_path, "LJSpeech_alignments", file)
        # print(file,align_path)
        try:
            df_align = pd.read_csv(align_path, sep='\t', header=None)
            print(df_align.head())

        except Exception as e:
            print(f"Error processing {align_path}: {e}")

extract_duration()
'''

import os
import pandas as pd

folder_path = 'dataset/LJSpeech/'
hop_length = 256
sample_rate = 22050
hop_length_sec = hop_length / sample_rate

df = pd.read_csv(folder_path + "metadata.csv", sep='|', usecols=['File_Name', 'Normalize_text'])
df.dropna(inplace=True)

aligner_files = [f"{fname}.csv" for fname in df['File_Name']]

def extract_duration(max_files=None):
    durations = []

    for file in aligner_files[:max_files] if max_files else aligner_files:
        align_path = os.path.join(folder_path, "LJSpeech_alignments", file)

        try:
            df = pd.read_csv(align_path)
            phone_df = df[df["Type"] == "words"]
            phone_df = phone_df[~phone_df["Label"].isin(["spn", "[bracketed]", "<eos>"])]
            # print(df_align.head())
            phone_df['duration'] = phone_df['End'] - phone_df['Begin']
            phone_df['duration_frames'] = round(phone_df['duration'] / hop_length_sec)
            print(phone_df)

            # durations.append({
            #     'file': file.replace('.csv', ''),
            #     'phonemes': df_align['phoneme'].tolist(),
            #     'durations': df_align['duration'].tolist()
            # })

        except Exception as e:
            print(f"Error processing {align_path}: {e}")
    
    return durations

# Example: process only the first 5 files
dur_data = extract_duration(max_files=5)
# for d in dur_data:
#     print(d['file'], list(zip(d['phonemes'], d['durations'])))

