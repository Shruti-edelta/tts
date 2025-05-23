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

def extract_duration(file):
    durations = []    
    # align_path = os.path.join(folder_path, "LJSpeech_alignments", file)

    try:
        df = pd.read_csv(file)
        phone_df = df[df["Type"] == "words"]
        phone_df = phone_df[~phone_df["Label"].isin(["spn", "[bracketed]", "<eos>"])]
        # print(df_align.head())
        phone_df['duration'] = phone_df['End'] - phone_df['Begin']
        phone_df['duration_frames'] = round(phone_df['duration'] / hop_length_sec)
        dura=phone_df['duration_frames'].tolist()
        print(dura)

    except Exception as e:
        print(f"**** Error processing {file}: {e}")
    
    return durations

# Example: process only the first 5 files
# print(df)
dur_data = extract_duration('dataset/LJSpeech/LJSpeech_alignments/LJ049-0097.csv')
# dur_data = extract_duration('dataset/LJSpeech/LJSpeech_alignments/LJ050-0278.csv')
# for d in dur_data:
#     print(d['file'], list(zip(d['phonemes'], d['durations'])))

# [3.0, 3.0, 8.0, 9.0, 3.0, 10.0, 3.0, 9.0, 11.0, 3.0, 10.0, 3.0, 7.0, 6.0, 3.0, 3.0, 9.0, 5.0, 10.0, 3.0, 5.0, 7.0, 6.0, 3.0, 7.0, 5.0, 12.0, 3.0, 14.0, 8.0, 11.0, 3.0, 5.0, 6.0, 3.0, 3.0, 6.0, 5.0, 3.0, 15.0, 5.0, 4.0, 3.0, 3.0, 10.0, 13.0, 9.0, 3.0, 13.0, 6.0, 6.0, 2.0, 9.0, 4.0, 3.0, 3.0, 3.0, 6.0, 3.0, 13.0, 16.0, 3.0, 8.0, 4.0, 7.0, 5.0, 3.0, 3.0, 12.0, 9.0, 11.0, 3.0, 6.0, 5.0, 3.0, 3.0, 3.0, 3.0, 3.0, 9.0, 5.0, 5.0, 3.0, 3.0, 3.0, 6.0, 3.0, 6.0, 4.0, 5.0, 7.0, 10.0, 4.0, 9.0, 8.0, 4.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# [3.0, 3.0, 8.0, 9.0, 3.0, 10.0, 3.0, 9.0, 11.0, 3.0, 10.0, 3.0, 7.0, 6.0, 3.0, 3.0, 9.0, 5.0, 10.0, 3.0, 5.0, 7.0, 6.0, 3.0, 7.0, 5.0, 12.0, 3.0, 14.0, 8.0, 11.0, 3.0, 5.0, 6.0, 3.0, 3.0, 6.0, 5.0, 3.0, 15.0, 5.0, 4.0, 3.0, 3.0, 10.0, 13.0, 9.0, 3.0, 13.0, 6.0, 6.0, 2.0, 9.0, 4.0, 3.0, 3.0, 3.0, 6.0, 3.0, 13.0, 16.0, 3.0, 8.0, 4.0, 7.0, 5.0, 3.0, 3.0, 12.0, 9.0, 11.0, 3.0, 6.0, 5.0, 3.0, 3.0, 3.0, 3.0, 3.0, 9.0, 5.0, 5.0, 3.0, 3.0, 3.0, 6.0, 3.0, 6.0, 4.0, 5.0, 7.0, 10.0, 4.0, 9.0, 8.0, 4.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# [3.0, 3.0, 10.0, 3.0, 6.0, 3.0, 6.0, 3.0, 4.0, 4.0, 6.0, 3.0, 3.0, 8.0, 3.0, 3.0, 9.0, 3.0, 3.0, 3.0, 5.0, 3.0, 3.0, 9.0, 11.0, 3.0, 11.0, 8.0, 3.0, 10.0, 5.0, 7.0, 9.0, 8.0, 3.0, 2.0, 3.0, 8.0, 7.0, 6.0, 9.0, 5.0, 3.0, 7.0, 6.0, 6.0, 9.0, 8.0, 10.0, 3.0, 3.0, 12.0, 3.0, 9.0, 3.0, 3.0, 9.0, 3.0, 7.0, 3.0, 6.0, 9.0, 3.0, 4.0, 3.0, 21.0, 3.0, 16.0, 10.0, 5.0, 4.0, 3.0, 4.0, 5.0, 8.0, 6.0, 3.0, 9.0, 7.0, 5.0, 8.0, 9.0, 3.0, 8.0, 9.0, 3.0, 7.0, 15.0, 3.0, 12.0, 3.0, 7.0, 3.0, 3.0, 9.0, 3.0, 8.0, 5.0, 3.0, 3.0, 11.0, 3.0, 6.0, 3.0, 7.0, 3.0, 3.0]