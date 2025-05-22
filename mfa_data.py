'''
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
'''


import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from pydub import AudioSegment, silence
import noisereduce as nr

class LJSpeechPreprocessor:
    def __init__(self,
                 dataset_path,
                 text_normalizer,
                 g2p_converter,
                 output_csv="dataset/acoustic_dataset/tts_data_LJ.csv",            
                 batch_size=32):

        self.dataset_path = dataset_path
        self.audio_folder = os.path.join(dataset_path, "wavs")
        self.metadata_path = os.path.join(dataset_path, "metadata.csv")
        self.batch_size = batch_size
        self.normalizer = text_normalizer
        self.g2p = g2p_converter

    def load_metadata(self):
        df = pd.read_csv(self.metadata_path, sep='|', usecols=['File_Name', 'Normalize_text'])
        df.dropna(inplace=True)
        print(df)
        # df=df['Normalize_text']
        return df

    def normalize_and_phonemize(self, df):
        df['Normalize_text'] = df['Normalize_text'].apply(self.normalizer.normalize_text)
        df['Normalize_text'] = [r['normalized_text'] for r in df['Normalize_text']]
        print(df['Normalize_text'].values)
        results = []

        for i in tqdm(range(0, len(df), self.batch_size), desc="Converting Text to Phonemes", unit="batch"):
            batch_texts = df['Normalize_text'].iloc[i:i + self.batch_size].tolist()
            try:
                phonemes = self.g2p.batch_predict(batch_texts)
            except Exception as e:
                print(f"Error during G2P conversion: {e}")
                phonemes = [[] for _ in batch_texts]
            results.extend(phonemes)

        df['Phoneme_text'] = results
        print(df)
        return df

    def run(self):
        print("📥 Loading metadata...")
        df = self.load_metadata()
        print("🔤 Normalizing and converting to phonemes...")
        df = self.normalize_and_phonemize(df)
        
        # 📤 Export for MFA
        export_mfa_files(df, output_dir=self.dataset_path)
    

# === Usage Example ===
if __name__ == "__main__":
    from acoustic.text_preprocess import TextNormalizer,G2PConverter
    from acoustic.mfa_export import export_mfa_files  

    preprocessor = LJSpeechPreprocessor(
        dataset_path="dataset/LJSpeech/",
        text_normalizer=TextNormalizer(),
        g2p_converter=G2PConverter("model/1/3model_cnn.keras")
    )

    preprocessor.run()
