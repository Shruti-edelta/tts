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
                 sample_rate=22050,
                 n_mels=80,
                 fmax=8000,
                 n_fft=2048,
                 hop_length=256,
                 win_length=2048,
                 batch_size=32,
                 max_time_frames=1024,
                 max_phoneme_length=256):

        self.dataset_path = dataset_path
        self.audio_folder = os.path.join(dataset_path, "wavs")
        self.metadata_path = os.path.join(dataset_path, "metadata.csv")
        self.output_csv = output_csv

        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.fmax = fmax
        self.max_time_frames = max_time_frames
        self.batch_size = batch_size
        self.max_phoneme_length = max_phoneme_length

        self.normalizer = text_normalizer
        self.g2p = g2p_converter

    def pre_emphasis(self, signal, coeff=0.97):
        return np.append(signal[0], signal[1:] - coeff * signal[:-1])

    def remove_silence_wav(self, input_path, silence_thresh=-40, min_silence_len=500):
        audio = AudioSegment.from_wav(input_path)
        chunks = silence.split_on_silence(audio, silence_thresh=silence_thresh, min_silence_len=min_silence_len)
        if len(chunks) == 0:
            return None, None  # No speech detected
        clean_audio = sum(chunks)
        samples = np.array(clean_audio.get_array_of_samples()).astype(np.float32)
        samples /= np.iinfo(clean_audio.array_type).max  # Normalize to [-1, 1]
        return samples, clean_audio.frame_rate

    def audio_to_mel_spectrogram(self, audio_file):
        audio, sr = self.remove_silence_wav(audio_file)
        if audio is None:
            print(f"⚠️ No speech detected in {audio_file}")
            return None

        audio, _ = librosa.effects.trim(audio, top_db=30)  # Optional: refine silence
        audio = nr.reduce_noise(y=audio, sr=sr)
        # audio = self.pre_emphasis(audio)
        
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=self.n_mels,
            fmax=self.fmax,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length
        )
        mel_db = librosa.power_to_db(mel_spec, ref=np.max).T  # shape: (T, 80)

        return mel_db

    def save_mel_spectrograms(self, file_list):
        mel_spectrograms = []
        for file in tqdm(file_list, desc="Saving Mel Spectrograms", unit="audio"):
            audio_path = os.path.join(self.audio_folder, file)
            npy_path = audio_path.replace('.wav', '.npy')
            try:
                mel = self.audio_to_mel_spectrogram(audio_path)
                mel_spectrograms.append(mel.shape[0])
                np.save(npy_path, mel)
                # print(f"Created {npy_path}")
            except Exception as e:
                print(f"❌ Error processing {audio_path}: {e}")
        self.max_time_frames=max(mel_spectrograms)
        print("self.max_time_frames: ",self.max_time_frames)

    def load_metadata(self):
        df = pd.read_csv(self.metadata_path, sep='|', usecols=['File_Name', 'Normalize_text'])
        df.dropna(inplace=True)
        # print(df)
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
        self.max_phoneme_length= max([len(seq) for seq in df['Phoneme_text'].values])
        print("max_input_length: ",self.max_phoneme_length)
        def pad_sequence(seq,input_length=self.max_phoneme_length):
            if len(seq) < input_length:
                return seq + [0] * (input_length - len(seq))
            else:
                return seq[:input_length]
            
        df['Phoneme_text'] = df['Phoneme_text'].apply(pad_sequence)
        print(" ** padded text lenghth: ",set(df['Phoneme_text'].apply(len)))
        return df

    def add_npy_paths(self, df):
        df['Read_npy'] = df['File_Name'].apply(lambda x: os.path.join(self.audio_folder, f"{x}.npy"))
        return df

    def split_and_save(self, df, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        df['Phoneme_text'] = df['Phoneme_text'].apply(str)  # make string before splitting
        texts = df['Phoneme_text'].values
        mels = df['Read_npy'].values

        texts_train, texts_temp, mel_train, mel_temp = train_test_split(texts, mels, test_size=0.2, random_state=42)
        texts_val, texts_test, mel_val, mel_test = train_test_split(texts_temp, mel_temp, test_size=0.3, random_state=42)

        pd.DataFrame({'Phoneme_text': texts_train, 'Read_npy': mel_train}).to_csv(os.path.join(output_dir, 'train.csv'), index=False)
        pd.DataFrame({'Phoneme_text': texts_val, 'Read_npy': mel_val}).to_csv(os.path.join(output_dir, 'val.csv'), index=False)
        pd.DataFrame({'Phoneme_text': texts_test, 'Read_npy': mel_test}).to_csv(os.path.join(output_dir, 'test.csv'), index=False)
        
        print("✅ Data split into train, val, and test.")
        return texts_train, mel_train,mels  # for normalization

    def compute_and_save_mel_stats(self, mel_paths, out_path="mel_mean_std.npy"):
        all_mels = []
        # for i, path in enumerate(mel_paths[:3]):
        #     mel = np.load(path)
        #     print(f"{path}: min={mel.min()}, max={mel.max()}, mean={mel.mean()}")

        for path in tqdm(mel_paths, desc="Load mel_files for global mean and std"):
            mel = np.load(path)
            all_mels.append(mel)
        all_mels_concat = np.concatenate(all_mels, axis=0)
        mean = np.mean(all_mels_concat)
        std = np.std(all_mels_concat)
        np.save(out_path, np.array([mean, std]))
        print(f"✅ Mel spectrogram mean/std saved to {out_path}")
        return mean,std

    def normalize_padd_mels(self, mel_paths, g_mean, g_std):
        print("🔊 Normalizing(std & mean) and padding mel spectrograms...")
        s=set()
        for path in tqdm(mel_paths, desc="Mel normalization"):
            try:
                mel = np.load(path).astype(np.float32)                
                # mel = np.log(mel + 1e-6)  # log compression                
                mel = (mel - g_mean) / g_std  # global normalization
                T, D = mel.shape
                if T > self.max_time_frames:
                    mel = mel[:self.max_time_frames]
                else:
                    pad_len = self.max_time_frames - T
                    mel = np.pad(mel, ((0, pad_len), (0, 0)), mode='constant')
                s.add(mel.shape[0])
                np.save(path, mel)  # overwrite with normalized mel
            except Exception as e:
                print(f"❌ Error normalizing {path}: {e}")

        print("** padded mel : ",s)

    def run(self):
        print("📥 Loading metadata...")
        df = self.load_metadata()

        print("🔤 Normalizing and converting to phonemes...")
        df = self.normalize_and_phonemize(df)
    
        print("🎧 Generating and saving mel spectrograms...")
        audio_files = [f"{fname}.wav" for fname in df['File_Name']]
        self.save_mel_spectrograms(audio_files)

        print("📁 Attaching .npy paths...")
        df = self.add_npy_paths(df)

        print(f"💾 Saving dataset to CSV: {self.output_csv}")
        os.makedirs(os.path.dirname(self.output_csv), exist_ok=True)
        df.to_csv(self.output_csv, index=False)
        # df.to_csv('dataset/acoustic_dataset/analysis.csv', index=False)

        texts_train, mel_train ,mels = self.split_and_save(df, 'dataset/acoustic_dataset')
        g_mean,g_std = self.compute_and_save_mel_stats(mel_train, out_path="dataset/acoustic_dataset/mel_mean_std.npy")

        self.normalize_padd_mels(mels,g_mean,g_std)

        print("✅ Preprocessing complete.")

# === Usage Example ===
if __name__ == "__main__":
    from acoustic.text_preprocess import TextNormalizer,G2PConverter

    preprocessor = LJSpeechPreprocessor(
        dataset_path="dataset/LJSpeech/",
        text_normalizer=TextNormalizer(),
        g2p_converter=G2PConverter("model/1/3model_cnn.keras")
    )

    preprocessor.run()



# mean,std:  [-53.679546  17.261095]
# log_mel = (norm_log_mel * std) + mean
# mel = np.exp(log_mel) - 1e-5





'''

import os
import numpy as np
import librosa
import pandas as pd
from tqdm import tqdm
from text_preprocess import TextNormalizer,G2PConverter
from a_tactron_ar import input_length
from dataset import df['Normalize_text']

# def pad_or_truncate(mel, max_len=1056):
#     T, D = mel.shape
#     if T > max_len:
#         mel = mel[:max_len, :]
#     elif T < max_len:
#         pad_width = max_len - T
#         mel = np.pad(mel, ((0, pad_width), (0, 0)), mode='constant')
#     return mel

# def normalize_mel(mel,g_mean,g_std):
#     # mean = np.mean(mel, axis=0, keepdims=True)
#     # std = np.std(mel, axis=0, keepdims=True) + 1e-6
#     g_mean=np.sum(g_mean)/len(file_names)
#     g_mean=np.sum(g_mean)/len(file_names)
#     return (mel - g_mean) / g_std

def audio_to_mel_spectrogram(audio_file,max_time_frames=1056): 
    audio, sr = librosa.load(audio_file,sr=22050)  # load at 22050 Hz consistant SR
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=80,fmax=8000,n_fft=1024,hop_length=256,win_length = 1024)     #range in 0.00000 somthing 7.4612461e-03 hop_length = 512
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)  # Convert to dB (decibels)unit scale 
    # print(mel_db.T.shape)       #(512, 80)
    return mel_db.T

def save_npyfile():
    for file in tqdm(file_names,desc="Processing audio", unit="audio"):
        audio_path = os.path.join(folder_path, "wavs", file)
        npy_path = audio_path.replace('.wav', '.npy')
        try:
            if not os.path.exists(npy_path):
                mel_spec = audio_to_mel_spectrogram(audio_path)
                np.save(npy_path, mel_spec)
                # print(f"Created {npy_path}")
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
                
folder_path='dataset/LJSpeech/'
file_names = os.listdir(folder_path+"wavs/")

df=pd.read_csv(folder_path+"metadata.csv",sep='|',usecols=['File_Name', 'Normalize_text'])
print(df)

# rows_all_null = df[df.isnull().any(axis=1)]
# print(rows_all_null)

normalizer = TextNormalizer()
g2p=G2PConverter("model/1/model_cnn.keras")

df.dropna(inplace=True,ignore_index=True)

df['Normalize_text'] = df['Normalize_text'].apply(normalizer.normalize_text)

batch_size = 32 
results = []
for i in tqdm(range(0, len(df), batch_size), desc="Processing Batches", unit="batch"):
    # print(i)
    batch_texts = df['Normalize_text'].iloc[i:i + batch_size].tolist()
    batch_phonemes = g2p.batch_predict(batch_texts)
    results.extend(batch_phonemes)
df['Phoneme_text'] = results


save_npyfile()
# audio_to_mel_spectrogram("dataset/LJSpeech/wavs/LJ048-0016.wav")

for i in range(len(df)):
    file_id = df.iloc[i, 0]
    path = os.path.join(folder_path, "wavs", f"{file_id}.npy")
    df.at[i, 'Read_npy'] = path

# # rows_all_null = df[df.isnull().any(axis=1)]
# # print(rows_all_null)

df.to_csv('dataset/acoustic_dataset/tts_data_LJ.csv',index=False)

'''

# import librosa
# import numpy as np
# import pandas as pd
# import os
# import pronouncing
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.preprocessing.text import Tokenizer

# def text_to_phonemes(text):
#     words = text.split()
#     phonemes = []
#     for word in words:
#         # Get phonemes for each word from the CMU Pronouncing Dictionary
#         word_phonemes = pronouncing.phones_for_word(word)
#         if word_phonemes:
#             phonemes.append(word_phonemes[0])  # Take the first pronunciation variant
#         else:
#             phonemes.append(word)  # If no phoneme found, keep the word as it is
#     return phonemes

# def audio_to_mel_spectrogram(audio_file):     
#     y, sr = librosa.load(audio_file,sr=None)  # load at 22050 Hz consistant SR
#     mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80,fmax=8000)     #range in 0.00000 somthing 7.4612461e-03
#     mel_spectrogram_db = librosa.power_to_db(mel_spec, ref=np.max)  # Convert to dB (decibels)unit scale 
#     mel_spectrogram_db = (mel_spectrogram_db - np.mean(mel_spectrogram_db)) / np.std(mel_spectrogram_db)       # Normalize the Mel spectrogram to a fixed range (e.g., -1 to 1)
#     return mel_spectrogram_db

# def save_npyfile():
#     for k,v in dic_w.items():
#         # print("speaker_id: ",k,len(v),v[0])
#         for file in v:
#             if not os.path.exists(file.replace('.wav','.npy')):
#                 mel_spec = audio_to_mel_spectrogram(file)
#                 # print(mel_spec.shape)
#                 np.save(file.replace('.wav','.npy'), mel_spec)  # save as .npy    
#                 print(f"create {file.replace('.wav','.npy')} npy file... ")

# def concat_speaker():
#     global final_df
#     for k,v in dic.items():
#         text_df = pd.read_csv(v, sep='\t')
#         text_df.dropna(inplace=True) 
#         text_df['Phoneme_text'] = text_df['Normalize_text'].apply(lambda x: text_to_phonemes(x))
#         text_df.drop(columns=["Original_text","Normalize_text"],inplace=True) 
#         temp_df=text_df
#         final_df=pd.concat([final_df, temp_df],ignore_index = True)

# # folder_path = 'dataset/libri_dataset/'
# dic_w={}
# dic={}
# # dic_t={}

# for speaker_id in os.listdir(folder_path):
#     if speaker_id!='.DS_Store':
#         # print(speaker_id)
#         for file in os.listdir(os.path.join(folder_path, speaker_id)):
#             # if file.endswith(".txt"):
#             #     file_path=os.path.join(os.path.join(folder_path, speaker_id),file)
#             #     dic_t.setdefault(speaker_id, []).append(file_path)
#             if file.endswith(".wav"):
#                 file_path=os.path.join(os.path.join(folder_path, speaker_id),file)
#                 dic_w.setdefault(speaker_id, []).append(file_path)
#             elif file.endswith(".trans.tsv"):
#                 file_path=os.path.join(os.path.join(folder_path, speaker_id),file)
#                 dic[speaker_id]=file_path

# # for speaker_id in os.listdir(folder_path):
# #     if speaker_id!='.DS_Store':
# #         # print(f'{speaker_id} = {len(dic_t[speaker_id])}')
# #         print(f'{speaker_id} = {len(dic_w[speaker_id])}')

# save_npyfile()
# final_df = pd.DataFrame(columns=['File_Name', 'Phoneme_text','Read_npy'])
# concat_speaker()

# for i in range(len(final_df)):
#     file_id=final_df.iloc[i, 0]
#     path=folder_path+file_id.split('_')[0]+'/'+file_id+'.npy'
#     final_df.at[i, 'Read_npy'] = path

# final_df.to_csv('train.csv',index=False)


# # texts=final_df['Phoneme_text'].values
# # token=tokenizer_text_word().tokenizer_text()
# # sequences = token.texts_to_sequences(texts)
# # padded_text = pad_sequences(sequences, maxlen=600, padding='post') 
# # padded_text=padded_text.tolist()
# # final_df['padded_texts'] = final_df['padded_texts'].astype(object)
# # for i in range(len(final_df)):
# #     final_df.at[i, 'padded_texts'] = padded_text[i]

# # print(final_df)
# # s_row=final_df.iloc[1075]
# # print(s_row)

# # path="dataset/libri_dataset/32/32_4137_000002_000000.npy"
# # arr=np.load(path)
# # print(type(arr))
# # print(arr)
# # print(arr.shape)  # (1, 32, 32)



