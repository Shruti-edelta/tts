import os

# # # Specify the folder where you want to remove .txt files
# folder_path = 'dataset/libri_dataset/'

# # # Iterate through all files in the specified folder
# # for speaker_id in os.listdir(folder_path):
# #     # Check if the file ends with '.txt'
# #     if speaker_id!='.DS_Store':
# #         print(speaker_id)
# #         for file in os.listdir(os.path.join(folder_path, speaker_id)):
# #             if file.endswith('.npy'):
# #                 file_path=os.path.join(os.path.join(folder_path, speaker_id),file)
# #                 # print(file_path)
# #                 os.remove(file_path)  # Remove the file
# # #                 print(f"Removed: {file_path}")

# folder_path='dataset/mfa_data1/'
# file_names = os.listdir(folder_path+"wavs/")
# for file in file_names:
#     if file.endswith('.npy'):
#         try:
#             file_path=os.path.join(folder_path+"wavs/",file)
#             print(file_path)
#             os.remove(file_path)  # Remove the file
#             print(f"Removed: {file_path}")  # Print the file path that was removed
#         except Exception as e:
#             print(f"❌ Error normalizing {file_path}: {e}")

# with open("dataset/mfa_data1/dict.txt", "r", encoding="utf-8") as f:
#     for i, line in enumerate(f, 1):
#         # print(i,line)
#         if line.strip() == "":
#             print(f"Blank line at line {i}")
#         elif "\t" in line:
#             print(f"Tab character found at line {i}")
#         elif not line[0].isalpha():
#             print(f"Non-alphabetic starting character at line {i}: {line!r}")


import numpy as np
import noisereduce as nr
import librosa
import soundfile as sf

y, sr = librosa.load("dataset/mfa_data1/LJ001-0001.wav", sr=None)
print(librosa.get_duration(y=y,sr=sr))
# # audio, _ = librosa.effects.trim(y, top_db=20)
# # reduced_noise = nr.reduce_noise(y=y, sr=sr)
# # sf.write("check_noise.wav", reduced_noise, sr)
# # sf.write("check_t.wav", audio, sr)
# # print(y,audio,reduced_noise)
# # # sf.write("checko.wav", y, sr)
# # print(np.array_equal(y,audio))

from pydub import AudioSegment, silence

# audio = AudioSegment.from_wav("LJ001-0001.wav")
# chunks = silence.split_on_silence(audio, silence_thresh=-40, min_silence_len=500)
# clean_audio = sum(chunks)
# samples = np.array(clean_audio.get_array_of_samples()).astype(np.float32)
# samples /= np.iinfo(clean_audio.array_type).max  # Normalize to [-1, 1]
# print(clean_audio,samples)
# clean_audio.export("check_clean.wav", format="wav")
# sf.write("check_cle.wav", samples, sr)
# # print(np.array_equal(y,audio))
# # import numpy as np

def pre_emphasis(signal, coeff=0.97):
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])

def remove_silence_wav(input_path, silence_thresh=-40, min_silence_len=500):
    audio = AudioSegment.from_wav(input_path)
    print(audio)
    # print(librosa.get_duration(y=audio))
    chunks = silence.split_on_silence(audio, silence_thresh=silence_thresh, min_silence_len=min_silence_len)
    if len(chunks) == 0:
        return None, None  # No speech detected
    clean_audio = sum(chunks)
    samples = np.array(clean_audio.get_array_of_samples()).astype(np.float32)
    samples /= np.iinfo(clean_audio.array_type).max  # Normalize to [-1, 1]
    
    return samples, clean_audio.frame_rate

def audio_to_mel_spectrogram(audio_file):
    # print(librosa.get_duration(y=y,sr=sr))
    y, sr = librosa.load(audio_file)
    # y, sr = remove_silence_wav(audio_file)
    # # y,sr=librosa
    print("silence remove 1: ",y)    
    # print(librosa.get_duration(y=y,sr=sr))
    # sf.write("check_clean1.wav", y, sr)
    if y is None:
        print(f"⚠️ No speech detected in {audio_file}")
        return None

    # y, _ = librosa.effects.trim(y, top_db=30)  # Optional: refine silence
    # print("trim: ",y)
    # sf.write("check_t2.wav", y, sr)
    # print(librosa.get_duration(y=y,sr=sr))

    reduced_noise = nr.reduce_noise(y=y, sr=sr)
    print("reduce_noise: ",reduced_noise)
    sf.write("check_noice3.wav", reduced_noise, sr)
    print(librosa.get_duration(y=reduced_noise,sr=sr))

    # y = pre_emphasis(reduced_noise)
    # print("pre_emphasis: ",y)
    # sf.write("check_e4.wav", y, sr)
    # print(librosa.get_duration(y=y,sr=sr))

    audio, sr = librosa.load(audio_file,sr=22050)
    mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=80,
            fmax=8000,
            n_fft=2048,
            hop_length=256,
            win_length=2048
        )
    mel_db = librosa.power_to_db(mel_spec, ref=np.max).T  # shape: (T, 80)
    print(mel_db.shape)
    return y,sr,mel_db

audio,sr,mel_db=audio_to_mel_spectrogram("dataset/mfa_data1/LJ001-0001.wav")
print("norm_audio : ",audio,sr)
sf.write("check_norm.wav", audio, sr)
print("Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition")


# def mel_to_audio_griffin_lim(mel_db, sr=22050, n_fft=2048, hop_length=256, win_length=2048, n_mels=80, fmax=8000):
#     # Step 1: Convert dB back to power
#     mel_spec = librosa.db_to_power(mel_db.T)

#     # Step 2: Invert mel to linear
#     mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmax=fmax)
#     inv_mel_basis = np.linalg.pinv(mel_basis)
#     linear_spec = np.dot(inv_mel_basis, mel_spec)

#     # Step 3: Griffin-Lim
#     audio = librosa.griffinlim(linear_spec, hop_length=hop_length, win_length=win_length, n_iter=60)

#     # Step 4: Normalize audio
#     audio = audio / np.max(np.abs(audio) + 1e-6)

#     return audio

# # Example use
# audio = mel_to_audio_griffin_lim(mel_db)
# sf.write("reconstructed_griffin.wav", audio, sr)



# audio *= 1.2  # Boost volume by 20%
# audio = np.clip(audio, -1.0, 1.0)


# Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition

# mel= np.load("dataset/LJSpeech/wavs/LJ048-0015.npy") # LJ048-0016
# print(mel,mel.shape)
# mean= np.load("dataset/acoustic_dataset/mel_mean_std.npy")
# print("mean,std: ",mean)

# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import ast

# df = pd.read_csv('dataset/acoustic_dataset/tts_data_LJ.csv', usecols=['Phoneme_text', 'Read_npy'])

# lengths = [len(ast.literal_eval(seq)) for seq in df['Phoneme_text']]
# print(lengths)
# plt.hist(lengths, bins=50)
# plt.title("Phoneme Sequence Lengths")
# plt.xlabel("Length")
# plt.ylabel("Count")
# plt.show()

# max_input_len = max(lengths)
# print(max_input_len)  # Output: 168
# percentile_95_input = np.percentile(lengths, 95)
# print(percentile_95_input)      # 129.0


# import pandas as pd
# from sklearn.model_selection import train_test_split

# # Step 1: Load the CSV file into a DataFrame
# # df = pd.read_csv('dataset/LJSpeech/metadata.csv')  # Replace with your CSV file path
# df=pd.read_csv('dataset/LJSpeech/metadata.csv',sep='|')

# # Step 2: Split the data into training and validation sets
# train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)  # 20% for validation, 80% for training

# # Step 3: Save the datasets to text files
# train_df.to_csv('dataset/LJSpeech/train_data.txt', sep='\t', index=False, header=True)
# val_df.to_csv('dataset/LJSpeech/val_data.txt', sep='\t', index=False, header=True)

# print(f'Training data saved to "train_data.txt"')
# print(f'Validation data saved to "val_data.txt"')



# import librosa
# import librosa.display
# import matplotlib.pyplot as plt
# import numpy as np
# import soundfile as sf
# from test import audio
# from test2 import are_equal

# # y, sr = librosa.load("dataset/LJSpeech/wavs/LJ001-0001.wav", sr=22050)
# mel_db=np.load("dataset/LJSpeech/wavs/LJ001-0001.npy")
# def mel_to_audio_griffin_lim(mel_db, sr=22050, n_fft=1024, hop_length=256, win_length=1024, n_mels=80, fmax=8000):
#     # Transpose if needed (librosa expects shape [n_mels, T])
#     # if mel_db.shape[1] < mel_db.shape[0]:
#     #     mel_db = mel_db.T

#     # Convert dB back to power
#     mel_spec = librosa.db_to_power(mel_db.T)
#     # mel_inv = librosa.feature.inverse.mel_to_audio(mel_spectrogram, sr=sr)
#     # Get Mel filterbank
#     mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmax=fmax)

#     # Pseudo-inverse to recover linear spectrogram
#     inv_mel_basis = np.linalg.pinv(mel_basis)
#     linear_spec = np.dot(inv_mel_basis, mel_spec)

#     # Reconstruct audio using Griffin-Lim
#     audio = librosa.griffinlim(linear_spec, n_iter=60, hop_length=hop_length, win_length=win_length)
#     return audio

# audio=mel_to_audio_griffin_lim(mel_db)
# sf.write('rebuild.wav', audio, 22050) 
# D = librosa.stft(y, n_fft=1024, hop_length=256)
# S_db = librosa.amplitude_to_db(abs(D))

# librosa.display.specshow(S_db, sr=sr, hop_length=256, x_axis='time', y_axis='log')
# plt.colorbar()
# plt.title("STFT Spectrogram")
# plt.show()