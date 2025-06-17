import numpy as np
import os
import librosa
import soundfile as sf
import noisereduce as nr
from pydub import AudioSegment, silence
from pydub.silence import split_on_silence
import pandas as pd
import matplotlib.pyplot as plt
# import noisereduce as nr # Uncomment if you're using noisereduce

# Configuration (adjust as needed)
# TARGET_SAMPLE_RATE = 22050 # Hz
# PROCESSED_AUDIO_DIR = 'my_tts_dataset/processed_audio'
# ... other paths and silence trimming parameters as defined previously ...

# Make sure the directory exists
# os.makedirs(PROCESSED_AUDIO_DIR, exist_ok=True)

# Assuming you have a list of raw audio paths and their transcriptions
# For each raw_audio_path in your dataset:
#     1. Load audio with pydub (for easy format handling and basic ops)
#        audio = AudioSegment.from_file(raw_audio_path)

#     2. (Optional) Noise Reduction (using noisereduce, example only)
#        # Convert pydub AudioSegment to numpy array
#        audio_np = np.array(audio.get_array_of_samples())
#        if audio.sample_width == 2: # 16-bit
#            audio_np = audio_np.astype(np.float32) / (2**15)
#        elif audio.sample_width == 4: # 32-bit
#            audio_np = audio_np.astype(np.float32) / (2**31)
#        reduced_noise = nr.reduce_noise(y=audio_np, sr=audio.frame_rate)
#        # Convert back to pydub AudioSegment (careful with sample width)
#        audio = AudioSegment(
#            (reduced_noise * (2**15)).astype(np.int16).tobytes(),
#            frame_rate=audio.frame_rate,
#            sample_width=2,
#            channels=audio.channels
#        )


#     3. Convert to Mono
#        if audio.channels > 1:
#            audio = audio.set_channels(1)

#     4. Normalize Volume
#        audio = audio.normalize(-1.0) # Peak normalization to -1 dBFS

#     5. Silence Trimming
#        chunks = split_on_silence(audio, min_silence_len=..., silence_thresh=..., keep_silence=...)
#        if not chunks: continue # Skip if no speech detected
#        processed_audio_segment = AudioSegment.empty()
#        for chunk in chunks:
#            processed_audio_segment += chunk

#     6. Save as WAV at Target Sample Rate (using temp file and librosa for quality resampling)
#        temp_path = os.path.join(PROCESSED_AUDIO_DIR, "temp_" + os.path.basename(raw_audio_path))
#        processed_audio_segment.export(temp_path, format="wav")
#        y, sr = librosa.load(temp_path, sr=TARGET_SAMPLE_RATE)
#        sf.write(os.path.join(PROCESSED_AUDIO_DIR, os.path.basename(raw_audio_path)), y, TARGET_SAMPLE_RATE)
#        os.remove(temp_path) # Clean up temp file

#     Update your metadata file with the path to the processed audio file.

def mel_to_audio_griffin_lim(mel_db, mean, std, sr=22050, n_fft=2048, hop_length=256, win_length=2048, n_mels=80, fmax=8000):

    log_mel = (mel_db * std) + mean

    mel_spec = librosa.db_to_power(log_mel.T)  # shape: (80, T)

    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmax=fmax)
    inv_mel_basis = np.linalg.pinv(mel_basis)
    linear_spec = np.dot(inv_mel_basis, mel_spec)  # shape: (1025, T)

    audio = librosa.griffinlim(linear_spec, hop_length=hop_length, win_length=win_length, n_iter=60)

    audio = audio / (np.max(np.abs(audio)) + 1e-6)

    print(librosa.get_duration(y=audio))
    return audio

# def audio_to_mel_spectrogram(audio_file):     
#     y, sr = librosa.load(audio_file)  # load at 22050 Hz consistant SR
#     print(y,sr)
#     print(librosa.get_duration(y=y))
#     mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80,fmax=8000)     #range in 0.00000 somthing 7.4612461e-03
#     mel_spectrogram_db = librosa.power_to_db(mel_spec, ref=np.max)  # Convert to dB (decibels)unit scale 
#     # mel_spectrogram_db = (mel_spectrogram_db - np.mean(mel_spectrogram_db)) / np.std(mel_spectrogram_db)       # Normalize the Mel spectrogram to a fixed range (e.g., -1 to 1)
#     return mel_spectrogram_db

def pre_emphasis(signal, coeff=0.97):
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])

def remove_silence_wav(input_path, silence_thresh=-40, min_silence_len=500):
    audio = AudioSegment.from_wav(input_path)
    chunks = silence.split_on_silence(audio, silence_thresh=silence_thresh, min_silence_len=min_silence_len)
    if len(chunks) == 0:
        return None, None  # No speech detected
    clean_audio = sum(chunks)
    samples = np.array(clean_audio.get_array_of_samples()).astype(np.float32)
    samples /= np.iinfo(clean_audio.array_type).max  # Normalize to [-1, 1]
    return samples, clean_audio.frame_rate

def audio_to_mel_spectrogram(audio_file,mean,std):
    # audio, sr =remove_silence_wav(audio_file)
    # print(audio,sr)
    # audio,sr=librosa.load(audio_file,sr=22050)
    # print(librosa.get_duration(y=audio))
    
    # if audio is None:
    #     print(f"⚠️ No speech detected in {audio_file}")
    #     return None

    # # audio, _ = librosa.effects.trim(audio, top_db=30)  # Optional: refine silence
    # # print(librosa.get_duration(y=audio))
    # # audio = self.pre_emphasis(audio)
        
    # # audio, sr = librosa.load(audio_file,sr=22050)
    # # audio = nr.reduce_noise(y=audio, sr=sr)
    # # print(librosa.get_duration(y=audio))
    # mel_spec = librosa.feature.melspectrogram(
    #     y=audio,
    #     sr=22050,
    #     n_mels=80,
    #     fmax=8000,
    #     n_fft=2048,
    #     hop_length=256,
    #     win_length=2048
    # )
    # mel_db = librosa.power_to_db(mel_spec, ref=np.max).T  # shape: (T, 80)
    # mel = (mel_db - mean) / std
    # T, D = mel.shape
    # if T > 865:
    #     mel = mel[:865]
    # else:
    #     pad_len = 865 - T
    #     mel = np.pad(mel, ((0, pad_len), (0, 0)), mode='constant')

    # audio = AudioSegment.from_file(audio_file)
    # audio_np = np.array(audio.get_array_of_samples())
    # if audio.sample_width == 2: # 16-bit
    #     audio_np = audio_np.astype(np.float32) / (2**15)
    # elif audio.sample_width == 4: # 32-bit
    #     audio_np = audio_np.astype(np.float32) / (2**31)
    # reduced_noise = nr.reduce_noise(y=audio_np, sr=audio.frame_rate)
    #    # Convert back to pydub AudioSegment (careful with sample width)
    # audio = AudioSegment(
    #     (reduced_noise * (2**15)).astype(np.int16).tobytes(),
    #     frame_rate=audio.frame_rate,
    #     sample_width=2,
    #     channels=audio.channels
    # )

    # if audio.channels > 1:
    #     audio = audio.set_channels(1)
    
    # audio = audio.normalize(-1.0)

    # chunks = split_on_silence(audio, min_silence_len=..., silence_thresh=..., keep_silence=...)
    # if not chunks: 
    #     continue # Skip if no speech detected
    # processed_audio_segment = AudioSegment.empty()
    # for chunk in chunks:
    #     processed_audio_segment += chunk

    # temp_path = os.path.join(PROCESSED_AUDIO_DIR, "temp_" + os.path.basename(raw_audio_path))
    # processed_audio_segment.export(temp_path, format="wav")
    # y, sr = librosa.load(temp_path, sr=TARGET_SAMPLE_RATE)
    # sf.write(os.path.join(PROCESSED_AUDIO_DIR, os.path.basename(raw_audio_path)), y, TARGET_SAMPLE_RATE)
    # os.remove(temp_path) # Clean up temp file

    return mel

mean,std= np.load("audio/mel_mean_std.npy")
mel=audio_to_mel_spectrogram("audio/LJ001-0001.wav",mean,std)
print(mel,mel.shape)

audio=mel_to_audio_griffin_lim(mel,mean,std)
print(librosa.get_duration(y=audio))
sf.write('audio/LJ001_before.wav', audio,22050)
audio,_=remove_silence_wav("audio/LJ001_before.wav")
sf.write('audio/LJ001_after_sile.wav', audio,22050)
print(librosa.get_duration(y=audio))
print(" Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition")


# plt.figure(figsize=(12, 4))
# plt.subplot(1, 2, 1)
# plt.title("padding")
# plt.imshow(mel.T, aspect='auto', origin='lower')
# # mel= np.load("audio/LJ001.npy")

# plt.subplot(1, 2, 2)
# plt.title("with_outpadd")
# plt.imshow(mel.T, aspect='auto', origin='lower')
# plt.tight_layout()
# plt.show()

