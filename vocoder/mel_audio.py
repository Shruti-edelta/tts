import numpy as np
import noisereduce as nr
import librosa
import soundfile as sf

y, sr = librosa.load("../LJ001-0001.wav", sr=None)
print(librosa.get_duration(y=y,sr=sr))
# audio, _ = librosa.effects.trim(y, top_db=20)
# reduced_noise = nr.reduce_noise(y=y, sr=sr)
# sf.write("check_noise.wav", reduced_noise, sr)
# sf.write("check_t.wav", audio, sr)
# print(y,audio,reduced_noise)
# # sf.write("checko.wav", y, sr)
# print(np.array_equal(y,audio))

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
    y, sr = remove_silence_wav(audio_file)
    print("silence remove 1: ",y)    
    print(librosa.get_duration(y=y,sr=sr))
    sf.write("check_clean1.wav", y, sr)
    if y is None:
        print(f"⚠️ No speech detected in {audio_file}")
        return None

    y, _ = librosa.effects.trim(y, top_db=30)  # Optional: refine silence
    print("trim: ",y)
    sf.write("check_t2.wav", y, sr)
    print(librosa.get_duration(y=y,sr=sr))

    reduced_noise = nr.reduce_noise(y=y, sr=sr)
    print("reduce_noise: ",reduced_noise)
    sf.write("check_noice3.wav", reduced_noise, sr)
    print(librosa.get_duration(y=reduced_noise,sr=sr))

    # y = pre_emphasis(reduced_noise)
    # print("pre_emphasis: ",y)
    # sf.write("check_e4.wav", y, sr)
    # print(librosa.get_duration(y=y,sr=sr))

    # audio, sr = librosa.load(audio_file,sr=22050)
    mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=80,
            fmax=8000,
            n_fft=2048,
            hop_length=256,
            win_length=2048
        )
    mel_db = librosa.power_to_db(mel_spec, ref=np.max).T  # shape: (T, 80)
    mean=np.mean(mel_db)
    std=np.std(mel_db)
    mel_db=(mel_db - mean) / std
    print(mel_db.shape)
    return y,sr,mel_db,mean ,std

audio,sr,mel_db,mean,std=audio_to_mel_spectrogram("../LJ001-0001.wav")
print("norm_audio : ",audio,sr)
sf.write("check_norm.wav", audio, sr)
print("Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition")

# def mel_to_audio_griffin_lim(mel_db,mean,std, sr=22050, n_fft=2048, hop_length=256, win_length=2048, n_mels=80, fmax=8000):
#     log_mel = (mel_db * std) + mean
#     # Step 1: Convert dB back to power
#     mel_spec = librosa.db_to_power(mel_db.T)

#     # Step 2: Invert mel to linear
#     mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmax=fmax)
#     inv_mel_basis = np.linalg.pinv(mel_basis)
#     linear_spec = np.dot(inv_mel_basis, mel_spec)

#     # Step 3: Griffin-Lim
#     audio = librosa.griffinlim(linear_spec, hop_length=hop_length, win_length=win_length, n_iter=60)

#     # Step 4: Normalize audio
#     # audio = audio / np.max(np.abs(audio) + 1e-6)

#     # reduced_noise = nr.reduce_noise(y=audio, sr=sr)
#     return audio

def mel_to_audio_griffin_lim(mel_db, mean, std, sr=22050, n_fft=2048, hop_length=256, win_length=2048, n_mels=80, fmax=8000):
    # ✅ Step 1: Denormalize
    log_mel = (mel_db * std) + mean
    # print(log_mel)
    # print(mel_db)
    # ✅ Step 2: Convert dB to power
    mel_spec = librosa.db_to_power(log_mel.T)  # shape: (80, T)

    # ✅ Step 3: Invert mel to linear spectrogram
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmax=fmax)
    inv_mel_basis = np.linalg.pinv(mel_basis)
    linear_spec = np.dot(inv_mel_basis, mel_spec)  # shape: (1025, T)

    # ✅ Step 4: Griffin-Lim
    audio = librosa.griffinlim(linear_spec, hop_length=hop_length, win_length=win_length, n_iter=60)

    # ✅ Step 5: Normalize to [-1, 1]
    audio = audio / (np.max(np.abs(audio)) + 1e-6)
    print(str(audio))
    print(librosa.get_duration(y=audio))
    return audio

# Example use
# mel_db = np.random.rand(100, 80)
mel_db = np.load("../dataset/LJSpeech/wavs/LJ001-0001.npy")
mean,std = np.load("../dataset/acoustic_dataset/mel_mean_std.npy")
audio = mel_to_audio_griffin_lim(mel_db,mean,std)
sf.write("reconstructed_griffin.wav", audio, sr)





'''
[ 1.3123261e-04 -5.1114777e-05 -2.2463559e-04 ...  2.7145108e-05
  2.6073210e-05  2.3655213e-05] without std mean

[-5.7033780e-05 -6.3179396e-05 -6.6548564e-05 ... -1.3446953e-06
 -1.2523996e-06 -1.0839307e-06]

[ 7.8037963e-05  8.4326675e-05  8.4445644e-05 ... -1.8026427e-06
 -1.7433061e-06 -1.6064243e-06]
'''

# Apply Post-Gain Boost (if needed)
# If it’s still quiet, you can boost gain manually:

# audio *= 1.2  # Boost volume by 20%
# audio = np.clip(audio, -1.0, 1.0)
