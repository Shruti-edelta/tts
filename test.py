import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf  

def mel_to_audio(mel_spectrogram, sr=22050,n_iter=32):
    mel_inv = librosa.feature.inverse.mel_to_audio(mel_spectrogram, sr=sr,n_iter=n_iter)
    return mel_inv

# # Load audio file (provide path to your audio file)
#I had no conception that vessels ever came so far north and was astounded at the sight.
file_path = 'dataset/libri_dataset/78/78_369_000022_000006.wav'

# y, sr = librosa.load(file_path, sr=None)  # y is the audio signal, sr is the sample rate

y, sr = librosa.load(file_path,sr=None)  # load at 22050 Hz consistant SR
mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,fmax=8000)     #range in 0.00000 somthing 7.4612461e-03
mel_spectrogram_db = librosa.power_to_db(mel_spec, ref=np.max)  # Convert to dB (decibels)unit scale 
# S_db = (mel_spectrogram_db - np.mean(mel_spectrogram_db)) / np.std(mel_spectrogram_db) 

# # Plot the Mel spectrogram
# plt.figure(figsize=(10, 6))
# librosa.display.specshow(S_db, x_axis='time', y_axis='mel', sr=sr)
# plt.colorbar(format='%+2.0f dB')
# plt.title('Mel Spectrogram')
# plt.show()

# audio_griffin_lim = librosa.griffinlim(S_db,n_iter=10)
audio = mel_to_audio(mel_spectrogram_db,sr=sr)
print(audio)
sf.write('test_audio_sample.wav', audio, 22050)
