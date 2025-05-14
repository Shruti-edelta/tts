import tensorflow as tf
from tensorflow.keras.metrics import CosineSimilarity
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, Conv1D, Conv1DTranspose, Bidirectional, LSTM, LayerNormalization, Dense, Add, Activation
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ast
from text_preprocess import G2PConverter
import soundfile as sf
import librosa
from keras.saving import register_keras_serializable
import keras


# @tf.keras.utils.register_keras_serializable()
@register_keras_serializable()
class CropLayer(tf.keras.layers.Layer):
    def __init__(self, target_length, **kwargs):
        super().__init__(**kwargs)
        self.length = target_length

    def call(self, inputs):
        return inputs[:, :self.length, :]

    def get_config(self):
        config = super().get_config()
        config.update({'target_length': self.length})
        return config

best_model = tf.keras.models.load_model("model/2/2best_model_cnn_9f.keras", compile=False,custom_objects={'CropLayer': CropLayer})

g2p = G2PConverter("model/1/3model_cnn.keras")

text = "This is a test"
phonemes = g2p.predict(text)  # use your G2P model
print(phonemes)
padded = pad_sequences([phonemes], maxlen=256, padding='post')[0]
# print(padded)
input_tensor = tf.convert_to_tensor([padded], dtype=tf.int32)

predicted_mel = best_model.predict(input_tensor,verbose=0)[0]
print("predicted_mel(input): ",predicted_mel)

# Plot it
plt.imshow(predicted_mel.T, aspect='auto', origin='lower')
plt.title("Predicted Mel for custom sentence")
plt.show()

audio=mel_to_audio(predicted_mel.T)
print("audio: ",audio)
# sf.write('audio/cnn/1best_model_t1.wav', audio, 22050) 


# =================== test_data ====================
import librosa
import numpy as np

def mel_to_audio_griffin_lim(mel_db, sr=22050, n_fft=1024, hop_length=256, win_length=1024, n_mels=80, fmax=8000):
    # Transpose if needed (librosa expects shape [n_mels, T])
    if mel_db.shape[1] < mel_db.shape[0]:
        mel_db = mel_db.T

    
    # Convert dB back to power
    mel_spec = librosa.db_to_power(mel_db.T)

    # Get Mel filterbank
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmax=fmax)

    # Pseudo-inverse to recover linear spectrogram
    inv_mel_basis = np.linalg.pinv(mel_basis)
    linear_spec = np.dot(inv_mel_basis, mel_spec)

    # Reconstruct audio using Griffin-Lim
    audio = librosa.griffinlim(linear_spec, n_iter=60, hop_length=hop_length, win_length=win_length)
    return audio


def preprocess_testdata(texts, mel_spectrograms, input_length=256,mel_max_len=1024):
    # print(texts[0])
    mel_spectrograms = [np.load(mel) for mel in mel_spectrograms]
    mel_spectrograms = np.array(mel_spectrograms)
    return texts, mel_spectrograms

df_test = pd.read_csv('dataset/acoustic_dataset/test.csv')
phoneme_seq = df_test['Phoneme_text'].apply(ast.literal_eval).values
mel_spectrograms_test = df_test['Read_npy'].values

test_text, test_mel = preprocess_testdata(phoneme_seq, mel_spectrograms_test)
x_test = np.expand_dims(test_text[0], axis=0)  # Adding a batch dimension
y_true = np.expand_dims(test_mel[0], axis=0)  # Adding a batch dimension)

# test_loss = model.evaluate(x_test, y_true)
# print(f"Test loss: {test_loss}")
print(x_test)
y_pred = best_model.predict(x_test)

print("Input shape:", x_test.shape)
print("True mel shape:", y_true.shape)
print("Predicted mel shape:", y_pred.shape)


# Visualize 1 example
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.title("Ground Truth")
plt.imshow(y_true[0].T, aspect='auto', origin='lower')

plt.subplot(1, 2, 2)
plt.title("Prediction")
plt.imshow(y_pred[0].T, aspect='auto', origin='lower')
plt.tight_layout()
plt.show()

audio=mel_to_audio(y_true[0].T)
print("y_true mel :", y_true)
print("Audio shape:", audio)
sf.write('org_lj1.wav', audio, 22050) 
audio = mel_to_audio(y_pred[0].T)
print("Predicted mel spectrograms:", y_pred)
print("Audio shape:", audio)
sf.write('testing_lj1.wav', audio, 22050) 

are_equal = np.allclose(y_true[0].T, y_pred[0].T)
print("is equal? ", are_equal)
are_equal = np.allclose(predicted_mel.T, y_pred[0].T)
print("is equal? ", are_equal)




