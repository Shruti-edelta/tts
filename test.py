from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.saving import register_keras_serializable
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ast
from acoustic.text_preprocess import G2PConverter,TextNormalizer  #relative path
import soundfile as sf
import librosa
import keras
from keras.config import enable_unsafe_deserialization
enable_unsafe_deserialization()

# @tf.keras.utils.register_keras_serializable()
@register_keras_serializable()
class CropLayer(tf.keras.layers.Layer):
    def __init__(self, length, **kwargs):
        super().__init__(**kwargs)
        self.length = length

    def call(self, inputs):
        return inputs[:, :self.length, :]

    def get_config(self):
        config = super().get_config()
        config.update({'length': self.length})
        return config
    
@register_keras_serializable()
# --- Simplified Additive Attention ---
class AdditiveAttention(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_query = tf.keras.layers.Dense(units)
        self.W_values = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values, mask=None):
        query = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W_query(query) + self.W_values(values)))
        score = tf.squeeze(score, axis=-1)

        if mask is not None:
            score += (1.0 - tf.cast(mask, tf.float32)) * -1e9

        attention_weights = tf.nn.softmax(score, axis=-1)
        context = tf.matmul(tf.expand_dims(attention_weights, 1), values)
        context = tf.squeeze(context, axis=1)
        return context, attention_weights

@register_keras_serializable()
class AttentionContextLayer(tf.keras.layers.Layer):
    def __init__(self, units, input_length, **kwargs):
        super(AttentionContextLayer, self).__init__(**kwargs)
        self.units = units
        self.input_length = input_length

    def build(self, input_shape):
        self.attention = AdditiveAttention(units=self.units)
        super(AttentionContextLayer, self).build(input_shape)

    def call(self, inputs):
        query, values, mask = inputs
        context_vector, _ = self.attention(query, values, mask=mask)
        
        # Expand context and concatenate with sequence
        context_expanded = tf.expand_dims(context_vector, axis=1)
        context_expanded = tf.tile(context_expanded, [1, self.input_length, 1])
        return tf.concat([values, context_expanded], axis=-1)
    
@tf.keras.utils.register_keras_serializable()
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, input_length, embed_dim, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.pos_embedding = tf.keras.layers.Embedding(input_dim=input_length, output_dim=embed_dim)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        positions = tf.expand_dims(positions, 0)
        pos_encoded = self.pos_embedding(positions)
        return pos_encoded

@tf.keras.utils.register_keras_serializable()
class LastTimestep(tf.keras.layers.Layer):
    def call(self, inputs):
        return inputs[:, -1, :]

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

best_model = tf.keras.models.load_model("model/2/best_model_cnn_9f_log.keras", compile=False,custom_objects={'CropLayer': CropLayer})
g2p = G2PConverter("model/1/3model_cnn.keras")
normalizer=TextNormalizer()

text = "This is a test"
text = "However, it now laid down in plain language and with precise details the requirements of a good jail system."
text = "The first step is to identify the problem and its root cause."
text="The second step we have taken in the restoration of normal business enterprise"
# text = "The second step "#is to develop a plan to address the problem."  # ****
# text = "The third step is to implement the plan."
# text="shruti mungra"
# text="hello world"

normalized_text = normalizer.normalize_text(text)
phonemes=g2p.predict(normalized_text['normalized_text'])
print(phonemes)

padded = pad_sequences([phonemes], maxlen=163, padding='post')[0]
input_tensor = tf.convert_to_tensor([padded], dtype=tf.int32)

predicted_mel = best_model.predict(input_tensor,verbose=0)[0]
print("predicted_mel(input): ",predicted_mel)

mean,std = np.load("dataset/acoustic_dataset/mel_mean_std.npy")
audio=mel_to_audio_griffin_lim(predicted_mel, mean, std)
print("audio: ",audio)
sf.write('audio/cnn/best_model_9f_t2_log_padd_e9_f.wav', audio, 22050) 
# sf.write('audio/cnn/best_model_9f_t2_ef.wav', audio, 22050) 

# Plot it
plt.imshow(predicted_mel.T, aspect='auto', origin='lower')
plt.title("Predicted Mel for custom sentence")
plt.show()

# =================== test_data ====================

def preprocess_testdata(texts, mel_spectrograms, input_length=256,mel_max_len=1024):
    # print(texts[0])
    mel_spectrograms = [np.load(mel) for mel in mel_spectrograms]
    mel_spectrograms = np.array(mel_spectrograms)
    return texts, mel_spectrograms

df_test = pd.read_csv('dataset/acoustic_dataset/test.csv')
phoneme_seq = df_test['Phoneme_text'].apply(ast.literal_eval).values
mel_spectrograms_test = df_test['Read_npy'].values

test_text, test_mel = preprocess_testdata(phoneme_seq, mel_spectrograms_test)
x_test = np.expand_dims(test_text[1], axis=0)  # Adding a batch dimension
y_true = np.expand_dims(test_mel[1], axis=0)  # Adding a batch dimension)

# test_loss = model.evaluate(x_test, y_true)
# print(f"Test loss: {test_loss}")
print(x_test)
y_pred = best_model.predict(x_test)

print("Input shape:", x_test.shape)
print("True mel shape:", y_true.shape)
print("Predicted mel shape:", y_pred.shape)

audio=mel_to_audio_griffin_lim(y_true[0],mean,std)
print("y_true mel :", y_true)
print("Audio shape:", audio)
sf.write('org_lj1.wav', audio, 22050)

audio = mel_to_audio_griffin_lim(y_pred[0],mean,std)
print("Predicted mel spectrograms:", y_pred)
print("Audio shape:", audio)
sf.write('testing_lj1.wav', audio, 22050) 

are_equal = np.allclose(y_true[0].T, y_pred[0].T)
print("is equal? ", are_equal)
are_equal = np.allclose(predicted_mel.T, y_pred[0].T)
print("is equal? ", are_equal)
print(text)

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


'''
 conda activate mfa-env
 conda deactivate
'''