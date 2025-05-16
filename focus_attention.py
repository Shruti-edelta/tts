import os
import datetime
import tensorflow as tf
import numpy as np
import pandas as pd
import ast
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from acoustic.text_preprocess import G2PConverter,TextNormalizer
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.metrics import CosineSimilarity
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.saving import register_keras_serializable
import seaborn as sns


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
class AdditiveAttention(tf.keras.layers.Layer):
    def __init__(self, units, return_attention=False, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_query = tf.keras.layers.Dense(units)
        self.W_values = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)
        self.return_attention = return_attention

    def call(self, query, values, mask=None):
        query = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W_query(query) + self.W_values(values)))
        score = tf.squeeze(score, axis=-1)

        if mask is not None:
            score += (1.0 - tf.cast(mask, tf.float32)) * -1e9

        attention_weights = tf.nn.softmax(score, axis=-1)
        context = tf.matmul(tf.expand_dims(attention_weights, 1), values)
        context = tf.squeeze(context, axis=1)

        if self.return_attention:
            return context, attention_weights
        return context, None

@register_keras_serializable()
class AttentionContextLayer(tf.keras.layers.Layer):
    def __init__(self, units, input_length, return_attention=False, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.input_length = input_length
        self.return_attention = return_attention

    def build(self, input_shape):
        self.attention = AdditiveAttention(units=self.units, return_attention=self.return_attention)
        super().build(input_shape)

    def call(self, inputs):
        query, values, mask = inputs
        context_vector, attention_weights = self.attention(query, values, mask=mask)

        context_expanded = tf.expand_dims(context_vector, axis=1)
        context_expanded = tf.tile(context_expanded, [1, self.input_length, 1])
        concat = tf.concat([values, context_expanded], axis=-1)

        if self.return_attention:
            return concat, attention_weights
        return concat

def build_debug_model(vocab_size, input_length=163, mel_dim=80, mel_len=865, embed_dim=256):
    inputs = layers.Input(shape=(None,), name='phoneme_input')

    embedding = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, mask_zero=True)
    x = embedding(inputs)
    mask = embedding.compute_mask(inputs)

    pos_embedding = layers.Embedding(input_dim=input_length, output_dim=embed_dim)

    def positional_encoding_layer(x):
        seq_len = tf.shape(x)[1]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        positions = tf.expand_dims(positions, 0)
        return pos_embedding(positions)

    pos_encoded = layers.Lambda(positional_encoding_layer)(x)
    x = layers.Add()([x, pos_encoded])

    x = layers.Bidirectional(layers.LSTM(512, return_sequences=True))(x, mask=mask)
    x = layers.Bidirectional(layers.LSTM(512, return_sequences=True))(x, mask=mask)
    x = layers.Dropout(0.3)(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x, mask=mask)

    for _ in range(3):
        residual = x
        x = layers.Conv1D(filters=256, kernel_size=5, padding='same', dilation_rate=2, activation='relu')(x)
        x = layers.LayerNormalization()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Add()([x, residual])

    query = layers.GlobalAveragePooling1D()(x)
    x, attention_weights = AttentionContextLayer(units=256, input_length=input_length, return_attention=True)([query, x, mask])

    x = layers.Conv1DTranspose(256, kernel_size=5, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv1DTranspose(128, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv1DTranspose(64, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    x = CropLayer(mel_len)(x)

    mel_output = layers.Conv1D(mel_dim, kernel_size=1, activation='linear', name="mel_output")(x)

    postnet = mel_output
    for _ in range(5):
        postnet = layers.Conv1D(filters=mel_dim, kernel_size=5, padding='same', activation='tanh')(postnet)
        postnet = layers.BatchNormalization()(postnet)

    mel_output = layers.Add(name="refined_mel_output")([mel_output, postnet])

    model = tf.keras.Model(inputs=inputs, outputs=[mel_output, attention_weights])
    return model

normalizer=TextNormalizer()
# g2p = G2PConverter(load_model=True)
g2p = G2PConverter(model_path="model/1/3model_cnn.keras")
print(g2p.phn2idx)
vocab_size = len(g2p.phn2idx)

debug_model = build_debug_model(vocab_size)
debug_model.load_weights("model/2/3acoustic_model_cnn_9f.weights.h5")


text="hello world how are you my name is shruti"
# text = "The third step is to implement the plan."

normalized_text = normalizer.normalize_text(text)
phonemes=g2p.predict(normalized_text['normalized_text'])
print(phonemes)

padded = pad_sequences([phonemes], maxlen=163, padding='post')[0]
input_tensor = tf.convert_to_tensor([padded], dtype=tf.int32)

predicted_mel, attention = debug_model.predict(input_tensor)
print(attention.shape)
print(predicted_mel.shape)

phoneme_strs = [g2p.idx2phn[idx] for idx in phonemes[:np.count_nonzero(phonemes)]]
attn_weights = attention[0][:len(phoneme_strs)]

plt.figure(figsize=(12, 2))
sns.heatmap([attn_weights], xticklabels=phoneme_strs, cmap='viridis', cbar=True)
plt.title("Global Attention over Phoneme Sequence")
plt.yticks([])  # optional: hide y-axis
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# plt.imshow(attention[0], cmap='viridis', aspect='auto')
# plt.title("Attention Weights")
# plt.xlabel("Input Phonemes")
# plt.ylabel("Attention Score")
# plt.colorbar()
# plt.show()

