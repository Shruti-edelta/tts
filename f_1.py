import pandas as pd
import numpy as np
from acoustic.text_preprocess import G2PConverter
import tensorflow as tf
from tensorflow.keras import layers, models

# === Hyperparameters ===
n_phonemes = 50
embedding_dim = 256
n_mel = 80

# === Inputs ===
phoneme_input = layers.Input(shape=(None,), dtype='int32', name='phonemes')
duration_input = layers.Input(shape=(None,), dtype='int32', name='durations')

# === Embedding + Encoder ===
x = layers.Embedding(n_phonemes, embedding_dim)(phoneme_input)
x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
x = layers.Conv1D(256, 3, padding='same', activation='relu')(x)

# === Length Regulator ===
def length_regulator(encoder_outputs, durations):
    def _regulate(enc_out, dur):
        # Repeat each encoder vector by its corresponding duration
        repeated = tf.repeat(enc_out, dur, axis=0)
        return repeated

    regulated = tf.map_fn(
        lambda x: _regulate(x[0], x[1]),
        (encoder_outputs, durations),
        dtype=tf.float32
    )
    return regulated

x = layers.Lambda(lambda args: length_regulator(*args))([x, duration_input])

# === Decoder ===
x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
x = layers.Conv1D(80, 1)(x)  # Predict mel spectrogram

# === Final Model ===
model = models.Model([phoneme_input, duration_input], x)


def mse_loss(y_true, y_pred):
    mask = tf.cast(tf.math.not_equal(tf.reduce_sum(y_true, axis=-1), 0.0), tf.float32)
    loss = tf.reduce_sum(tf.square(y_true - y_pred), axis=-1)
    loss = loss * mask
    return tf.reduce_sum(loss) / tf.reduce_sum(mask)

model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=mse_loss)

# ==================== Load Data =====================


df_train = pd.read_csv('dataset/acoustic_dataset/train.csv', usecols=['Phoneme_text', 'duration','Read_npy'])
df_val = pd.read_csv('dataset/acoustic_dataset/val.csv', usecols=['Phoneme_text', 'duration','Read_npy'])
df_test = pd.read_csv('dataset/acoustic_dataset/test.csv', usecols=['Phoneme_text', 'duration','Read_npy'])

texts_train = df_train['Phoneme_text'].values
dura_train = df_train['duration'].values
mel_train = df_train['Read_npy'].values

texts_val = df_val['Phoneme_text'].values
dura_val = df_val['duration'].values
mel_val = df_val['Read_npy'].values

texts_test= df_test['Phoneme_text'].values
dura_test= df_test['duration'].values
mel_test = df_test['Read_npy'].values

g2p = G2PConverter(load_model=False)
# print(g2p.phn2idx)
vocab_size = len(g2p.phn2idx)


# model.fit(
#     x={'phonemes': phoneme_seqs, 'durations': duration_seqs},
#     y=mel_targets,
#     batch_size=32,
#     epochs=50,
#     validation_split=0.1,
#     callbacks=[
#         tf.keras.callbacks.ModelCheckpoint("tts_model.h5", save_best_only=True),
#         tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
#         tf.keras.callbacks.TensorBoard(log_dir="./logs")
#     ]
# )
