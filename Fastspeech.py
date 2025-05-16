import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model

# ---------- Config ----------
MEL_DIM = 80
MAX_PHONEME_LEN = 200
MAX_MEL_LEN = 1000
EMBED_DIM = 256

# ---------- Length Regulator ----------
class LengthRegulator(layers.Layer):
    def call(self, encoder_output, durations):
        reps = tf.repeat(encoder_output, durations, axis=1)
        return reps

# ---------- Duration Predictor ----------
def build_duration_predictor():
    return tf.keras.Sequential([
        layers.Conv1D(256, kernel_size=3, padding='same', activation='relu'),
        layers.LayerNormalization(),
        layers.ReLU(),
        layers.Conv1D(256, kernel_size=3, padding='same', activation='relu'),
        layers.LayerNormalization(),
        layers.ReLU(),
        layers.Dense(1)
    ])

# ---------- FastSpeech-style Model ----------
def build_fastspeech(vocab_size):
    phoneme_input = layers.Input(shape=(MAX_PHONEME_LEN,), dtype=tf.int32, name="phoneme_input")
    duration_input = layers.Input(shape=(MAX_PHONEME_LEN,), dtype=tf.int32, name="duration_input")

    # Embedding
    x = layers.Embedding(input_dim=vocab_size, output_dim=EMBED_DIM)(phoneme_input)

    # Encoder (2-layer Transformer block style)
    for _ in range(2):
        residual = x
        x = layers.LayerNormalization()(x)
        x = layers.MultiHeadAttention(num_heads=2, key_dim=EMBED_DIM)(x, x)
        x = layers.Add()([x, residual])
        x = layers.LayerNormalization()(x)
        ff = layers.Dense(EMBED_DIM * 4, activation='relu')(x)
        ff = layers.Dense(EMBED_DIM)(ff)
        x = layers.Add()([x, ff])

    # Duration predictor (for training supervision only)
    duration_pred = build_duration_predictor()(x)
    duration_pred = tf.squeeze(duration_pred, -1)

    # Length regulator
    length_regulator = LengthRegulator()
    expanded = length_regulator(x, duration_input)

    # Decoder
    y = expanded
    for _ in range(2):
        residual = y
        y = layers.LayerNormalization()(y)
        y = layers.MultiHeadAttention(num_heads=2, key_dim=EMBED_DIM)(y, y)
        y = layers.Add()([y, residual])
        y = layers.LayerNormalization()(y)
        ff = layers.Dense(EMBED_DIM * 4, activation='relu')(y)
        ff = layers.Dense(EMBED_DIM)(ff)
        y = layers.Add()([y, ff])

    # Mel output
    mel_output = layers.Dense(MEL_DIM)(y)

    model = Model(inputs=[phoneme_input, duration_input], outputs=[mel_output, duration_pred], name="FastSpeech")
    return model
