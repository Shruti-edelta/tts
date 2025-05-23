import tensorflow as tf
from tensorflow.keras import layers, models

class PositionalEncoding(layers.Layer):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pos = tf.range(max_len)[:, tf.newaxis]
        i = tf.range(d_model)[tf.newaxis, :]
        angle_rates = 1 / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        angle_rads = tf.cast(pos, tf.float32) * angle_rates
        pos_encoding = tf.where(i % 2 == 0, tf.sin(angle_rads), tf.cos(angle_rads))
        self.pos_encoding = pos_encoding[tf.newaxis, ...]
        
    def call(self, x):
        length = tf.shape(x)[1]
        return x + self.pos_encoding[:, :length, :]

class DurationPredictor(layers.Layer):
    def __init__(self, d_model):
        super().__init__()
        self.net = models.Sequential([
            layers.Conv1D(d_model, kernel_size=3, padding="same", activation="relu"),
            layers.LayerNormalization(),
            layers.Dropout(0.1),
            layers.Conv1D(1, kernel_size=3, padding="same")
        ])

    def call(self, x):
        return self.net(x)  # [B, T, 1]

class LengthRegulator(layers.Layer):
    def call(self, x, durations):
        reps = tf.repeat(tf.range(tf.shape(x)[1]), durations, axis=0)
        return tf.gather(tf.reshape(x, [-1, tf.shape(x)[-1]]), reps)

class FastSpeechMini(tf.keras.Model):
    def __init__(self, vocab_size, d_model=192, mel_dim=80):
        super().__init__()
        self.embedding = layers.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)

        self.encoder = layers.Bidirectional(layers.LSTM(d_model, return_sequences=True))
        self.duration_predictor = DurationPredictor(d_model)

        self.length_regulator = LengthRegulator()
        self.decoder = layers.Bidirectional(layers.LSTM(d_model, return_sequences=True))

        self.mel_linear = layers.Dense(mel_dim)

    def call(self, phonemes, durations=None, training=True):
        x = self.embedding(phonemes)
        x = self.pos_encoding(x)
        x = self.encoder(x)

        log_dur_pred = tf.squeeze(self.duration_predictor(x), -1)
        if training:
            x = self.length_regulator(x, durations)
        else:
            predicted_durations = tf.cast(tf.math.round(tf.exp(log_dur_pred) - 1), tf.int32)
            x = self.length_regulator(x, predicted_durations)

        x = self.decoder(x)
        mel = self.mel_linear(x)
        return mel, log_dur_pred
    
def compute_loss(mel_pred, mel_true, log_dur_pred, log_dur_true):
    mel_loss = tf.reduce_mean(tf.abs(mel_pred - mel_true))
    duration_loss = tf.reduce_mean(tf.square(log_dur_pred - log_dur_true))
    return mel_loss + duration_loss