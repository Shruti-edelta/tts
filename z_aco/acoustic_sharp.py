import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.metrics import CosineSimilarity
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import ast
from acoustic.text_preprocess import G2PConverter
from keras.saving import register_keras_serializable


def create_dataset_fast(texts, mel_paths, input_length=168, mel_dim=80, mel_max_len=900, batch_size=32):
    def load_and_preprocess_py(text, mel_path):
        text = text.numpy().decode("utf-8")
        mel_path = mel_path.numpy().decode("utf-8")
        # print(text)
        phoneme_seq = ast.literal_eval(text)
        padded_text = pad_sequences([phoneme_seq], maxlen=input_length, padding='post')[0].astype(np.int32)
        mel = np.load(mel_path).astype(np.float32)
        T, D = mel.shape
        if T > mel_max_len:
            mel = mel[:mel_max_len, :]
        elif T < mel_max_len:
            pad_len = mel_max_len - T
            mel = np.pad(mel, ((0, pad_len), (0, 0)), mode='constant')
        return padded_text, mel

    def tf_wrapper(text, mel_path):
        text_tensor, mel_tensor = tf.py_function(
            func=load_and_preprocess_py,
            inp=[text, mel_path],
            Tout=[tf.int32, tf.float32]
        )
        text_tensor.set_shape([input_length])
        mel_tensor.set_shape([mel_max_len, mel_dim])
        return text_tensor, mel_tensor

    dataset = tf.data.Dataset.from_tensor_slices((texts, mel_paths))
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.map(tf_wrapper, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

@register_keras_serializable()
class CropLayer(tf.keras.layers.Layer):
    def __init__(self, length, **kwargs):
        super().__init__(**kwargs)
        self.length = length

    def call(self, inputs):
        return inputs[:, :self.length, :]

    def get_config(self):
        config = super().get_config()
        config.update({'target_length': self.length})
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

def build_acoustic_model(vocab_size, input_length, mel_dim=80, mel_len=900, embed_dim=256):
    inputs = layers.Input(shape=(input_length,), name='phoneme_input')

    # Embedding + Positional Encoding
    embedding = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, mask_zero=True)
    x = embedding(inputs)
    mask = embedding.compute_mask(inputs)

    # Positional embeddings (learnable)
    pos_embedding = layers.Embedding(input_dim=input_length, output_dim=embed_dim)
    positions = tf.range(start=0, limit=input_length, delta=1)
    pos_encoded = pos_embedding(positions)
    x = x + pos_encoded

    # BiLSTM stack
    x = layers.Bidirectional(layers.LSTM(512, return_sequences=True))(x, mask=mask)
    # x = layers.Dropout(0.3)(x)
    x = layers.Bidirectional(layers.LSTM(512, return_sequences=True))(x, mask=mask)
    x = layers.Dropout(0.3)(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x, mask=mask)
    # x = layers.Dropout(0.3)(x)

    # Residual CNN blocks
    for _ in range(3):
        residual = x
        x = layers.Conv1D(filters=256, kernel_size=5, padding='same', dilation_rate=2, activation='relu')(x)
        x = layers.LayerNormalization()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Add()([x, residual])

    # # Attention over encoded sequence
    # query = layers.GlobalAveragePooling1D()(x)
    # context_vector, _ = AdditiveAttention(units=256)(query, x, mask=mask)

    # # Expand context and concatenate with sequence
    # context_expanded = tf.expand_dims(context_vector, axis=1)
    # context_expanded = tf.tile(context_expanded, [1, tf.shape(x)[1], 1])
    # x = layers.Concatenate()([x, context_expanded])

    query = layers.GlobalAveragePooling1D()(x)
    x = AttentionContextLayer(units=256, input_length=input_length)([query, x, mask])

    # Upsample
    x = layers.Conv1DTranspose(256, kernel_size=5, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv1DTranspose(128, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv1DTranspose(128, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    x = CropLayer(mel_len)(x)

    # Initial mel output
    mel_output = layers.Conv1D(mel_dim, kernel_size=1, activation='linear', name="mel_output")(x)

    # Post-net for refinement
    postnet = mel_output
    for _ in range(7):
        postnet = layers.Conv1D(filters=mel_dim, kernel_size=5, padding='same', activation='tanh')(postnet)
        postnet = layers.BatchNormalization()(postnet)
    mel_output = layers.Add(name="refined_mel_output")([mel_output, postnet])

    model = tf.keras.Model(inputs=inputs, outputs=mel_output)
    return model

class LRSchedulerLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.learning_rate
        current_lr = lr(epoch).numpy() if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule) else tf.keras.backend.get_value(lr)
        print(f"\nEpoch {epoch+1}: Learning rate is {current_lr:.6f}")

class LearningRatePlotter(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.learning_rate
        self.lrs.append(lr)

    def on_train_begin(self, logs=None):
        self.lrs = []

# ===== Delta Feature Computation =====
def delta_features(x):
    delta = x[:, 2:, :] - x[:, :-2, :]
    delta = tf.pad(delta, [[0, 0], [1, 1], [0, 0]])
    return delta

def double_delta_features(x):
    delta = delta_features(x)
    double_delta = delta_features(delta)
    return double_delta

# ===== Spectral Convergence Loss =====
def spectral_convergence_loss(y_true, y_pred):
    true_stft = tf.signal.stft(y_true, frame_length=1024, frame_step=256, fft_length=1024, pad_end=True)
    pred_stft = tf.signal.stft(y_pred, frame_length=1024, frame_step=256, fft_length=1024, pad_end=True)

    true_mag = tf.abs(true_stft)
    pred_mag = tf.abs(pred_stft)

    sc_loss = tf.norm(true_mag - pred_mag, ord='fro', axis=[-2, -1]) / (tf.norm(true_mag, ord='fro', axis=[-2, -1]) + 1e-6)
    return tf.reduce_mean(sc_loss)

# ===== Multi-Resolution STFT Loss =====
def multi_resolution_stft_loss(y_true, y_pred):
    fft_sizes = [512, 1024, 2048]
    hop_sizes = [128, 256, 512]
    win_lengths = [400, 800, 1600]

    total_loss = 0.0
    for fft_size, hop_size, win_length in zip(fft_sizes, hop_sizes, win_lengths):
        true_stft = tf.signal.stft(y_true, frame_length=win_length, frame_step=hop_size, fft_length=fft_size, pad_end=True)
        pred_stft = tf.signal.stft(y_pred, frame_length=win_length, frame_step=hop_size, fft_length=fft_size, pad_end=True)

        true_mag = tf.abs(true_stft)
        pred_mag = tf.abs(pred_stft)

        sc_loss = tf.norm(true_mag - pred_mag, ord='fro', axis=[-2, -1]) / (tf.norm(true_mag, ord='fro', axis=[-2, -1]) + 1e-6)
        mag_loss = tf.reduce_mean(tf.abs(true_mag - pred_mag))

        total_loss += sc_loss + mag_loss

    return total_loss / len(fft_sizes)

# ===== Final Full Combined Loss =====
def full_combined_loss(y_true, y_pred):
    # Basic losses
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    # log_mel_loss = tf.reduce_mean(tf.abs(tf.math.log(y_true + 1e-6) - tf.math.log(y_pred + 1e-6)))
    spec_convergence_loss = spectral_convergence_loss(y_true, y_pred)

    # Delta losses
    delta_true = delta_features(y_true)
    delta_pred = delta_features(y_pred)
    delta_loss = tf.reduce_mean(tf.square(delta_true - delta_pred))

    double_delta_true = double_delta_features(y_true)
    double_delta_pred = double_delta_features(y_pred)
    double_delta_loss = tf.reduce_mean(tf.square(double_delta_true - double_delta_pred))

    # MR-STFT loss
    # mr_stft_loss = multi_resolution_stft_loss(y_true, y_pred)

    # Combine all
    total_loss = (
        mse_loss +
        0.5 * spec_convergence_loss +
        0.2 * delta_loss 
        # 0.1 * double_delta_loss +
        # 0.5 * mr_stft_loss
    )

    return total_loss

def compile_model(model):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)

    model.compile(optimizer=optimizer,
                  loss=full_combined_loss,
                  metrics=['mae', CosineSimilarity(axis=-1)])
    return model


# ==================== Load Data =====================

df = pd.read_csv('dataset/acoustic_dataset/tts_data_LJ.csv', usecols=['Phoneme_text', 'Read_npy'])
texts = df['Phoneme_text'].apply(ast.literal_eval).values
mel_spectrograms = df['Read_npy'].values
input_length = max([len(seq) for seq in texts])

texts_str = [str(seq) for seq in texts]
texts_train, texts_temp, mel_train, mel_temp = train_test_split(texts_str, mel_spectrograms, test_size=0.2, random_state=33)
texts_val, texts_test, mel_val, mel_test = train_test_split(texts_temp, mel_temp, test_size=0.3, random_state=33)

train_dataset = create_dataset_fast(texts_train, mel_train, input_length=input_length)
val_dataset = create_dataset_fast(texts_val, mel_val, input_length=input_length)
test_dataset = create_dataset_fast(texts_test, mel_test, input_length=input_length)

g2p = G2PConverter(load_model=False)
vocab_size = len(g2p.phn2idx)

# ==================== Build & Train =====================

model = build_acoustic_model(vocab_size, input_length)
model = compile_model(model)
# model.compile(optimizer='adam', loss='mse', metrics=['mae',CosineSimilarity(axis=-1)])
model.summary()


callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ModelCheckpoint('model/2/best_model_cnn_sjarp.keras', monitor='val_loss', save_best_only=True, verbose=1),
    LRSchedulerLogger(),
    LearningRatePlotter()
]

history = model.fit(
    train_dataset,
    epochs=50,
    validation_data=val_dataset,
    callbacks=callbacks
)

# Save model & history
model.save('model/2/acoustic_model_cnn.keras')
model.save_weights('model/2/acoustic_model_cnn.weights.h5')
history_df = pd.DataFrame(history.history)
history_df.to_csv('model/2/acoustic_model_cnn.csv', index=False)

# ==================== Evaluation =====================
test_loss = model.evaluate(test_dataset)
print(f"Test loss: {test_loss}")




'''
add Multi-Resolution STFT Loss


'''