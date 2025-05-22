import os
import datetime
import tensorflow as tf
import numpy as np
import pandas as pd
import ast
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from acoustic.text_preprocess import G2PConverter
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.metrics import CosineSimilarity
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.saving import register_keras_serializable
# import tensorflow_addons as tfa

def create_dataset_fast(texts, mel_paths, batch_size=32, shuffle=True):
    def parse_data(phoneme_str, mel_path):
        def _parse_numpy(phoneme_str_, mel_path_):
            phonemes = np.array(ast.literal_eval(phoneme_str_.decode()), dtype=np.int32)  # Convert string to list, then to array
            mel = np.load(mel_path_.decode()).astype(np.float32)  # Load .npy file
            return phonemes, mel

        phonemes, mel = tf.numpy_function(_parse_numpy, [phoneme_str, mel_path], [tf.int32, tf.float32])
        phonemes.set_shape([None])      # Shape: (phoneme_seq_len,)
        mel.set_shape([None, 80])       # Shape: (mel_frame_len, 80)
        return phonemes, mel

    dataset = tf.data.Dataset.from_tensor_slices((texts, mel_paths))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)    
    dataset = dataset.map(lambda x, y: parse_data(x, y), num_parallel_calls=tf.data.AUTOTUNE)
    # dataset=dataset.cache()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

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

# def build_acoustic_model(vocab_size, input_length, mel_dim=80, mel_len=1024, embed_dim=256):
def build_acoustic_model(vocab_size, input_length=163, mel_dim=80, mel_len=865, embed_dim=256):
    inputs = layers.Input(shape=(None,), name='phoneme_input')

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
    x = layers.Bidirectional(layers.LSTM(512, return_sequences=True,dropout=0.3))(x, mask=mask)
    # x = layers.Dropout(0.3)(x)
    x = layers.Bidirectional(layers.LSTM(512, return_sequences=True,dropout=0.3))(x, mask=mask)
    # x = layers.Dropout(0.3)(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True,dropout=0.3))(x, mask=mask)
    # x = layers.Dropout(0.3)(x)

    # Residual CNN blocks
    for _ in range(3):
        residual = x
        x = layers.Conv1D(filters=256, kernel_size=5, padding='same', dilation_rate=2, activation='relu')(x)
        x = layers.LayerNormalization()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Add()([x, residual])

    query = layers.GlobalAveragePooling1D()(x)
    x = AttentionContextLayer(units=256, input_length=input_length)([query, x, mask])

    # Upsample
    x = layers.Conv1DTranspose(256, kernel_size=5, strides=2, padding='same', activation='relu')(x) # (None, 336, 256) (None, 400, 256)
    x = layers.Conv1DTranspose(128, kernel_size=3, strides=2, padding='same', activation='relu')(x) # (None, 800, 128)
    x = layers.Conv1DTranspose(64, kernel_size=3, strides=2, padding='same', activation='relu')(x) # (None, 1600, 128) 
    x = CropLayer(mel_len)(x)

    # Initial mel output
    mel_output = layers.Conv1D(mel_dim, kernel_size=1, activation='linear', name="mel_output")(x)

    # Post-net for refinement
    postnet = mel_output
    for _ in range(5):
        postnet = layers.Conv1D(filters=mel_dim, kernel_size=5, padding='same', activation='tanh')(postnet)
        postnet = layers.BatchNormalization()(postnet)

    for i in range(5):
        activation = 'tanh' if i < 4 else None
        postnet = layers.Conv1D(mel_dim, kernel_size=5, padding='same', activation=activation)(postnet)
        if i < 4:
            postnet = layers.BatchNormalization()(postnet)
            postnet = layers.Dropout(0.1)(postnet)

    mel_output = layers.Add(name="refined_mel_output")([mel_output, postnet])

    model = tf.keras.Model(inputs=inputs, outputs=mel_output)
    return model

def compile_model(model):
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.001,
        decay_steps=50 * len(train_dataset),
        alpha=0.1
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)

    def spectral_convergence_loss(y_true, y_pred, mask):
        diff = y_true - y_pred
        numerator = tf.norm(diff * mask[..., tf.newaxis], ord='fro', axis=[-2, -1])
        denominator = tf.norm(y_true * mask[..., tf.newaxis], ord='fro', axis=[-2, -1])
        return tf.reduce_mean(numerator / (denominator + 1e-6))

    def log_mel_loss(y_true, y_pred, mask):
        epsilon = 1e-5
        y_true = tf.nn.relu(y_true)
        y_pred = tf.nn.relu(y_pred)

        log_true = tf.math.log(y_true + epsilon)
        log_pred = tf.math.log(y_pred + epsilon)

        l1 = tf.abs(log_true - log_pred)
        l1 = tf.reduce_sum(l1 * mask[..., tf.newaxis]) / tf.reduce_sum(mask)
        return l1

    def masked_mse_loss(y_true, y_pred, mask):
        mse = tf.square(y_true - y_pred)  # shape: (batch, time, mel)
        mse = tf.reduce_sum(mse, axis=-1)  # (batch, time)
        masked_mse = tf.reduce_sum(mse * mask) / tf.reduce_sum(mask)
        return masked_mse

    def combined_loss(y_true, y_pred):
        # Create mask: 1 where y_true != 0 (any mel bin), shape: (batch, time)
        mask = tf.reduce_any(tf.not_equal(y_true, 0.0), axis=-1)  # axis=-1 = mel dimension
        mask = tf.cast(mask, tf.float32)

        mse = masked_mse_loss(y_true, y_pred, mask)
        sc_loss = spectral_convergence_loss(y_true, y_pred, mask)
        log_loss = log_mel_loss(y_true, y_pred, mask)

        return mse + 0.5 * sc_loss + 0.1 * log_loss

    model.compile(
        optimizer=optimizer,
        loss=combined_loss,
        metrics=['mae', tf.keras.metrics.CosineSimilarity(axis=-1)]
    )
    return model

# ==================== Load Data =====================

df_train = pd.read_csv('dataset/acoustic_dataset/train.csv', usecols=['Phoneme_text', 'Read_npy'])
df_val = pd.read_csv('dataset/acoustic_dataset/val.csv', usecols=['Phoneme_text', 'Read_npy'])
df_test = pd.read_csv('dataset/acoustic_dataset/test.csv', usecols=['Phoneme_text', 'Read_npy'])

texts_train = df_train['Phoneme_text'].values
mel_train = df_train['Read_npy'].values

texts_val = df_val['Phoneme_text'].values
mel_val = df_val['Read_npy'].values

texts_test= df_test['Phoneme_text'].values
mel_test = df_test['Read_npy'].values

train_dataset = create_dataset_fast(texts_train, mel_train )
val_dataset = create_dataset_fast(texts_val, mel_val )
test_dataset = create_dataset_fast(texts_test, mel_test)

g2p = G2PConverter(load_model=False)
print(g2p.phn2idx)
vocab_size = len(g2p.phn2idx)

# ==================== Build & Train =====================


# model = build_acoustic_model(vocab_size, input_length)
model = build_acoustic_model(vocab_size)
model = compile_model(model)
model.summary()

# Set up log directory with timestamp
log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1,write_graph=True,write_images=True)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    ModelCheckpoint('model/2/best_model_cnn_9f_log.keras', monitor='val_loss', save_best_only=True, verbose=1),
    LRSchedulerLogger(),
    LearningRatePlotter(),
    tensorboard_callback , # 👈 Add this line,
    # TensorBoard(log_dir="logs", histogram_freq=1)  # 👈 This logs training info
]

history = model.fit(
    train_dataset,
    epochs=150,
    validation_data=val_dataset,
    callbacks=callbacks
)

# Save model & history
model.save('model/2/model_cnn_9f_log.keras')
model.save_weights('model/2/best_model_cnn_9f_log.weights.h5')
history_df = pd.DataFrame(history.history)
history_df.to_csv('model/2/best_model_cnn_9f_log.csv', index=False)

# ==================== Evaluation =====================
test_loss = model.evaluate(test_dataset)
print(f"Test loss: {test_loss}")
