import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
from tensorflow.keras.metrics import CosineSimilarity
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import ast
from acoustic.text_preprocess import G2PConverter
from keras.saving import register_keras_serializable
import numpy as np
import os

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
    dataset=dataset.cache()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

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

def build_acoustic_model(vocab_size,batch_size=32, input_length=256, mel_dim=80, mel_len=1024, embed_dim=256):
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
    for _ in range(5):
        residual = x
        x = layers.Conv1D(256, 5, padding='same', activation='relu')(x)
        x = layers.LayerNormalization()(x)
        x = layers.Conv1D(256, 1, padding='same')(x)
        x = layers.Add()([x, residual])
        x = layers.ReLU()(x)
    # print(tf.shape(x)[0])
    print(x.shape[0])

    # Learned decoder queries
    x = layers.Conv1DTranspose(256, 5, strides=2, padding='valid', activation='relu')(x)
    x = layers.Conv1DTranspose(128, 3, strides=3, padding='same', activation='relu')(x)
    x = layers.Conv1DTranspose(64, 3, strides=3, padding='same', activation='relu')(x)
    # x = layers.Conv1DTranspose(64, 3, strides=1, padding='same', activation='relu')(x)
    x = CropLayer(mel_len)(x)

    query_positions = tf.range(mel_len)
    query_embedding = layers.Embedding(input_dim=mel_len, output_dim=256)(query_positions)
    query_embedding = tf.expand_dims(query_embedding, axis=0)  # [1, T, D]
    query_embedding = tf.tile(query_embedding, [batch_size, 1, 1])  # [B, T, D]
    print(query_embedding)
    context_vectors= layers.AdditiveAttention(256)([query_embedding, x])
    # x = tf.concat([context_vectors, query_embedding], axis=-1)


    # Initial mel output
    mel_output = layers.Conv1D(mel_dim, kernel_size=1, activation='linear', name="mel_output")(x)

    # Post-net for refinement
    postnet = mel_output
    for _ in range(20):
        postnet = layers.Conv1D(filters=mel_dim, kernel_size=5, padding='same', activation='tanh')(postnet)
        postnet = layers.BatchNormalization()(postnet)
    mel_output = layers.Add(name="refined_mel_output")([mel_output, postnet])
    model = tf.keras.Model(inputs=inputs, outputs=mel_output)
    return model

class DelayedLogMelLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.4, beta=0.3, log_start_epoch=10):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.log_start_epoch = log_start_epoch
        self.current_epoch = 0

    def update_epoch(self, epoch):
        self.current_epoch = epoch

    def spectral_convergence_loss(self, y_true, y_pred):
        return tf.norm(y_true - y_pred, ord='fro', axis=[-2, -1]) / (tf.norm(y_true, ord='fro', axis=[-2, -1]) + 1e-6)

    def log_mel_loss(self, y_true, y_pred):
        return tf.reduce_mean(tf.abs(tf.math.log(y_true + 1e-6) - tf.math.log(y_pred + 1e-6)))

    def call(self, y_true, y_pred):
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        l1 = tf.reduce_mean(tf.abs(y_true - y_pred))
        sc = self.spectral_convergence_loss(y_true, y_pred)

        loss = self.alpha * mse + (1 - self.alpha) * l1 + self.beta * sc

        if self.current_epoch >= self.log_start_epoch:
            log_loss = self.log_mel_loss(tf.abs(y_true), tf.abs(y_pred))
            loss += self.beta * log_loss

        return loss

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
vocab_size = len(g2p.phn2idx)

# ==================== Build & Train =====================

model = build_acoustic_model(vocab_size)
loss_fn = DelayedLogMelLoss(alpha=0.4, beta=0.3, log_start_epoch=10)
model.compile(optimizer='adam', loss=loss_fn, metrics=['mae', tf.keras.metrics.CosineSimilarity(axis=-1)])
model.summary()

callbacks = [
    EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1),
    ModelCheckpoint('model/2/best_model_cnn.keras', monitor='val_loss', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4,verbose=1),
    # LossEpochUpdater(loss_fn),
    LRSchedulerLogger(),
    LearningRatePlotter()
]

history = model.fit(
    train_dataset,
    epochs=100,
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




