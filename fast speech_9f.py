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



# def create_dataset_fast(texts, mel_paths, batch_size=32, shuffle=True):
#     def parse_data(phoneme_str, mel_path):
#         def _parse_numpy(phoneme_str_, mel_path_):
#             phonemes = np.array(ast.literal_eval(phoneme_str_.decode()), dtype=np.int32)  # Convert string to list, then to array
#             mel = np.load(mel_path_.decode()).astype(np.float32)  # Load .npy file
#             return phonemes, mel

#         phonemes, mel = tf.numpy_function(_parse_numpy, [phoneme_str, mel_path], [tf.int32, tf.float32])
#         phonemes.set_shape([None])      # Shape: (phoneme_seq_len,)
#         mel.set_shape([None, 80])       # Shape: (mel_frame_len, 80)
#         return phonemes, mel

#     dataset = tf.data.Dataset.from_tensor_slices((texts, mel_paths))
#     if shuffle:
#         dataset = dataset.shuffle(buffer_size=10000)    
#     dataset = dataset.map(lambda x, y: parse_data(x, y), num_parallel_calls=tf.data.AUTOTUNE)
#     # dataset=dataset.cache()
#     dataset = dataset.batch(batch_size)
#     dataset = dataset.prefetch(tf.data.AUTOTUNE)
#     return dataset

def create_dataset_fast(texts, durations, mel_paths, batch_size=32, shuffle=True):
    def parse_data(phoneme_str, duration_str, mel_path):
        def _parse_numpy(phoneme_str_, duration_str_, mel_path_):
            phonemes = np.array(ast.literal_eval(phoneme_str_.decode()), dtype=np.int32)
            durations = np.array(ast.literal_eval(duration_str_.decode()), dtype=np.float32)
            mel = np.load(mel_path_.decode()).astype(np.float32)
            return phonemes, durations, mel

        phonemes, durations, mel = tf.numpy_function(
            _parse_numpy, [phoneme_str, duration_str, mel_path],
            [tf.int32, tf.float32, tf.float32]
        )

        phonemes.set_shape([None])       # (T_phoneme,)
        durations.set_shape([None])      # (T_phoneme,)
        mel.set_shape([None, 80])        # (T_mel, 80)

        return {
            "phoneme_input": phonemes,
            "log_duration_input": durations
        }, mel

    dataset = tf.data.Dataset.from_tensor_slices((texts, durations, mel_paths))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)

    dataset = dataset.map(parse_data, num_parallel_calls=tf.data.AUTOTUNE)
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

class LengthRegulator(tf.keras.layers.Layer):
    def call(self, inputs):
        x, durations = inputs  # x: (B, T, D), durations: (B, T)
        durations = tf.cast(tf.round(durations), tf.int32)

        # Flatten the inputs to shape: (total_phonemes, D)
        batch_size = tf.shape(x)[0]
        max_phonemes = tf.shape(x)[1]
        embed_dim = tf.shape(x)[2]

        flat_x = tf.reshape(x, [-1, embed_dim])                   # (B*T, D)
        flat_durations = tf.reshape(durations, [-1])              # (B*T,)

        # Repeat each phoneme representation according to its duration
        repeated = tf.repeat(flat_x, flat_durations, axis=0)      # (sum(durations), D)

        # Build segment IDs to regroup into batches
        repeats_per_batch = tf.reduce_sum(durations, axis=1)      # (B,)
        segment_ids = tf.ragged.range(repeats_per_batch).with_row_splits_dtype(tf.int32)
        segment_ids = tf.repeat(tf.range(batch_size), repeats_per_batch)

        # Use ragged tensor to build output batch
        ragged = tf.RaggedTensor.from_value_rowids(repeated, segment_ids)
        padded = ragged.to_tensor()  # (B, T_mel, D)
        return padded

def build_fastspeech_model(vocab_size, max_phoneme_len=132, mel_dim=80, max_mel_len=865, embed_dim=256):
    phoneme_input = tf.keras.Input(shape=(None,), name="phoneme_input")
    duration_input = tf.keras.Input(shape=(None,), name="log_duration_input")  # log durations

    # Embedding
    x = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, mask_zero=True)(phoneme_input)
    x += layers.Embedding(input_dim=max_phoneme_len, output_dim=embed_dim, mask_zero=True)(tf.range(max_phoneme_len))  # positional
    # mask = x.compute_mask(phoneme_input)
    # mask = tf.not_equal(phoneme_input, 0)
    # Encoder (transformer or BiLSTM + CNN like yours)
    # x = layers.Bidirectional(layers.LSTM(512, return_sequences=True))(x)
      # BiLSTM stack
    x = layers.Bidirectional(layers.LSTM(512, return_sequences=True,dropout=0.3))(x)
    # x = layers.Dropout(0.3)(x)
    x = layers.Bidirectional(layers.LSTM(512, return_sequences=True,dropout=0.3))(x)
    # x = layers.Dropout(0.3)(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True,dropout=0.3))(x)
    # x = layers.Dropout(0.3)(x)
    
    for _ in range(2):
        residual = x
        x = layers.Conv1D(256, kernel_size=5,  padding="same", activation="relu")(x)
        x = layers.LayerNormalization()(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Add()([x, residual])

    # Duration predictor
    dur_pred = layers.Conv1D(256, 3, padding="same", activation="relu")(x)
    dur_pred = layers.LayerNormalization()(dur_pred)
    dur_pred = layers.Conv1D(1, 1)(dur_pred)
    dur_pred = layers.Flatten(name="log_duration_input")(dur_pred)

    # Length regulator (upsample encoder outputs)
    x = LengthRegulator()([x, duration_input])  # (B, T_mel, embed_dim)

    # Decoder
    x = layers.Conv1D(256, kernel_size=5, padding='same', activation='relu')(x)
    x = layers.Conv1D(128, kernel_size=5, padding='same', activation='relu')(x)

    mel_output = layers.Conv1D(mel_dim, kernel_size=1, activation='linear', name="mel_output")(x)

    # Post-net
    postnet = mel_output
    for _ in range(1):
        postnet = layers.Conv1D(mel_dim, kernel_size=5, padding='same', activation='tanh')(postnet)
        postnet = layers.LayerNormalization()(postnet)
        postnet = layers.Dropout(0.1)(postnet)

    mel_output = layers.Add(name="refined_mel_output")([mel_output, postnet])

    return tf.keras.Model(inputs=[phoneme_input, duration_input], outputs=mel_output)


def compile_model(model):
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.001,
        decay_steps=50 * len(train_dataset),
        alpha=0.1
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)

    def spectral_convergence_loss(y_true, y_pred):
        return tf.norm(y_true - y_pred, ord='fro', axis=[-2, -1]) / tf.norm(y_true, ord='fro', axis=[-2, -1])

    def log_mel_loss(y_true, y_pred):
        epsilon = 1e-5
        y_true = tf.nn.relu(y_true)
        y_pred = tf.nn.relu(y_pred)
        log_true = tf.math.log(y_true + epsilon)
        log_pred = tf.math.log(y_pred + epsilon)
        return tf.reduce_mean(tf.abs(log_true - log_pred))  # L1 in log-mel space

    def combined_loss(y_true, y_pred):
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        sc_loss = spectral_convergence_loss(y_true, y_pred)
        log_loss = log_mel_loss(y_true, y_pred)
        return mse + 0.5 * sc_loss + 0.1 * log_loss

    # Secondary duration prediction loss (e.g., MSE in log space)
    def duration_loss(y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred))

    model.compile(optimizer=optimizer,loss=combined_loss,metrics=['mae', tf.keras.metrics.CosineSimilarity(axis=-1)]
    )
    return model
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

train_dataset = create_dataset_fast(texts_train,dura_train, mel_train )
val_dataset = create_dataset_fast(texts_val, dura_val,mel_val )
test_dataset = create_dataset_fast(texts_test,dura_test, mel_test)

g2p = G2PConverter(load_model=False)
# print(g2p.phn2idx)
vocab_size = len(g2p.phn2idx)

# ==================== Build & Train =====================

# model = build_acoustic_model(vocab_size, input_length)
model = build_fastspeech_model(vocab_size,870)
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
print(train_dataset) 
for x, y in train_dataset.take(1):  # Show 5 samples
    tf.print("Input:", str(x['log_duration_input'][0]),str(x['phoneme_input'][0]))
    tf.print("Target:", y.shape)

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

