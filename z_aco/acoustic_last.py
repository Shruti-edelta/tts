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
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,TensorBoard,ReduceLROnPlateau
from tensorflow.keras.metrics import CosineSimilarity
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.saving import register_keras_serializable
from tensorflow.keras.layers import Layer
# import tensorflow.keras.backend as K

def create_dataset_fast(texts, mel_paths, batch_size=32, shuffle=True):
    def parse_data(phoneme_str, mel_path):
        def _parse_numpy(phoneme_str_, mel_path_):
            phonemes = np.array(ast.literal_eval(phoneme_str_.decode()), dtype=np.int32)  # Convert string to list, then to array
            mel = np.load(mel_path_.decode()).astype(np.float32)  # Load .npy file
            return phonemes, mel

        phonemes, mel = tf.numpy_function(_parse_numpy, [phoneme_str, mel_path], [tf.int32, tf.float32])
        phonemes.set_shape([None])      # Shape: (phoneme_seq_len,)
        mel.set_shape([None, 80])       # Shape: (mel_frame_len, 80)
        return ((phonemes, mel),mel)

    dataset = tf.data.Dataset.from_tensor_slices((texts, mel_paths))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)    
    dataset = dataset.map(lambda x, y: parse_data(x, y), num_parallel_calls=tf.data.AUTOTUNE)
    # dataset=dataset.cache()
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

# Custom CropLayer to match output length to target mel
@tf.keras.utils.register_keras_serializable()
class CropLayer(Layer):
    def call(self, inputs, target):
        return inputs[:, :tf.shape(target)[1], :]

# Positional Encoding
class PositionalEncoding(Layer):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim

    def call(self, x):
        position = tf.range(tf.shape(x)[1], dtype=tf.float32)[:, tf.newaxis]
        i = tf.range(self.dim, dtype=tf.float32)[tf.newaxis, :]
        angle_rates = 1 / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(self.dim, tf.float32))
        angle_rads = position * angle_rates
        pos_encoding = tf.where(i % 2 == 0, tf.sin(angle_rads), tf.cos(angle_rads))
        return x + pos_encoding

# PostNet block (as in Tacotron2)
def PostNet(n_mels):
    model = tf.keras.Sequential(name="PostNet")
    model.add(layers.Conv1D(512, 5, padding="same", activation="tanh"))
    for _ in range(3):
        model.add(layers.Conv1D(512, 5, padding="same", activation="tanh"))
    model.add(layers.Conv1D(n_mels, 5, padding="same", activation=None))
    return model

# Build the model
def build_tts_model(vocab_size, n_mels=80, embedding_dim=256, rnn_units=256, max_mel_len=1000):
    inputs = layers.Input(shape=(None,), dtype='int32', name='phoneme_input',mask_zero=True)
    target_mel = layers.Input(shape=(None, n_mels), dtype='float32', name='mel_target')

    # Embedding + Positional
    x = layers.Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs)
    x = PositionalEncoding(embedding_dim)(x)

    # CNN Encoder
    for _ in range(3):
        x = layers.Conv1D(256, 5, padding='same', activation='relu')(x)
        x = layers.LayerNormalization()(x)

    # BiLSTM
    x = layers.Bidirectional(layers.LSTM(rnn_units, return_sequences=True))(x)

    # Attention
    attention = tf.keras.layers.AdditiveAttention()
    context_vector = attention([x, x])
    # context_vector, attention_weights = layers.Attention()([x, x]), None

    # Decoder
    x = layers.Conv1DTranspose(128, 4, strides=6, padding="same", activation="relu")(context_vector)
    x = layers.Conv1D(n_mels, 3, padding="same")(x)

    # Crop to match length
    cropped = CropLayer()(x, target_mel)

    # PostNet
    postnet = PostNet(n_mels)
    refined = postnet(cropped)
    final_output = layers.Add()([cropped, refined])

    model = tf.keras.Model([inputs, target_mel], final_output)
    return model

def spectral_convergence(y_true, y_pred):
    stft_true = tf.signal.stft(y_true[..., 0], frame_length=256, frame_step=128)
    stft_pred = tf.signal.stft(y_pred[..., 0], frame_length=256, frame_step=128)
    return tf.norm(tf.abs(stft_true) - tf.abs(stft_pred), ord='fro', axis=[-2, -1]) / tf.norm(tf.abs(stft_true), ord='fro', axis=[-2, -1])

def log_mel_loss(y_true, y_pred):
    return tf.reduce_mean(tf.abs(tf.math.log(y_true + 1e-5) - tf.math.log(y_pred + 1e-5)))

def combined_loss(y_true, y_pred):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    log_mel = log_mel_loss(y_true, y_pred)
    sc = spectral_convergence(y_true, y_pred)
    return mse + 0.1 * log_mel + 0.5 * sc


def compile_model(model):
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=0.001,
            decay_steps=50 * len(train_dataset),
            alpha=0.1
        )
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)    
    # optimizer = tfa.optimizers.AdamW(weight_decay=1e-4, learning_rate=lr_schedule)

    def spectral_convergence_loss(y_true, y_pred):
        return tf.norm(y_true - y_pred, ord='fro', axis=[-2, -1]) / tf.norm(y_true, ord='fro', axis=[-2, -1])

    # def log_mel_loss(y_true, y_pred):
    #     return tf.reduce_mean(tf.math.log(y_true + 1e-6) - tf.math.log(y_pred + 1e-6))

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
    
    model.compile(optimizer=optimizer,
                  loss=combined_loss,
                  metrics=['mae', CosineSimilarity(axis=-1)])
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
model = build_tts_model(vocab_size)
model = compile_model(model)
model.summary()

print(train_dataset) 
for x, y in train_dataset.take(1):  # Show 5 samples
    tf.print("Input:", x)
    tf.print("Target:", y.shape)

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

