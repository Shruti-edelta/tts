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
            phonemes = np.array(ast.literal_eval(phoneme_str_.decode()), dtype=np.int32)
            mel = np.load(mel_path_.decode()).astype(np.float32)
            return phonemes, mel

        phonemes, mel = tf.numpy_function(_parse_numpy, [phoneme_str, mel_path], [tf.int32, tf.float32])
        phonemes.set_shape([None])       # already padded
        mel.set_shape([None, 80])        # already padded
        return phonemes, mel

    def add_attention_input(phonemes, mel):
        att_input = tf.zeros((tf.shape(phonemes)[0]), dtype=tf.float32)  # shape (mel_len, 1)
        return (phonemes, att_input), mel

    dataset = tf.data.Dataset.from_tensor_slices((texts, mel_paths))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)

    dataset = dataset.map(parse_data, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(add_attention_input, num_parallel_calls=tf.data.AUTOTUNE)

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
class LocationSensitiveAttention(tf.keras.layers.Layer):
    def __init__(self, units, filters=32, kernel_size=31, **kwargs):
        super(LocationSensitiveAttention, self).__init__(**kwargs)
        self.units = units
        self.filters = filters
        self.kernel_size = kernel_size
        self.W_query = layers.Dense(units)
        self.W_values = layers.Dense(units)
        self.W_location = layers.Dense(units)
        self.V = layers.Dense(1)
        self.location_conv = layers.Conv1D(filters, kernel_size, padding='same', use_bias=False)

    def call(self, query, values, prev_attention, mask=None):
        # prev_attention: (batch_size, seq_len)
        query = tf.expand_dims(query, 1)  # (batch_size, 1, dim)

        # Location features
        f = self.location_conv(tf.expand_dims(prev_attention, -1))  # (batch_size, seq_len, filters)
        location_features = self.W_location(f)

        # Score calculation
        score = self.V(tf.nn.tanh(
            self.W_query(query) + self.W_values(values) + location_features))  # (batch, seq_len, 1)
        score = tf.squeeze(score, -1)

        if mask is not None:
            score += (1.0 - tf.cast(mask, tf.float32)) * -1e9

        attention_weights = tf.nn.softmax(score, axis=-1)
        context_vector = tf.matmul(tf.expand_dims(attention_weights, 1), values)
        context_vector = tf.squeeze(context_vector, axis=1)

        return context_vector, attention_weights

@register_keras_serializable()
class LocationAttentionContextLayer(tf.keras.layers.Layer):
    def __init__(self, units, input_length, **kwargs):
        super(LocationAttentionContextLayer, self).__init__(**kwargs)
        self.units = units
        self.input_length = input_length

    def build(self, input_shape):
        self.attention = LocationSensitiveAttention(units=self.units)
        super().build(input_shape)

    def call(self, inputs):
        query, values, prev_attention, mask = inputs
        context_vector, attention_weights = self.attention(query, values, prev_attention, mask=mask)
        context_expanded = tf.expand_dims(context_vector, axis=1)
        context_expanded = tf.tile(context_expanded, [1, self.input_length, 1])
        return tf.concat([values, context_expanded], axis=-1), attention_weights


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
def build_acoustic_model(vocab_size, input_length=163, mel_dim=80, mel_len=870, embed_dim=256):
    # inputs = layers.Input(shape=(None,), name='phoneme_input')
    phoneme_input = layers.Input(shape=(None,), name='phoneme_input')
    prev_attention_input = layers.Input(shape=(input_length,), name='prev_attention')
    # Embedding + Positional Encoding
    embedding = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, mask_zero=True)
    x = embedding(phoneme_input)
    mask = embedding.compute_mask(phoneme_input)
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
    prev_attention_input = layers.Input(shape=(input_length,), name="prev_attention")  # New input

    attention_output, attention_weights = LocationAttentionContextLayer(
        units=256, input_length=input_length)([query, x, prev_attention_input, mask])

    x = attention_output

    # Upsample
    x = layers.Conv1DTranspose(256, kernel_size=5, strides=2, padding='same', activation='relu')(x) # (None, 336, 256) (None, 400, 256)
    x = layers.Conv1DTranspose(128, kernel_size=3, strides=2, padding='same', activation='relu')(x) # (None, 800, 128)
    x = layers.Conv1DTranspose(64, kernel_size=3, strides=2, padding='same', activation='relu')(x) # (None, 1600, 128) 
    x = CropLayer(mel_len)(x)
    # x = crop_layer([x, mel_inputs])  # Align output to match target mel length


    # Initial mel output
    mel_output = layers.Conv1D(mel_dim, kernel_size=1, activation='linear', name="mel_output")(x)

    # Post-net for refinement
    postnet = mel_output
    # for _ in range(5):
    #     postnet = layers.Conv1D(filters=mel_dim, kernel_size=5, padding='same', activation='tanh')(postnet)
    #     postnet = layers.BatchNormalization()(postnet)

    for i in range(5):
        activation = 'tanh' if i < 4 else None
        postnet = layers.Conv1D(mel_dim, kernel_size=5, padding='same', activation=activation)(postnet)
        if i < 4:
            postnet = layers.BatchNormalization()(postnet)
            postnet = layers.Dropout(0.1)(postnet)

    mel_output = layers.Add(name="refined_mel_output")([mel_output, postnet])

    # model = tf.keras.Model(inputs=phoneme_input, outputs=mel_output)
    model = tf.keras.Model(inputs=[phoneme_input, prev_attention_input], outputs=mel_output)
    return model

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
        return mse + 0.5 * sc_loss + 0.2 * log_loss
    
    model.compile(optimizer=optimizer,
                  loss=combined_loss,
                  metrics=['mae', CosineSimilarity(axis=-1)])
    return model
# ==================== Load Data =====================


df_train = pd.read_csv('dataset/acoustic_dataset(9f_eos_notduration)/train.csv', usecols=['Phoneme_text', 'Read_npy'])
df_val = pd.read_csv('dataset/acoustic_dataset(9f_eos_notduration)/val.csv', usecols=['Phoneme_text', 'Read_npy'])
df_test = pd.read_csv('dataset/acoustic_dataset(9f_eos_notduration)/test.csv', usecols=['Phoneme_text', 'Read_npy'])

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


    # def stft_loss(y_true, y_pred, fft_size=1024, hop_size=256, win_length=1024):

    #     print("===")
    #     print("y_true: ",y_true.shape)
    #     print("y_pred: ",y_pred.shape)
    #     # STFT
    #     y_true_stft = tf.signal.stft(y_true, frame_length=win_length, frame_step=hop_size, fft_length=fft_size)
    #     y_pred_stft = tf.signal.stft(y_pred, frame_length=win_length, frame_step=hop_size, fft_length=fft_size)
        
    #     print("y_true_stft: ",y_true_stft.shape)
    #     print("y_pred_stft: ",y_pred_stft.shape)

    #     # Magnitudes
    #     y_true_mag = tf.abs(y_true_stft)
    #     y_pred_mag = tf.abs(y_pred_stft)
        
    #     # Small epsilon for stability
    #     eps = 1e-6
    #     print(y_pred_stft)
    #     # Spectral Convergence (Stable)
    #     diff = y_true_mag - y_pred_mag
    #     diff_norm = tf.sqrt(tf.reduce_sum(tf.square(diff), axis=[-2, -1]))
    #     true_norm = tf.sqrt(tf.reduce_sum(tf.square(y_true_mag), axis=[-2, -1]))
    #     sc_loss = tf.reduce_mean(diff_norm / true_norm)
        
    #     # Magnitude Loss
    #     mag_loss = tf.reduce_mean(tf.abs(y_true_mag - y_pred_mag))
        
    #     return sc_loss + mag_loss


