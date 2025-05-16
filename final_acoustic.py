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


'''
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

class LocationSensitiveAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.query_layer = layers.Dense(units)
        self.memory_layer = layers.Dense(units)
        self.location_layer = layers.Conv1D(units, kernel_size=31, padding='same')
        self.V = layers.Dense(1)

    def call(self, query, memory, prev_attention):
        # query: (batch, units)
        # memory: (batch, time, units)
        # prev_attention: (batch, time, 1)

        processed_query = self.query_layer(tf.expand_dims(query, 1))  # (batch, 1, units)
        processed_memory = self.memory_layer(memory)  # (batch, time, units)
        processed_location = self.location_layer(prev_attention)  # (batch, time, units)

        score = self.V(tf.nn.tanh(processed_query + processed_memory + processed_location))  # (batch, time, 1)
        attention_weights = tf.nn.softmax(score, axis=1)  # (batch, time, 1)

        context_vector = tf.reduce_sum(attention_weights * memory, axis=1)  # (batch, units)
        return context_vector, attention_weights

# @register_keras_serializable()
# class AdditiveAttention(tf.keras.layers.Layer):
#     def __init__(self, units, dropout_rate=0.1, **kwargs):
#         super(AdditiveAttention, self).__init__(**kwargs)
#         self.W_query = layers.Dense(units)
#         self.W_values = layers.Dense(units)
#         self.V = layers.Dense(1)
#         self.dropout = layers.Dropout(dropout_rate)
#         self.units = units

#     def call(self, query, values, mask=None):
#         query = tf.expand_dims(query, 1)  # Shape: (B, 1, D)
#         score = self.V(tf.nn.tanh(self.W_query(query) + self.W_values(values)))  # (B, T, 1)
#         score = tf.squeeze(score, axis=-1)  # (B, T)
#         score = score / tf.math.sqrt(tf.cast(self.units, tf.float32))  # Scale

#         if mask is not None:
#             score += (1.0 - tf.cast(mask, tf.float32)) * -1e9  # Masking

#         score = self.dropout(score)
#         attention_weights = tf.nn.softmax(score, axis=-1)  # (B, T)
#         context = tf.matmul(tf.expand_dims(attention_weights, 1), values)  # (B, 1, D)
#         return tf.squeeze(context, axis=1), attention_weights  # (B, D)

# @register_keras_serializable()
# class AttentionContextLayer(tf.keras.layers.Layer):
#     def __init__(self, units, input_length, **kwargs):
#         super(AttentionContextLayer, self).__init__(**kwargs)
#         self.attention = AdditiveAttention(units)
#         self.input_length = input_length

#     def call(self, inputs):
#         query, values, mask = inputs
#         context_vector, _ = self.attention(query, values, mask=mask)
#         context_expanded = tf.expand_dims(context_vector, axis=1)
#         context_expanded = tf.tile(context_expanded, [1, self.input_length, 1])
#         return tf.concat([values, context_expanded], axis=-1)  # (B, T, D + context)


@tf.keras.utils.register_keras_serializable()
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, input_length, embed_dim, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.pos_embedding = tf.keras.layers.Embedding(input_dim=input_length, output_dim=embed_dim)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        positions = tf.expand_dims(positions, 0)
        pos_encoded = self.pos_embedding(positions)
        return pos_encoded

@tf.keras.utils.register_keras_serializable()
class LastTimestep(tf.keras.layers.Layer):
    def call(self, inputs):
        return inputs[:, -1, :]

def build_acoustic_model(vocab_size, input_length=163, mel_dim=80, mel_len=865, embed_dim=256):
    inputs = layers.Input(shape=(None,), name='phoneme_input')

    # --- Embedding ---
    embedding = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, mask_zero=True)
    x = embedding(inputs)
    mask = embedding.compute_mask(inputs)

    # # --- Dynamic Positional Embedding ---
    pos_embedding = layers.Embedding(input_dim=input_length, output_dim=embed_dim)  # Larger input dim

    # pos_encoded = PositionalEncoding(input_length, embed_dim)(x)
    # x = layers.Add()([x, pos_encoded])

    positions = tf.range(start=0, limit=tf.shape(x)[1])
    positions = tf.expand_dims(positions, 0)
    pos_encoded = pos_embedding(positions)
    x = x + pos_encoded

    # --- BiLSTM Encoder ---
    x = layers.Bidirectional(layers.LSTM(512, return_sequences=True))(x, mask=mask)
    x = layers.Bidirectional(layers.LSTM(512, return_sequences=True))(x, mask=mask)
    x = layers.Dropout(0.3)(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x, mask=mask)

    # --- Residual CNN Blocks ---
    for _ in range(3):
        residual = x
        x = layers.Conv1D(256, kernel_size=5, padding='same', dilation_rate=2, activation='relu')(x)
        x = layers.LayerNormalization()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Add()([x, residual])

    # --- Attention ---
    # query = layers.Lambda(lambda x: x[:, -1, :])(x)  # Use last timestep as query
    query = LastTimestep()(x)
    # x = AttentionContextLayer(units=256, input_length=input_length)([query, x, mask])


    # --- Decoder (LSTM stack) ---
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)
    x = layers.LayerNormalization()(x)

    # --- Upsampling ---
    x = layers.Conv1DTranspose(256, kernel_size=5, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv1DTranspose(128, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv1DTranspose(64, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    x = CropLayer(mel_len)(x)

    # --- Initial Mel Output ---
    mel_output = layers.Conv1D(mel_dim, kernel_size=1, activation='linear', name="mel_output")(x)

    # --- Post-net ---
    postnet = mel_output
    for _ in range(5):
        postnet = layers.Conv1D(mel_dim, kernel_size=5, padding='same', activation='tanh')(postnet)
        postnet = layers.BatchNormalization()(postnet)
    mel_output = layers.Add(name="refined_mel_output")([mel_output, postnet])

    model = tf.keras.Model(inputs=inputs, outputs=mel_output)
    return model'''

import tensorflow as tf
from tensorflow.keras import layers, models, Model
from tensorflow.keras.layers import LayerNormalization, Dropout, Add, UpSampling1D
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.layers import LayerNormalization, Dropout, MultiHeadAttention, Dense, Add
from tensorflow.keras.layers import Conv1D, Activation, Multiply, Add, Dense

class GLU(layers.Layer):
    def call(self, x):
        a, b = tf.split(x, num_or_size_splits=2, axis=-1)
        return a * tf.sigmoid(b)

class CropLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        x, target = inputs
        target_length = tf.shape(target)[1]
        return x[:, :target_length, :]


def residual_glu_cnn_block(x, filters, kernel_size, dilation_rate=1, dropout=0.1):
    residual = x  # Save for residual connection

    # Gated Linear Unit (GLU)
    conv1 = Conv1D(filters, kernel_size, padding='same', dilation_rate=dilation_rate)(x)
    conv2 = Conv1D(filters, kernel_size, padding='same', dilation_rate=dilation_rate)(x)
    conv2 = Activation('sigmoid')(conv2)

    x = Multiply()([conv1, conv2])
    x = tf.keras.layers.Dropout(dropout)(x)

    # 🔧 Project `residual` to match `x` shape if needed
    if residual.shape[-1] != x.shape[-1]:
        residual = Dense(x.shape[-1])(residual)

    x = Add()([x, residual])
    return x

def transformer_block(x, num_heads, key_dim, ff_dim=None, dropout=0.1):
    # Multi-head attention
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x)
    attn_output = Dropout(dropout)(attn_output)
    out1 = LayerNormalization()(x + attn_output)

    # Feed-forward network
    if ff_dim is None:
        ff_dim = x.shape[-1]  # default to match input dim

    ffn = Dense(ff_dim, activation='relu')(out1)
    ffn = Dense(x.shape[-1])(ffn)  # 👈 project back to original dimension
    ffn = Dropout(dropout)(ffn)

    out2 = LayerNormalization()(out1 + ffn)
    return out2

def build_acoustic_model(input_dim, output_dim, max_input_len, max_output_len,
                         cnn_filters=512, cnn_kernel_size=5, cnn_layers=4,
                         lstm_units=256, dropout_rate=0.2, num_heads=4, head_dim=64):

    phoneme_inputs = layers.Input(shape=(max_input_len,), name='phoneme_input')

    x = layers.Embedding(input_dim=input_dim, output_dim=256, mask_zero=True)(phoneme_inputs)
    x = layers.Dropout(dropout_rate)(x)

    # Positional Encoding
    positions = tf.range(start=0, limit=max_input_len, delta=1)
    pos_embedding = layers.Embedding(input_dim=max_input_len, output_dim=256)(positions)
    x = x + pos_embedding

    # Transformer Block
    x = transformer_block(x, num_heads=num_heads, key_dim=head_dim, dropout=dropout_rate)

    # Residual GLU CNN Blocks
    for i in range(cnn_layers):
        x = residual_glu_cnn_block(x, filters=cnn_filters,
                                   kernel_size=cnn_kernel_size,
                                   dilation_rate=2**i,
                                   dropout=dropout_rate)

    # BiLSTM
    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True))(x)
    x = layers.LayerNormalization()(x)

    # Additive Attention
    attention_output = layers.Attention()([x, x])  # You can replace with LocationSensitiveAttention if desired
    x = Add()([x, attention_output])

    # Decoder: Upsampling + Conv1D
    x = UpSampling1D(size=2)(x)
    x = layers.Conv1D(512, kernel_size=5, padding='same', activation='relu')(x)
    x = UpSampling1D(size=2)(x)
    x = layers.Conv1D(256, kernel_size=5, padding='same', activation='relu')(x)
    x = layers.Conv1D(output_dim, kernel_size=1, padding='same')(x)

    mel_outputs = CropLayer()([x, x])  # crops to match mel target shape

    # Post-net
    postnet = layers.Conv1D(512, kernel_size=5, padding='same', activation='tanh')(mel_outputs)
    postnet = Dropout(dropout_rate)(postnet)
    postnet = layers.Conv1D(output_dim, kernel_size=5, padding='same')(postnet)
    mel_outputs_postnet = layers.Add(name="mel_outputs_postnet")([mel_outputs, postnet])

    return Model(inputs=phoneme_inputs, outputs=[mel_outputs_postnet])



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
        # Ensure non-negative values before log
        y_true = tf.maximum(y_true, 1e-5)
        y_pred = tf.maximum(y_pred, 1e-5)
        return tf.reduce_mean(tf.abs(tf.math.log(y_true) - tf.math.log(y_pred)))

    def combined_loss(y_true, y_pred):
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        sc_loss = spectral_convergence_loss(y_true, y_pred)
        log_loss = log_mel_loss(y_true, y_pred)
        return mse + 0.5 * sc_loss + 0.5 * log_loss  # Weighted sum

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

# model = build_acoustic_model(vocab_size)
model = build_acoustic_model(
    input_dim=vocab_size,
    output_dim=80,  # 👈 Make sure this matches your mel dimension
    max_input_len=245,
    max_output_len=1045
)
model = compile_model(model)
model.summary()

# Set up log directory with timestamp
log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1,write_graph=True,write_images=True)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    ModelCheckpoint('model/2/best_model_cnn_9f.keras', monitor='val_loss', save_best_only=True, verbose=1),
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
model.save('model/2/acoustic_model_cnn_9f.keras')
model.save_weights('model/2/acoustic_model_cnn_9f.weights.h5')
history_df = pd.DataFrame(history.history)
history_df.to_csv('model/2/acoustic_model_cnn_9f.csv', index=False)

# ==================== Evaluation =====================
test_loss = model.evaluate(test_dataset)
print(f"Test loss: {test_loss}")


