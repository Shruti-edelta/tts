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
class AdditiveAttention(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_query = tf.keras.layers.Dense(units)
        self.W_values = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values, mask=None):
        score = self.V(tf.nn.tanh(self.W_query(query) + self.W_values(values)))
        score = tf.squeeze(score, axis=-1)
        if mask is not None:
            score += (1.0 - tf.cast(mask, tf.float32)) * -1e9
        attention_weights = tf.nn.softmax(score, axis=-1)
        context = tf.matmul(tf.expand_dims(attention_weights, 1), values)
        context = tf.squeeze(context, axis=1)
        return context, attention_weights

@register_keras_serializable()
class AttentionStep(tf.keras.layers.Layer):
    def __init__(self, attention, query_proj, value_dim, **kwargs):
        super().__init__(**kwargs)
        self.attention = attention
        self.query_proj = query_proj
        self.value_dim = value_dim

        # Required: define state_size (here, values and mask are not changing, so we set state_size as needed)
        self.state_size = [tf.TensorShape([None, value_dim]),  # shape for values
                           tf.TensorShape([None])]             # shape for mask

    def call(self, inputs, states):
        # inputs: (batch_size, query_dim) at each time step
        query_t = inputs  # (B, query_dim)
        values, mask = states

        query_t_proj = self.query_proj(query_t)  # (B, value_dim)
        context_vector, _ = self.attention(query_t_proj, values, mask)  # (B, value_dim)
        output = tf.concat([query_t, context_vector], axis=-1)  # (B, query_dim + value_dim)

        return output, states  # states remain unchanged


@register_keras_serializable()
class AttentionContextLayer(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.attention = AdditiveAttention(units)
        self.query_proj = tf.keras.layers.Dense(units)
        self.units = units  # store for reuse

    def call(self, inputs):
        query_seq, values, mask = inputs  # (B, T_q, D_q), (B, T_v, D_v), (B, T_v)

        # Setup RNN with attention step cell
        cell = AttentionStep(self.attention, self.query_proj, value_dim=values.shape[-1])
        rnn = tf.keras.layers.RNN(cell, return_sequences=True)

        # Run RNN
        context_seq = rnn(query_seq, initial_state=[values, mask])
        return context_seq



# ==== MAIN MODEL FUNCTION ====
def build_acoustic_model(vocab_size, input_length=163, mel_dim=80, mel_len=865, embed_dim=256):
    inputs = layers.Input(shape=(None,), name='phoneme_input')

    # Embedding + Positional Encoding
    embedding = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, mask_zero=True)
    x = embedding(inputs)
    mask = embedding.compute_mask(inputs)

    pos_embedding = layers.Embedding(input_dim=input_length, output_dim=embed_dim)
    positions = tf.range(start=0, limit=input_length, delta=1)
    pos_encoded = pos_embedding(positions)
    x = x + pos_encoded

    # BiLSTM stack
    x = layers.Bidirectional(layers.LSTM(512, return_sequences=True, dropout=0.3))(x, mask=mask)
    x = layers.Bidirectional(layers.LSTM(512, return_sequences=True, dropout=0.3))(x, mask=mask)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.3))(x, mask=mask)

    # Residual CNN blocks
    for _ in range(3):
        residual = x
        x = layers.Conv1D(256, kernel_size=5, padding='same', dilation_rate=2, activation='relu')(x)
        x = layers.LayerNormalization()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Add()([x, residual])

    # Define Query (Learnable Embedding)
    def get_query(tensor):
        batch_size = tf.shape(tensor)[0]
        query_positions = tf.range(start=0, limit=mel_len, delta=1)
        query_embedding = layers.Embedding(input_dim=mel_len, output_dim=x.shape[-1])
        query = query_embedding(query_positions)  # (mel_len, dim)
        query = tf.expand_dims(query, axis=0)     # (1, mel_len, dim)
        query = tf.tile(query, [batch_size, 1, 1]) # (B, mel_len, dim)
        return query

    query = layers.Lambda(get_query)(inputs)  # Safe dynamic query

    # Apply Attention for each time-step
    context_layer = AttentionContextLayer(units=256)
    x = context_layer([query, x, mask])  # (B, mel_len, context_dim + value_dim)

    # # Upsampling
    # x = layers.Conv1DTranspose(256, kernel_size=5, strides=2, padding='same', activation='relu')(x)
    # x = layers.Conv1DTranspose(128, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    # x = layers.Conv1DTranspose(64, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    # x = CropLayer(mel_len)(x)

    # Mel output
    mel_output = layers.Conv1D(mel_dim, kernel_size=1, activation='linear', name="mel_output")(x)

    # PostNet
    postnet = mel_output
    for i in range(5):
        activation = 'tanh' if i < 4 else None
        postnet = layers.Conv1D(mel_dim, kernel_size=5, padding='same', activation=activation)(postnet)
        if i < 4:
            postnet = layers.BatchNormalization()(postnet)
            postnet = layers.Dropout(0.1)(postnet)

    mel_output = layers.Add(name="refined_mel_output")([mel_output, postnet])

    return tf.keras.Model(inputs=inputs, outputs=mel_output)


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
    tf.print("Input:", x.shape)
    tf.print("Target:", y.shape)

# Set up log directory with timestamp
log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1,write_graph=True,write_images=True)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    ModelCheckpoint('model/2/best_model_cnn_9f_log-att.keras', monitor='val_loss', save_best_only=True, verbose=1),
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
model.save('model/2/model_cnn_9f_log-att.keras')
model.save_weights('model/2/best_model_cnn_9f_log-att.weights.h5')
history_df = pd.DataFrame(history.history)
history_df.to_csv('model/2/best_model_cnn_9f_log-att.csv', index=False)

# ==================== Evaluation =====================
test_loss = model.evaluate(test_dataset)
print(f"Test loss: {test_loss}")


