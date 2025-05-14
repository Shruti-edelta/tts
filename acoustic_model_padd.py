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

@tf.keras.utils.register_keras_serializable()
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

def build_acoustic_model(vocab_size, input_length, mel_dim=80, mel_len=900, embed_dim=256):
    inputs = layers.Input(shape=(input_length,), name='phoneme_input')

    embedding = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, mask_zero=True)
    x = embedding(inputs)
    mask = embedding.compute_mask(inputs)

    # BiLSTM Stack
    x = layers.Bidirectional(layers.LSTM(512, return_sequences=True))(x, mask=mask)
    x = layers.Dropout(0.1)(x)
    # print(x.shape)
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x, mask=mask)
    x = layers.Dropout(0.1)(x)
    # print(x.shape)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x, mask=mask) 
    x = layers.Dropout(0.1)(x)
    # print(x.shape)

    # Residual CNN blocks
    for i in range(3):
        residual = x
        x = layers.Conv1D(filters=256, kernel_size=5, padding='same', dilation_rate=2, activation='relu')(x)
        x = layers.LayerNormalization()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Add()([x, residual])

    # Multi-Head Self-Attention
    # expanded_mask =tf.expand_dims(mask, axis=1)
    attn = layers.MultiHeadAttention(num_heads=4, key_dim=128)(x, x)
    x = layers.Add()([x, attn])
    x = layers.LayerNormalization()(x)

    # Upsampling with Conv1DTranspose
    x = layers.Conv1DTranspose(256, kernel_size=5, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv1DTranspose(128, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv1DTranspose(128, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    x = CropLayer(mel_len)(x)

    # Initial Mel Output
    mel_output = layers.Conv1D(mel_dim, kernel_size=1, activation='linear', name="mel_output")(x)

    # Post-Net for refinement
    postnet = mel_output
    for i in range(5):
        postnet = layers.Conv1D(filters=mel_dim, kernel_size=5, padding='same', activation='tanh')(postnet)
        postnet = layers.BatchNormalization()(postnet)
    mel_output = layers.Add(name="refined_mel_output")([mel_output, postnet])

    model = tf.keras.Model(inputs=inputs, outputs=mel_output)
    return model

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
    dataset = dataset.map(tf_wrapper, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def masked_loss(y_true, y_pred):
    # Mask is 1 where y_true is not zero (valid mel frames)
    mask = tf.reduce_any(tf.not_equal(y_true, 0.0), axis=-1)  # (batch, time)
    mask = tf.cast(mask, tf.float32)

    # Compute MSE
    mse = tf.reduce_sum(tf.square(y_true - y_pred), axis=-1)
    mse = tf.reduce_sum(mse * mask) / tf.reduce_sum(mask)

    # Cosine similarity
    cos_sim = tf.keras.losses.cosine_similarity(y_true, y_pred, axis=-1)
    cos_sim = tf.reduce_sum(cos_sim * mask) / tf.reduce_sum(mask)

    return mse + 0.3 * cos_sim

def compile_model(model, train_dataset):
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.001,
        decay_steps=50 * len(train_dataset),
        alpha=0.1
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)    

    model.compile(
        optimizer=optimizer,
        loss=masked_loss,
        metrics=['mae', CosineSimilarity(axis=-1)]
    )
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

df = pd.read_csv('dataset/acoustic_dataset/tts_data_LJ.csv', usecols=['Phoneme_text', 'Read_npy'])
texts = df['Phoneme_text'].apply(ast.literal_eval).values
mel_spectrograms = df['Read_npy'].values
input_length = max([len(seq) for seq in texts])

texts_str = [str(seq) for seq in texts]
texts_train, texts_temp, mel_train, mel_temp = train_test_split(texts_str, mel_spectrograms, test_size=0.1, random_state=33)
texts_val, texts_test, mel_val, mel_test = train_test_split(texts_temp, mel_temp, test_size=0.3, random_state=33)

train_dataset = create_dataset_fast(texts_train, mel_train, input_length=input_length)
val_dataset = create_dataset_fast(texts_val, mel_val, input_length=input_length)
test_dataset = create_dataset_fast(texts_test, mel_test, input_length=input_length)

g2p = G2PConverter(load_model=False)
vocab_size = len(g2p.phn2idx)

# ==================== Build & Train =====================
model = build_acoustic_model(vocab_size, input_length=input_length)
model = compile_model(model, train_dataset)
model.summary()

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ModelCheckpoint('model/2/best_model_cnn.keras', monitor='val_loss', save_best_only=True, verbose=1),
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




# def build_acoustic_model(vocab_size, input_length, mel_dim=80, mel_len=900, embed_dim=256):
#     inputs = layers.Input(shape=(input_length,), name='phoneme_input')

#     embedding = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, mask_zero=True)
#     x = embedding(inputs)
#     mask = embedding.compute_mask(inputs)

#     for i in range(3):
#         residual = x
#         x = layers.Conv1D(filters=256, kernel_size=5, padding='same', dilation_rate=2, activation='relu')(x)
#         x = layers.LayerNormalization()(x)
#         x = layers.Dropout(0.2)(x)
#         x = layers.Add()([x, residual])

#     x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x, mask=mask)
#     x = layers.Dropout(0.3)(x)

#     attention = layers.Dense(1, activation='tanh')(x)
#     attention = layers.Softmax(axis=1)(attention)
#     x = layers.Multiply()([x, attention])

#     x = layers.Conv1DTranspose(256, kernel_size=5, strides=2, padding='same', activation='relu')(x)
#     x = layers.Conv1DTranspose(128, kernel_size=3, strides=2, padding='same', activation='relu')(x)
#     x = layers.Conv1DTranspose(128, kernel_size=3, strides=2, padding='same', activation='relu')(x)
#     x = CropLayer(mel_len)(x)

#     mel_output = layers.Conv1D(mel_dim, kernel_size=1, activation='linear', name="mel_output")(x)
#     refinement = layers.Conv1D(mel_dim, kernel_size=5, padding='same', activation='tanh')(mel_output)
#     mel_output = layers.Add()([mel_output, refinement])

#     model = tf.keras.Model(inputs=inputs, outputs=mel_output)
#     return model