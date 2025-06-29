import os
import datetime
import tensorflow as tf
import numpy as np
import pandas as pd
import ast
from acoustic.text_preprocess import G2PConverter
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.metrics import CosineSimilarity
from keras.saving import register_keras_serializable
# import tensorflow_addons as tfa

# def create_dataset_fast(texts, mel_paths, input_length=200, mel_dim=80, mel_max_len=1024, batch_size=32):
#     def load_and_preprocess_py(text, mel_path):
#         text = text.numpy().decode("utf-8")
#         # print(text)
#         mel_path = mel_path.numpy().decode("utf-8")
#         phoneme_seq = ast.literal_eval(text)
#         padded_text = pad_sequences([phoneme_seq], maxlen=input_length, padding='post')[0].astype(np.int32)
#         mel = np.load(mel_path).astype(np.float32)
#         T, D = mel.shape
#         if T > mel_max_len:
#             mel = mel[:mel_max_len, :]
#         elif T < mel_max_len:
#             pad_len = mel_max_len - T
#             mel = np.pad(mel, ((0, pad_len), (0, 0)), mode='constant')
#         return padded_text, mel

#     def tf_wrapper(text, mel_path):
#         text_tensor, mel_tensor = tf.py_function(
#             func=load_and_preprocess_py,
#             inp=[text, mel_path],
#             Tout=[tf.int32, tf.float32]
#         )
#         text_tensor.set_shape([input_length])
#         mel_tensor.set_shape([mel_max_len, mel_dim])
#         return text_tensor, mel_tensor

#     dataset = tf.data.Dataset.from_tensor_slices((texts, mel_paths))
#     dataset = dataset.shuffle(buffer_size=10000)
#     dataset = dataset.map(tf_wrapper, num_parallel_calls=tf.data.AUTOTUNE)
#     dataset = dataset.batch(batch_size)
#     dataset = dataset.prefetch(tf.data.AUTOTUNE)
#     return dataset

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
def build_acoustic_model(vocab_size, input_length=165, mel_dim=80, mel_len=865, embed_dim=256):
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
    x = layers.Bidirectional(layers.LSTM(512, return_sequences=True,dropout=0.2))(x, mask=mask)
    # x = layers.Dropout(0.3)(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True,dropout=0.2))(x, mask=mask)
    # x = layers.Dropout(0.3)(x)

    # Residual CNN blocks
    for _ in range(3):
        residual = x
        x = layers.Conv1D(filters=256, kernel_size=5, padding='same', dilation_rate=2, activation='relu')(x)
        x = layers.LayerNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Add()([x, residual])

    query = layers.GlobalAveragePooling1D()(x)
    x = AttentionContextLayer(units=256, input_length=input_length)([query, x, mask])

    # # Upsample
    x = layers.Conv1DTranspose(256, kernel_size=5, strides=2, padding='same', activation='relu')(x) # (None, 336, 256) (None, 400, 256) (None, 330, 256)
    x = layers.Conv1DTranspose(128, kernel_size=3, strides=2, padding='same', activation='relu')(x) # (None, 800, 128) (None, 660, 128)
    x = layers.Conv1DTranspose(128, kernel_size=3, strides=2, padding='same', activation='relu')(x) # (None, 1600, 128) (None, 1320, 128)
    # x = layers.Conv1DTranspose(128, kernel_size=3, strides=1, padding='same', activation='relu')(x) # (None, 1600, 128)  (None, 1320, 128)
    x = CropLayer(mel_len)(x)

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
            postnet = layers.Dropout(0.3)(postnet)

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
        return (0.8*mse) + (0.7 * sc_loss) + (0.1 * log_loss)
    
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

train_dataset = create_dataset_fast(texts_train, mel_train)
val_dataset = create_dataset_fast(texts_val, mel_val)
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
    # EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    ModelCheckpoint('model/2/3best_model_cnn_r_EarlyStopping.keras', monitor='loss', save_best_only=True, verbose=1),
    LRSchedulerLogger(),
    LearningRatePlotter(),
    tensorboard_callback , # 👈 Add this line,
    # TensorBoard(log_dir="logs", histogram_freq=1)  # 👈 This logs training info
]

history = model.fit(
    train_dataset,
    epochs=500,
    validation_data=val_dataset,
    callbacks=callbacks
)

# Save model & history
model.save('model/2/2best_model_cnn_r_EarlyStopping.keras')
model.save_weights('model/2/2best_model_cnn_r_EarlyStopping.weights.h5')
history_df = pd.DataFrame(history.history)
history_df.to_csv('model/2/2best_model_cnn_r_EarlyStopping.csv', index=False)

# ==================== Evaluation =====================
test_loss = model.evaluate(test_dataset)
print(f"Test loss: {test_loss}")






# df = pd.read_csv('dataset/acoustic_dataset/tts_data_LJ.csv', usecols=['Phoneme_text', 'Read_npy'])
# texts = df['Phoneme_text'].apply(ast.literal_eval).values
# mel_spectrograms = df['Read_npy'].values
# input_length = max([len(seq) for seq in texts])+32

# texts_str = [str(seq) for seq in texts]
# texts_train, texts_temp, mel_train, mel_temp = train_test_split(texts_str, mel_spectrograms, test_size=0.2, random_state=33)
# texts_val, texts_test, mel_val, mel_test = train_test_split(texts_temp, mel_temp, test_size=0.3, random_state=33)

# train_dataset = create_dataset_fast(texts_train, mel_train, input_length=input_length)
# val_dataset = create_dataset_fast(texts_val, mel_val, input_length=input_length)
# test_dataset = create_dataset_fast(texts_test, mel_test, input_length=input_length)

# g2p = G2PConverter(load_model=False)
# vocab_size = len(g2p.phn2idx)





'''Model: "functional"   making custom attention layer 

These bankers wishing for more specific information

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                  ┃ Output Shape              ┃         Param # ┃ Connected to               ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ phoneme_input (InputLayer)    │ (None, 168)               │               0 │ -                          │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ embedding (Embedding)         │ (None, 168, 256)          │          10,752 │ phoneme_input[0][0]        │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ add (Add)                     │ (None, 168, 256)          │               0 │ embedding[0][0]            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ not_equal_1 (NotEqual)        │ (None, 168)               │               0 │ phoneme_input[0][0]        │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ bidirectional (Bidirectional) │ (None, 168, 1024)         │       3,149,824 │ add[0][0],                 │
│                               │                           │                 │ not_equal_1[0][0]          │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ bidirectional_1               │ (None, 168, 1024)         │       6,295,552 │ bidirectional[0][0],       │
│ (Bidirectional)               │                           │                 │ not_equal_1[0][0]          │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dropout (Dropout)             │ (None, 168, 1024)         │               0 │ bidirectional_1[0][0]      │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ bidirectional_2               │ (None, 168, 256)          │       1,180,672 │ dropout[0][0],             │
│ (Bidirectional)               │                           │                 │ not_equal_1[0][0]          │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ conv1d (Conv1D)               │ (None, 168, 256)          │         327,936 │ bidirectional_2[0][0]      │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ layer_normalization           │ (None, 168, 256)          │             512 │ conv1d[0][0]               │
│ (LayerNormalization)          │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dropout_1 (Dropout)           │ (None, 168, 256)          │               0 │ layer_normalization[0][0]  │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ add_1 (Add)                   │ (None, 168, 256)          │               0 │ dropout_1[0][0],           │
│                               │                           │                 │ bidirectional_2[0][0]      │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ conv1d_1 (Conv1D)             │ (None, 168, 256)          │         327,936 │ add_1[0][0]                │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ layer_normalization_1         │ (None, 168, 256)          │             512 │ conv1d_1[0][0]             │
│ (LayerNormalization)          │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dropout_2 (Dropout)           │ (None, 168, 256)          │               0 │ layer_normalization_1[0][… │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ add_2 (Add)                   │ (None, 168, 256)          │               0 │ dropout_2[0][0],           │
│                               │                           │                 │ add_1[0][0]                │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ conv1d_2 (Conv1D)             │ (None, 168, 256)          │         327,936 │ add_2[0][0]                │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ layer_normalization_2         │ (None, 168, 256)          │             512 │ conv1d_2[0][0]             │
│ (LayerNormalization)          │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dropout_3 (Dropout)           │ (None, 168, 256)          │               0 │ layer_normalization_2[0][… │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ add_3 (Add)                   │ (None, 168, 256)          │               0 │ dropout_3[0][0],           │
│                               │                           │                 │ add_2[0][0]                │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ global_average_pooling1d      │ (None, 256)               │               0 │ add_3[0][0]                │
│ (GlobalAveragePooling1D)      │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ attention_context_layer       │ (None, 168, 512)          │         131,841 │ global_average_pooling1d[… │
│ (AttentionContextLayer)       │                           │                 │ add_3[0][0],               │
│                               │                           │                 │ not_equal_1[0][0]          │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ conv1d_transpose              │ (None, 336, 256)          │         655,616 │ attention_context_layer[0… │
│ (Conv1DTranspose)             │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ conv1d_transpose_1            │ (None, 672, 128)          │          98,432 │ conv1d_transpose[0][0]     │
│ (Conv1DTranspose)             │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ conv1d_transpose_2            │ (None, 1344, 128)         │          49,280 │ conv1d_transpose_1[0][0]   │
│ (Conv1DTranspose)             │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ crop_layer (CropLayer)        │ (None, 900, 128)          │               0 │ conv1d_transpose_2[0][0]   │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ mel_output (Conv1D)           │ (None, 900, 80)           │          10,320 │ crop_layer[0][0]           │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ conv1d_3 (Conv1D)             │ (None, 900, 80)           │          32,080 │ mel_output[0][0]           │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ batch_normalization           │ (None, 900, 80)           │             320 │ conv1d_3[0][0]             │
│ (BatchNormalization)          │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ conv1d_4 (Conv1D)             │ (None, 900, 80)           │          32,080 │ batch_normalization[0][0]  │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ batch_normalization_1         │ (None, 900, 80)           │             320 │ conv1d_4[0][0]             │
│ (BatchNormalization)          │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ conv1d_5 (Conv1D)             │ (None, 900, 80)           │          32,080 │ batch_normalization_1[0][… │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ batch_normalization_2         │ (None, 900, 80)           │             320 │ conv1d_5[0][0]             │
│ (BatchNormalization)          │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ conv1d_6 (Conv1D)             │ (None, 900, 80)           │          32,080 │ batch_normalization_2[0][… │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ batch_normalization_3         │ (None, 900, 80)           │             320 │ conv1d_6[0][0]             │
│ (BatchNormalization)          │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ conv1d_7 (Conv1D)             │ (None, 900, 80)           │          32,080 │ batch_normalization_3[0][… │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ batch_normalization_4         │ (None, 900, 80)           │             320 │ conv1d_7[0][0]             │
│ (BatchNormalization)          │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ refined_mel_output (Add)      │ (None, 900, 80)           │               0 │ mel_output[0][0],          │
│                               │                           │                 │ batch_normalization_4[0][… │
└───────────────────────────────┴───────────────────────────┴─────────────────┴────────────────────────────┘
 Total params: 12,729,633 (48.56 MB)
 Trainable params: 12,728,833 (48.56 MB)
 Non-trainable params: 800 (3.12 KB)
 
  Non-trainable params: 800 (3.12 KB)
Epoch 1/50
328/328 ━━━━━━━━━━━━━━━━━━━━ 0s 3s/step - cosine_similarity: 0.3314 - loss: 1.0056 - mae: 0.5173  Learning_rate=0.001000
328/328 ━━━━━━━━━━━━━━━━━━━━ 948s 3s/step - cosine_similarity: 0.3315 - loss: 1.0052 - mae: 0.5171 - val_cosine_similarity: 0.3677 - val_loss: 0.8308 - val_mae: 0.4363
Epoch 2/50
 73/328 ━━━━━━━━━━━━━━━━━━━━ 11:33 3s/step - cosine_similarity: 0.3778 - loss: 0.8382 - mae: 0.4413
328/328 ━━━━━━━━━━━━━━━━━━━━ 948s 3s/step - cosine_similarity: 0.3755 - loss: 0.8290 - mae: 0.4347 - val_cosine_similarity: 0.3732 - val_loss: 0.8197 - val_mae: 0.4264
Epoch 3/50
328/328 ━━━━━━━━━━━━━━━━━━━━ 910s 3s/step - cosine_similarity: 0.3808 - loss: 0.8194 - mae: 0.4292 - val_cosine_similarity: 0.3785 - val_loss: 0.8134 - val_mae: 0.4217
Epoch 4/50
270/328 ━━━━━━━━━━━━━━━━━━━━ 2:33 3s/step - cosine_similarity: 0.3834 - loss: 0.8160 - mae: 0.4284 
328/328 ━━━━━━━━━━━━━━━━━━━━ 927s 3s/step - cosine_similarity: 0.3829 - loss: 0.8156 - mae: 0.4280 - val_cosine_similarity: 0.3811 - val_loss: 0.8063 - val_mae: 0.4188
Epoch 5/50
328/328 ━━━━━━━━━━━━━━━━━━━━ 0s 3s/step - cosine_similarity: 0.3864 - loss: 0.8084 - mae: 0.4239   
Epoch 5: val_loss improved from 0.80634 to 0.80206, saving model to model/2/best_model_cnn.keras

Epoch 5: Learning rate is 0.001000
328/328 ━━━━━━━━━━━━━━━━━━━━ 959s 3s/step - cosine_similarity: 0.3864 - loss: 0.8084 - mae: 0.4238 - val_cosine_similarity: 0.3839 - val_loss: 0.8021 - val_mae: 0.4200
Epoch 6/50
328/328 ━━━━━━━━━━━━━━━━━━━━ 0s 3s/step - cosine_similarity: 0.3883 - loss: 0.8044 - mae: 0.4215   
Epoch 6: val_loss did not improve from 0.80206

Epoch 6: Learning rate is 0.001000
328/328 ━━━━━━━━━━━━━━━━━━━━ 972s 3s/step - cosine_similarity: 0.3883 - loss: 0.8044 - mae: 0.4215 - val_cosine_similarity: 0.3795 - val_loss: 0.8088 - val_mae: 0.4176
Epoch 7/50
328/328 ━━━━━━━━━━━━━━━━━━━━ 0s 3s/step - cosine_similarity: 0.3873 - loss: 0.8072 - mae: 0.4225   
Epoch 7: val_loss improved from 0.80206 to 0.80028, saving model to model/2/best_model_cnn.keras

Epoch 7: Learning rate is 0.001000
328/328 ━━━━━━━━━━━━━━━━━━━━ 935s 3s/step - cosine_similarity: 0.3873 - loss: 0.8072 - mae: 0.4225 - val_cosine_similarity: 0.3850 - val_loss: 0.8003 - val_mae: 0.4197
Epoch 8/50
328/328 ━━━━━━━━━━━━━━━━━━━━ 0s 3s/step - cosine_similarity: 0.3907 - loss: 0.7992 - mae: 0.4185   
Epoch 8: val_loss improved from 0.80028 to 0.79575, saving model to model/2/best_model_cnn.keras

Epoch 8: Learning rate is 0.001000
328/328 ━━━━━━━━━━━━━━━━━━━━ 998s 3s/step - cosine_similarity: 0.3906 - loss: 0.7992 - mae: 0.4185 - val_cosine_similarity: 0.3867 - val_loss: 0.7957 - val_mae: 0.4157
Epoch 9/50
328/328 ━━━━━━━━━━━━━━━━━━━━ 0s 3s/step - cosine_similarity: 0.3925 - loss: 0.7948 - mae: 0.4163   
Epoch 9: val_loss improved from 0.79575 to 0.79207, saving model to model/2/best_model_cnn.keras

Epoch 9: Learning rate is 0.001000
328/328 ━━━━━━━━━━━━━━━━━━━━ 940s 3s/step - cosine_similarity: 0.3925 - loss: 0.7948 - mae: 0.4163 - val_cosine_similarity: 0.3874 - val_loss: 0.7921 - val_mae: 0.4140
Epoch 10/50
328/328 ━━━━━━━━━━━━━━━━━━━━ 0s 3s/step - cosine_similarity: 0.3944 - loss: 0.7900 - mae: 0.4137   
Epoch 10: val_loss did not improve from 0.79207

Epoch 10: Learning rate is 0.001000
328/328 ━━━━━━━━━━━━━━━━━━━━ 971s 3s/step - cosine_similarity: 0.3944 - loss: 0.7900 - mae: 0.4137 - val_cosine_similarity: 0.3869 - val_loss: 0.7966 - val_mae: 0.4195
Epoch 11/50
328/328 ━━━━━━━━━━━━━━━━━━━━ 0s 3s/step - cosine_similarity: 0.3961 - loss: 0.7858 - mae: 0.4117   
Epoch 11: val_loss did not improve from 0.79207

Epoch 11: Learning rate is 0.001000
328/328 ━━━━━━━━━━━━━━━━━━━━ 966s 3s/step - cosine_similarity: 0.3960 - loss: 0.7858 - mae: 0.4117 - val_cosine_similarity: 0.3870 - val_loss: 0.7968 - val_mae: 0.4145
Epoch 12/50
328/328 ━━━━━━━━━━━━━━━━━━━━ 0s 3s/step - cosine_similarity: 0.3980 - loss: 0.7809 - mae: 0.4094   
Epoch 12: val_loss improved from 0.79207 to 0.79083, saving model to model/2/best_model_cnn.keras

Epoch 12: Learning rate is 0.001000
328/328 ━━━━━━━━━━━━━━━━━━━━ 904s 3s/step - cosine_similarity: 0.3980 - loss: 0.7809 - mae: 0.4094 - val_cosine_similarity: 0.3887 - val_loss: 0.7908 - val_mae: 0.4088
Epoch 13/50
328/328 ━━━━━━━━━━━━━━━━━━━━ 0s 3s/step - cosine_similarity: 0.4001 - loss: 0.7760 - mae: 0.4071   
Epoch 13: val_loss did not improve from 0.79083

Epoch 13: Learning rate is 0.001000
328/328 ━━━━━━━━━━━━━━━━━━━━ 963s 3s/step - cosine_similarity: 0.4000 - loss: 0.7759 - mae: 0.4071 - val_cosine_similarity: 0.3849 - val_loss: 0.7974 - val_mae: 0.4078
Epoch 14/50
328/328 ━━━━━━━━━━━━━━━━━━━━ 0s 3s/step - cosine_similarity: 0.4021 - loss: 0.7709 - mae: 0.4046   
Epoch 14: val_loss did not improve from 0.79083

Epoch 14: Learning rate is 0.001000
328/328 ━━━━━━━━━━━━━━━━━━━━ 961s 3s/step - cosine_similarity: 0.4021 - loss: 0.7709 - mae: 0.4046 - val_cosine_similarity: 0.3829 - val_loss: 0.7997 - val_mae: 0.4092
Epoch 15/50
328/328 ━━━━━━━━━━━━━━━━━━━━ 0s 3s/step - cosine_similarity: 0.4044 - loss: 0.7650 - mae: 0.4019   
Epoch 15: val_loss did not improve from 0.79083

Epoch 15: Learning rate is 0.001000
328/328 ━━━━━━━━━━━━━━━━━━━━ 985s 3s/step - cosine_similarity: 0.4044 - loss: 0.7650 - mae: 0.4019 - val_cosine_similarity: 0.3866 - val_loss: 0.7976 - val_mae: 0.4112
Epoch 16/50
328/328 ━━━━━━━━━━━━━━━━━━━━ 0s 3s/step - cosine_similarity: 0.4072 - loss: 0.7582 - mae: 0.3987   
Epoch 16: val_loss did not improve from 0.79083

Epoch 16: Learning rate is 0.001000
328/328 ━━━━━━━━━━━━━━━━━━━━ 931s 3s/step - cosine_similarity: 0.4072 - loss: 0.7582 - mae: 0.3987 - val_cosine_similarity: 0.3781 - val_loss: 0.8074 - val_mae: 0.4069
Epoch 17/50
328/328 ━━━━━━━━━━━━━━━━━━━━ 0s 3s/step - cosine_similarity: 0.4092 - loss: 0.7541 - mae: 0.3970   
Epoch 17: val_loss did not improve from 0.79083

Epoch 17: Learning rate is 0.001000
328/328 ━━━━━━━━━━━━━━━━━━━━ 907s 3s/step - cosine_similarity: 0.4092 - loss: 0.7541 - mae: 0.3970 - val_cosine_similarity: 0.3745 - val_loss: 0.8054 - val_mae: 0.4066
Epoch 17: early stopping
Restoring model weights from the end of the best epoch: 12.
25/25 ━━━━━━━━━━━━━━━━━━━━ 20s 808ms/step - cosine_similarity: 0.3851 - loss: 0.7904 - mae: 0.4071
Test loss: [0.7902094125747681, 0.4065769910812378, 0.3852331340312958]

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
    for _ in range(5):
        postnet = layers.Conv1D(filters=mel_dim, kernel_size=5, padding='same', activation='linear')(postnet)
        postnet = layers.BatchNormalization()(postnet)
    mel_output = layers.Add(name="refined_mel_output")([mel_output, postnet])

    model = tf.keras.Model(inputs=inputs, outputs=mel_output)
    return model

 '''

'''

Model: "functional"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                  ┃ Output Shape              ┃         Param # ┃ Connected to               ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ input_layer (InputLayer)      │ (None, None)              │               0 │ -                          │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ embedding (Embedding)         │ (None, None, 256)         │          10,752 │ input_layer[0][0]          │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ add (Add)                     │ (None, 200, 256)          │               0 │ embedding[0][0]            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ residual_block                │ (None, 200, 256)          │         656,896 │ add[0][0]                  │
│ (ResidualBlock)               │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ residual_block_1              │ (None, 200, 256)          │         656,896 │ residual_block[0][0]       │
│ (ResidualBlock)               │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ residual_block_2              │ (None, 200, 256)          │         656,896 │ residual_block_1[0][0]     │
│ (ResidualBlock)               │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ not_equal_1 (NotEqual)        │ (None, None)              │               0 │ input_layer[0][0]          │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ bidirectional (Bidirectional) │ (None, 200, 512)          │       1,050,624 │ residual_block_2[0][0],    │
│                               │                           │                 │ not_equal_1[0][0]          │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ attention (Attention)         │ (None, 200, 512)          │               0 │ bidirectional[0][0],       │
│                               │                           │                 │ bidirectional[0][0]        │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ conv1d_transpose              │ (None, 400, 256)          │         524,544 │ attention[0][0]            │
│ (Conv1DTranspose)             │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ conv1d_transpose_1            │ (None, 800, 128)          │         131,200 │ conv1d_transpose[0][0]     │
│ (Conv1DTranspose)             │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ conv1d_transpose_2            │ (None, 1600, 128)         │          65,664 │ conv1d_transpose_1[0][0]   │
│ (Conv1DTranspose)             │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ crop_layer (CropLayer)        │ (None, 1024, 128)         │               0 │ conv1d_transpose_2[0][0]   │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ mel_output (Conv1D)           │ (None, 1024, 80)          │          10,320 │ crop_layer[0][0]           │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ conv1d_6 (Conv1D)             │ (None, 1024, 80)          │          32,080 │ mel_output[0][0]           │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ layer_normalization_6         │ (None, 1024, 80)          │             160 │ conv1d_6[0][0]             │
│ (LayerNormalization)          │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ conv1d_7 (Conv1D)             │ (None, 1024, 80)          │          32,080 │ layer_normalization_6[0][… │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ layer_normalization_7         │ (None, 1024, 80)          │             160 │ conv1d_7[0][0]             │
│ (LayerNormalization)          │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ conv1d_8 (Conv1D)             │ (None, 1024, 80)          │          32,080 │ layer_normalization_7[0][… │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ layer_normalization_8         │ (None, 1024, 80)          │             160 │ conv1d_8[0][0]             │
│ (LayerNormalization)          │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ conv1d_9 (Conv1D)             │ (None, 1024, 80)          │          32,080 │ layer_normalization_8[0][… │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ layer_normalization_9         │ (None, 1024, 80)          │             160 │ conv1d_9[0][0]             │
│ (LayerNormalization)          │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ conv1d_10 (Conv1D)            │ (None, 1024, 80)          │          32,080 │ layer_normalization_9[0][… │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ layer_normalization_10        │ (None, 1024, 80)          │             160 │ conv1d_10[0][0]            │
│ (LayerNormalization)          │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ refined_mel_output (Add)      │ (None, 1024, 80)          │               0 │ mel_output[0][0],          │
│                               │                           │                 │ layer_normalization_10[0]… │
└───────────────────────────────┴───────────────────────────┴─────────────────┴────────────────────────────┘
 Total params: 3,924,992 (14.97 MB)
 Trainable params: 3,924,992 (14.97 MB)
 Non-trainable params: 0 (0.00 B)
Epoch 1/50
 26/328 ━━━━━━━━━━━━━━━━━━━━ 9:55 2s/step - cosine_similarity: 0.2151 - loss: 1.6342 - mae: 0.7572      

 change pos-net activation 'tanh'->'linear'

 Epoch 1/50
178/328 ━━━━━━━━━━━━━━━━━━━━ 3:19 1s/step - cosine_similarity: 0.2931 - loss: 1.1487 - mae: 0.5928  


'''

'''



'''