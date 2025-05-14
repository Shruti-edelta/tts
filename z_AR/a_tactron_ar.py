
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
from tensorflow.keras.metrics import CosineSimilarity
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import ast
from acoustic.text_preprocess import G2PConverter
from keras.saving import register_keras_serializable
'''

def build_encoder(vocab_size, embed_dim=256, encoder_dim=512, input_length=None):
    inputs = tf.keras.Input(shape=(input_length,), name="phoneme_input")

    # Embedding with masking
    x = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, mask_zero=True)(inputs)

    # Convolutional Bank (3 Conv1D layers with ReLU + BatchNorm)
    for i in range(3):
        x = layers.Conv1D(filters=embed_dim, kernel_size=5, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.1)(x)

    # BiLSTM to create encoder outputs
    x = layers.Bidirectional(layers.LSTM(encoder_dim, return_sequences=True))(x)

    return tf.keras.Model(inputs=inputs, outputs=x, name="Encoder")

encoder = build_encoder(vocab_size=50, input_length=200)
encoder.summary()

# phoneme_input [None, 200] → [None, 200, 1024]


class LocationSensitiveAttention(layers.Layer):
    def __init__(self, units, **kwargs):
        super(LocationSensitiveAttention, self).__init__(**kwargs)
        self.query_layer = layers.Dense(units)
        self.values_layer = layers.Dense(units)
        self.location_conv = layers.Conv1D(filters=32, kernel_size=31, padding='same', activation='relu')
        self.location_dense = layers.Dense(units)
        self.score_dense = layers.Dense(1)

    def call(self, query, values, prev_alignments, mask=None):
        # query: [batch, decoder_dim]
        # values: [batch, time, encoder_dim]
        # prev_alignments: [batch, time]

        # Expand and transform query
        query = tf.expand_dims(query, 1)  # → [batch, 1, decoder_dim]
        processed_query = self.query_layer(query)

        # Process encoder outputs
        processed_values = self.values_layer(values)

        # Process alignment history
        prev_alignments_exp = tf.expand_dims(prev_alignments, axis=-1)
        location_features = self.location_conv(prev_alignments_exp)
        processed_location = self.location_dense(location_features)

        # Combine all sources
        score = self.score_dense(tf.nn.tanh(processed_query + processed_values + processed_location))
        score = tf.squeeze(score, axis=-1)

        if mask is not None:
            score += (1.0 - tf.cast(mask, tf.float32)) * -1e9

        # Attention weights (alignment)
        alignments = tf.nn.softmax(score, axis=-1)

        # Context vector
        context_vector = tf.matmul(tf.expand_dims(alignments, 1), values)
        context_vector = tf.squeeze(context_vector, axis=1)

        return context_vector, alignments


class TacotronDecoderCell(tf.keras.layers.Layer):
    def __init__(self, lstm_units=1024, mel_dim=80, **kwargs):
        super(TacotronDecoderCell, self).__init__(**kwargs)
        self.prenet = tf.keras.Sequential([
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5)
        ])
        self.lstm = layers.LSTMCell(lstm_units)
        self.projection = layers.Dense(mel_dim)  # predict next mel frame

    def call(self, inputs, states):
        # Unpack inputs
        prev_mel, context = inputs

        # 1. Prenet on prev mel frame
        x = self.prenet(prev_mel)

        # 2. Concatenate with context vector from attention
        x = tf.concat([x, context], axis=-1)

        # 3. Pass through LSTM
        output, new_states = self.lstm(x, states)

        # 4. Project to mel frame
        mel_frame = self.projection(output)

        return mel_frame, new_states

class TacotronDecoder(tf.keras.Model):
    def __init__(self, mel_dim=80, decoder_dim=1024, attn_units=256):
        super().__init__()
        self.attention = LocationSensitiveAttention(units=attn_units)
        self.decoder_cell = TacotronDecoderCell(lstm_units=decoder_dim, mel_dim=mel_dim)
        self.mel_dim = mel_dim

    def call(self, encoder_outputs, decoder_inputs, input_mask=None):
        # encoder_outputs: [B, T_enc, D]
        # decoder_inputs: [B, T_dec, mel_dim]
        batch_size = tf.shape(decoder_inputs)[0]
        time_steps = tf.shape(decoder_inputs)[1]

        # Initial decoder states and attention alignment
        decoder_state = self.decoder_cell.lstm.get_initial_state(
            batch_size=batch_size, dtype=tf.float32)
        attention_weights = tf.zeros([batch_size, tf.shape(encoder_outputs)[1]])
        context_vector = tf.zeros([batch_size, tf.shape(encoder_outputs)[-1]])

        outputs = []

        # Go through each timestep
        for t in range(time_steps):
            # Teacher forcing: use ground-truth mel frame t
            prev_mel = decoder_inputs[:, t, :]  # [B, mel_dim]

            # Attention: get context and updated attention
            context_vector, attention_weights = self.attention(
                query=decoder_state[0],  # hidden state from LSTM
                values=encoder_outputs,
                prev_alignments=attention_weights,
                mask=input_mask
            )

            # Decoder: predict next mel frame
            mel_frame, decoder_state = self.decoder_cell((prev_mel, context_vector), decoder_state)
            outputs.append(mel_frame)

        outputs = tf.stack(outputs, axis=1)  # [B, T, mel_dim]
        return outputs

def build_postnet(mel_dim=80):
    layers_list = []
    for i in range(5):
        layers_list.append(layers.Conv1D(
            filters=mel_dim,
            kernel_size=5,
            padding='same',
            activation='tanh' if i < 4 else None
        ))
        layers_list.append(layers.BatchNormalization())
    return tf.keras.Sequential(layers_list, name="postnet")

# def build_tacotron_model(vocab_size, input_length, mel_len, mel_dim=80):
#     encoder = build_encoder(vocab_size=vocab_size, input_length=input_length)
#     decoder = TacotronDecoder(mel_dim=mel_dim)

#     # Inputs
#     phoneme_input = tf.keras.Input(shape=(input_length,), name="phoneme_input")
#     mel_input = tf.keras.Input(shape=(mel_len, mel_dim), name="mel_input")

#     # Mask from embedding
#     encoder_output = encoder(phoneme_input)
#     mask = encoder.layers[1].compute_mask(phoneme_input)

#     # Decoder
#     mel_output = decoder(encoder_output, mel_input, input_mask=mask)

#     # Post-Net (optional cleanup after this)
#     return tf.keras.Model(inputs=[phoneme_input, mel_input], outputs=mel_output)


def build_tacotron_model(vocab_size, input_length, mel_len=900, mel_dim=80):
    encoder = build_encoder(vocab_size=vocab_size, input_length=input_length)
    decoder = TacotronDecoder(mel_dim=mel_dim)
    postnet = build_postnet(mel_dim)

    # Inputs
    phoneme_input = tf.keras.Input(shape=(input_length,), name="phoneme_input")
    mel_input = tf.keras.Input(shape=(mel_len, mel_dim), name="mel_input")

    # Encoder
    encoder_output = encoder(phoneme_input)
    mask = encoder.layers[1].compute_mask(phoneme_input)

    # Decoder
    mel_output = decoder(encoder_output, mel_input, input_mask=mask)

    # Post-Net
    post_output = postnet(mel_output)
    final_output = layers.Add(name="refined_mel")([mel_output, post_output])

    return tf.keras.Model(inputs=[phoneme_input, mel_input], outputs=final_output)


def combined_mel_loss(y_true, y_pred):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    
    # Cosine similarity: higher is better, so subtract from 1
    cos_sim = tf.keras.losses.cosine_similarity(y_true, y_pred)
    cos_sim = tf.reduce_mean(1.0 + cos_sim)  # 1 - (-1) = max = 2

    return mse + 0.1 * cos_sim


# ==================== Load Data =====================

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

model = build_tacotron_model(vocab_size, input_length, mel_len=900, mel_dim=80)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=combined_mel_loss)
model.summary()

# callbacks = [
#     EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
#     ModelCheckpoint('model/2/best_model_cnn.keras', monitor='val_loss', save_best_only=True, verbose=1),
#     ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3,verbose=1)
# ]

# history = model.fit(
#     train_dataset,
#     epochs=50,
#     validation_data=val_dataset,
#     callbacks=callbacks
# )

'''



import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.layers import Embedding, LSTM, Dense, Conv1D, Bidirectional


def create_dataset_fast(texts, mel_paths, input_length=245, mel_dim=80, mel_max_len=1045, batch_size=32):
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


def positional_encoding(length, depth):
    depth = depth // 2
    positions = tf.cast(tf.range(length)[:, tf.newaxis], dtype=tf.float64)  # Ensure positions are float32
    depths = tf.range(depth)[tf.newaxis, :] / depth
    angle_rates = 1 / (10000**depths)

    # Check data types before multiplication
    print(f"positions dtype: {positions.dtype}")
    print(f"angle_rates dtype: {angle_rates.dtype}")

    angle_rads = positions * angle_rates  # No type mismatch here

    # Check dtype of angle_rads
    print(f"angle_rads dtype after multiplication: {angle_rads.dtype}")
    
    pos_encoding = tf.concat([tf.sin(angle_rads), tf.cos(angle_rads)], axis=-1)
    return pos_encoding

# # Positional Encoding (you can adjust if you have a better one)
# def positional_encoding(length, depth):
#     depth = depth // 2
#     # positions = tf.range(length)[:, tf.newaxis]    # (seq, 1)
#     positions = tf.cast(tf.range(length)[:, tf.newaxis], dtype=tf.float32)
#     depths = tf.range(depth)[tf.newaxis, :] / depth # (1, depth)
#     angle_rates = 1 / (10000**depths)
#     angle_rads = positions * angle_rates
#     pos_encoding = tf.concat([tf.sin(angle_rads), tf.cos(angle_rads)], axis=-1)
#     return pos_encoding

# Residual Block (simple version)
def ResidualBlock(filters, kernel_size):
    def block(x):
        shortcut = x
        x = Conv1D(filters, kernel_size, padding='same', activation='relu')(x)
        x = Conv1D(filters, kernel_size, padding='same')(x)
        x = layers.Add()([shortcut, x])
        x = layers.Activation('relu')(x)
        return x
    return block

import tensorflow as tf
from tensorflow.keras import layers

from tensorflow.keras import layers, Model
import tensorflow as tf

class ARDecoderCell(layers.Layer):
    def __init__(self, encoder_dim, decoder_dim, mel_dim, **kwargs):
        super().__init__(**kwargs)
        self.decoder_lstm = layers.LSTMCell(decoder_dim)
        self.attention = layers.AdditiveAttention()
        self.projection = layers.Dense(mel_dim)
        self.state_size = self.decoder_lstm.state_size

    def call(self, inputs, states, constants):
        prev_mel_frame, encoder_outputs = inputs
        context_vector = self.attention(
            [tf.expand_dims(states[0][0], 1), encoder_outputs]
        )
        lstm_input = tf.concat([prev_mel_frame, context_vector], axis=-1)
        output, new_states = self.decoder_lstm(lstm_input, states)
        mel_frame = self.projection(output)
        return mel_frame, new_states

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return self.decoder_lstm.get_initial_state(batch_size=batch_size, dtype=dtype)

def build_ar_model(vocab_size, mel_dim, encoder_dim, decoder_dim, max_phoneme_length, max_mel_length):
    # Inputs
    phoneme_inputs = layers.Input(shape=(max_phoneme_length,), name="phonemes")

    # Encoder (Bidirectional LSTM)
    x = layers.Embedding(vocab_size, encoder_dim)(phoneme_inputs)
    encoder_outputs = layers.Bidirectional(layers.LSTM(encoder_dim, return_sequences=True))(x)

    # Decoder (previous mel frames)
    decoder_inputs = layers.Input(shape=(None, mel_dim), name="decoder_inputs")

    # Decoder RNN Cell
    decoder_cell = ARDecoderCell(encoder_dim * 2, decoder_dim, mel_dim)
    decoder_rnn = layers.RNN(decoder_cell, return_sequences=True)

    # Decoder output connected to RNN layer
    decoder_outputs = decoder_rnn([decoder_inputs, encoder_outputs])

    # Build the model
    model = Model([phoneme_inputs, decoder_inputs], decoder_outputs)
    return model

# Example of building the model
model = build_ar_model(
    vocab_size=5000,  # Replace with actual vocab size
    mel_dim=80,  # Mel-spectrogram dimension
    encoder_dim=512,  # Encoder LSTM size
    decoder_dim=256,  # Decoder LSTM size
    max_phoneme_length=245,  # Max phoneme sequence length
    max_mel_length=1045  # Max mel length
)

model.summary()

# Compile the model with an optimizer and loss function
model.compile(optimizer='adam', loss='mse')

# # Full Autoregressive Acoustic Model
# def build_ar_acoustic_model(vocab_size, 
#                              max_phoneme_length, 
#                              mel_dim=80, 
#                              encoder_dim=256, 
#                              decoder_dim=256,
#                              max_mel_length=1024):
#     # Encoder: Phoneme inputs
#     phoneme_inputs = Input(shape=(max_phoneme_length,), dtype=tf.int32, name='phoneme_inputs')
#     phoneme_embed = Embedding(input_dim=vocab_size, output_dim=encoder_dim, mask_zero=True)(phoneme_inputs)
    
#     # Add Positional Encoding
#     pos_enc = positional_encoding(max_phoneme_length, encoder_dim)
#     phoneme_embed += pos_enc[tf.newaxis, :max_phoneme_length, :]

#     # Convolutional encoder
#     x = ResidualBlock(encoder_dim, kernel_size=5)(phoneme_embed)
#     x = ResidualBlock(encoder_dim, kernel_size=5)(x)

#     # BiLSTM
#     encoder_outputs = Bidirectional(LSTM(encoder_dim, return_sequences=True))(x)

#     # Decoder: Mel inputs (shifted mel spectrograms)
#     mel_inputs = Input(shape=(None, mel_dim), dtype=tf.float32, name='mel_inputs')

#     decoder_lstm = LSTM(decoder_dim, return_sequences=True, return_state=True)

#     # Attention
#     attention_layer = layers.Attention()

#     # Output projection
#     projection = Dense(mel_dim)

#     # Prepare AR decoding
#     all_outputs = []
#     state_h, state_c = None, None

#     # Loop over time steps
#     for t in range(max_mel_length):
#         mel_slice = mel_inputs[:, t:t+1, :]  # (batch, 1, mel_dim)

#         if t == 0:
#             decoder_output, state_h, state_c = decoder_lstm(mel_slice)
#         else:
#             decoder_output, state_h, state_c = decoder_lstm(mel_slice, initial_state=[state_h, state_c])

#         context_vector = attention_layer([decoder_output, encoder_outputs])
#         context_concat = layers.Concatenate(axis=-1)([decoder_output, context_vector])

#         mel_frame = projection(context_concat)  # (batch, 1, mel_dim)
#         all_outputs.append(mel_frame)

#     # Combine all time steps
#     decoder_outputs = layers.Concatenate(axis=1)(all_outputs)  # (batch, max_mel_length, mel_dim)

#     # Build Model
#     model = Model(inputs=[phoneme_inputs, mel_inputs], outputs=decoder_outputs)

#     return model


# ==================== Load Data =====================

df = pd.read_csv('dataset/acoustic_dataset/tts_data_LJ.csv', usecols=['Phoneme_text', 'Read_npy'])
texts = df['Phoneme_text'].apply(ast.literal_eval).values
mel_spectrograms = df['Read_npy'].values
# input_length = max([len(seq) for seq in texts])+32
input_length = 245

texts_str = [str(seq) for seq in texts]
texts_train, texts_temp, mel_train, mel_temp = train_test_split(texts_str, mel_spectrograms, test_size=0.2, random_state=33)
texts_val, texts_test, mel_val, mel_test = train_test_split(texts_temp, mel_temp, test_size=0.3, random_state=33)

# train_df = pd.DataFrame({'Phoneme_text': texts_train, 'Read_npy': mel_train})
# val_df = pd.DataFrame({'Phoneme_text': texts_val, 'Read_npy': mel_val})
# test_df = pd.DataFrame({'Phoneme_text': texts_test, 'Read_npy': mel_test})

# train_df.to_csv('dataset/acoustic_dataset/train.csv', index=False)
# val_df.to_csv('dataset/acoustic_dataset/val.csv', index=False)
# test_df.to_csv('dataset/acoustic_dataset/test.csv', index=False)

train_dataset = create_dataset_fast(texts_train, mel_train, input_length=input_length)
val_dataset = create_dataset_fast(texts_val, mel_val, input_length=input_length)
test_dataset = create_dataset_fast(texts_test, mel_test, input_length=input_length)

g2p = G2PConverter(load_model=False)
vocab_size = len(g2p.phn2idx)


# ==================== Build & Train =====================

max_phoneme_length = 245  # (your max phoneme length)
mel_dim = 80
max_mel_length = 1045  # (your max mel length)

model = build_ar_model(
    vocab_size=vocab_size,
    max_phoneme_length=max_phoneme_length,
    mel_dim=mel_dim,
    encoder_dim=256,
    decoder_dim=256,
    max_mel_length=max_mel_length
)
model.summary()
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='mse',
    metrics=['mae']
)
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ModelCheckpoint('model/2/best_model_AR.keras', monitor='val_loss', save_best_only=True, verbose=1),
    LRSchedulerLogger(),
    LearningRatePlotter()
]

for phonemes, mels in train_dataset.take(1):
    print(f"Phonemes shape: {phonemes.shape}")
    print(f"Mels shape: {mels.shape}")

# Train it
history=model.fit(
    train_dataset.map(lambda x, y: {'phoneme_inputs': x, 'mel_inputs': y}),  # Map the inputs to named keys
    validation_data=val_dataset.map(lambda x, y: {'phoneme_inputs': x, 'mel_inputs': y}),
    epochs=100,
    callbacks=callbacks
)

# Save model & history
model.save('model/2/acoustic_model_AR.keras')
model.save_weights('model/2/acoustic_model_AR.weights.h5')
history_df = pd.DataFrame(history.history)
history_df.to_csv('model/2/acoustic_model_AR.csv', index=False)

# ==================== Evaluation =====================
test_loss = model.evaluate(test_dataset)
print(f"Test loss: {test_loss}")




