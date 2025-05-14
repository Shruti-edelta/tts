import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model


def pad_or_truncate(mel_spectrogram, max_time_frames):
    if mel_spectrogram.shape[1] < max_time_frames:      # If the Mel spectrogram has fewer time frames than required, pad with zeros
        pad_width = max_time_frames - mel_spectrogram.shape[1]
        mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, pad_width)), mode='constant')
    elif mel_spectrogram.shape[1] > max_time_frames:        # If the Mel spectrogram has more time frames than required, truncate it
        mel_spectrogram = mel_spectrogram[:, :max_time_frames]
    return mel_spectrogram

def seq2seq_model_GRU(vocab_size, input_length, mel_dim=80, embed_dim=128, rnn_hidden_units=256):      
    # Encoder
    encoder_inputs = layers.Input(shape=(input_length,))
    encoder_embedding = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)(encoder_inputs)
    
    # Replace LSTM with GRU here
    encoder_gru = layers.Bidirectional(layers.GRU(rnn_hidden_units, return_sequences=True))(encoder_embedding)

    # Attention Layer
    attention = layers.Attention()([encoder_gru, encoder_gru])  # Self-attention

    # Decoder
    decoder_gru = layers.GRU(rnn_hidden_units, return_sequences=True)(attention)
    mel_output = layers.Dense(mel_dim, activation='linear')(decoder_gru)  # Linear output for Mel-spectrogram prediction

    mel_output = layers.Reshape((mel_dim, input_length))(mel_output)

    model = tf.keras.Model(inputs=encoder_inputs, outputs=mel_output)
    return model

# Load your data
df = pd.read_csv('tts_data.csv')
texts = df['Phoneme_text'].values
mel_spectrograms = df['Read_npy'].values

# Tokenize the phoneme sequences
tokenizer = Tokenizer(char_level=False)  # Assume each phoneme is a "word"
tokenizer.fit_on_texts(texts)
# Convert the phoneme sequences to integer sequences
sequences = tokenizer.texts_to_sequences(texts)
padded_texts = pad_sequences(sequences, maxlen=600, padding='post')    # (1211, 245)

# Load Mel spectrograms as NumPy arrays
mel_spectrograms = [np.load(mel) for mel in mel_spectrograms] 
# output_length = mel_spectrograms[0].shape[0]
# Pad or truncate the Mel spectrograms to ensure they have the same number of time frames
mel_spectrograms = [pad_or_truncate(mel, max_time_frames=600) for mel in mel_spectrograms]
mel_spectrograms = np.array(mel_spectrograms)

# Split the data into train, validation, and test
texts_train, texts_temp, mel_train, mel_temp = train_test_split(padded_texts, mel_spectrograms, test_size=0.2, random_state=42)
texts_val, texts_test, mel_val, mel_test = train_test_split(texts_temp, mel_temp, test_size=0.5, random_state=42)

# print(texts_train.shape,texts_test.shape,texts_val.shape)       # (968, 245) (122, 245) (121, 245)
# print(mel_train.shape,mel_test.shape,mel_val.shape)     # (968, 80, 1000) (122, 80, 1000) (121, 80, 1000)

# Model configuration
vocab_size = len(tokenizer.word_index) + 1  # Vocabulary size (add 1 for padding token) 1997+1  # Number of unique phonemes
input_length=600

model = seq2seq_model_GRU(vocab_size, input_length)
model.summary()
plot_model(model, to_file='model_architecture.png', show_shapes=True, show_layer_names=True)


# for stop overfitting 
e_s=EarlyStopping(monitor="val_loss",patience=5,restore_best_weights=True)

# Compile & Fit the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
# model.compile(optimizer='adam', loss="mse")
model.fit(np.array(texts_train), np.array(mel_train), batch_size=32, epochs=15, validation_data=(np.array(texts_val), np.array(mel_val)),callbacks=[e_s])

test_loss = model.evaluate(np.array(texts_test), np.array(mel_test))
print(f"Test loss: {test_loss}")

model.save('tts_model_GRU.keras')

# def build_tacotron_model(vocab_size, mel_dim, embedding_dim=256):
#     # Input for text sequence
#     inputs = layers.Input(shape=(None,), dtype=tf.int32, name="text_input")
#     # Embedding layer
#     x = layers.Embedding(vocab_size, embedding_dim)(inputs)
#     # Encoder: GRU + Attention
#     x, state = layers.GRU(512, return_state=True, return_sequences=True)(x)
#     attention_layer = layers.Attention()([x, x])

#     # Decoder: GRU to produce Mel Spectrogra
#     mel_output = layers.GRU(512, return_sequences=True)(attention_layer)
#     # Output Mel spectrogram
#     mel_output = layers.Dense(mel_dim, activation='linear', name="mel_output")(mel_output)
#     # Build the model
#     model = tf.keras.Model(inputs, mel_output)
#     return model
