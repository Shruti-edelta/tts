import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

def pad_or_truncate(mel_spectrogram, max_time_frames):
    if mel_spectrogram.shape[1] < max_time_frames:      # If the Mel spectrogram has fewer time frames than required, pad with zeros
        pad_width = max_time_frames - mel_spectrogram.shape[1]
        mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, pad_width)), mode='constant')
    elif mel_spectrogram.shape[1] > max_time_frames:        # If the Mel spectrogram has more time frames than required, truncate it
        mel_spectrogram = mel_spectrogram[:, :max_time_frames]
    return mel_spectrogram

def seq2seq_model_LSTM(vocab_size,input_length, mel_dim=80, embed_dim=128, rnn_hidden_units=256):      
    # Encoder
    encoder_inputs = layers.Input(shape=(input_length,))
    print(encoder_inputs.shape)
    encoder_embedding = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)(encoder_inputs)
    encoder_lstm = layers.Bidirectional(layers.LSTM(rnn_hidden_units, return_sequences=True))(encoder_embedding)

    # Attention Layer
    attention = layers.Attention()([encoder_lstm, encoder_lstm])  # Self-attention

    # Decoder
    decoder_lstm = layers.LSTM(rnn_hidden_units, return_sequences=True,dropout=0.3)(attention)
    mel_output = layers.Dense(mel_dim, activation='linear')(decoder_lstm)  # Linear output for Mel-spectrogram prediction # dropout=0.3

    mel_output = layers.Reshape((mel_dim,input_length))(mel_output)

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
# input_length = max([len(seq) for seq in sequences])  # Determine max sequence length 245
padded_texts = pad_sequences(sequences, maxlen=600, padding='post')    # (1211, 245)
print(padded_texts)

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

# Build the seq2seq model
model = seq2seq_model_LSTM(vocab_size, input_length)
model.summary()

e_s=EarlyStopping(monitor="val_loss",patience=5,restore_best_weights=True)

# Compile & Fit the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
model.fit(np.array(texts_train), np.array(mel_train), batch_size=32, epochs=15, validation_data=(np.array(texts_val), np.array(mel_val)),callbacks=[e_s])

# # Visualize the loss curves
# plt.plot(history.history['loss'], label='Train Loss')
# plt.plot(history.history['val_loss'], label='Val Loss')
# plt.title('Train and Validation Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

test_loss = model.evaluate(np.array(texts_test), np.array(mel_test))
print(f"Test loss: {test_loss}")

model.save('tts_model_LSTM.keras')




# def improved_tts_model(vocab_size, input_length, mel_dim=128, rnn_units=512, embed_dim=256):    
#     # Encoder
#     encoder_inputs = layers.Input(shape=(input_length,))
#     masked_inputs = layers.Masking(mask_value=0)(encoder_inputs)

#     encoder_embedding = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)(masked_inputs)

#     encoder_conv = layers.Conv1D(filters=64, kernel_size=5, activation='relu', padding='same')(encoder_embedding)
#     encoder_conv = layers.Conv1D(filters=64, kernel_size=5, activation='relu', padding='same')(encoder_conv)

#     encoder_lstm1 = layers.Bidirectional(layers.LSTM(rnn_units, return_sequences=True),)(encoder_conv)
#     encoder_lstm2 = layers.Bidirectional(layers.LSTM(rnn_units, return_sequences=True))(encoder_lstm1)
#     # encoder_lstm2 = layers.Dropout(0.5)(encoder_lstm2)

#     # Attention Layer
#     # attention = layers.Attention()([encoder_lstm2, encoder_lstm2])
#     attention = layers.MultiHeadAttention(num_heads=4, key_dim=rnn_units)(encoder_lstm2, encoder_lstm2)

#     # Decoder
#     decoder_lstm = layers.LSTM(rnn_units, return_sequences=True,dropout=0.3)(attention)
#     # decoder_lstm = layers.BatchNormalization()(decoder_lstm)
#     mel_output = layers.Dense(mel_dim, activation='linear')(decoder_lstm)
#     mel_output = layers.Reshape((mel_dim, input_length))(mel_output)

#     model = tf.keras.Model(inputs=encoder_inputs, outputs=mel_output)
#     return model