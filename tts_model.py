import pickle
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.metrics import CosineSimilarity
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# def pad_or_truncate(mel_spectrogram, max_time_frames):
#     if mel_spectrogram.shape[1] < max_time_frames:      # If the Mel spectrogram has fewer time frames than required, pad with zeros
#         pad_width = max_time_frames - mel_spectrogram.shape[1]
#         mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, pad_width)), mode='constant')
#     elif mel_spectrogram.shape[1] > max_time_frames:        # If the Mel spectrogram has more time frames than required, truncate it
#         mel_spectrogram = mel_spectrogram[:, :max_time_frames]
#     return mel_spectrogram

def load_tokenizer(filename='tokenizer_LJ.pickle'):
    with open(filename, 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer

def preprocess_data(texts, mel_spectrograms, tokenizer, input_length=512):
    # Tokenize and pad texts
    sequences = tokenizer.texts_to_sequences(texts)
    padded_texts = pad_sequences(sequences, maxlen=input_length, padding='post')
    
    # Load and preprocess mel spectrograms
    mel_spectrograms = [np.load(mel) for mel in mel_spectrograms]
    mel_spectrograms = np.array(mel_spectrograms)

    return padded_texts, mel_spectrograms


def improved_tts_model(vocab_size, input_length, mel_dim=128, rnn_units=512, embed_dim=256):    # 512,256
    # Encoder
    encoder_inputs = layers.Input(shape=(input_length,))
    # masked_inputs = layers.Masking(mask_value=0)(encoder_inputs)

    encoder_embedding = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)(encoder_inputs)

    encoder_conv = layers.Conv1D(filters=32, kernel_size=5, activation='relu', padding='same')(encoder_embedding)
    encoder_conv = layers.Conv1D(filters=32, kernel_size=5, activation='relu', padding='same')(encoder_conv)

    encoder_lstm1 = layers.Bidirectional(layers.LSTM(rnn_units, return_sequences=True),)(encoder_conv)
    encoder_lstm2 = layers.Bidirectional(layers.LSTM(rnn_units, return_sequences=True))(encoder_lstm1)
    # encoder_lstm2 = layers.Dropout(0.5)(encoder_lstm2)

    # Attention Layer
    attention = layers.Attention()([encoder_lstm2, encoder_lstm2])
    # attention = layers.AdditiveAttention()([encoder_lstm2, encoder_lstm2])
    # attention = layers.MultiHeadAttention(num_heads=4, key_dim=rnn_units)(encoder_lstm2, encoder_lstm2)

    # Decoder
    decoder_lstm = layers.LSTM(rnn_units, return_sequences=True,dropout=0.3)(attention)
    # decoder_lstm = layers.BatchNormalization()(decoder_lstm)
    mel_output = layers.Dense(mel_dim, activation='linear')(decoder_lstm)
    mel_output = layers.Reshape((mel_dim, input_length))(mel_output)

    cosine_similarity_metric =CosineSimilarity(axis=-1)

    model = tf.keras.Model(inputs=encoder_inputs, outputs=mel_output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse',metrics=['mae',cosine_similarity_metric])
    return model

# Training Callbacks for model improvement
e_s = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1)

checkpoint = ModelCheckpoint(
    'best_model.h5',  # File path to save the model
    monitor='val_loss',  # Monitor validation loss
    verbose=1,  # Print a message when the model is saved
    save_best_only=True,  # Save the model only if the validation loss improves
    mode='min',  # 'min' means save when the monitored metric decreases (for loss)
    save_weights_only=False,  # Save the entire model (not just weights)
)

reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1)

df = pd.read_csv('tts_data_LJ.csv')
texts = df['Phoneme_text'].values
mel_spectrograms = df['Read_npy'].values

tokenizer = load_tokenizer()
padded_texts, mel_spectrograms = preprocess_data(texts, mel_spectrograms, tokenizer)

texts_train, texts_temp, mel_train, mel_temp = train_test_split(padded_texts, mel_spectrograms, test_size=0.2, random_state=42) # (10472, 600) (1309, 600) (1309, 600)
texts_val, texts_test, mel_val, mel_test = train_test_split(texts_temp, mel_temp, test_size=0.5, random_state=42)   # (10472, 128, 600) (1309, 128, 600) (1309, 128, 600)

vocab_size = len(tokenizer.word_index) + 1 

model = improved_tts_model(vocab_size,input_length=512)
model.summary()

# model.compile(optimizer='adam', loss="mse")
history = model.fit(
    np.array(texts_train), np.array(mel_train), 
    batch_size=32, epochs=20, 
    validation_data=(np.array(texts_val), np.array(mel_val)),
    callbacks=[e_s, checkpoint, reduce_lr]
)

model.save('tts_model_lj_LSTM_bandunau_att.keras')
model.save_weights('weights_LSTM_bandunau_att.h5')

test_loss = model.evaluate(np.array(texts_test), np.array(mel_test))
print(f"Test loss: {test_loss}")

for layer in model.layers:
    print(f"Layer: {layer.name}")
    weights = layer.get_weights()
    if len(weights) > 0:
        print(f"  Weights: {weights[0].shape}")
        print(f"  Weights Values: {weights[0]}")  # Print the actual weight values
        if len(weights) > 1:
            print(f"  Biases: {weights[1].shape}")
            print(f"  Biases Values: {weights[1]}")  # Print the actual bias values
    else:
        print("  No weights or biases.")
    print("-" * 50)

history_df = pd.DataFrame(history.history)
history_df['epoch'] = range(1, len(history_df) + 1)
history_df.to_csv('training_metrics.csv', index=False)

