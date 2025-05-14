import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
import pickle
import keras_tuner as kt


# def pad_or_truncate(mel_spectrogram, max_time_frames):
#     if mel_spectrogram.shape[1] < max_time_frames:      # If the Mel spectrogram has fewer time frames than required, pad with zeros
#         pad_width = max_time_frames - mel_spectrogram.shape[1]
#         mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, pad_width)), mode='constant')
#     elif mel_spectrogram.shape[1] > max_time_frames:        # If the Mel spectrogram has more time frames than required, truncate it
#         mel_spectrogram = mel_spectrogram[:, :max_time_frames]
#     return mel_spectrogram

def build_model(hp):
    vocab_size = len(tokenizer.word_index) + 1  # Vocabulary size (add 1 for padding token)
    input_length = 512
    mel_dim = 128

    # Encoder
    encoder_inputs = layers.Input(shape=(input_length,))
    encoder_embedding = layers.Embedding(input_dim=vocab_size, output_dim=hp.Int('embed_dim', min_value=128, max_value=256, step=64))(encoder_inputs)
    # masked_inputs = layers.Masking(mask_value=0)(encoder_conv)

    # Encoder Conv1D Layers
    encoder_conv = layers.Conv1D(filters=hp.Int('conv_filters', min_value=32, max_value=64, step=32), 
                                  kernel_size=5, activation='relu', padding='same')(encoder_embedding)
    encoder_conv = layers.Conv1D(filters=hp.Int('conv_filters', min_value=32, max_value=64, step=32), 
                                  kernel_size=5, activation='relu', padding='same')(encoder_conv)
    
    encoder_lstm = layers.Bidirectional(layers.LSTM(hp.Int('rnn_units', min_value=256, max_value=512, step=128), 
                                                    return_sequences=True))(encoder_conv)

    # Attention Layer
    attention = layers.MultiHeadAttention(num_heads=4, key_dim=hp.Int('rnn_units', min_value=256, max_value=512, step=128))(encoder_lstm, encoder_lstm)

    # Decoder
    decoder_lstm = layers.LSTM(hp.Int('rnn_units', min_value=256, max_value=512, step=128), return_sequences=True)(attention)
    mel_output = layers.Dense(mel_dim, activation='linear')(decoder_lstm)

    mel_output = layers.Reshape((mel_dim, input_length))(mel_output)
    model = tf.keras.Model(inputs=encoder_inputs, outputs=mel_output)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='LOG')), 
                  loss='mse', metrics=['mae'])

    return model

def load_tokenizer(filename='tokenizer_LJ.pickle'):
    with open(filename, 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer

df = pd.read_csv('tts_data_LJ.csv')
texts = df['Phoneme_text'].values
mel_spectrograms = df['Read_npy'].values

tokenizer = load_tokenizer()
sequences = tokenizer.texts_to_sequences(texts)
input_length=512
padded_texts = pad_sequences(sequences, maxlen=input_length, padding='post')    # (1211, 245)

mel_spectrograms = [np.load(mel) for mel in mel_spectrograms] 
# mel_spectrograms = [pad_or_truncate(mel,input_length) for mel in mel_spectrograms]
mel_spectrograms = np.array(mel_spectrograms)

texts_train, texts_temp, mel_train, mel_temp = train_test_split(padded_texts, mel_spectrograms, test_size=0.2, random_state=42)
texts_val, texts_test, mel_val, mel_test = train_test_split(texts_temp, mel_temp, test_size=0.5, random_state=42)

# Initialize the tuner
tuner = kt.Hyperband(build_model,
                     objective='val_loss',  # or 'val_accuracy' depending on your problem
                     max_epochs=20,         # Number of epochs for each trial
                     hyperband_iterations=2,  # Number of iterations to run the hyperband algorithm
                     directory='my_dir',   # Directory to store the results
                     project_name='TTS_model_tuning')

early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True,verbose=1)

tuner.search(np.array(texts_train), np.array(mel_train), epochs=20, validation_data=(np.array(texts_val), np.array(mel_val)), callbacks=[early_stopping])

best_hp = tuner.get_best_hyperparameters(num_trials=1)
print("====",best_hp.values)

best_model = tuner.hypermodel.build(best_hp)
best_model.summary()

history=best_model.fit(np.array(texts_train), np.array(mel_train), epochs=20, validation_data=(np.array(texts_val), np.array(mel_val)), callbacks=[early_stopping])

best_model.save_weights('model_weights.h5')


for layer in best_model.layers:
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

best_model.save('best_tts_model.keras')

# # Example function to build your model with hyperparameters
# def build_model(hp):
#     vocab_size = len(tokenizer.word_index) + 1  # Vocabulary size (add 1 for padding token) 1997+1  # Number of unique phonemes
#     input_length = 512
#     mel_dim = 128

#     # Encoder
#     encoder_inputs = layers.Input(shape=(input_length,))
#     masked_inputs = layers.Masking(mask_value=0)(encoder_inputs)
#     encoder_embedding = layers.Embedding(input_dim=vocab_size, output_dim=hp.Int('embed_dim', min_value=128, max_value=512, step=64))(masked_inputs)

#     # Encoder Conv1D Layers
#     encoder_conv = layers.Conv1D(filters=hp.Int('conv_filters', min_value=32, max_value=128, step=32), 
#                                   kernel_size=5, activation='relu', padding='same')(encoder_embedding)
#     encoder_conv = layers.Conv1D(filters=hp.Int('conv_filters', min_value=32, max_value=128, step=32), 
#                                   kernel_size=5, activation='relu', padding='same')(encoder_conv)

#     encoder_lstm = layers.Bidirectional(layers.LSTM(hp.Int('rnn_units', min_value=256, max_value=1024, step=256), 
#                                                     return_sequences=True))(encoder_conv)

#     # Attention Layer
#     attention = layers.MultiHeadAttention(num_heads=4, key_dim=hp.Int('rnn_units', min_value=256, max_value=1024, step=256))(encoder_lstm, encoder_lstm)

#     # Decoder
#     decoder_lstm = layers.LSTM(hp.Int('rnn_units', min_value=256, max_value=1024, step=256), return_sequences=True)(attention)
#     mel_output = layers.Dense(mel_dim, activation='linear')(decoder_lstm)
#     mel_output = layers.Reshape((mel_dim, input_length))(mel_output)

#     model = tf.keras.Model(inputs=encoder_inputs, outputs=mel_output)

#     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='LOG')), 
#                   loss='mse', metrics=['mae'])
#     return model