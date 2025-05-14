# import nltk
# from nltk.corpus import cmudict

# # # Load the CMU Pronouncing Dictionary
# # # nltk.download('cmudict')
# d = cmudict.dict()
# # # d={"a":[2,3,1],"b":[3.45,53,2]}
# # print(len(d.keys()))
# # print(d["for"])

# # Example function to extract phonemes from a word
# def get_phonemes(word):
#     word = word.lower()
#     if word in d:
#         return d[word][0]  # Return the first pronunciation variant
#     else:
#         return None

# # Example: Get phonemes for "cat"
# phonemes = get_phonemes("doctor")
# print(phonemes)  # Output: ['K', 'AE', 'T']

# # Now, build a phoneme vocabulary from this
# phoneme_vocab = {}
# index = 0

# # Loop through all words in the CMU dictionary and collect unique phonemes
# for word in d:
#     for phoneme_sequence in d[word]:
#         for phoneme in phoneme_sequence:
#             if phoneme not in phoneme_vocab:
#                 phoneme_vocab[phoneme] = index
#                 index += 1

# print(phoneme_vocab)

# import tensorflow as tf
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from tensorflow.keras import layers
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.callbacks import EarlyStopping
# import pickle

# # def pad_or_truncate(mel_spectrogram, max_time_frames):
# #     if mel_spectrogram.shape[1] < max_time_frames:      # If the Mel spectrogram has fewer time frames than required, pad with zeros
# #         pad_width = max_time_frames - mel_spectrogram.shape[1]
# #         mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, pad_width)), mode='constant')
# #     elif mel_spectrogram.shape[1] > max_time_frames:        # If the Mel spectrogram has more time frames than required, truncate it
# #         mel_spectrogram = mel_spectrogram[:, :max_time_frames]
# #     return mel_spectrogram


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

# def load_tokenizer(filename='tokenizer_LJ.pickle'):
#     with open(filename, 'rb') as handle:
#         tokenizer = pickle.load(handle)
#     return tokenizer

# df = pd.read_csv('tts_data_LJ.csv')
# texts = df['Phoneme_text'].values
# mel_spectrograms = df['Read_npy'].values

# tokenizer = load_tokenizer()
# sequences = tokenizer.texts_to_sequences(texts)
# # input_length=max(len(sequences[i]) for i in range(len(sequences)))   # 145
# input_length=512
# print(texts.shape,len(sequences),len(sequences[0]))
# print(type(sequences[0]))
# padded_texts = pad_sequences(sequences, maxlen=input_length, padding='post')    # (1211, 245)
# # print(padded_texts)

# mel_spectrograms = [np.load(mel) for mel in mel_spectrograms] 
# # mel_spectrograms = [pad_or_truncate(mel,input_length) for mel in mel_spectrograms]
# mel_spectrograms = np.array(mel_spectrograms)
# print(mel_spectrograms[0].shape)

# texts_train, texts_temp, mel_train, mel_temp = train_test_split(padded_texts, mel_spectrograms, test_size=0.2, random_state=42)
# texts_val, texts_test, mel_val, mel_test = train_test_split(texts_temp, mel_temp, test_size=0.5, random_state=42)

# print(texts_train.shape,texts_test.shape,texts_val.shape)       # (10472, 600) (1309, 600) (1309, 600)
# print(mel_train.shape,mel_test.shape,mel_val.shape)     # (10472, 128, 600) (1309, 128, 600) (1309, 128, 600)
# vocab_size = len(tokenizer.word_index) + 1  # Vocabulary size (add 1 for padding token) 1997+1  # Number of unique phonemes

# model = improved_tts_model(vocab_size,input_length)
# model.summary()

# e_s=EarlyStopping(monitor="val_loss",patience=5,restore_best_weights=True)      # for stop overfitting

# # model.compile(optimizer='adam', loss="mse")
# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse',metrics=['mae'])
# model.fit(np.array(texts_train), np.array(mel_train), batch_size=32, epochs=20, validation_data=(np.array(texts_val), np.array(mel_val)),callbacks=[e_s])

# test_loss = model.evaluate(np.array(texts_test), np.array(mel_test))
# print(f"Test loss: {test_loss}")

# model.save('tts_model_lj_LSTM_attmulti.keras')


import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.metrics import CosineSimilarity
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# def improved_tts_model(vocab_size, input_length, mel_dim=128, rnn_units=512, embed_dim=256):    
#     # Encoder
#     encoder_inputs = layers.Input(shape=(input_length,))
    
#     encoder_embedding = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)(encoder_inputs)
    
#     encoder_conv = layers.Conv1D(filters=64, kernel_size=5, activation='relu', padding='same')(encoder_embedding)
#     encoder_conv = layers.Conv1D(filters=64, kernel_size=5, activation='relu', padding='same')(encoder_conv)
    
#     encoder_lstm1 = layers.Bidirectional(layers.LSTM(rnn_units, return_sequences=True, 
#                                                       kernel_regularizer=regularizers.l2(0.001)))(encoder_conv)
#     encoder_lstm2 = layers.Bidirectional(layers.LSTM(rnn_units, return_sequences=True, 
#                                                       kernel_regularizer=regularizers.l2(0.001)))(encoder_lstm1)

#     # Attention Layer
#     attention = layers.MultiHeadAttention(num_heads=8, key_dim=rnn_units)(encoder_lstm2, encoder_lstm2)
#     attention = layers.Add()([encoder_lstm2, attention])  # Residual connection

#     # Decoder (Add a few more LSTM layers here)
#     decoder_lstm = layers.LSTM(rnn_units, return_sequences=True )(attention)
#     decoder_lstm = layers.LSTM(rnn_units, return_sequences=True )(decoder_lstm)  # Additional LSTM layer
#     decoder_lstm = layers.BatchNormalization()(decoder_lstm) 
    
#     # Add Dense Layer(s) after the LSTM layer(s)
#     dense1 = layers.Dense(1024, activation='relu')(decoder_lstm)
#     dense1 = layers.Dropout(0.3)(dense1)  # Apply dropout to dense1
    
#     dense2 = layers.Dense(512, activation='relu')(dense1)
#     dense2 = layers.Dropout(0.3)(dense2)  # Apply dropout to dense2
    
#     dense3 = layers.Dense(256, activation='relu')(dense2)
#     dense3 = layers.Dropout(0.3)(dense3)  # Apply dropout to dense3
    
#     # Output Layer
#     mel_output = layers.Dense(mel_dim, activation='linear')(dense3)
#     mel_output = layers.Reshape((mel_dim, input_length))(mel_output)
#     # Cosine Similarity Metric
#     cosine_similarity_metric = CosineSimilarity(axis=-1)

#     # Build the model
#     model = tf.keras.Model(inputs=encoder_inputs, outputs=mel_output)
#     optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)     # clip for stabilizing the training process
#     model.compile(optimizer=optimizer, loss='mse', metrics=['mae', cosine_similarity_metric])
#     return model

def improved_tts_model(vocab_size, input_length, mel_dim=128, rnn_units=512, embed_dim=256):
    # Encoder
    encoder_inputs = layers.Input(shape=(input_length,))
    
    encoder_embedding = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)(encoder_inputs)
    
    encoder_conv = layers.Conv1D(filters=64, kernel_size=5, activation='relu', padding='same')(encoder_embedding)
    encoder_conv = layers.Conv1D(filters=64, kernel_size=5, activation='relu', padding='same')(encoder_conv)
    
    encoder_lstm1 = layers.Bidirectional(layers.LSTM(rnn_units, return_sequences=True, 
                                                      kernel_regularizer=regularizers.l2(0.01)))(encoder_conv)
    encoder_lstm2 = layers.Bidirectional(layers.LSTM(rnn_units, return_sequences=True, 
                                                      kernel_regularizer=regularizers.l2(0.01)))(encoder_lstm1)
    
    # Attention Layer (Multi-Head Attention)
    attention = layers.MultiHeadAttention(num_heads=8, key_dim=rnn_units)(encoder_lstm2, encoder_lstm2)
    attention = layers.Add()([encoder_lstm2, attention])  # Residual connection
    
    # Decoder
    decoder_lstm = layers.LSTM(rnn_units, return_sequences=True, dropout=0.3)(attention)
    
    mel_output = layers.Dense(mel_dim, activation='linear')(decoder_lstm)
    mel_output = layers.Reshape((mel_dim, input_length))(mel_output)

    # Cosine similarity metric
    cosine_similarity_metric = CosineSimilarity(axis=-1)

    model = tf.keras.Model(inputs=encoder_inputs, outputs=mel_output)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)     # clip for stabilizing the training process
    model.compile(optimizer=optimizer,loss='mse', metrics=['mae', cosine_similarity_metric])
    return model

def preprocess_data(texts, mel_spectrograms, tokenizer, input_length=512):
    sequences = tokenizer.texts_to_sequences(texts)
    padded_texts = pad_sequences(sequences, maxlen=input_length, padding='post')

    mel_spectrograms = [np.load(mel) for mel in mel_spectrograms]
    mel_spectrograms = np.array(mel_spectrograms)
    return padded_texts, mel_spectrograms


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

texts_train, texts_temp, mel_train, mel_temp = train_test_split(padded_texts, mel_spectrograms, test_size=0.2, random_state=42)
texts_val, texts_test, mel_val, mel_test = train_test_split(texts_temp, mel_temp, test_size=0.5, random_state=42)

vocab_size = len(tokenizer.word_index) + 1
model = improved_tts_model(vocab_size, input_length=512)
model.summary()

from tensorflow.keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt

# Callback for ReduceLROnPlateau
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

# Track learning rate
class LearningRatePlotter(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.learning_rate
        self.lrs.append(lr)

    def on_train_begin(self, logs=None):
        self.lrs = []

# Instantiate the callback
lr_plotter = LearningRatePlotter()

# Train the model
history = model.fit(
    np.array(texts_train), np.array(mel_train),
    batch_size=32, epochs=20,
    validation_data=(np.array(texts_val), np.array(mel_val)),
    callbacks=[lr_plotter, e_s, checkpoint, reduce_lr]
)

model.save('tts_model_lj_LSTM_bandunau_att.keras')
model.save_weights('metrics/weights_LSTM_bandunau_att.h5')

# Evaluate on test data
test_loss = model.evaluate(np.array(texts_test), np.array(mel_test))
print(f"Test loss: {test_loss}")

# Plot learning rate vs epoch
plt.plot(range(1, len(lr_plotter.lrs) + 1), lr_plotter.lrs)
plt.xlabel('Epochs')
plt.ylabel('Learning Rate')
plt.title('Learning Rate vs Epoch')
plt.grid(True)
plt.show()

import matplotlib.pyplot as plt

# Plot training & validation loss
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Plot training & validation cosine similarity
plt.figure(figsize=(12, 6))
plt.plot(history.history['cosine_similarity'], label='Training Cosine Similarity')
plt.plot(history.history['val_cosine_similarity'], label='Validation Cosine Similarity')
plt.title('Cosine Similarity')
plt.xlabel('Epochs')
plt.ylabel('Cosine Similarity')
plt.legend()
plt.grid(True)
plt.show()

import librosa.display
import matplotlib.pyplot as plt

def plot_spectrogram(predicted_mel, actual_mel, epoch):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    librosa.display.specshow(actual_mel.T, x_axis='time', cmap='Blues')
    plt.title(f'Actual Spectrogram - Epoch {epoch}')
    plt.colorbar()

    plt.subplot(1, 2, 2)
    librosa.display.specshow(predicted_mel.T, x_axis='time', cmap='Blues')
    plt.title(f'Predicted Spectrogram - Epoch {epoch}')
    plt.colorbar()
    plt.tight_layout()
    plt.show()

# Visualize after the training epoch
predicted_mel = model.predict(np.array(texts_test[:1]))
plot_spectrogram(predicted_mel[0], mel_test[0], epoch=1)

