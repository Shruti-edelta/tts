import numpy as np
import pandas as pd
import tensorflow as tf
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
import pickle
from acoustic.text_preprocess  import TextNormalizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def load_tokenizer(filename='tokenizer_LJ.pickle'):
    with open(filename, 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer

def combined_loss(y_true, y_pred):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    cos_sim = tf.reduce_mean(tf.keras.losses.cosine_similarity(y_true, y_pred, axis=-1))  # returns negative
    return mse + 0.1 * cos_sim

def plot_mel_spectrogram(mel_spectrogram):
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(mel_spectrogram, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Generated Mel Spectrogram")
    plt.show()

def preprocess_text(text, tokenizer, input_length):
    phonemes = TextNormalizer().text_to_phonemes(text)  # Get phonemes directly from the TextNormalizer
    print(phonemes)
    sequence = tokenizer.texts_to_sequences(phonemes)  # Pass as list for correct tokenizer processing
    # print(sequence)
    flat_list = [item for sublist in sequence for item in sublist]
    flattened_array = np.array(flat_list)
    # print(flattened_array.shape)
    padded_sequence = pad_sequences([flattened_array], maxlen=input_length, padding='post')  # Pad to fixed length
    # print(type(padded_sequence))
    return padded_sequence

def generate_mel_spectrogram(text_input, model, tokenizer, input_length=512):
    processed_input = preprocess_text(text_input, tokenizer, input_length)
    print(processed_input)
    predicted_mel = model.predict(processed_input)  # Shape: (1, mel_dim, time_steps)
    # print(predicted_mel,predicted_mel.shape)
    predicted_mel = predicted_mel[0].T  # Shape: (mel_dim, time_steps) T=(512,128)
    # print(predicted_mel.shape)
    return predicted_mel

def mel_to_audio(mel_spectrogram, sr=22050):
    mel_inv = librosa.feature.inverse.mel_to_audio(mel_spectrogram, sr=sr)
    return mel_inv

tokenizer = load_tokenizer()
model = tf.keras.models.load_model('checkpoints/first/model_epoch_03_valLoss_0.4616.keras',custom_objects={'combined_loss': combined_loss})  # Load your trained model
# model= tf.keras.models.load_model('metrics/epoch2.h5')  # Load your trained model

# text_input = "hello world"  # Your own text input
# text_input="I had no conception that vessels ever came so far north and was astounded at the sight."
# text_input="He likewise indicated he was disenchanted with Russia."
text_input="However, it now laid down in plain language and with precise details the requirements of a good jail system."
# text_input = 's ah'  # Input text (phoneme sequence)
# text_input='HH AH0 L OW1'

predicted_mel = generate_mel_spectrogram(text_input, model, tokenizer)
# print("predicted mel single: ",predicted_mel)
# plot_mel_spectrogram(predicted_mel)
# audio = mel_to_audio(predicted_mel)
# sf.write('text_testing_lj1.wav', audio, 22050) 

# =================== test_data ====================

def preprocess_testdata(texts, mel_spectrograms, tokenizer, input_length=512):
    # Preprocess text (phoneme transcription)
    sequences = tokenizer.texts_to_sequences(texts)
    print(texts[0])
    padded_texts = pad_sequences(sequences, maxlen=input_length, padding='post')
    
    # Preprocess mel spectrograms (assuming mel_spectrograms are paths to .npy files)
    mel_spectrograms = [np.load(mel) for mel in mel_spectrograms]
    mel_spectrograms = np.array(mel_spectrograms)
    
    return padded_texts, mel_spectrograms

df_test = pd.read_csv('test_dataset1.csv')
texts_test = df_test['Text'].values
mel_spectrograms_test = df_test['Mel_Spectrogram'].values
test_text, test_mel = preprocess_testdata(texts_test, mel_spectrograms_test, tokenizer)

x_test = np.expand_dims(test_text[0], axis=0)  # Adding a batch dimension
y_true = np.expand_dims(test_mel[0], axis=0)  # Adding a batch dimension

# test_loss = model.evaluate(x_test, y_true)
# print(f"Test loss: {test_loss}")

print(x_test)
y_pred = model.predict(x_test)
print("Predicted mel spectrograms:", y_pred)

print("Input shape:", x_test.shape)
print("True mel shape:", y_true.shape)
print("Predicted mel shape:", y_pred.shape)

# Visualize 1 example

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.title("Ground Truth")
plt.imshow(y_true[0].T, aspect='auto', origin='lower')

plt.subplot(1, 2, 2)
plt.title("Prediction")
plt.imshow(y_pred[0].T, aspect='auto', origin='lower')
plt.tight_layout()
plt.show()

audio=mel_to_audio(y_true[0].T)
sf.write('org_lj1.wav', audio, 22050) 
audio = mel_to_audio(y_pred[0].T)
sf.write('testing_lj1.wav', audio, 22050) 

are_equal = np.allclose(y_true[0].T, y_pred[0].T)
are_equal = np.allclose(predicted_mel, y_pred[0].T)
print("is equal? ", are_equal)





# [[126 127  11  47  41   9   7   2  70  13  31  34  34  70   2  15   2  29
#    13  31   2  13  17  86  67  65   1  76   1   2  34  65   9  82  29   6
#    15  14  45  14  34  15   7  31  13  48  82   1   6  15  18  65  45   6
#    24   1   2   7  14  27  47   1  67  73  34  76  31  13  14   9  14   7
#     1  24   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
#     0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0

# [[ 19 127  11  47  39  60   5  56 133  61  31   4  49  70   8  46   8  33
#    13  31   8  61  17  86  67  65   1 105  22   2   4  16   9  85  33   6
#    15  14  45  25  49  15   7  31  13  10   3  12  50  15  18  65  45   6
#    24   1   2   7  25  30  26  62  74  73   4  88  31  40  21   9  14   7
#     1  57   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
#     0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
#

