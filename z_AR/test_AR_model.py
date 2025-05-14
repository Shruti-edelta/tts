
import matplotlib.pyplot as plt
import numpy as np
from acoustic.text_preprocess import G2PConverter
from tensorflow.keras.preprocessing.sequence import pad_sequences
import librosa
import ast
import tensorflow as tf
import pandas as pd
import soundfile as sf

def mel_to_audio(mel_spectrogram, sr=22050):
    mel_inv = librosa.feature.inverse.mel_to_audio(mel_spectrogram, sr=sr)
    return mel_inv

best_model = tf.keras.models.load_model(
    "model/2/best_model_cnn.keras",
    compile=False
    # custom_objects={
    #     'CropLayer': CropLayer,
    #     'ResidualBlock': ResidualBlock,
    #     'AttentionContextLayer': AttentionContextLayer,
    #     'AdditiveAttention': AdditiveAttention,
    #     'add_positional_encoding': add_positional_encoding,
    #     'positional_encoding': positional_encoding,  # if used inside
    # }
)

g2p = G2PConverter("model/1/model_cnn.keras")

text = "This is a test "
text = "However, it now laid down in plain language and with precise details the requirements of a good jail system."
text = "These bankers wishing for more specific information"
text="and into his possible motivation for the assassination"
phonemes = g2p.predict(text)  # use your G2P model
phoneme_input = pad_sequences([phonemes], maxlen=200, padding='post')
print(phoneme_input.shape)


max_mel_length = 1024  # or any stopping condition you want
mel_dim = 80           # depends on your mel dimension

generated_mel = []
current_input = np.zeros((1, 1, mel_dim), dtype=np.float32)  # initial empty mel

print(current_input[0][0].shape)

for _ in range(max_mel_length):
    # Model expects (phoneme input, previous mels)
    next_mel_frame = best_model.predict([phoneme_input, current_input], verbose=0)
    
    # Get only the last predicted frame
    next_frame = next_mel_frame[:, -1, :]  # shape (1, mel_dim)
    
    generated_mel.append(next_frame.numpy()[0])  # Save

    # Update current_input to include new frame
    current_input = np.concatenate([current_input, next_frame[:, np.newaxis, :]], axis=1)


final_mel = np.stack(generated_mel, axis=0)  # shape (num_frames, mel_dim)


# =================== test_data ====================

def preprocess_testdata(texts, mel_spectrograms, input_length=200,mel_max_len=1024):
    def pad_or_truncate(mel, max_len=mel_max_len):
        T, D = mel.shape
        if T > max_len:
            print("============== truncate ========")
            mel = mel[:max_len, :]
        elif T < max_len:
            pad_width = max_len - T
            mel = np.pad(mel, ((0, pad_width), (0, 0)), mode='constant')
        return mel
    # print(texts[0])
    padded_texts = pad_sequences(texts, maxlen=input_length, padding='post')
    # print(padded_texts[0])
    mel_spectrograms = [np.load(mel) for mel in mel_spectrograms]
    mel_spectrograms = [pad_or_truncate(mel) for mel in mel_spectrograms]
    mel_spectrograms = np.array(mel_spectrograms)
    return padded_texts, mel_spectrograms


df_test = pd.read_csv('dataset/acoustic_dataset/test.csv')
phoneme_seq = df_test['Phoneme_text'].apply(ast.literal_eval).values
mel_spectrograms_test = df_test['Read_npy'].values

test_text, test_mel = preprocess_testdata(phoneme_seq, mel_spectrograms_test)
x_test = np.expand_dims(test_text[0], axis=0)  # Adding a batch dimension
y_true = np.expand_dims(test_mel[0], axis=0)  # Adding a batch dimension)

# test_loss = model.evaluate(x_test, y_true)
# print(f"Test loss: {test_loss}")
# print(x_test)

generated_mel_t = []
current_input_t = np.zeros((1, 1, mel_dim), dtype=np.float32)  # initial empty mel

print(current_input_t[0][0].shape)

for _ in range(max_mel_length):
    # Model expects (phoneme input, previous mels)
    next_mel_frame = best_model.predict([x_test, current_input_t], verbose=0)
    
    # Get only the last predicted frame
    next_frame = next_mel_frame[:, -1, :]  # shape (1, mel_dim)
    
    generated_mel_t.append(next_frame.numpy()[0])  # Save

    # Update current_input to include new frame
    current_input_t = np.concatenate([current_input_t, next_frame[:, np.newaxis, :]], axis=1)


y_pred = np.stack(generated_mel_t, axis=0)  # shape (num_frames, mel_dim)


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
print("y_true mel :", y_true)
print("Audio shape:", audio)
sf.write('org_lj1.wav', audio, 22050) 
audio = mel_to_audio(y_pred[0].T)
print("Predicted mel spectrograms:", y_pred)
print("Audio shape:", audio)
sf.write('testing_lj1.wav', audio, 22050) 

are_equal = np.allclose(y_true[0].T, y_pred[0].T,)
print("is equal? ", are_equal)
are_equal = np.allclose(final_mel.T, y_pred[0].T,)
print("is equal? ", are_equal)


# are_equal = np.allclose(y_true[0].T, y_pred[0].T, atol=1e-1)







'''
import numpy as np
import tensorflow as tf

def generate_mel(model, phoneme_seq, max_frames=1000):
    """
    Autoregressively generate mel-spectrogram from input phoneme sequence.

    Args:
        model (tf.keras.Model): Trained AR acoustic model.
        phoneme_seq (list or np.ndarray): Input phoneme sequence (list of int IDs).
        max_frames (int): Maximum number of mel frames to generate.

    Returns:
        np.ndarray: Generated mel-spectrogram of shape (T, mel_dim).
    """
    # Ensure phoneme_seq is a batch of size 1
    phoneme_seq = np.array(phoneme_seq, dtype=np.int32)
    phoneme_seq = np.expand_dims(phoneme_seq, axis=0)  # [1, T_phonemes]

    # Initial mel input: zeros (first frame)
    mel_dim = model.output_shape[-1]
    generated = []

    prev_mel = np.zeros((1, 1, mel_dim), dtype=np.float32)  # [batch, time, mel_dim]

    # Autoregressive loop
    for _ in range(max_frames):
        # Predict next frame
        prediction = model.predict([phoneme_seq, prev_mel], verbose=0)
        next_frame = prediction[:, -1:, :]  # Only take the last frame output

        generated.append(next_frame.numpy())

        # Update prev_mel: concatenate to simulate growing sequence
        prev_mel = tf.concat([prev_mel, next_frame], axis=1)

        # Optional stopping condition: if next_frame energy is too small (model-dependent)
        # energy = tf.reduce_mean(tf.abs(next_frame))
        # if energy < 0.001:
        #     break

    # Combine generated frames
    generated = tf.concat(generated, axis=1)  # [1, T, mel_dim]
    return generated[0].numpy()  # Remove batch dimension

    
'''