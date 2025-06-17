import re

# Example: simple G2P using CMUdict
import nltk
nltk.download('cmudict')
from nltk.corpus import cmudict
cmu = cmudict.dict()

def g2p(word):
    word = word.lower()
    if word in cmu:
        return cmu[word][0]  # Return first pronunciation
    else:
        return ['SPN']  # Spoken noise or unknown token

def text_to_phonemes(text):
    words = re.findall(r"\b[\w']+\b", text)
    phonemes = ['<sos>']
    for word in words:
        phonemes.extend(g2p(word))
    phonemes.append('<eos>')
    return phonemes

# Build vocabulary from dataset
phoneme_vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, 'AA': 3, 'AE': 4, ..., 'SPN': 60}
phoneme_to_id = phoneme_vocab
id_to_phoneme = {v: k for k, v in phoneme_vocab.items()}

def phonemes_to_ids(phonemes):
    return [phoneme_to_id.get(p, phoneme_to_id['SPN']) for p in phonemes]

import librosa
import numpy as np

def wav_to_mel(path, sr=22050, n_fft=1024, hop_length=256, n_mels=80):
    y, _ = librosa.load(path, sr=sr)
    mel = librosa.feature.melspectrogram(y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db.T  # shape: (frames, n_mels)


import tensorflow as tf

def load_data(csv_path):
    import pandas as pd
    df = pd.read_csv(csv_path)
    return df  # should have columns: 'phonemes' (list), 'mel_path'

def preprocess(row):
    phoneme_ids = phonemes_to_ids(eval(row['phonemes']))
    mel = np.load(row['mel_path'])  # already extracted and saved
    return phoneme_ids, mel

def tf_preprocess(row):
    phoneme_ids, mel = tf.numpy_function(preprocess, [row], [tf.int32, tf.float32])
    return phoneme_ids, mel



from tensorflow.keras import layers, Model

def build_acoustic_model(vocab_size, mel_dim=80):
    input_seq = layers.Input(shape=(None,), dtype='int32')
    x = layers.Embedding(vocab_size, 256)(input_seq)
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)
    x = layers.Dense(512, activation='relu')(x)
    mel_output = layers.TimeDistributed(layers.Dense(mel_dim))(x)
    return Model(inputs=input_seq, outputs=mel_output)

model = build_acoustic_model(vocab_size=len(phoneme_vocab))
model.compile(optimizer='adam', loss='mse')

# tf.data pipeline (example)
def generator(df):
    for _, row in df.iterrows():
        yield preprocess(row)

ds = tf.data.Dataset.from_generator(lambda: generator(df),
                                    output_types=(tf.int32, tf.float32),
                                    output_shapes=((None,), (None, 80)))
ds = ds.padded_batch(16, padded_shapes=([None], [None, 80]))
model.fit(ds, epochs=20)


import torch
import torchaudio

hifigan = torch.hub.load('bshall/hifi-gan:main', 'hifi_gan', model_name='vctk')
hifigan.eval()

def mel_to_audio(mel_np):
    mel = torch.tensor(mel_np.T).unsqueeze(0)  # (1, n_mels, T)
    with torch.no_grad():
        audio = hifigan(mel).squeeze().cpu().numpy()
    return audio


def tts_infer(text, acoustic_model, vocoder):
    phonemes = text_to_phonemes(text)
    ids = phonemes_to_ids(phonemes)
    ids_tensor = tf.constant([ids])
    
    mel = acoustic_model.predict(ids_tensor)[0]  # (T, 80)
    audio = mel_to_audio(mel)
    return audio



