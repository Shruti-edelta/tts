import numpy as np
import pandas as pd
import tensorflow as tf
import ast
from Fastspeech_1 import FastSpeechMini
from acoustic.text_preprocess import G2PConverter

def load_training_data(df, base_path=""):
    data = []

    for idx, row in df.iterrows():
        # Parse phonemes and durations from string to list
        phonemes = np.array(ast.literal_eval(row["Phoneme_text"]), dtype=np.int32)
        durations = np.array(ast.literal_eval(row["duration"]), dtype=np.int32)

        # Load mel spectrogram
        mel_path = row['Read_npy']
        mel = np.load(mel_path)

        # Sanity check (optional)
        if durations.sum() != mel.shape[0]:
            print(f"Warning: mismatch at {row['Read_npy']} - mel frames: {mel.shape[0]}, durations sum: {durations.sum()}")

        data.append({
            "phonemes": phonemes,
            "durations": durations,
            "mel": mel
        })
    
    return data

def load_padded_data(df):
    data = []

    for _, row in df.iterrows():
        phonemes = np.array(ast.literal_eval(row["Phoneme_text"]), dtype=np.int32)
        durations = np.array(ast.literal_eval(row["duration"]), dtype=np.int32)
        log_durations = np.log(durations + 1.0).astype(np.float32)

        mel = np.load(row['Read_npy'])  # shape: [T_pad, 80]

        data.append({
            "phonemes": phonemes,
            "durations": durations,
            "log_durations": log_durations,
            "mel": mel
        })

    return data

def make_fixed_dataset(data, batch_size):
    def gen():
        for item in data:
            yield item

    T_phoneme = len(data[0]["phonemes"])
    T_mel = data[0]["mel"].shape[0]

    output_signature = {
        "phonemes": tf.TensorSpec([T_phoneme], tf.int32),
        "durations": tf.TensorSpec([T_phoneme], tf.int32),
        "log_durations": tf.TensorSpec([T_phoneme], tf.float32),
        "mel": tf.TensorSpec([T_mel, 80], tf.float32),
    }

    return tf.data.Dataset.from_generator(gen, output_signature=output_signature).batch(batch_size).prefetch(tf.data.AUTOTUNE)



# Usage
df=pd.read_csv('dataset/acoustic_dataset/train.csv')
dataset = load_training_data(df)
# data = load_padded_data(df)
# dataset = make_fixed_dataset(data, batch_size=16)


def preprocess_example(phonemes, durations, mel):
    # durations are [T], mel is [L, 80]
    log_durations = np.log(durations + 1).astype(np.int32)
    return {
        "phonemes": phonemes,
        "durations": durations,
        "log_durations": log_durations,
        "mel": mel
    }

def make_dataset(data_list, batch_size):
    def generator():
        for item in data_list:
            yield preprocess_example(item["phonemes"], item["durations"], item["mel"])

    output_signature = {
        "phonemes": tf.TensorSpec([None], tf.int32),
        "durations": tf.TensorSpec([None], tf.int32),
        "log_durations": tf.TensorSpec([None], tf.float32),
        "mel": tf.TensorSpec([None, 80], tf.float32),
    }

    ds = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    ds = ds.padded_batch(batch_size, padded_shapes={
        "phonemes": [None],
        "durations": [None],
        "log_durations": [None],
        "mel": [None, 80]
    })
    return ds.prefetch(tf.data.AUTOTUNE)

@tf.function
def train_step(model, batch, optimizer):
    with tf.GradientTape() as tape:
        mel_pred, log_dur_pred = model(batch["phonemes"], durations=batch["durations"], training=True)

        mel_loss = tf.reduce_mean(tf.abs(mel_pred - batch["mel"]))
        dur_loss = tf.reduce_mean(tf.square(log_dur_pred - batch["log_durations"]))

        loss = mel_loss + dur_loss

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss, mel_loss, dur_loss


g2p = G2PConverter(load_model=False)
print(g2p.phn2idx)
vocab_size = len(g2p.phn2idx)
epoch=100

model = FastSpeechMini(vocab_size=vocab_size)
optimizer = tf.keras.optimizers.Adam(1e-3)
print("start traioning")
for epoch in range(epoch):
    for batch in dataset:
        loss, mel_loss, dur_loss = train_step(model, batch, optimizer)
    print(f"Epoch {epoch} - Loss: {loss:.4f} | Mel: {mel_loss:.4f} | Dur: {dur_loss:.4f}")

