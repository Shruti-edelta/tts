import tensorflow as tf
import pandas as pd
import numpy as np
import ast

class FastSpeechDataset:
    def __init__(self, csv_path, mel_base_path, batch_size=32, shuffle=True, buffer_size=1000):
        print(csv_path)
        self.df = pd.read_csv(csv_path)
        self.mel_base_path = mel_base_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.buffer_size = buffer_size

    def _parse_row(self, phonemes, durations, mel_path):

        phonemes = np.array(ast.literal_eval(phonemes.decode()), dtype=np.int32)  # Convert string to list, then to array
        durations = np.array(ast.literal_eval(durations.decode()), dtype=np.float32)  # Convert string to list, then to array
        mel = np.load(mel_path.decode()).astype(np.float32)  # Load .npy file
        return phonemes,durations, mel
    
    def tf_parse_row(self, phonemes, durations, mel_path):
        phoneme_ids, duration_vals, mel_tensor = tf.numpy_function(
            self._parse_row,
            [phonemes, durations, mel_path],
            (tf.int32, tf.float32, tf.float32)
        )
        # Reshape or set shapes if known
        phoneme_ids.set_shape([None])
        duration_vals.set_shape([None])
        mel_tensor.set_shape([None, 80])  # assuming 80-dim mel spectrograms
        print(phoneme_ids.shape,duration_vals.shape,mel_tensor.shape)
        return (phoneme_ids, duration_vals), mel_tensor


    def create(self):
        phonemes = tf.convert_to_tensor(self.df['Phoneme_text'].values, dtype=tf.string)
        durations = tf.convert_to_tensor(self.df['duration'].values, dtype=tf.string)
        mel_paths = tf.convert_to_tensor(self.df['Read_npy'].values, dtype=tf.string)

        dataset = tf.data.Dataset.from_tensor_slices((phonemes, durations, mel_paths))
        if self.shuffle:
            dataset = dataset.shuffle(self.buffer_size)

        # dataset = dataset.map(self.tf_parse_row, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(lambda x, y,z: self.tf_parse_row(x, y,z), num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset
