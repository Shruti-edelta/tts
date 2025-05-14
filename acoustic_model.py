import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.metrics import CosineSimilarity
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import ast
from text_preprocess import G2PConverter
from keras.saving import register_keras_serializable

def create_dataset_fast(texts, mel_paths, input_length=168, mel_dim=80, mel_max_len=900, batch_size=32):
    def load_and_preprocess_py(text, mel_path):
        # Convert bytes to Python types
        text = text.numpy().decode("utf-8")
        mel_path = mel_path.numpy().decode("utf-8")
        # Parse phoneme sequence from string
        phoneme_seq = ast.literal_eval(text)
        padded_text = pad_sequences([phoneme_seq], maxlen=input_length, padding='post')[0].astype(np.int32)
        # Load mel-spectrogram
        mel = np.load(mel_path).astype(np.float32)
        T, D = mel.shape
        # Pad or truncate mel to mel_max_len
        if T > mel_max_len:
            mel = mel[:mel_max_len, :]
        elif T < mel_max_len:
            pad_len = mel_max_len - T
            mel = np.pad(mel, ((0, pad_len), (0, 0)), mode='constant')
        return padded_text, mel
    
    def tf_wrapper(text, mel_path):
        text_tensor, mel_tensor = tf.py_function(
            func=load_and_preprocess_py,
            inp=[text, mel_path],
            Tout=[tf.int32, tf.float32]
        )
        text_tensor.set_shape([input_length])
        mel_tensor.set_shape([mel_max_len, mel_dim])
        return text_tensor, mel_tensor
    
    dataset = tf.data.Dataset.from_tensor_slices((texts, mel_paths))
    dataset = dataset.map(tf_wrapper, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def compile_model(model):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
    def combined_loss(y_true, y_pred):
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        cos_sim = tf.reduce_mean(tf.keras.losses.cosine_similarity(y_true, y_pred, axis=-1))
        return mse + 0.1 * cos_sim
    
    model.compile(optimizer=optimizer,
                  loss=combined_loss,
                  metrics=['mae', CosineSimilarity(axis=-1)])
    return model

@register_keras_serializable()
class CropLayer(tf.keras.layers.Layer):
    def __init__(self, length, **kwargs):
        super().__init__(**kwargs)
        self.length = length

    def call(self, inputs):
        return inputs[:, :self.length, :]

    def get_config(self):
        config = super().get_config()
        config.update({'length': self.length})
        return config

def build_acoustic_model(vocab_size, input_length, mel_dim=80, mel_len=900, embed_dim=256):
    inputs = layers.Input(shape=(input_length,), name='phoneme_input')
    x = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)(inputs)

    # Initial Reshape
    x = layers.Reshape((input_length, embed_dim))(x)

    # Conv1D Stack with Residuals
    for _ in range(3):  # fewer residual blocks = better for CPU training
        residual = x
        x = layers.Conv1D(filters=256, kernel_size=5, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Add()([x, residual])  # residual connection

    # BiLSTM Layer for sequence modeling
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)

    # Optional attention (lightweight)
    attention = layers.Dense(1, activation='tanh')(x)
    attention = layers.Softmax(axis=1)(attention)
    x = layers.Multiply()([x, attention])

    # Upsampling stack
    x = layers.Conv1DTranspose(256, kernel_size=5, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv1DTranspose(128, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv1DTranspose(128, kernel_size=3, strides=2, padding='same', activation='relu')(x)

    # Crop to mel length
    # x = layers.Lambda(lambda t: t[:, :mel_len, :])(x)
    x = CropLayer(mel_len)(x)

    # Output layer
    mel_output = layers.Conv1D(mel_dim, kernel_size=1, activation='linear', name="mel_output")(x)

    model = tf.keras.Model(inputs=inputs, outputs=mel_output)
    return model


df = pd.read_csv('dataset/acoustic_dataset/tts_data_LJ.csv',usecols=['Phoneme_text', 'Read_npy'])
texts = df['Phoneme_text'].apply(ast.literal_eval).values  # Now it's list of lists
mel_spectrograms = df['Read_npy'].values

input_length =max([len(list(seq)) for seq in texts])
print(input_length)

texts_str = [str(seq) for seq in texts]

texts_train, texts_temp, mel_train, mel_temp = train_test_split(texts_str, mel_spectrograms, test_size=0.2, random_state=42)
texts_val, texts_test, mel_val, mel_test = train_test_split(texts_temp, mel_temp, test_size=0.3, random_state=42)

# train_df = pd.DataFrame({'Phoneme_text': texts_train, 'Read_npy': mel_train})
# val_df = pd.DataFrame({'Phoneme_text': texts_val, 'Read_npy': mel_val})
# test_df = pd.DataFrame({'Phoneme_text': texts_test, 'Read_npy': mel_test})

# train_df.to_csv('dataset/acoustic_dataset/train.csv', index=False)
# val_df.to_csv('dataset/acoustic_dataset/val.csv', index=False)
# test_df.to_csv('dataset/acoustic_dataset/test.csv', index=False)

train_dataset = create_dataset_fast(texts_train, mel_train,input_length=input_length)
# for batch in train_dataset.take(1):
#     inputs, targets = batch
#     print("Inputs shape:", inputs.shape)
#     print("Targets shape:", targets.shape)
#     print("Inputs:", inputs.numpy()[0])   # Show first example
#     print("Targets:", targets.numpy()[0]) # Show first mel-spectrogram
val_dataset = create_dataset_fast(texts_val, mel_val,input_length=input_length)
test_dataset=create_dataset_fast(texts_test,mel_test,input_length=input_length)

g2p = G2PConverter(load_model=False)
vocab_size = len(g2p.phn2idx)
print(vocab_size)

model = build_acoustic_model(vocab_size,input_length=input_length)
model = compile_model(model)
model.summary()

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True,verbose=1),
    ModelCheckpoint('model/2/best_model_cnn.keras', monitor='val_loss', save_best_only=True,save_weights_only=False,verbose=1,mode='min'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3,verbose=1)
]

history=model.fit(train_dataset,
          epochs=50,
          validation_data=val_dataset,
          callbacks=callbacks)

model.save('model/2/acoustic_model_cnn.keras')
model.save_weights('model/2/acoustic_model_cnn.weights.h5')

history_df = pd.DataFrame(history.history)
history_df['epoch'] = range(1, len(history_df) + 1)
history_df.to_csv('model/2/acoustic_model_cnn.csv', index=False)

test_loss = model.evaluate(test_dataset)
print(f"Test loss: {test_loss}")




# def build_acoustic_model(vocab_size, input_length=168, mel_dim=80, mel_len=900, embed_dim=256):
#     inputs = layers.Input(shape=(input_length,), name='phoneme_input')
#     x = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)(inputs)
#     x = layers.Reshape((input_length, embed_dim))(x)
#     # Conv stack
#     for _ in range(6):
#         x = layers.Conv1D(filters=256, kernel_size=5, padding='same', activation='relu')(x)
#         x = layers.BatchNormalization()(x)
#         x = layers.Dropout(0.2)(x)
#     # Upsample — aim for 336 → 900
#     x = layers.Conv1DTranspose(256, kernel_size=5, strides=2, padding='same', activation='relu')(x)  # 336 → 672
#     x = layers.Conv1DTranspose(128, kernel_size=3, strides=2, padding='same', activation='relu')(x)  # 672 → 1344
#     x = layers.Conv1DTranspose(128, kernel_size=3, strides=2, padding='same', activation='relu')(x)  # 1344 → 2688
#     # Crop to exactly 900
#     # x = layers.Lambda(lambda t: t[:, :mel_len, :])(x)
#     x = CropLayer(mel_len)(x)
#     mel_output = layers.Conv1D(mel_dim, kernel_size=1, activation='linear', name="mel_output")(x)
#     model = tf.keras.Model(inputs=inputs, outputs=mel_output)
#     return model

