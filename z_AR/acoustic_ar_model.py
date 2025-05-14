import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.metrics import CosineSimilarity
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import ast
from text_preprocess import G2PConverter
from keras.saving import register_keras_serializable
import json
import numpy as np

from tensorflow.keras.callbacks import TensorBoard
import datetime


# ==================== Create Dataset =====================
def create_dataset_fast(texts, mel_paths, input_length=200, mel_dim=80, mel_max_len=1024, batch_size=32):
    def load_and_preprocess_py(text, mel_path):
        try:
            text = text.numpy().decode("utf-8")
            mel_path = mel_path.numpy().decode("utf-8")
            phoneme_seq = ast.literal_eval(text)
            # print(text)
        except Exception as e:
            print("\n\n🔥🔥🔥 Error parsing text 🔥🔥🔥")
            print("Text that caused error:", text)
            raise e
        padded_text = pad_sequences([phoneme_seq], maxlen=input_length, padding='post')[0].astype(np.int32)
        mel = np.load(mel_path).astype(np.float32)
        T, D = mel.shape
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
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.map(tf_wrapper, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    # dataset = dataset.cache()
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# ==================== Define Model Parts =====================
class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        # print("encoder init....")
        self.embedding = layers.Embedding(vocab_size, embed_dim)
        self.conv1 = layers.Conv1D(256, 5, padding='same', activation='relu')
        self.bilstm = layers.Bidirectional(layers.LSTM(256, return_sequences=True))
    
    def build(self, input_shape):
        super().build(input_shape)
        # print("encoder buil....")
        self.embedding.build(input_shape)
        self.conv1.build((input_shape[0], input_shape[1], self.embedding.output_dim))
        self.bilstm.build((input_shape[0], input_shape[1], 256))
    
    def call(self, x):
        # print("encoder call....")
        x = self.embedding(x)
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)
        x = self.bilstm(x)
        # print(x.shape)
        return x
    

class Attention(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        # print("attention init....")
        self.W1 = layers.Dense(units)
        self.W2 = layers.Dense(units)
        self.V = layers.Dense(1)
    
    def build(self, input_shape):
        # input_shape: [query_shape, values_shape]
        query_shape, values_shape = input_shape
        self.W1.build(query_shape)
        self.W2.build(values_shape)
        self.V.build((values_shape[0], values_shape[1], self.W1.units))
        super().build(input_shape)

    def call(self, query, values):
        # print("attention call....")
        query = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W1(query) + self.W2(values)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights


class Decoder(tf.keras.Model):
    def __init__(self, mel_dim):
        super().__init__()
        # print("deoder init....")
        self.lstm = layers.LSTM(512, return_state=True)
        self.fc = layers.Dense(mel_dim)

    def build(self, input_shape):
        # input_shape: [decoder_input, context_vector]
        decoder_input_shape, context_vector_shape = input_shape
        concat_shape = (decoder_input_shape[0], decoder_input_shape[1], decoder_input_shape[2] + context_vector_shape[1])
        self.lstm.build(concat_shape)
        self.fc.build((concat_shape[0], concat_shape[1], 512))
        super().build(input_shape)
    
    def call(self, x, context_vector, hidden):
        # print("decoder call....")
        x = tf.concat([x, context_vector[:, None, :]], axis=-1)
        output, h, c = self.lstm(x, initial_state=hidden)
        frame = self.fc(output)
        return frame, (h, c)

class AcousticARModel(tf.keras.Model):
    def __init__(self, encoder, decoder, attention):
        super().__init__()
        # print("model init")
        self.encoder = encoder
        self.decoder = decoder
        self.attention = attention

    def build(self, input_shape):
        # print("model build...")
        phoneme_input_shape, mel_input_shape = input_shape
        # print(input_shape)
        self.encoder.build(phoneme_input_shape)
        self.decoder.build([
            (None, 1, mel_input_shape[-1]),  # decoder input
            (None, 512)                     # context vector
        ])
        self.attention.build([
            (None, 512),                         # query
            (None, phoneme_input_shape[1], 512)  # values
        ])
        super().build(input_shape)


    def call(self, inputs, training=False):
        # print("model call")
        phoneme_inputs, mel_inputs = inputs
        batch_size = tf.shape(phoneme_inputs)[0]
        mel_dim = tf.shape(mel_inputs)[-1]
        encoder_outputs = self.encoder(phoneme_inputs)
        # print("******* encoder_outputs: ",encoder_outputs)
        hidden_state = [tf.zeros((batch_size, 512)), tf.zeros((batch_size, 512))]
        decoder_input = tf.expand_dims(tf.zeros((batch_size, mel_dim)), 1)
        # print("===***====",)
        return encoder_outputs, decoder_input, hidden_state    

    @tf.function
    def train_step(self, data):
        # print("model train .....")
        (phoneme_inputs, mel_inputs) = data
        # print(phoneme_inputs)
        with tf.GradientTape() as tape:
            batch_size = tf.shape(phoneme_inputs)[0]
            mel_dim = tf.shape(mel_inputs)[-1]
            seq_len = tf.shape(mel_inputs)[1]

            encoder_outputs, decoder_input, hidden_state = self((phoneme_inputs, mel_inputs), training=True)

            t = tf.constant(0)
            outputs = tf.TensorArray(dtype=tf.float32, size=seq_len - 1)

            def condition(t, outputs, decoder_input, hidden_state):
                return tf.less(t, seq_len - 1)

            def body(t, outputs, decoder_input, hidden_state):
                context_vector, _ = self.attention(hidden_state[0], encoder_outputs)
                pred_frame, hidden_state = self.decoder(decoder_input, context_vector, hidden_state)
                outputs = outputs.write(t, pred_frame)  # ✅ No squeeze needed!
                next_input = tf.expand_dims(mel_inputs[:, t, :], 1)  # Teacher forcing
                return t + 1, outputs, next_input, hidden_state

            t, outputs, _, _ = tf.while_loop(
                condition,
                body,
                loop_vars=[t, outputs, decoder_input, hidden_state]
            )

            outputs = outputs.stack()  # (seq_len-1, batch, mel_dim)
            outputs = tf.transpose(outputs, [1, 0, 2])  # (batch, seq_len-1, mel_dim)

            loss = self.compiled_loss(mel_inputs[:, 1:, :], outputs)

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.compiled_metrics.update_state(mel_inputs[:, 1:, :], outputs)
        return {m.name: m.result() for m in self.metrics}


# ==================== Load Data =====================
df = pd.read_csv('dataset/acoustic_dataset/tts_data_LJ.csv', usecols=['Phoneme_text', 'Read_npy'])
texts = df['Phoneme_text'].apply(ast.literal_eval).values
mel_spectrograms = df['Read_npy'].values

input_length = max([len(seq) for seq in texts]) + 32
texts_str = [str(seq) for seq in texts]

texts_train, texts_temp, mel_train, mel_temp = train_test_split(texts_str, mel_spectrograms, test_size=0.2, random_state=33)
texts_val, texts_test, mel_val, mel_test = train_test_split(texts_temp, mel_temp, test_size=0.3, random_state=33)

train_dataset = create_dataset_fast(texts_train, mel_train, input_length=input_length)
val_dataset = create_dataset_fast(texts_val, mel_val, input_length=input_length)
test_dataset = create_dataset_fast(texts_test, mel_test, input_length=input_length)

# ==================== Build & Compile =====================
vocab_size = len(G2PConverter(load_model=False).phn2idx)
encoder = Encoder(vocab_size=vocab_size, embed_dim=256)
decoder = Decoder(mel_dim=80)
attention = Attention(units=256)

model = AcousticARModel(encoder, decoder, attention)

# Build model by running a dummy batch
print("======== 1 ===========")

sample_batch = next(iter(train_dataset))
sample_phonemes, sample_mels = sample_batch
# print(sample_phonemes)

# model(sample_batch)
model.build([(None, input_length), (None, 1024, 80)])

print("======== 2 ===========")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=[CosineSimilarity()]
)
print("======== 3 ===========")
model.summary()

# Define the TensorBoard callback.
log_dir = "model/2/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1) #histogram_freq logs histograms of model weights.

# # ==================== Train =====================
callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ModelCheckpoint("acoustic_ar_model_best.h5", save_best_only=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),tensorboard_callback
]
print("======== 4 ===========")

history = model.fit(
    train_dataset,
    epochs=50,
    callbacks=callbacks
)


# Save model & history
model.save('model/2/acoustic_AR_model_cnn_9f.keras')
model.save_weights('model/2/acoustic_AR_model_cnn_9f.weights.h5')
history_df = pd.DataFrame(history.history)
history_df.to_csv('model/2/acoustic_AR_model_cnn_9f.csv', index=False)

# ==================== Evaluation =====================
test_loss = model.evaluate(test_dataset)
print(f"Test loss: {test_loss}")





# def load_and_preprocess_py(text, mel_path):
#     # Decode inputs
#     if isinstance(text, bytes):
#         text = text.decode('utf-8')
#     if isinstance(mel_path, bytes):
#         mel_path = mel_path.decode('utf-8')

#     # Parse the phoneme list using json (much safer)
#     phoneme_seq = json.loads(text)

#     # Load mel spectrogram
#     mel_spectrogram = np.load(mel_path)

#     T, D = mel.shape
#     if T > mel_max_len:
#         mel = mel[:mel_max_len, :]
#     elif T < mel_max_len:
#         pad_len = mel_max_len - T
#         mel = np.pad(mel, ((0, pad_len), (0, 0)), mode='constant')

#     return phoneme_seq, mel_spectrogram


# import tensorflow as tf

# optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
# loss_fn = tf.keras.losses.MeanSquaredError()

# @tf.function
# def train_step(phoneme_batch, mel_batch):
#     batch_size = tf.shape(phoneme_batch)[0]
#     mel_dim = tf.shape(mel_batch)[-1]
    
#     # Teacher-forcing: inputs
#     decoder_inputs = mel_batch[:, :-1, :]    # everything except last frame
#     decoder_targets = mel_batch[:, 1:, :]     # everything except first frame

#     with tf.GradientTape() as tape:
#         # Encode
#         encoder_outputs = encoder(phoneme_batch)

#         # Initialize decoder
#         hidden_state = [tf.zeros((batch_size, 512)), tf.zeros((batch_size, 512))]
        
#         outputs = []
#         attention_layer = Attention(units=256)
        
#         # First decoder input: usually zero frame
#         decoder_input = tf.zeros((batch_size, 1, mel_dim))
        
#         for t in range(tf.shape(decoder_inputs)[1]):
#             context_vector, _ = attention_layer(hidden_state[0], encoder_outputs)
            
#             # Expand dims because decoder expects (batch, 1, features)
#             dec_input = tf.concat([decoder_input, tf.expand_dims(context_vector, 1)], axis=-1)
            
#             pred_frame, hidden_state = decoder(dec_input, context_vector, hidden_state)
#             outputs.append(pred_frame)
            
#             # Teacher forcing: use ground-truth mel frame for next step
#             decoder_input = tf.expand_dims(decoder_inputs[:, t, :], 1)
        
#         # Stack all outputs
#         outputs = tf.concat(outputs, axis=1)  # (batch, frames, mel_dim)

#         # Calculate loss
#         loss = loss_fn(decoder_targets, outputs)
    
#     # Backpropagation
#     gradients = tape.gradient(loss, encoder.trainable_variables + decoder.trainable_variables)
#     optimizer.apply_gradients(zip(gradients, encoder.trainable_variables + decoder.trainable_variables))

#     return loss


# EPOCHS = 2

# for epoch in range(EPOCHS):
#     total_loss = 0
#     for phoneme_batch, mel_batch in train_dataset:  # tf.data.Dataset
#         print(phoneme_batch,mel_batch)
#         batch_loss = train_step(phoneme_batch, mel_batch)
#         total_loss += batch_loss
    
#     print(f'Epoch {epoch+1} Loss {total_loss:.4f}')





'''
Model: "acoustic_ar_model"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ encoder (Encoder)                    │ ?                           │       1,389,312 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ decoder (Decoder)                    │ ?                           │     0 (unbuilt) │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ attention (Attention)                │ ?                           │     0 (unbuilt) │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘


 Total params: 1,389,312 (5.30 MB)
 Trainable params: 1,389,312 (5.30 MB)
 Non-trainable params: 0 (0.00 B)
Epoch 1/50
/Users/edelta/Desktop/shruti/TTS/env/lib/python3.11/site-packages/keras/src/backend/tensorflow/trainer.py:665: UserWarning: `model.compiled_loss()` is deprecated. Instead, use `model.compute_loss(x, y, y_pred, sample_weight, training)`.
  warnings.warn(
/Users/edelta/Desktop/shruti/TTS/env/lib/python3.11/site-packages/keras/src/backend/tensorflow/trainer.py:640: UserWarning: `model.compiled_metrics()` is deprecated. Instead, use e.g.:
```
for metric in self.metrics:
    metric.update_state(y, y_pred)
```

  return self._compiled_metrics_update_state(
 14/328 ━━━━━━━━━━━━━━━━━━━━ 9:57:10 114s/step - cosine_similarity: 0.3239 - loss: 0.0045 
 50/328 ━━━━━━━━━━━━━━━━━━━━ 8:04:49 105s/step - cosine_similarity: 0.4299 - loss: 0.0020
 61/328 ━━━━━━━━━━━━━━━━━━━━ 7:40:14 103s/step - cosine_similarity: 0.4422 - loss: 0.0017
 80/328 ━━━━━━━━━━━━━━━━━━━━ 7:00:17 102s/step - cosine_similarity: 0.4578 - loss: 0.0014
 88/328 ━━━━━━━━━━━━━━━━━━━━ 6:44:52 101s/step - cosine_similarity: 0.4630 - loss: 0.0013
 99/328 ━━━━━━━━━━━━━━━━━━━━ 6:25:16 101s/step - cosine_similarity: 0.4692 - loss: 0.0012
 
 '''