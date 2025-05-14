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

def load_tokenizer(filename='tokenizer_LJ.pickle'):
    with open(filename, 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer

def create_dataset_fast(texts, mel_paths, tokenizer, batch_size=32, input_length=512):
    def load_and_preprocess_py(text, mel_path):
        text = text.numpy().decode("utf-8")
        mel_path = mel_path.numpy().decode("utf-8")
        
        sequence = tokenizer.texts_to_sequences([text])
        padded_text = pad_sequences(sequence, maxlen=input_length, padding='post')[0]
        mel = np.load(mel_path).astype(np.float32)
        return padded_text.astype(np.int32), mel

    def tf_wrapper(text, mel_path):
        result_text, result_mel = tf.py_function(
            func=load_and_preprocess_py,
            inp=[text, mel_path],
            Tout=[tf.int32, tf.float32]
        )
        result_text.set_shape([input_length])
        result_mel.set_shape([None, 128])  # adjust mel_dim if needed
        return result_text, result_mel

    dataset = tf.data.Dataset.from_tensor_slices((texts, mel_paths))
    dataset = dataset.map(tf_wrapper, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# Combined loss: MSE + weighted cosine similarity
def combined_loss(y_true, y_pred):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    cos_sim = tf.reduce_mean(tf.keras.losses.cosine_similarity(y_true, y_pred, axis=-1))  # returns negative
    return mse + 0.1 * cos_sim

def improved_tts_model(vocab_size, input_length, mel_dim=128, rnn_units=512, embed_dim=256):    
    encoder_inputs = layers.Input(shape=(input_length,))
    encoder_embedding = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, mask_zero=True)(encoder_inputs)
    # encoder_conv = layers.Conv1D(filters=64, kernel_size=5, activation='relu', padding='same')(encoder_embedding)
    # encoder_conv = layers.Conv1D(filters=64, kernel_size=5, activation='relu', padding='same')(encoder_conv)
    
    encoder_lstm1 = layers.Bidirectional(layers.LSTM(rnn_units, return_sequences=True, 
                                                      kernel_regularizer=regularizers.l2(0.001)))(encoder_embedding)
    encoder_lstm2 = layers.Bidirectional(layers.LSTM(rnn_units, return_sequences=True, 
                                                      kernel_regularizer=regularizers.l2(0.001)))(encoder_lstm1)

    attention = layers.MultiHeadAttention(num_heads=8, key_dim=rnn_units)(encoder_lstm2, encoder_lstm2)
    attention = layers.Add()([encoder_lstm2, attention])

    decoder_lstm = layers.LSTM(rnn_units, return_sequences=True, kernel_regularizer=regularizers.l2(0.001))(attention)
    decoder_lstm = layers.LSTM(rnn_units, return_sequences=True, kernel_regularizer=regularizers.l2(0.001))(decoder_lstm)
    decoder_lstm = layers.BatchNormalization()(decoder_lstm)

    decoder_conv1 = layers.Conv1D(filters=80, kernel_size=3, activation='relu', padding='same')(decoder_lstm)
    decoder_conv2 = layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(decoder_conv1)
    decoder_conv3 = layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(decoder_conv2)
    decoder_conv4 = layers.Conv1D(filters=256, kernel_size=3, activation='relu', padding='same')(decoder_conv3)
    decoder_conv5 = layers.Conv1D(filters=256, kernel_size=3, activation='relu', padding='same')(decoder_conv4)

    dense1 = layers.Dense(1024, activation='relu')(decoder_conv5)
    dense1 = layers.Dropout(0.3)(dense1)
    dense2 = layers.Dense(512, activation='relu')(dense1)
    dense2 = layers.Dropout(0.3)(dense2)

    norm = layers.LayerNormalization()(dense2)
    mel_output = layers.Dense(mel_dim, activation='linear')(norm)

    model = tf.keras.Model(inputs=encoder_inputs, outputs=mel_output)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss=combined_loss, metrics=['mae', CosineSimilarity(axis=-1)])
    return model

class LearningRatePlotter(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.learning_rate
        self.lrs.append(lr)

    def on_train_begin(self, logs=None):
        self.lrs = []

# def scheduler(epoch, lr):
#     if epoch > 13:
#         return lr * 0.5
#     return lr

# Load and prepare data
df = pd.read_csv('tts_data_LJ.csv')
texts = df['Phoneme_text'].values
mel_spectrograms = df['Read_npy'].values

tokenizer = load_tokenizer()
# padded_texts, mel_spectrograms = preprocess_data(texts, mel_spectrograms, tokenizer)

texts_train, texts_temp, mel_train, mel_temp = train_test_split(texts, mel_spectrograms, test_size=0.2, random_state=42)
texts_val, texts_test, mel_val, mel_test = train_test_split(texts_temp, mel_temp, test_size=0.3, random_state=42)

train_dataset = create_dataset_fast(texts_train, mel_train, tokenizer)
val_dataset = create_dataset_fast(texts_val, mel_val, tokenizer)
test_dataset=create_dataset_fast(texts_test,mel_test,tokenizer)

vocab_size = len(tokenizer.word_index) + 1
model = improved_tts_model(vocab_size, input_length=512)
model.summary()

# for batch in train_dataset.take(1):
#     inputs, targets = batch
#     print("Inputs shape:", inputs.shape)
#     print("Targets shape:", targets.shape)
#     print("Inputs:", inputs.numpy()[0])   # Show first example
#     print("Targets:", targets.numpy()[0]) # Show first mel-spectrogram
# print(test_dataset)

# Training Callbacks
e_s = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1)

checkpoint = ModelCheckpoint(
    filepath='checkpoints/model_epoch_{epoch:02d}_valLoss_{val_loss:.4f}.keras',
    monitor='val_loss',
    verbose=1,
    save_best_only=False,
    save_weights_only=False,
    mode='min',
)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1)
lr_plotter = LearningRatePlotter()
# lr_scheduler = LearningRateScheduler(scheduler)

# history = model.fit(np.array(texts_train), np.array(mel_train),batch_size=16, epochs=25,validation_data=(np.array(texts_val), np.array(mel_val)),callbacks=[e_s, checkpoint, reduce_lr,lr_plotter])
history = model.fit(
    train_dataset,
    epochs=25,
    validation_data=val_dataset,
    callbacks=[e_s, checkpoint, reduce_lr, lr_plotter])

model.save('tts_model_lj_LSTM_cosine.keras')
model.save_weights('weights_LSTM_cosine.weights.h5')

history_df = pd.DataFrame(history.history)
history_df['epoch'] = range(1, len(history_df) + 1)
history_df.to_csv('training_metrics.csv', index=False)

test_loss = model.evaluate(test_dataset)
print(f"Test loss: {test_loss}")

# Plot learning rate
plt.plot(range(1, len(lr_plotter.lrs) + 1), lr_plotter.lrs)
plt.xlabel('Epochs')
plt.ylabel('Learning Rate')
plt.title('Learning Rate vs Epoch')
plt.grid(True)
plt.show()

for batch in test_dataset.take(1):  # take 1 batch
    x_test, y_true = batch
    y_pred = model.predict(x_test)
    
    print("Input shape:", x_test.shape)
    print("True mel shape:", y_true.shape)
    print("Predicted mel shape:", y_pred.shape)
    
    # Visualize 1 example
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.title("Ground Truth")
    plt.imshow(y_true[0].numpy().T, aspect='auto', origin='lower')
    
    plt.subplot(1, 2, 2)
    plt.title("Prediction")
    plt.imshow(y_pred[0].T, aspect='auto', origin='lower')
    plt.tight_layout()
    plt.show()



