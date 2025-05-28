# from TTS.Fastspeech_1 import FastSpeechAcousticModel
# from TTS.losses import total_acoustic_loss

# from TTS.fastspeech_dataset import FastSpeechDataset

# dataset = FastSpeechDataset(
#     csv_path="train.csv",
#     mel_base_path="/mnt/data/",  # path where .npy files are located
#     batch_size=8
# )

# train_ds = dataset.create()

# val_dataset = FastSpeechDataset(
#     csv_path="val.csv",
#     mel_base_path="/mnt/data/",
#     batch_size=8,
#     shuffle=False  # No shuffling for validation
# ).create()
# # model, optimizer setup
# model = FastSpeechAcousticModel(vocab_size)
# optimizer = tf.keras.optimizers.Adam()

# # in training loop
# with tf.GradientTape() as tape:
#     mel_pred = model(inputs, durations, training=True)
#     loss = total_acoustic_loss(mel_target, mel_pred, duration_true, duration_pred)

# grads = tape.gradient(loss, model.trainable_variables)
# optimizer.apply_gradients(zip(grads, model.trainable_variables))

from fastspeech_dataset import FastSpeechDataset
from Fastspeech_1 import FastSpeechAcousticModel
from losses import TotalAcousticLoss
import tensorflow as tf
from tensorflow.keras.metrics import CosineSimilarity

# Load datasets
train_ds = FastSpeechDataset("dataset/acoustic_dataset/train.csv", mel_base_path="dataset/LJSpeech/wavs").create()
val_ds = FastSpeechDataset("dataset/acoustic_dataset/val.csv", mel_base_path="dataset/LJSpeech/wavs", shuffle=False).create()

# Create and compile model
model = FastSpeechAcousticModel(vocab_size=72)  # Set correct vocab size


lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=0.001,
            decay_steps=50 * len(train_ds),
            alpha=0.1
        )
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)    
model.compile(optimizer=optimizer,
                  loss=TotalAcousticLoss(),
                  metrics=['mae', CosineSimilarity(axis=-1)])
# model = compile_model(model)
model.build(input_shape=[(None, 132), (None, 132)])
model.summary()
# # Example dummy input
# dummy_phonemes = tf.constant([[1, 2, 3, 4]], dtype=tf.int32)  # (batch, time)
# dummy_durations = tf.constant([[1.0, 2.0, 1.0, 3.0]], dtype=tf.float32)

# # Run one forward pass to build the model
# _ = model((dummy_phonemes, dummy_durations))
# model.summary()

# # model.summary()

# Train
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint("fastspeech_model.h5", save_best_only=True),
        tf.keras.callbacks.TensorBoard(log_dir="./logs"),
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    ]
)

