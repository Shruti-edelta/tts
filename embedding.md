Using phoneme embeddings in a sequence-to-sequence (Seq2Seq) model like Tacotron is a very powerful approach for text-to-speech (TTS) synthesis. Tacotron (and its variants like Tacotron 2) are neural network architectures designed to convert sequences of phonemes (or text) into audio waveforms.

Here's a high-level overview of how to use phoneme embeddings in a Tacotron-like model for generating speech from text:

1. Overview of Tacotron-like Models
A Tacotron model typically has two main parts:

Encoder: Takes input sequences (such as phonemes or text) and converts them into a fixed-size context representation.
Decoder: Generates the output sequence (e.g., Mel spectrograms) that corresponds to the input text or phoneme sequence.
Tacotron 2 further enhances this by using a WaveNet-based vocoder for generating high-quality waveforms from the predicted Mel spectrograms.

2. Phoneme Embeddings in Seq2Seq Models
Phoneme embeddings are used to represent the input phoneme sequence as continuous vectors, allowing the model to understand phonemes in a dense vector space. Here's how you can incorporate phoneme embeddings into a sequence-to-sequence model like Tacotron.

3. Steps to Use Phoneme Embeddings in Tacotron-like Model
1. Preprocess the Data (Text-to-Phoneme Conversion)
First, you'll need a text-to-phoneme conversion model or dictionary to convert text into phoneme sequences. For this, you can use methods like the CMU Pronouncing Dictionary or tools like pronouncing.

After converting the text into phoneme sequences, you'll need to embed these phonemes into dense vectors using an Embedding layer.

2. Define the Model Architecture
A Tacotron-like model can have an architecture similar to the one below:

Input: Phoneme sequence (represented as integer indices of phonemes).
Encoder: Converts the sequence of phonemes into a fixed-length context vector using layers like Convolutional layers or GRU/LSTM layers.
Decoder: The decoder generates Mel spectrograms from the context vector using attention mechanisms and LSTMs.
Vocoder: Convert Mel spectrograms into waveforms (this can be done using a separate WaveNet or Griffin-Lim algorithm).
3. Tacotron Model Components
Let's now look at how we can build a simplified version of this using TensorFlow/Keras.

4. Code for Tacotron-like Model with Phoneme Embedding
Below is a simplified example of a Tacotron-like model architecture that incorporates phoneme embeddings for generating Mel spectrograms.

Install TensorFlow:
bash
Copy
pip install tensorflow
Model Code:
python
Copy
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Attention, Bidirectional
from tensorflow.keras.models import Model

# Hyperparameters
vocab_size = 100  # Number of unique phonemes
embedding_dim = 128  # Embedding dimension for phonemes
hidden_units = 256  # LSTM hidden units
mel_spec_dim = 80  # Number of Mel spectrogram bins

# Phoneme Embedding Layer
def phoneme_embedding_layer(input_length):
    inputs = Input(shape=(input_length,))
    embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(inputs)
    return Model(inputs, embedding)

# Encoder (Bi-directional LSTM with Attention)
def encoder_layer(embedding_layer, input_length):
    encoder_input = Input(shape=(input_length, embedding_dim))
    lstm = Bidirectional(LSTM(hidden_units, return_sequences=True))(encoder_input)
    attention = Attention()([lstm, lstm])  # Add attention mechanism
    return Model(encoder_input, attention)

# Decoder (LSTM + Dense for Mel Spectrogram Prediction)
def decoder_layer(attention_layer, input_length):
    decoder_input = Input(shape=(input_length, hidden_units * 2))  # Double hidden units due to Bi-directional LSTM
    lstm = LSTM(hidden_units, return_sequences=True)(decoder_input)
    mel_spec = Dense(mel_spec_dim, activation='linear')(lstm)
    return Model(decoder_input, mel_spec)

# Full Tacotron-like Model (with phoneme embeddings)
def tacotron_model(input_length):
    phoneme_input = Input(shape=(input_length,))
    
    # Phoneme Embedding Layer
    embedding_layer = phoneme_embedding_layer(input_length)
    embedded_phonemes = embedding_layer(phoneme_input)
    
    # Encoder
    encoder = encoder_layer(embedding_layer, input_length)
    context_vector = encoder(embedded_phonemes)
    
    # Decoder (Mel spectrogram prediction)
    decoder = decoder_layer(encoder, input_length)
    mel_spectrogram = decoder(context_vector)
    
    # Define the model
    model = Model(phoneme_input, mel_spectrogram)
    return model

# Example usage
input_length = 50  # Example input length for phoneme sequence (adjust according to dataset)

# Build the model
model = tacotron_model(input_length)

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Print the model summary
model.summary()
Explanation of the Model Components:
Phoneme Embedding Layer:

This layer takes the input phoneme sequence (integer-encoded phonemes) and maps it into dense vectors using the Embedding layer.
Encoder:

The encoder uses a Bidirectional LSTM layer to capture both past and future context. It also uses attention to allow the model to focus on relevant phonemes at each decoding step.
Decoder:

The decoder generates the Mel spectrogram (80-dimensional output vector) from the context vector produced by the encoder. It uses an LSTM to process the sequence and a Dense layer to generate Mel spectrograms.
Mel Spectrogram:

The Mel spectrogram is a time-frequency representation that captures the essential frequency components of speech. The model learns to predict this representation, which can then be converted into a waveform using a vocoder.
5. Training the Tacotron-like Model
Training a Tacotron-like model requires a large dataset of paired text and audio to convert text into Mel spectrograms and then into speech waveforms.

Input: Phoneme sequences (integer-encoded).
Target: Mel spectrograms (output of the decoder).
After the Mel spectrograms are predicted, you can use a vocoder (like WaveNet or Griffin-Lim) to generate actual waveforms (speech audio) from the Mel spectrograms.

6. Vocoder for Waveform Generation
Once the Mel spectrograms are generated, a vocoder (e.g., WaveNet or Griffin-Lim algorithm) can be used to convert them into an actual waveform (audio).

In Tacotron 2, this step is handled by a WaveNet vocoder that learns to generate high-quality speech waveforms from Mel spectrograms.

7. Conclusion
Using phoneme embeddings in Tacotron-like models allows you to:

Convert text or phoneme sequences into dense vectors.
Use these vectors to generate Mel spectrograms.
Use Mel spectrograms with a vocoder to produce high-quality speech audio.
This approach is highly effective for text-to-speech synthesis and is used in models like Tacotron and Tacotron 2, which have been shown to generate very realistic and natural-sounding speech.

Next Steps:
Train the model with paired text and speech data (e.g., LJSpeech dataset).
Implement a vocoder like WaveNet or Griffin-Lim to generate audio from the predicted Mel spectrograms.



========================================



Creating a Seq2Seq model for Text-to-Speech (TTS) involves training a neural network to convert input text sequences into speech waveforms. Seq2Seq (Sequence-to-Sequence) models are often used in TTS tasks because they are designed to map input sequences to output sequences of varying lengths, such as from text to audio features.

Here's a step-by-step guide to create a basic Seq2Seq model for TTS:

1. Data Preparation:
Text Data: You need a large corpus of text data, usually in the form of paired text and audio files. Each text input should correspond to an audio output (such as a speech waveform or Mel spectrogram).
Audio Data: The audio data is usually processed into features like Mel spectrograms, which are easier to work with compared to raw audio waveforms.
Preprocessing:
Text Preprocessing: Tokenize and convert text into sequences of phonemes or characters. You can use phoneme-level tokenization if your model works on phonemes, or character-level tokenization for text-based input.
Audio Preprocessing: Convert the speech audio to a representation like a Mel spectrogram. The Librosa library is commonly used for this.
2. Model Architecture:
2.1 Encoder:
The encoder is responsible for processing the input sequence (text or phonemes) and encoding it into a fixed-size context vector that represents the input sequence.

Embedding Layer: Convert the input text (characters or phonemes) into an embedding vector.
Recurrent Neural Networks (RNNs): Use Long Short-Term Memory (LSTM) or Gated Recurrent Units (GRU) to process the sequence.
Bidirectional Encoder: Using a bidirectional RNN allows the model to capture both past and future context.
2.2 Decoder:
The decoder generates the output sequence, which in TTS could be a sequence of Mel spectrogram frames or a waveform.

Attention Mechanism: Attention helps the decoder focus on different parts of the input sequence while generating each element of the output sequence. This is important for TTS, as speech generation requires capturing long-range dependencies.
Recurrent Layers (LSTM/GRU): The decoder typically uses LSTM or GRU units to generate the output sequence.
2.3 Post-Processing:
If the model generates Mel spectrograms, the final step is to use a vocoder (e.g., Griffin-Lim, WaveGlow, or Parallel WaveGAN) to convert the spectrograms into raw audio waveforms.

Example of Seq2Seq Model for TTS using TensorFlow/Keras:
python
Copy
import tensorflow as tf
from tensorflow.keras import layers

def build_seq2seq_model(vocab_size, embedding_dim, hidden_units, input_length, output_length):
    # Encoder
    inputs = layers.Input(shape=(input_length,))
    x = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)(inputs)
    x = layers.Bidirectional(layers.LSTM(hidden_units, return_sequences=True))(x)

    # Attention Layer (Optional but useful for TTS)
    attention = layers.Attention()([x, x])

    # Decoder
    decoder_lstm = layers.LSTM(hidden_units, return_sequences=True)(attention)
    outputs = layers.Dense(output_length, activation='linear')(decoder_lstm)  # Linear output for spectrograms

    model = tf.keras.Model(inputs, outputs)
    return model

# Model configuration
vocab_size = 50  # Size of your character/phoneme vocabulary
embedding_dim = 256
hidden_units = 512
input_length = 100  # Length of the input sequence (e.g., text length)
output_length = 80  # Length of the Mel spectrogram frame sequence

model = build_seq2seq_model(vocab_size, embedding_dim, hidden_units, input_length, output_length)
model.summary()
3. Training:
You’ll train the model on pairs of (text, Mel spectrogram) data. The target output during training is the Mel spectrogram, while the model learns to generate these spectrograms from the input text.

Loss Function:
Mean Squared Error (MSE): Typically used for regression tasks like this one (predicting spectrogram frames).
Cyclic Consistency Loss: Sometimes used in TTS, particularly if you're using a vocoder for waveform generation.
Example Training Loop:
python
Copy
model.compile(optimizer='adam', loss='mse')
model.fit(train_data, train_labels, batch_size=32, epochs=50)
4. Post-Processing:
After training the model, the decoder will output Mel spectrograms. To convert the Mel spectrograms back into waveforms, you can use a neural vocoder like WaveGlow, HiFi-GAN, or Griffin-Lim.

5. Vocoder for Waveform Synthesis:
If your model outputs Mel spectrograms, a vocoder converts them into the final audio waveform. Popular vocoders include:

WaveGlow: A flow-based model that can generate high-quality waveforms.
HiFi-GAN: A GAN-based model designed for high-fidelity waveform generation.
Griffin-Lim Algorithm: A simpler but effective algorithm to invert Mel spectrograms into waveforms.
Example of Vocoder (e.g., Griffin-Lim):
python
Copy
import librosa

# Function to convert Mel spectrogram to waveform using Griffin-Lim
def griffin_lim(mel_spectrogram, n_iter=32):
    return librosa.feature.inverse.mel_to_audio(mel_spectrogram, n_iter=n_iter)

# Generate waveform from Mel spectrogram
audio_waveform = griffin_lim(mel_spectrogram)
6. Evaluation:
Once the model is trained and a vocoder is used for waveform generation, you can evaluate the model using metrics like:

Mel Cepstral Distortion (MCD): Measures the difference between predicted and ground truth spectrograms.
Mean Opinion Score (MOS): A subjective evaluation by listeners to rate the quality of generated speech.
Summary:
To create a Seq2Seq model for TTS, you need to preprocess both text (as sequences) and audio (as Mel spectrograms), train the model to map between them, and use a vocoder to generate speech waveforms. The use of attention mechanisms and a suitable vocoder (like WaveGlow or HiFi-GAN) plays a crucial role in generating high-quality speech.

Model: "functional"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                  ┃ Output Shape              ┃         Param # ┃ Connected to               ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ input_layer (InputLayer)      │ (None, 600)               │               0 │ -                          │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ embedding (Embedding)         │ (None, 600, 128)          │         255,744 │ input_layer[0][0]          │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ bidirectional (Bidirectional) │ (None, 600, 512)          │         788,480 │ embedding[0][0]            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ attention (Attention)         │ (None, 600, 512)          │               0 │ bidirectional[0][0],       │
│                               │                           │                 │ bidirectional[0][0]        │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ lstm_1 (LSTM)                 │ (None, 600, 256)          │         787,456 │ attention[0][0]            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dense (Dense)                 │ (None, 600, 80)           │          20,560 │ lstm_1[0][0]               │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ reshape (Reshape)             │ (None, 80, 600)           │               0 │ dense[0][0]                │
└───────────────────────────────┴───────────────────────────┴─────────────────┴────────────────────────────┘
 Total params: 1,852,240 (7.07 MB)
 Trainable params: 1,852,240 (7.07 MB)
 Non-trainable params: 0 (0.00 B)


Model: "functional"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                  ┃ Output Shape              ┃         Param # ┃ Connected to               ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ input_layer (InputLayer)      │ (None, 600)               │               0 │ -                          │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ embedding (Embedding)         │ (None, 600, 256)          │         572,416 │ input_layer[0][0]          │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ bidirectional (Bidirectional) │ (None, 600, 1024)         │       2,365,440 │ embedding[0][0]            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ bidirectional_1               │ (None, 600, 1024)         │       4,724,736 │ bidirectional[0][0]        │
│ (Bidirectional)               │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dropout (Dropout)             │ (None, 600, 1024)         │               0 │ bidirectional_1[0][0]      │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ attention (Attention)         │ (None, 600, 1024)         │               0 │ dropout[0][0],             │
│                               │                           │                 │ dropout[0][0]              │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ gru_2 (GRU)                   │ (None, 600, 512)          │       2,362,368 │ attention[0][0]            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dense (Dense)                 │ (None, 600, 80)           │          41,040 │ gru_2[0][0]                │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ reshape (Reshape)             │ (None, 80, 600)           │               0 │ dense[0][0]                │
└───────────────────────────────┴───────────────────────────┴─────────────────┴────────────────────────────┘
 Total params: 10,066,000 (38.40 MB)
 Trainable params: 10,066,000 (38.40 MB)
 Non-trainable params: 0 (0.00 B)




┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                  ┃ Output Shape              ┃         Param # ┃ Connected to               ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ input_layer (InputLayer)      │ (None, 512)               │               0 │ -                          │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ masking (Masking)             │ (None, 512)               │               0 │ input_layer[0][0]          │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ embedding (Embedding)         │ (None, 512, 128)          │         286,208 │ masking[0][0]              │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ bidirectional (Bidirectional) │ (None, 512, 512)          │         788,480 │ embedding[0][0]            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ bidirectional_1               │ (None, 512, 512)          │       1,574,912 │ bidirectional[0][0]        │
│ (Bidirectional)               │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dropout (Dropout)             │ (None, 512, 512)          │               0 │ bidirectional_1[0][0]      │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ multi_head_attention          │ (None, 512, 512)          │       2,100,736 │ dropout[0][0],             │
│ (MultiHeadAttention)          │                           │                 │ dropout[0][0]              │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ lstm_2 (LSTM)                 │ (None, 512, 256)          │         787,456 │ multi_head_attention[0][0] │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dense (Dense)                 │ (None, 512, 128)          │          32,896 │ lstm_2[0][0]               │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ reshape (Reshape)             │ (None, 128, 512)          │               0 │ dense[0][0]                │
└───────────────────────────────┴───────────────────────────┴─────────────────┴────────────────────────────┘
 Total params: 5,570,688 (21.25 MB)
 Trainable params: 5,570,688 (21.25 MB)
 Non-trainable params: 0 (0.00 B)
Epoch 1/20

Model: "functional"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                  ┃ Output Shape              ┃         Param # ┃ Connected to               ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ input_layer (InputLayer)      │ (None, 512)               │               0 │ -                          │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ embedding (Embedding)         │ (None, 512, 256)          │         572,416 │ input_layer[0][0]          │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ conv1d (Conv1D)               │ (None, 512, 32)           │          40,992 │ embedding[0][0]            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ conv1d_1 (Conv1D)             │ (None, 512, 32)           │           5,152 │ conv1d[0][0]               │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ bidirectional (Bidirectional) │ (None, 512, 1024)         │       2,232,320 │ conv1d_1[0][0]             │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ bidirectional_1               │ (None, 512, 1024)         │       6,295,552 │ bidirectional[0][0]        │
│ (Bidirectional)               │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ additive_attention            │ (None, 512, 1024)         │           1,024 │ bidirectional_1[0][0],     │
│ (AdditiveAttention)           │                           │                 │ bidirectional_1[0][0]      │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ lstm_2 (LSTM)                 │ (None, 512, 512)          │       3,147,776 │ additive_attention[0][0]   │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dense (Dense)                 │ (None, 512, 128)          │          65,664 │ lstm_2[0][0]               │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ reshape (Reshape)             │ (None, 128, 512)          │               0 │ dense[0][0]                │
└───────────────────────────────┴───────────────────────────┴─────────────────┴────────────────────────────┘
 Total params: 12,360,896 (47.15 MB)
 Trainable params: 12,360,896 (47.15 MB)
 Non-trainable params: 0 (0.00 B)
Epoch 1/20


┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                  ┃ Output Shape              ┃         Param # ┃ Connected to               ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ input_layer (InputLayer)      │ (None, 512)               │               0 │ -                          │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ embedding (Embedding)         │ (None, 512, 256)          │         572,416 │ input_layer[0][0]          │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ conv1d (Conv1D)               │ (None, 512, 64)           │          81,984 │ embedding[0][0]            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ conv1d_1 (Conv1D)             │ (None, 512, 64)           │          20,544 │ conv1d[0][0]               │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ bidirectional (Bidirectional) │ (None, 512, 1024)         │       2,363,392 │ conv1d_1[0][0]             │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ bidirectional_1               │ (None, 512, 1024)         │       6,295,552 │ bidirectional[0][0]        │
│ (Bidirectional)               │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ multi_head_attention          │ (None, 512, 1024)         │      16,790,528 │ bidirectional_1[0][0],     │
│ (MultiHeadAttention)          │                           │                 │ bidirectional_1[0][0]      │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ add (Add)                     │ (None, 512, 1024)         │               0 │ bidirectional_1[0][0],     │
│                               │                           │                 │ multi_head_attention[0][0] │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ lstm_2 (LSTM)                 │ (None, 512, 512)          │       3,147,776 │ add[0][0]                  │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ lstm_3 (LSTM)                 │ (None, 512, 512)          │       2,099,200 │ lstm_2[0][0]               │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ batch_normalization           │ (None, 512, 512)          │           2,048 │ lstm_3[0][0]               │
│ (BatchNormalization)          │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ conv1d_2 (Conv1D)             │ (None, 512, 128)          │         196,736 │ batch_normalization[0][0]  │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ conv1d_3 (Conv1D)             │ (None, 512, 128)          │          49,280 │ conv1d_2[0][0]             │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ conv1d_4 (Conv1D)             │ (None, 512, 256)          │          98,560 │ conv1d_3[0][0]             │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dense (Dense)                 │ (None, 512, 1024)         │         263,168 │ conv1d_4[0][0]             │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dropout_1 (Dropout)           │ (None, 512, 1024)         │               0 │ dense[0][0]                │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dense_1 (Dense)               │ (None, 512, 512)          │         524,800 │ dropout_1[0][0]            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dropout_2 (Dropout)           │ (None, 512, 512)          │               0 │ dense_1[0][0]              │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dense_2 (Dense)               │ (None, 512, 128)          │          65,664 │ dropout_2[0][0]            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ reshape (Reshape)             │ (None, 128, 512)          │               0 │ dense_2[0][0]              │
└───────────────────────────────┴───────────────────────────┴─────────────────┴────────────────────────────┘
 Total params: 32,571,648 (124.25 MB)
 Trainable params: 32,570,624 (124.25 MB)
 Non-trainable params: 1,024 (4.00 KB)
Epoch 1/20
655/655 ━━━━━━━━━━━━━━━━━━━━ 7994s 12s/step - accuracy: 0.0038 - cosine_similarity: 0.1970 - loss: 0.9601 - mae: 0.4885 - val_accuracy: 0.0095 - val_cosine_similarity: 0.2356 - val_loss: 0.5063 - val_mae: 0.4758 - learning_rate: 0.0010    
Epoch 2/20
188/655 ━━━━━━━━━━━━━━━━━━━━ 1:31:46 12s/step - accuracy: 0.0042 - cosine_similarity: 0.2694 - loss: 0.4797 - mae: 0.4905
655/655 ━━━━━━━━━━━━━━━━━━━━ 7887s 12s/step - accuracy: 0.0044 - cosine_similarity: 0.2695 - loss: 0.4762 - mae: 0.4884 - val_accuracy: 8.1169e-04 - val_cosine_similarity: 0.0525 - val_loss: 0.5563 - val_mae: 0.4667 - learning_rate: 0.0010
Epoch 3/20
265/655 ━━━━━━━━━━━━━━━━━━━━ 1:16:39 12s/step - accuracy: 0.0048 - cosine_similarity: 0.2715 - loss: 0.4757 - mae: 0.4890
655/655 ━━━━━━━━━━━━━━━━━━━━ 7874s 12s/step - accuracy: 0.0048 - cosine_similarity: 0.2685 - loss: 0.4909 - mae: 0.4916 - val_accuracy: 0.0030 - val_cosine_similarity: 0.0216 - val_loss: 1.5446 - val_mae: 1.0591 - learning_rate: 0.0010
Epoch 4/20
 54/655 ━━━━━━━━━━━━━━━━━━━━ 1:54:25 11s/step - accuracy: 0.0049 - cosine_similarity: 0.2667 - loss: 0.5107 - mae: 0.4997
655/655 ━━━━━━━━━━━━━━━━━━━━ 7928s 12s/step - accuracy: 0.0047 - cosine_similarity: 0.2648 - loss: 0.5176 - mae: 0.4999 - val_accuracy: 0.0072 - val_cosine_similarity: 0.1592 - val_loss: 0.5530 - val_mae: 0.4719 - learning_rate: 5.0000e-04

Model: "functional"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                  ┃ Output Shape              ┃         Param # ┃ Connected to               ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ input_layer (InputLayer)      │ (None, 512)               │               0 │ -                          │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ embedding (Embedding)         │ (None, 512, 256)          │         572,416 │ input_layer[0][0]          │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ conv1d (Conv1D)               │ (None, 512, 64)           │          81,984 │ embedding[0][0]            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ conv1d_1 (Conv1D)             │ (None, 512, 64)           │          20,544 │ conv1d[0][0]               │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ bidirectional (Bidirectional) │ (None, 512, 1024)         │       2,363,392 │ conv1d_1[0][0]             │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ bidirectional_1               │ (None, 512, 1024)         │       6,295,552 │ bidirectional[0][0]        │
│ (Bidirectional)               │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ multi_head_attention          │ (None, 512, 1024)         │      16,790,528 │ bidirectional_1[0][0],     │
│ (MultiHeadAttention)          │                           │                 │ bidirectional_1[0][0]      │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ add (Add)                     │ (None, 512, 1024)         │               0 │ bidirectional_1[0][0],     │
│                               │                           │                 │ multi_head_attention[0][0] │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ lstm_2 (LSTM)                 │ (None, 512, 512)          │       3,147,776 │ add[0][0]                  │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ lstm_3 (LSTM)                 │ (None, 512, 512)          │       2,099,200 │ lstm_2[0][0]               │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ batch_normalization           │ (None, 512, 512)          │           2,048 │ lstm_3[0][0]               │
│ (BatchNormalization)          │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ conv1d_2 (Conv1D)             │ (None, 512, 128)          │         196,736 │ batch_normalization[0][0]  │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ conv1d_3 (Conv1D)             │ (None, 512, 128)          │          49,280 │ conv1d_2[0][0]             │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ conv1d_4 (Conv1D)             │ (None, 512, 256)          │          98,560 │ conv1d_3[0][0]             │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dense (Dense)                 │ (None, 512, 1024)         │         263,168 │ conv1d_4[0][0]             │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dropout_1 (Dropout)           │ (None, 512, 1024)         │               0 │ dense[0][0]                │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dense_1 (Dense)               │ (None, 512, 512)          │         524,800 │ dropout_1[0][0]            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dropout_2 (Dropout)           │ (None, 512, 512)          │               0 │ dense_1[0][0]              │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ layer_normalization           │ (None, 512, 512)          │           1,024 │ dropout_2[0][0]            │
│ (LayerNormalization)          │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dense_2 (Dense)               │ (None, 512, 128)          │          65,664 │ layer_normalization[0][0]  │
└───────────────────────────────┴───────────────────────────┴─────────────────┴────────────────────────────┘
 Total params: 32,572,672 (124.25 MB)
 Trainable params: 32,571,648 (124.25 MB)
 Non-trainable params: 1,024 (4.00 KB)


Model: "functional"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                  ┃ Output Shape              ┃         Param # ┃ Connected to               ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ input_layer (InputLayer)      │ (None, 512)               │               0 │ -                          │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ embedding (Embedding)         │ (None, 512, 256)          │         572,416 │ input_layer[0][0]          │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ conv1d (Conv1D)               │ (None, 512, 64)           │          81,984 │ embedding[0][0]            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ conv1d_1 (Conv1D)             │ (None, 512, 64)           │          20,544 │ conv1d[0][0]               │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ bidirectional (Bidirectional) │ (None, 512, 1024)         │       2,363,392 │ conv1d_1[0][0]             │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ bidirectional_1               │ (None, 512, 1024)         │       6,295,552 │ bidirectional[0][0]        │
│ (Bidirectional)               │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ multi_head_attention          │ (None, 512, 1024)         │      16,790,528 │ bidirectional_1[0][0],     │
│ (MultiHeadAttention)          │                           │                 │ bidirectional_1[0][0]      │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ add (Add)                     │ (None, 512, 1024)         │               0 │ bidirectional_1[0][0],     │
│                               │                           │                 │ multi_head_attention[0][0] │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ lstm_2 (LSTM)                 │ (None, 512, 512)          │       3,147,776 │ add[0][0]                  │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ lstm_3 (LSTM)                 │ (None, 512, 512)          │       2,099,200 │ lstm_2[0][0]               │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ batch_normalization           │ (None, 512, 512)          │           2,048 │ lstm_3[0][0]               │
│ (BatchNormalization)          │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ conv1d_2 (Conv1D)             │ (None, 512, 128)          │         196,736 │ batch_normalization[0][0]  │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ conv1d_3 (Conv1D)             │ (None, 512, 128)          │          49,280 │ conv1d_2[0][0]             │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ conv1d_4 (Conv1D)             │ (None, 512, 256)          │          98,560 │ conv1d_3[0][0]             │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ conv1d_5 (Conv1D)             │ (None, 512, 256)          │         196,864 │ conv1d_4[0][0]             │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ conv1d_6 (Conv1D)             │ (None, 512, 256)          │         196,864 │ conv1d_5[0][0]             │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dense (Dense)                 │ (None, 512, 1024)         │         263,168 │ conv1d_6[0][0]             │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dropout_1 (Dropout)           │ (None, 512, 1024)         │               0 │ dense[0][0]                │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dense_1 (Dense)               │ (None, 512, 512)          │         524,800 │ dropout_1[0][0]            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dropout_2 (Dropout)           │ (None, 512, 512)          │               0 │ dense_1[0][0]              │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ layer_normalization           │ (None, 512, 512)          │           1,024 │ dropout_2[0][0]            │
│ (LayerNormalization)          │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dense_2 (Dense)               │ (None, 512, 128)          │          65,664 │ layer_normalization[0][0]  │
└───────────────────────────────┴───────────────────────────┴─────────────────┴────────────────────────────┘
 Total params: 32,966,400 (125.76 MB)
 Trainable params: 32,965,376 (125.75 MB)
 Non-trainable params: 1,024 (4.00 KB)

Epoch 1/25
165/655 ━━━━━━━━━━━━━━━━━━━━ 1:34:59 12s/step - cosine_similarity: 0.2449 - loss: 0.6822 - mae: 0.4874 
655/655 ━━━━━━━━━━━━━━━━━━━━ 8095s 12s/step - cosine_similarity: 0.2606 - loss: 0.5478 - mae: 0.4687 - val_cosine_similarity: 0.2729 - val_loss: 0.4278 - val_mae: 0.4517 - learning_rate: 0.0010
Epoch 2/25
Epoch 2: saving model to checkpoints/model_epoch_02_valLoss_0.4277.keras
95/655 ━━━━━━━━━━━━━━━━━━━━ 1:43:41 11s/step - cosine_similarity: 0.2640 - loss: 0.4175 - mae: 0.4446
437/655 ━━━━━━━━━━━━━━━━━━━━ 43:05 12s/step - cosine_similarity: 0.2684 - loss: 0.4209 - mae: 0.4481  
655/655 ━━━━━━━━━━━━━━━━━━━━ 8121s 12s/step - cosine_similarity: 0.2691 - loss: 0.4211 - mae: 0.4488 - val_cosine_similarity: 0.2726 - val_loss: 0.4277 - val_mae: 0.4838 - learning_rate: 0.0010
Epoch 3/25
75/655 ━━━━━━━━━━━━━━━━━━━━ 1:54:49 12s/step - cosine_similarity: 0.2694 - loss: 0.4168 - mae: 0.4471
643/655 ━━━━━━━━━━━━━━━━━━━━ 2:44 14s/step - cosine_similarity: 0.2709 - loss: 0.4185 - mae: 0.4486  
655/655 ━━━━━━━━━━━━━━━━━━━━ 9503s 15s/step - cosine_similarity: 0.2709 - loss: 0.4185 - mae: 0.4486 - val_cosine_similarity: 0.2691 - val_loss: 0.4616 - val_mae: 0.5207 - learning_rate: 0.0010
Epoch 4/25
 16/655 ━━━━━━━━━━━━━━━━━━━━ 3:06:21 17s/step - cosine_similarity: 0.2705 - loss: 0.4198 - mae: 0.4480
 42/655 ━━━━━━━━━━━━━━━━━━━━ 3:01:02 18s/step - cosine_similarity: 0.2698 - loss: 0.4172 - mae: 0.4467


Model: "functional"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                  ┃ Output Shape              ┃         Param # ┃ Connected to               ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ input_layer (InputLayer)      │ (None, 512)               │               0 │ -                          │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ embedding (Embedding)         │ (None, 512, 256)          │         572,416 │ input_layer[0][0]          │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ conv1d (Conv1D)               │ (None, 512, 64)           │          81,984 │ embedding[0][0]            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ conv1d_1 (Conv1D)             │ (None, 512, 64)           │          20,544 │ conv1d[0][0]               │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ bidirectional (Bidirectional) │ (None, 512, 1024)         │       2,363,392 │ conv1d_1[0][0]             │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ bidirectional_1               │ (None, 512, 1024)         │       6,295,552 │ bidirectional[0][0]        │
│ (Bidirectional)               │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ multi_head_attention          │ (None, 512, 1024)         │      16,790,528 │ bidirectional_1[0][0],     │
│ (MultiHeadAttention)          │                           │                 │ bidirectional_1[0][0]      │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ add (Add)                     │ (None, 512, 1024)         │               0 │ bidirectional_1[0][0],     │
│                               │                           │                 │ multi_head_attention[0][0] │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ lstm_2 (LSTM)                 │ (None, 512, 512)          │       3,147,776 │ add[0][0]                  │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ lstm_3 (LSTM)                 │ (None, 512, 512)          │       2,099,200 │ lstm_2[0][0]               │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ batch_normalization           │ (None, 512, 512)          │           2,048 │ lstm_3[0][0]               │
│ (BatchNormalization)          │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ conv1d_2 (Conv1D)             │ (None, 512, 80)           │         122,960 │ batch_normalization[0][0]  │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ conv1d_3 (Conv1D)             │ (None, 512, 128)          │          30,848 │ conv1d_2[0][0]             │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ conv1d_4 (Conv1D)             │ (None, 512, 128)          │          49,280 │ conv1d_3[0][0]             │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ conv1d_5 (Conv1D)             │ (None, 512, 256)          │          98,560 │ conv1d_4[0][0]             │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ conv1d_6 (Conv1D)             │ (None, 512, 256)          │         196,864 │ conv1d_5[0][0]             │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dense (Dense)                 │ (None, 512, 1024)         │         263,168 │ conv1d_6[0][0]             │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dropout_1 (Dropout)           │ (None, 512, 1024)         │               0 │ dense[0][0]                │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dense_1 (Dense)               │ (None, 512, 512)          │         524,800 │ dropout_1[0][0]            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dropout_2 (Dropout)           │ (None, 512, 512)          │               0 │ dense_1[0][0]              │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ layer_normalization           │ (None, 512, 512)          │           1,024 │ dropout_2[0][0]            │
│ (LayerNormalization)          │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dense_2 (Dense)               │ (None, 512, 128)          │          65,664 │ layer_normalization[0][0]  │
└───────────────────────────────┴───────────────────────────┴─────────────────┴────────────────────────────┘
 Total params: 32,726,608 (124.84 MB)
 Trainable params: 32,725,584 (124.84 MB)
 Non-trainable params: 1,024 (4.00 KB)
Epoch 1/25
 73/655 ━━━━━━━━━━━━━━━━━━━━ 1:51:56 12s/step - cosine_similarity: 0.2207 - loss: 2.8913 - mae: 0.4927 
 271/655 ━━━━━━━━━━━━━━━━━━━━ 1:15:50 12s/step - cosine_similarity: 0.2473 - loss: 1.5113 - mae: 0.4842 
 655/655 ━━━━━━━━━━━━━━━━━━━━ 8210s 13s/step - cosine_similarity: 0.2577 - loss: 0.9925 - mae: 0.4720 - val_cosine_similarity: 0.0422 - val_loss: 0.5291 - val_mae: 0.4754 - learning_rate: 0.0010
Epoch 2/25
340/655 ━━━━━━━━━━━━━━━━━━━━ 1:05:44 13s/step - cosine_similarity: 0.2682 - loss: 0.4177 - mae: 0.4466
655/655 ━━━━━━━━━━━━━━━━━━━━ 8363s 13s/step - cosine_similarity: 0.2691 - loss: 0.4178 - mae: 0.4468 - val_cosine_similarity: -0.0534 - val_loss: 0.5632 - val_mae: 0.4853 - learning_rate: 0.0010
Epoch 3/25
  5/655 ━━━━━━━━━━━━━━━━━━━━ 2:11:39 12s/step - cosine_similarity: 0.2742 - loss: 0.4439 - mae: 0.4564
655/655 ━━━━━━━━━━━━━━━━━━━━ 7992s 12s/step - cosine_similarity: 0.2693 - loss: 0.4168 - mae: 0.4460 - val_cosine_similarity: 0.2649 - val_loss: 0.4357 - val_mae: 0.4317 - learning_rate: 0.0010
Epoch 4/25
246/655 ━━━━━━━━━━━━━━━━━━━━ 1:25:07 12s/step - cosine_similarity: 0.2657 - loss: 0.5145 - mae: 0.4596





Search: Running Trial #1

Value             |Best Value So Far |Hyperparameter
128               |128               |embed_dim
32                |32                |conv_filters
384               |384               |rnn_units
1.1341e-05        |1.1341e-05        |learning_rate
3                 |3                 |tuner/epochs
0                 |0                 |tuner/initial_epoch
2                 |2                 |tuner/bracket
0                 |0                 |tuner/round

/Users/edelta/Desktop/shruti/TTS/env/lib/python3.11/site-packages/keras/src/layers/layer.py:939: UserWarning: Layer 'reshape' (of type Reshape) was passed an input with a mask attached to it. However, this layer does not support masking and will therefore destroy the mask information. Downstream layers will not see the mask.
  warnings.warn(
Epoch 1/3
328/328 ━━━━━━━━━━━━━━━━━━━━ 1607s 5s/step - loss: 0.5532 - mae: 0.4561 - val_loss: 0.5579 - val_mae: 0.4633
Epoch 2/3
328/328 ━━━━━━━━━━━━━━━━━━━━ 1623s 5s/step - loss: 0.5497 - mae: 0.4581 - val_loss: 0.5576 - val_mae: 0.4650
Epoch 3/3
328/328 ━━━━━━━━━━━━━━━━━━━━ 1662s 5s/step - loss: 0.5490 - mae: 0.4629 - val_loss: 0.5545 - val_mae: 0.4655


Trial 1 Complete [01h 21m 35s]
val_loss: 0.554463803768158

Best val_loss So Far: 0.554463803768158
Total elapsed time: 01h 21m 35s

Search: Running Trial #2

Value             |Best Value So Far |Hyperparameter
256               |128               |embed_dim
64                |32                |conv_filters
256               |384               |rnn_units
7.35e-05          |1.1341e-05        |learning_rate
3                 |3                 |tuner/epochs
0                 |0                 |tuner/initial_epoch
2                 |2                 |tuner/bracket
0                 |0                 |tuner/round

Epoch 1/3
328/328 ━━━━━━━━━━━━━━━━━━━━ 944s 3s/step - loss: 0.5517 - mae: 0.4580 - val_loss: 0.5564 - val_mae: 0.4670
Epoch 2/3
328/328 ━━━━━━━━━━━━━━━━━━━━ 978s 3s/step - loss: 0.5414 - mae: 0.4725 - val_loss: 0.4980 - val_mae: 0.5119
Epoch 3/3
328/328 ━━━━━━━━━━━━━━━━━━━━ 967s 3s/step - loss: 0.4933 - mae: 0.5096 - val_loss: 0.4930 - val_mae: 0.4973

Trial 2 Complete [00h 48m 16s]
val_loss: 0.4929933547973633

Best val_loss So Far: 0.4929933547973633
Total elapsed time: 02h 09m 52s

Search: Running Trial #3

Value             |Best Value So Far |Hyperparameter
128               |256               |embed_dim
32                |64                |conv_filters
384               |256               |rnn_units
0.00052007        |7.35e-05          |learning_rate
3                 |3                 |tuner/epochs
0                 |0                 |tuner/initial_epoch
2                 |2                 |tuner/bracket
0                 |0                 |tuner/round

Epoch 1/3
328/328 ━━━━━━━━━━━━━━━━━━━━ 1600s 5s/step - loss: 0.5498 - mae: 0.4620 - val_loss: 0.5512 - val_mae: 0.4782
Epoch 2/3
328/328 ━━━━━━━━━━━━━━━━━━━━ 1659s 5s/step - loss: 0.5458 - mae: 0.4727 - val_loss: 0.5549 - val_mae: 0.4661
Epoch 3/3
328/328 ━━━━━━━━━━━━━━━━━━━━ 1652s 5s/step - loss: 0.5455 - mae: 0.4616 - val_loss: 0.5586 - val_mae: 0.4738

Trial 3 Complete [01h 21m 59s]
val_loss: 0.5512418150901794

Best val_loss So Far: 0.4929933547973633
Total elapsed time: 03h 31m 51s

Search: Running Trial #4

Value             |Best Value So Far |Hyperparameter
192               |256               |embed_dim
32                |64                |conv_filters
256               |256               |rnn_units
0.0015149         |7.35e-05          |learning_rate
3                 |3                 |tuner/epochs
0                 |0                 |tuner/initial_epoch
2                 |2                 |tuner/bracket
0                 |0                 |tuner/round


================= CNN based model =================

12312  He likewise indicated he was disenchanted with Russia

✅ Summary Recommendation:
Goal	Recommended Model
Highest speech quality	RNN + Attention (Tacotron)
Fastest training/inference	Fully Convolutional (ConvTTS/FastSpeech-like)
Using only CPU	Fully Convolutional
Smaller dataset	RNN may generalize better

If you're training on CPU only, I recommend starting with the convolutional model, get it working well, then try LSTM+attention later for quality tuning.

Model: "functional"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ phoneme_input (InputLayer)           │ (None, 168)                 │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ embedding (Embedding)                │ (None, 168, 256)            │          10,752 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ reshape (Reshape)                    │ (None, 168, 256)            │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv1d (Conv1D)                      │ (None, 168, 256)            │         327,936 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ batch_normalization                  │ (None, 168, 256)            │           1,024 │
│ (BatchNormalization)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout (Dropout)                    │ (None, 168, 256)            │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv1d_1 (Conv1D)                    │ (None, 168, 256)            │         327,936 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ batch_normalization_1                │ (None, 168, 256)            │           1,024 │
│ (BatchNormalization)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_1 (Dropout)                  │ (None, 168, 256)            │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv1d_2 (Conv1D)                    │ (None, 168, 256)            │         327,936 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ batch_normalization_2                │ (None, 168, 256)            │           1,024 │
│ (BatchNormalization)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_2 (Dropout)                  │ (None, 168, 256)            │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv1d_3 (Conv1D)                    │ (None, 168, 256)            │         327,936 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ batch_normalization_3                │ (None, 168, 256)            │           1,024 │
│ (BatchNormalization)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_3 (Dropout)                  │ (None, 168, 256)            │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv1d_4 (Conv1D)                    │ (None, 168, 256)            │         327,936 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ batch_normalization_4                │ (None, 168, 256)            │           1,024 │
│ (BatchNormalization)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_4 (Dropout)                  │ (None, 168, 256)            │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv1d_5 (Conv1D)                    │ (None, 168, 256)            │         327,936 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ batch_normalization_5                │ (None, 168, 256)            │           1,024 │
│ (BatchNormalization)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_5 (Dropout)                  │ (None, 168, 256)            │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv1d_transpose (Conv1DTranspose)   │ (None, 336, 256)            │         327,936 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv1d_transpose_1 (Conv1DTranspose) │ (None, 672, 128)            │          98,432 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv1d_transpose_2 (Conv1DTranspose) │ (None, 1344, 128)           │          49,280 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ lambda (Lambda)                      │ (None, 900, 128)            │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ mel_output (Conv1D)                  │ (None, 900, 80)             │          10,320 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 2,470,480 (9.42 MB)
 Trainable params: 2,467,408 (9.41 MB)
 Non-trainable params: 3,072 (12.00 KB)
Epoch 1/50
 90/328 ━━━━━━━━━━━━━━━━━━━━ 2:02 516ms/step - cosine_similarity: 0.2461 - loss: 0.5016 - mae: 0.5039
 328/328 ━━━━━━━━━━━━━━━━━━━━ 194s 579ms/step - cosine_similarity: 0.2829 - loss: 0.4717 - mae: 0.4894 - val_cosine_similarity: 0.3051 - val_loss: 0.4794 - val_mae: 0.5228 - learning_rate: 0.0010
Epoch 2/50
120/328 ━━━━━━━━━━━━━━━━━━━━ 2:02 591ms/step - cosine_similarity: 0.3093 - loss: 0.4453 - mae: 0.4749
328/328 ━━━━━━━━━━━━━━━━━━━━ 194s 591ms/step - cosine_similarity: 0.3114 - loss: 0.4447 - mae: 0.4750 - val_cosine_similarity: 0.3173 - val_loss: 0.4516 - val_mae: 0.4788 - learning_rate: 0.0010
Epoch 3/50
328/328 ━━━━━━━━━━━━━━━━━━━━ 193s 587ms/step - cosine_similarity: 0.3136 - loss: 0.4416 - mae: 0.4729 - val_cosine_similarity: 0.3186 - val_loss: 0.4460 - val_mae: 0.4773 - learning_rate: 0.0010
Epoch 4/50
 58/328 ━━━━━━━━━━━━━━━━━━━━ 2:26 543ms/step - cosine_similarity: 0.3126 - loss: 0.4410 - mae: 0.4720
 328/328 ━━━━━━━━━━━━━━━━━━━━ 193s 587ms/step - cosine_similarity: 0.3155 - loss: 0.4390 - mae: 0.4713 - val_cosine_similarity: 0.3204 - val_loss: 0.4435 - val_mae: 0.4766 - learning_rate: 0.0010
Epoch 5/50
328/328 ━━━━━━━━━━━━━━━━━━━━ 0s 590ms/step - cosine_similarity: 0.3176 - loss: 0.4365 - mae: 0.4698  
328/328 ━━━━━━━━━━━━━━━━━━━━ 203s 620ms/step - cosine_similarity: 0.3176 - loss: 0.4365 - mae: 0.4698 - val_cosine_similarity: 0.3221 - val_loss: 0.4415 - val_mae: 0.4739 - learning_rate: 0.0010
Epoch 6/50
328/328 ━━━━━━━━━━━━━━━━━━━━ 196s 596ms/step - cosine_similarity: 0.3189 - loss: 0.4349 - mae: 0.4688 - val_cosine_similarity: 0.3216 - val_loss: 0.4410 - val_mae: 0.4761 - learning_rate: 0.0010
Epoch 7/50
328/328 ━━━━━━━━━━━━━━━━━━━━ 193s 587ms/step - cosine_similarity: 0.3199 - loss: 0.4335 - mae: 0.4679 - val_cosine_similarity: 0.3228 - val_loss: 0.4403 - val_mae: 0.4752 - learning_rate: 0.0010
Epoch 8/50
328/328 ━━━━━━━━━━━━━━━━━━━━ 190s 578ms/step - cosine_similarity: 0.3209 - loss: 0.4322 - mae: 0.4671 - val_cosine_similarity: 0.3237 - val_loss: 0.4400 - val_mae: 0.4750 - learning_rate: 0.0010
Epoch 9/50
328/328 ━━━━━━━━━━━━━━━━━━━━ 193s 590ms/step - cosine_similarity: 0.3219 - loss: 0.4309 - mae: 0.4663 - val_cosine_similarity: 0.3240 - val_loss: 0.4393 - val_mae: 0.4738 - learning_rate: 0.0010
Epoch 10/50
191/328 ━━━━━━━━━━━━━━━━━━━━ 1:14 544ms/step - cosine_similarity: 0.3220 - loss: 0.4289 - mae: 0.4648
328/328 ━━━━━━━━━━━━━━━━━━━━ 191s 582ms/step - cosine_similarity: 0.3229 - loss: 0.4295 - mae: 0.4655 - val_cosine_similarity: 0.3229 - val_loss: 0.4403 - val_mae: 0.4757 - learning_rate: 0.0010
Epoch 11/50
328/328 ━━━━━━━━━━━━━━━━━━━━ 185s 564ms/step - cosine_similarity: 0.3238 - loss: 0.4282 - mae: 0.4648 - val_cosine_similarity: 0.3228 - val_loss: 0.4403 - val_mae: 0.4752 - learning_rate: 0.0010
Epoch 12/50
328/328 ━━━━━━━━━━━━━━━━━━━━ 190s 578ms/step - cosine_similarity: 0.3247 - loss: 0.4269 - mae: 0.4640 - val_cosine_similarity: 0.3211 - val_loss: 0.4415 - val_mae: 0.4747 - learning_rate: 0.0010
Epoch 13/50
328/328 ━━━━━━━━━━━━━━━━━━━━ 185s 564ms/step - cosine_similarity: 0.3269 - loss: 0.4238 - mae: 0.4622 - val_cosine_similarity: 0.3219 - val_loss: 0.4408 - val_mae: 0.4730 - learning_rate: 5.0000e-04
Epoch 14/50
328/328 ━━━━━━━━━━━━━━━━━━━━ 184s 561ms/step - cosine_similarity: 0.3287 - loss: 0.4213 - mae: 0.4607 - val_cosine_similarity: 0.3205 - val_loss: 0.4428 - val_mae: 0.4739 - learning_rate: 5.0000e-04




Model: "functional"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ phoneme_input (InputLayer)           │ (None, 168)                 │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ embedding (Embedding)                │ (None, 168, 256)            │          10,752 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ reshape (Reshape)                    │ (None, 168, 256)            │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv1d (Conv1D)                      │ (None, 168, 256)            │         327,936 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ batch_normalization                  │ (None, 168, 256)            │           1,024 │
│ (BatchNormalization)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout (Dropout)                    │ (None, 168, 256)            │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv1d_1 (Conv1D)                    │ (None, 168, 256)            │         327,936 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ batch_normalization_1                │ (None, 168, 256)            │           1,024 │
│ (BatchNormalization)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_1 (Dropout)                  │ (None, 168, 256)            │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv1d_2 (Conv1D)                    │ (None, 168, 256)            │         327,936 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ batch_normalization_2                │ (None, 168, 256)            │           1,024 │
│ (BatchNormalization)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_2 (Dropout)                  │ (None, 168, 256)            │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv1d_3 (Conv1D)                    │ (None, 168, 256)            │         327,936 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ batch_normalization_3                │ (None, 168, 256)            │           1,024 │
│ (BatchNormalization)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_3 (Dropout)                  │ (None, 168, 256)            │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv1d_4 (Conv1D)                    │ (None, 168, 256)            │         327,936 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ batch_normalization_4                │ (None, 168, 256)            │           1,024 │
│ (BatchNormalization)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_4 (Dropout)                  │ (None, 168, 256)            │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv1d_5 (Conv1D)                    │ (None, 168, 256)            │         327,936 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ batch_normalization_5                │ (None, 168, 256)            │           1,024 │
│ (BatchNormalization)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_5 (Dropout)                  │ (None, 168, 256)            │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv1d_transpose (Conv1DTranspose)   │ (None, 336, 256)            │         327,936 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv1d_transpose_1 (Conv1DTranspose) │ (None, 672, 128)            │          98,432 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv1d_transpose_2 (Conv1DTranspose) │ (None, 1344, 128)           │          49,280 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ crop_layer (CropLayer)               │ (None, 900, 128)            │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ mel_output (Conv1D)                  │ (None, 900, 80)             │          10,320 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 2,470,480 (9.42 MB)
 Trainable params: 2,467,408 (9.41 MB)
 Non-trainable params: 3,072 (12.00 KB)
Epoch 1/50
328/328 ━━━━━━━━━━━━━━━━━━━━ 203s 607ms/step - cosine_similarity: 0.2839 - loss: 0.4711 - mae: 0.4894 - val_cosine_similarity: 0.3078 - val_loss: 0.5012 - val_mae: 0.5552 - learning_rate: 0.0010
Epoch 2/50
328/328 ━━━━━━━━━━━━━━━━━━━━ 198s 603ms/step - cosine_similarity: 0.3116 - loss: 0.4444 - mae: 0.4749 - val_cosine_similarity: 0.3169 - val_loss: 0.4509 - val_mae: 0.4798 - learning_rate: 0.0010
Epoch 3/50
263/328 ━━━━━━━━━━━━━━━━━━━━ 36s 555ms/step - cosine_similarity: 0.3133 - loss: 0.4411 - mae: 0.4723 
328/328 ━━━━━━━━━━━━━━━━━━━━ 190s 578ms/step - cosine_similarity: 0.3138 - loss: 0.4413 - mae: 0.4727 - val_cosine_similarity: 0.3196 - val_loss: 0.4469 - val_mae: 0.4774 - learning_rate: 0.0010
Epoch 4/50
328/328 ━━━━━━━━━━━━━━━━━━━━ 206s 627ms/step - cosine_similarity: 0.3160 - loss: 0.4386 - mae: 0.4710 - val_cosine_similarity: 0.3204 - val_loss: 0.4440 - val_mae: 0.4753 - learning_rate: 0.0010
Epoch 5/50
328/328 ━━━━━━━━━━━━━━━━━━━━ 214s 653ms/step - cosine_similarity: 0.3177 - loss: 0.4364 - mae: 0.4696 - val_cosine_similarity: 0.3223 - val_loss: 0.4409 - val_mae: 0.4744 - learning_rate: 0.0010
Epoch 6/50
328/328 ━━━━━━━━━━━━━━━━━━━━ 197s 600ms/step - cosine_similarity: 0.3190 - loss: 0.4348 - mae: 0.4687 - val_cosine_similarity: 0.3227 - val_loss: 0.4400 - val_mae: 0.4745 - learning_rate: 0.0010
Epoch 7/50
328/328 ━━━━━━━━━━━━━━━━━━━━ 213s 648ms/step - cosine_similarity: 0.3201 - loss: 0.4333 - mae: 0.4679 - val_cosine_similarity: 0.3238 - val_loss: 0.4395 - val_mae: 0.4743 - learning_rate: 0.0010
Epoch 8/50
Epoch 8: val_loss did not improve from 0.43952
328/328 ━━━━━━━━━━━━━━━━━━━━ 202s 615ms/step - cosine_similarity: 0.3211 - loss: 0.4320 - mae: 0.4671 - val_cosine_similarity: 0.3233 - val_loss: 0.4396 - val_mae: 0.4760 - learning_rate: 0.0010
Epoch 9/50
Epoch 9: val_loss improved from 0.43952 to 0.43951, saving model to model/2/best_model_cnn.keras
328/328 ━━━━━━━━━━━━━━━━━━━━ 202s 615ms/step - cosine_similarity: 0.3220 - loss: 0.4307 - mae: 0.4663 - val_cosine_similarity: 0.3232 - val_loss: 0.4395 - val_mae: 0.4743 - learning_rate: 0.0010
Epoch 10/50
265/328 ━━━━━━━━━━━━━━━━━━━━ 37s 593ms/step - cosine_similarity: 0.3224 - loss: 0.4292 - mae: 0.4652 
Epoch 10: val_loss improved from 0.43951 to 0.43894, saving model to model/2/best_model_cnn.keras
328/328 ━━━━━━━━━━━━━━━━━━━━ 205s 623ms/step - cosine_similarity: 0.3229 - loss: 0.4295 - mae: 0.4656 - val_cosine_similarity: 0.3238 - val_loss: 0.4389 - val_mae: 0.4745 - learning_rate: 0.0010

Epoch 13/50
Epoch 13: val_loss did not improve from 0.43894
Epoch 13: ReduceLROnPlateau reducing learning rate to 0.0005000000237487257.
328/328 ━━━━━━━━━━━━━━━━━━━━ 205s 626ms/step - cosine_similarity: 0.3259 - loss: 0.4253 - mae: 0.4632 - val_cosine_similarity: 0.3217 - val_loss: 0.4412 - val_mae: 0.4754 - learning_rate: 0.0010
Epoch 14/50
Epoch 14: val_loss did not improve from 0.43894
328/328 ━━━━━━━━━━━━━━━━━━━━ 205s 625ms/step - cosine_similarity: 0.3280 - loss: 0.4224 - mae: 0.4615 - val_cosine_similarity: 0.3215 - val_loss: 0.4420 - val_mae: 0.4749 - learning_rate: 5.0000e-04
Epoch 15/50
Epoch 15: val_loss did not improve from 0.43894
328/328 ━━━━━━━━━━━━━━━━━━━━ 187s 569ms/step - cosine_similarity: 0.3297 - loss: 0.4199 - mae: 0.4599 - val_cosine_similarity: 0.3207 - val_loss: 0.4429 - val_mae: 0.4733 - learning_rate: 5.0000e-04
Epoch 15: early stopping
Restoring model weights from the end of the best epoch: 10.
25/25 ━━━━━━━━━━━━━━━━━━━━ 4s 142ms/step - cosine_similarity: 0.3261 - loss: 0.4448 - mae: 0.4795
Test loss: [0.4388144612312317, 0.4751414954662323, 0.32419657707214355]




┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                  ┃ Output Shape              ┃         Param # ┃ Connected to               ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ phoneme_input (InputLayer)    │ (None, 168)               │               0 │ -                          │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ embedding (Embedding)         │ (None, 168, 256)          │          10,752 │ phoneme_input[0][0]        │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ reshape (Reshape)             │ (None, 168, 256)          │               0 │ embedding[0][0]            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ conv1d (Conv1D)               │ (None, 168, 256)          │         327,936 │ reshape[0][0]              │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ batch_normalization           │ (None, 168, 256)          │           1,024 │ conv1d[0][0]               │
│ (BatchNormalization)          │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dropout (Dropout)             │ (None, 168, 256)          │               0 │ batch_normalization[0][0]  │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ add (Add)                     │ (None, 168, 256)          │               0 │ dropout[0][0],             │
│                               │                           │                 │ reshape[0][0]              │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ conv1d_1 (Conv1D)             │ (None, 168, 256)          │         327,936 │ add[0][0]                  │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ batch_normalization_1         │ (None, 168, 256)          │           1,024 │ conv1d_1[0][0]             │
│ (BatchNormalization)          │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dropout_1 (Dropout)           │ (None, 168, 256)          │               0 │ batch_normalization_1[0][… │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ add_1 (Add)                   │ (None, 168, 256)          │               0 │ dropout_1[0][0], add[0][0] │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ conv1d_2 (Conv1D)             │ (None, 168, 256)          │         327,936 │ add_1[0][0]                │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ batch_normalization_2         │ (None, 168, 256)          │           1,024 │ conv1d_2[0][0]             │
│ (BatchNormalization)          │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dropout_2 (Dropout)           │ (None, 168, 256)          │               0 │ batch_normalization_2[0][… │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ add_2 (Add)                   │ (None, 168, 256)          │               0 │ dropout_2[0][0],           │
│                               │                           │                 │ add_1[0][0]                │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ bidirectional (Bidirectional) │ (None, 168, 256)          │         394,240 │ add_2[0][0]                │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dense (Dense)                 │ (None, 168, 1)            │             257 │ bidirectional[0][0]        │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ softmax (Softmax)             │ (None, 168, 1)            │               0 │ dense[0][0]                │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ multiply (Multiply)           │ (None, 168, 256)          │               0 │ bidirectional[0][0],       │
│                               │                           │                 │ softmax[0][0]              │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ conv1d_transpose              │ (None, 336, 256)          │         327,936 │ multiply[0][0]             │
│ (Conv1DTranspose)             │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ conv1d_transpose_1            │ (None, 672, 128)          │          98,432 │ conv1d_transpose[0][0]     │
│ (Conv1DTranspose)             │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ conv1d_transpose_2            │ (None, 1344, 128)         │          49,280 │ conv1d_transpose_1[0][0]   │
│ (Conv1DTranspose)             │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ crop_layer (CropLayer)        │ (None, 900, 128)          │               0 │ conv1d_transpose_2[0][0]   │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ mel_output (Conv1D)           │ (None, 900, 80)           │          10,320 │ crop_layer[0][0]           │
└───────────────────────────────┴───────────────────────────┴─────────────────┴────────────────────────────┘
 Total params: 1,878,097 (7.16 MB)
 Trainable params: 1,876,561 (7.16 MB)
 Non-trainable params: 1,536 (6.00 KB)


Epoch 17/50
328/328 ━━━━━━━━━━━━━━━━━━━━ 0s 626ms/step - cosine_similarity: 0.3368 - loss: 0.4045 - mae: 0.4452  
Epoch 17: val_loss did not improve from 0.42575

Epoch 17: ReduceLROnPlateau reducing learning rate to 0.0005000000237487257.
328/328 ━━━━━━━━━━━━━━━━━━━━ 216s 658ms/step - cosine_similarity: 0.3369 - loss: 0.4045 - mae: 0.4452 - val_cosine_similarity: 0.3302 - val_loss: 0.4274 - val_mae: 0.4610 - learning_rate: 0.0010
Epoch 18/50
328/328 ━━━━━━━━━━━━━━━━━━━━ 0s 648ms/step - cosine_similarity: 0.3392 - loss: 0.4010 - mae: 0.4427  
Epoch 18: val_loss did not improve from 0.42575
328/328 ━━━━━━━━━━━━━━━━━━━━ 224s 683ms/step - cosine_similarity: 0.3392 - loss: 0.4010 - mae: 0.4427 - val_cosine_similarity: 0.3296 - val_loss: 0.4269 - val_mae: 0.4569 - learning_rate: 5.0000e-04
Epoch 19/50
328/328 ━━━━━━━━━━━━━━━━━━━━ 0s 644ms/step - cosine_similarity: 0.3413 - loss: 0.3976 - mae: 0.4403  
Epoch 19: val_loss did not improve from 0.42575
328/328 ━━━━━━━━━━━━━━━━━━━━ 223s 679ms/step - cosine_similarity: 0.3413 - loss: 0.3976 - mae: 0.4403 - val_cosine_similarity: 0.3300 - val_loss: 0.4274 - val_mae: 0.4600 - learning_rate: 5.0000e-04
Epoch 19: early stopping
Restoring model weights from the end of the best epoch: 14.
25/25 ━━━━━━━━━━━━━━━━━━━━ 6s 226ms/step - cosine_similarity: 0.3324 - loss: 0.4336 - mae: 0.4667 
Test loss: [0.42622050642967224, 0.46131065487861633, 0.33129048347473145]




Model: "functional"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                  ┃ Output Shape              ┃         Param # ┃ Connected to               ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ phoneme_input (InputLayer)    │ (None, 168)               │               0 │ -                          │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ embedding (Embedding)         │ (None, 168, 256)          │          10,752 │ phoneme_input[0][0]        │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ reshape (Reshape)             │ (None, 168, 256)          │               0 │ embedding[0][0]            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ conv1d (Conv1D)               │ (None, 168, 256)          │         327,936 │ reshape[0][0]              │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ layer_normalization           │ (None, 168, 256)          │             512 │ conv1d[0][0]               │
│ (LayerNormalization)          │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dropout (Dropout)             │ (None, 168, 256)          │               0 │ layer_normalization[0][0]  │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ add (Add)                     │ (None, 168, 256)          │               0 │ dropout[0][0],             │
│                               │                           │                 │ reshape[0][0]              │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ conv1d_1 (Conv1D)             │ (None, 168, 256)          │         327,936 │ add[0][0]                  │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ layer_normalization_1         │ (None, 168, 256)          │             512 │ conv1d_1[0][0]             │
│ (LayerNormalization)          │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dropout_1 (Dropout)           │ (None, 168, 256)          │               0 │ layer_normalization_1[0][… │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ add_1 (Add)                   │ (None, 168, 256)          │               0 │ dropout_1[0][0], add[0][0] │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ conv1d_2 (Conv1D)             │ (None, 168, 256)          │         327,936 │ add_1[0][0]                │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ layer_normalization_2         │ (None, 168, 256)          │             512 │ conv1d_2[0][0]             │
│ (LayerNormalization)          │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dropout_2 (Dropout)           │ (None, 168, 256)          │               0 │ layer_normalization_2[0][… │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ add_2 (Add)                   │ (None, 168, 256)          │               0 │ dropout_2[0][0],           │
│                               │                           │                 │ add_1[0][0]                │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ bidirectional (Bidirectional) │ (None, 168, 256)          │         394,240 │ add_2[0][0]                │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dropout_3 (Dropout)           │ (None, 168, 256)          │               0 │ bidirectional[0][0]        │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dense (Dense)                 │ (None, 168, 1)            │             257 │ dropout_3[0][0]            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ softmax (Softmax)             │ (None, 168, 1)            │               0 │ dense[0][0]                │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ multiply (Multiply)           │ (None, 168, 256)          │               0 │ dropout_3[0][0],           │
│                               │                           │                 │ softmax[0][0]              │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ conv1d_transpose              │ (None, 336, 256)          │         327,936 │ multiply[0][0]             │
│ (Conv1DTranspose)             │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ conv1d_transpose_1            │ (None, 672, 128)          │          98,432 │ conv1d_transpose[0][0]     │
│ (Conv1DTranspose)             │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ conv1d_transpose_2            │ (None, 1344, 128)         │          49,280 │ conv1d_transpose_1[0][0]   │
│ (Conv1DTranspose)             │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ crop_layer (CropLayer)        │ (None, 900, 128)          │               0 │ conv1d_transpose_2[0][0]   │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ mel_output (Conv1D)           │ (None, 900, 80)           │          10,320 │ crop_layer[0][0]           │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ conv1d_3 (Conv1D)             │ (None, 900, 80)           │          32,080 │ mel_output[0][0]           │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ add_3 (Add)                   │ (None, 900, 80)           │               0 │ mel_output[0][0],          │
│                               │                           │                 │ conv1d_3[0][0]             │
└───────────────────────────────┴───────────────────────────┴─────────────────┴────────────────────────────┘
 Total params: 1,908,641 (7.28 MB)
 Trainable params: 1,908,641 (7.28 MB)
 Non-trainable params: 0 (0.00 B)
Epoch 1/50
Epoch 1: Learning rate is 0.000999
328/328 ━━━━━━━━━━━━━━━━━━━━ 233s 701ms/step - cosine_similarity: 0.3554 - loss: 0.3615 - mae: 0.4585 - val_cosine_similarity: 0.3763 - val_loss: 0.3106 - val_mae: 0.4312 - learning_rate: 9.9911e-04
Epoch 2/50
Epoch 2: Learning rate is 0.000996
328/328 ━━━━━━━━━━━━━━━━━━━━ 214s 653ms/step - cosine_similarity: 0.3734 - loss: 0.3029 - mae: 0.4231 - val_cosine_similarity: 0.3833 - val_loss: 0.2966 - val_mae: 0.4241 - learning_rate: 9.9645e-04
Epoch 3/50
Epoch 3: Learning rate is 0.000992
328/328 ━━━━━━━━━━━━━━━━━━━━ 233s 710ms/step - cosine_similarity: 0.3785 - loss: 0.2931 - mae: 0.4168 - val_cosine_similarity: 0.3864 - val_loss: 0.2924 - val_mae: 0.4218 - learning_rate: 9.9203e-04
Epoch 4/50
Epoch 4: Learning rate is 0.000986
328/328 ━━━━━━━━━━━━━━━━━━━━ 222s 678ms/step - cosine_similarity: 0.3817 - loss: 0.2877 - mae: 0.4139 - val_cosine_similarity: 0.3879 - val_loss: 0.2909 - val_mae: 0.4227 - learning_rate: 9.8586e-04
Epoch 5/50
Epoch 5: Learning rate is 0.000978
328/328 ━━━━━━━━━━━━━━━━━━━━ 220s 670ms/step - cosine_similarity: 0.3834 - loss: 0.2845 - mae: 0.4118 - val_cosine_similarity: 0.3894 - val_loss: 0.2873 - val_mae: 0.4191 - learning_rate: 9.7798e-04
Epoch 6/50
Epoch 6: Learning rate is 0.000968
328/328 ━━━━━━━━━━━━━━━━━━━━ 205s 626ms/step - cosine_similarity: 0.3849 - loss: 0.2815 - mae: 0.4099 - val_cosine_similarity: 0.3889 - val_loss: 0.2880 - val_mae: 0.4219 - learning_rate: 9.6840e-04
Epoch 7/50
Epoch 7: Learning rate is 0.000957
328/328 ━━━━━━━━━━━━━━━━━━━━ 221s 673ms/step - cosine_similarity: 0.3860 - loss: 0.2794 - mae: 0.4087 - val_cosine_similarity: 0.3897 - val_loss: 0.2862 - val_mae: 0.4182 - learning_rate: 9.5717e-04
Epoch 8/50
Epoch 8: Learning rate is 0.000944
328/328 ━━━━━━━━━━━━━━━━━━━━ 244s 744ms/step - cosine_similarity: 0.3872 - loss: 0.2768 - mae: 0.4070 - val_cosine_similarity: 0.3903 - val_loss: 0.2853 - val_mae: 0.4176 - learning_rate: 9.4434e-04
Epoch 9/50
Epoch 9: Learning rate is 0.000930
328/328 ━━━━━━━━━━━━━━━━━━━━ 235s 716ms/step - cosine_similarity: 0.3882 - loss: 0.2748 - mae: 0.4058 - val_cosine_similarity: 0.3900 - val_loss: 0.2853 - val_mae: 0.4162 - learning_rate: 9.2995e-04
Epoch 10/50
 93/328 ━━━━━━━━━━━━━━━━━━━━ 2:43 697ms/step - cosine_similarity: 0.3865 - loss: 0.2758 - mae: 0.4068
 





 6==========

 Model: "functional"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                  ┃ Output Shape              ┃         Param # ┃ Connected to               ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ phoneme_input (InputLayer)    │ (None, 168)               │               0 │ -                          │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ embedding (Embedding)         │ (None, 168, 256)          │          10,752 │ phoneme_input[0][0]        │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ conv1d (Conv1D)               │ (None, 168, 256)          │         327,936 │ embedding[0][0]            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ layer_normalization           │ (None, 168, 256)          │             512 │ conv1d[0][0]               │
│ (LayerNormalization)          │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dropout (Dropout)             │ (None, 168, 256)          │               0 │ layer_normalization[0][0]  │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ add (Add)                     │ (None, 168, 256)          │               0 │ dropout[0][0],             │
│                               │                           │                 │ embedding[0][0]            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ conv1d_1 (Conv1D)             │ (None, 168, 256)          │         327,936 │ add[0][0]                  │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ layer_normalization_1         │ (None, 168, 256)          │             512 │ conv1d_1[0][0]             │
│ (LayerNormalization)          │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dropout_1 (Dropout)           │ (None, 168, 256)          │               0 │ layer_normalization_1[0][… │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ add_1 (Add)                   │ (None, 168, 256)          │               0 │ dropout_1[0][0], add[0][0] │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ conv1d_2 (Conv1D)             │ (None, 168, 256)          │         327,936 │ add_1[0][0]                │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ layer_normalization_2         │ (None, 168, 256)          │             512 │ conv1d_2[0][0]             │
│ (LayerNormalization)          │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dropout_2 (Dropout)           │ (None, 168, 256)          │               0 │ layer_normalization_2[0][… │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ add_2 (Add)                   │ (None, 168, 256)          │               0 │ dropout_2[0][0],           │
│                               │                           │                 │ add_1[0][0]                │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ not_equal_1 (NotEqual)        │ (None, 168)               │               0 │ phoneme_input[0][0]        │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ bidirectional (Bidirectional) │ (None, 168, 256)          │         394,240 │ add_2[0][0],               │
│                               │                           │                 │ not_equal_1[0][0]          │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dropout_3 (Dropout)           │ (None, 168, 256)          │               0 │ bidirectional[0][0]        │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dense (Dense)                 │ (None, 168, 1)            │             257 │ dropout_3[0][0]            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ softmax (Softmax)             │ (None, 168, 1)            │               0 │ dense[0][0]                │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ multiply (Multiply)           │ (None, 168, 256)          │               0 │ dropout_3[0][0],           │
│                               │                           │                 │ softmax[0][0]              │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ conv1d_transpose              │ (None, 336, 256)          │         327,936 │ multiply[0][0]             │
│ (Conv1DTranspose)             │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ conv1d_transpose_1            │ (None, 672, 128)          │          98,432 │ conv1d_transpose[0][0]     │
│ (Conv1DTranspose)             │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ conv1d_transpose_2            │ (None, 1344, 128)         │          49,280 │ conv1d_transpose_1[0][0]   │
│ (Conv1DTranspose)             │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ crop_layer (CropLayer)        │ (None, 900, 128)          │               0 │ conv1d_transpose_2[0][0]   │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ mel_output (Conv1D)           │ (None, 900, 80)           │          10,320 │ crop_layer[0][0]           │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ conv1d_3 (Conv1D)             │ (None, 900, 80)           │          32,080 │ mel_output[0][0]           │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ add_3 (Add)                   │ (None, 900, 80)           │               0 │ mel_output[0][0],          │
│                               │                           │                 │ conv1d_3[0][0]             │
└───────────────────────────────┴───────────────────────────┴─────────────────┴────────────────────────────┘
 Total params: 1,908,641 (7.28 MB)
 Trainable params: 1,908,641 (7.28 MB)
 Non-trainable params: 0 (0.00 B)
Epoch 1/50
328/328 ━━━━━━━━━━━━━━━━━━━━ 0s 452ms/step - cosine_similarity: 0.3524 - loss: 55.3678 - mae: 0.5525  
Epoch 1: val_loss improved from inf to 50.92939, saving model to model/2/best_model_cnn.keras

Epoch 1: Learning rate is 0.000999
328/328 ━━━━━━━━━━━━━━━━━━━━ 161s 484ms/step - cosine_similarity: 0.3525 - loss: 55.3591 - mae: 0.5526 - val_cosine_similarity: 0.3786 - val_loss: 50.9294 - val_mae: 0.5655
Epoch 2/50
328/328 ━━━━━━━━━━━━━━━━━━━━ 0s 470ms/step - cosine_similarity: 0.3750 - loss: 50.7710 - mae: 0.5665  
Epoch 2: val_loss improved from 50.92939 to 50.30367, saving model to model/2/best_model_cnn.keras

Epoch 2: Learning rate is 0.000996
328/328 ━━━━━━━━━━━━━━━━━━━━ 164s 502ms/step - cosine_similarity: 0.3750 - loss: 50.7702 - mae: 0.5665 - val_cosine_similarity: 0.3826 - val_loss: 50.3037 - val_mae: 0.5709
Epoch 3/50
270/328 ━━━━━━━━━━━━━━━━━━━━ 28s 498ms/step - cosine_similarity: 0.3774 - loss: 50.2891 - mae: 0.5727^[[A
328/328 ━━━━━━━━━━━━━━━━━━━━ 0s 493ms/step - cosine_similarity: 0.3781 - loss: 50.2350 - mae: 0.5724 
Epoch 3: val_loss improved from 50.30367 to 49.66397, saving model to model/2/best_model_cnn.keras

Epoch 3: Learning rate is 0.000992
328/328 ━━━━━━━━━━━━━━━━━━━━ 173s 527ms/step - cosine_similarity: 0.3781 - loss: 50.2341 - mae: 0.5724 - val_cosine_similarity: 0.3864 - val_loss: 49.6640 - val_mae: 0.5612
Epoch 4/50
328/328 ━━━━━━━━━━━━━━━━━━━━ 0s 486ms/step - cosine_similarity: 0.3810 - loss: 49.7563 - mae: 0.5691  
Epoch 4: val_loss improved from 49.66397 to 49.41357, saving model to model/2/best_model_cnn.keras

Epoch 4: Learning rate is 0.000986
328/328 ━━━━━━━━━━━━━━━━━━━━ 169s 516ms/step - cosine_similarity: 0.3810 - loss: 49.7557 - mae: 0.5691 - val_cosine_similarity: 0.3880 - val_loss: 49.4136 - val_mae: 0.5605
Epoch 5/50
328/328 ━━━━━━━━━━━━━━━━━━━━ 0s 469ms/step - cosine_similarity: 0.3827 - loss: 49.4648 - mae: 0.5724  
Epoch 5: val_loss improved from 49.41357 to 49.27781, saving model to model/2/best_model_cnn.keras

Epoch 5: Learning rate is 0.000978
328/328 ━━━━━━━━━━━━━━━━━━━━ 164s 501ms/step - cosine_similarity: 0.3827 - loss: 49.4643 - mae: 0.5724 - val_cosine_similarity: 0.3886 - val_loss: 49.2778 - val_mae: 0.5659
Epoch 6/50
328/328 ━━━━━━━━━━━━━━━━━━━━ 0s 455ms/step - cosine_similarity: 0.3841 - loss: 49.2222 - mae: 0.5737  
Epoch 6: val_loss improved from 49.27781 to 49.10455, saving model to model/2/best_model_cnn.keras

Epoch 6: Learning rate is 0.000968
328/328 ━━━━━━━━━━━━━━━━━━━━ 159s 484ms/step - cosine_similarity: 0.3841 - loss: 49.2217 - mae: 0.5737 - val_cosine_similarity: 0.3896 - val_loss: 49.1045 - val_mae: 0.5691
Epoch 7/50
328/328 ━━━━━━━━━━━━━━━━━━━━ 0s 450ms/step - cosine_similarity: 0.3854 - loss: 48.9883 - mae: 0.5752  
Epoch 7: val_loss improved from 49.10455 to 49.03452, saving model to model/2/best_model_cnn.keras

Epoch 7: Learning rate is 0.000957
328/328 ━━━━━━━━━━━━━━━━━━━━ 158s 481ms/step - cosine_similarity: 0.3854 - loss: 48.9879 - mae: 0.5752 - val_cosine_similarity: 0.3900 - val_loss: 49.0345 - val_mae: 0.5670
Epoch 8/50
328/328 ━━━━━━━━━━━━━━━━━━━━ 0s 450ms/step - cosine_similarity: 0.3865 - loss: 48.7919 - mae: 0.5732  
Epoch 8: val_loss improved from 49.03452 to 48.99873, saving model to model/2/best_model_cnn.keras

Epoch 8: Learning rate is 0.000944
328/328 ━━━━━━━━━━━━━━━━━━━━ 157s 480ms/step - cosine_similarity: 0.3866 - loss: 48.7914 - mae: 0.5732 - val_cosine_similarity: 0.3902 - val_loss: 48.9987 - val_mae: 0.5712
Epoch 9/50
328/328 ━━━━━━━━━━━━━━━━━━━━ 0s 450ms/step - cosine_similarity: 0.3877 - loss: 48.6045 - mae: 0.5710  
Epoch 9: val_loss improved from 48.99873 to 48.91460, saving model to model/2/best_model_cnn.keras

Epoch 9: Learning rate is 0.000930
328/328 ━━━━━━━━━━━━━━━━━━━━ 157s 480ms/step - cosine_similarity: 0.3877 - loss: 48.6041 - mae: 0.5710 - val_cosine_similarity: 0.3906 - val_loss: 48.9146 - val_mae: 0.5710
Epoch 10/50
328/328 ━━━━━━━━━━━━━━━━━━━━ 0s 456ms/step - cosine_similarity: 0.3887 - loss: 48.4156 - mae: 0.5716  
Epoch 10: val_loss improved from 48.91460 to 48.90816, saving model to model/2/best_model_cnn.keras

Epoch 10: Learning rate is 0.000914
328/328 ━━━━━━━━━━━━━━━━━━━━ 159s 485ms/step - cosine_similarity: 0.3887 - loss: 48.4153 - mae: 0.5716 - val_cosine_similarity: 0.3906 - val_loss: 48.9082 - val_mae: 0.5725
Epoch 11/50
328/328 ━━━━━━━━━━━━━━━━━━━━ 0s 446ms/step - cosine_similarity: 0.3898 - loss: 48.2369 - mae: 0.5712  
Epoch 11: val_loss did not improve from 48.90816

Epoch 11: Learning rate is 0.000897
328/328 ━━━━━━━━━━━━━━━━━━━━ 156s 475ms/step - cosine_similarity: 0.3898 - loss: 48.2365 - mae: 0.5712 - val_cosine_similarity: 0.3900 - val_loss: 49.0063 - val_mae: 0.5691
Epoch 12/50
328/328 ━━━━━━━━━━━━━━━━━━━━ 0s 449ms/step - cosine_similarity: 0.3909 - loss: 48.0375 - mae: 0.5710  
Epoch 12: val_loss did not improve from 48.90816

Epoch 12: Learning rate is 0.000878
328/328 ━━━━━━━━━━━━━━━━━━━━ 157s 478ms/step - cosine_similarity: 0.3910 - loss: 48.0372 - mae: 0.5710 - val_cosine_similarity: 0.3890 - val_loss: 49.1382 - val_mae: 0.5641
Epoch 13/50
328/328 ━━━━━━━━━━━━━━━━━━━━ 0s 462ms/step - cosine_similarity: 0.3921 - loss: 47.8416 - mae: 0.5694  
Epoch 13: val_loss did not improve from 48.90816

Epoch 13: Learning rate is 0.000858
328/328 ━━━━━━━━━━━━━━━━━━━━ 161s 490ms/step - cosine_similarity: 0.3921 - loss: 47.8413 - mae: 0.5694 - val_cosine_similarity: 0.3892 - val_loss: 49.1223 - val_mae: 0.5681
Epoch 14/50
328/328 ━━━━━━━━━━━━━━━━━━━━ 0s 444ms/step - cosine_similarity: 0.3929 - loss: 47.7144 - mae: 0.5702  
Epoch 14: val_loss did not improve from 48.90816

Epoch 14: Learning rate is 0.000837
328/328 ━━━━━━━━━━━━━━━━━━━━ 155s 472ms/step - cosine_similarity: 0.3929 - loss: 47.7141 - mae: 0.5702 - val_cosine_similarity: 0.3890 - val_loss: 49.1425 - val_mae: 0.5654
Epoch 15/50
328/328 ━━━━━━━━━━━━━━━━━━━━ 0s 452ms/step - cosine_similarity: 0.3937 - loss: 47.5654 - mae: 0.5687  
Epoch 15: val_loss did not improve from 48.90816

Epoch 15: Learning rate is 0.000815
328/328 ━━━━━━━━━━━━━━━━━━━━ 157s 479ms/step - cosine_similarity: 0.3937 - loss: 47.5651 - mae: 0.5687 - val_cosine_similarity: 0.3884 - val_loss: 49.2412 - val_mae: 0.5652
Epoch 15: early stopping
Restoring model weights from the end of the best epoch: 10.
25/25 ━━━━━━━━━━━━━━━━━━━━ 4s 161ms/step - cosine_similarity: 0.3938 - loss: 49.1186 - mae: 0.5741
Test loss: [48.90032196044922, 0.5724843144416809, 0.3910948634147644]


Model: "functional"  ->7
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                  ┃ Output Shape              ┃         Param # ┃ Connected to               ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ phoneme_input (InputLayer)    │ (None, 168)               │               0 │ -                          │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ embedding (Embedding)         │ (None, 168, 256)          │          10,752 │ phoneme_input[0][0]        │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ conv1d (Conv1D)               │ (None, 168, 256)          │         327,936 │ embedding[0][0]            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ layer_normalization           │ (None, 168, 256)          │             512 │ conv1d[0][0]               │
│ (LayerNormalization)          │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dropout (Dropout)             │ (None, 168, 256)          │               0 │ layer_normalization[0][0]  │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ add (Add)                     │ (None, 168, 256)          │               0 │ dropout[0][0],             │
│                               │                           │                 │ embedding[0][0]            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ conv1d_1 (Conv1D)             │ (None, 168, 256)          │         327,936 │ add[0][0]                  │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ layer_normalization_1         │ (None, 168, 256)          │             512 │ conv1d_1[0][0]             │
│ (LayerNormalization)          │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dropout_1 (Dropout)           │ (None, 168, 256)          │               0 │ layer_normalization_1[0][… │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ add_1 (Add)                   │ (None, 168, 256)          │               0 │ dropout_1[0][0], add[0][0] │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ conv1d_2 (Conv1D)             │ (None, 168, 256)          │         327,936 │ add_1[0][0]                │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ layer_normalization_2         │ (None, 168, 256)          │             512 │ conv1d_2[0][0]             │
│ (LayerNormalization)          │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dropout_2 (Dropout)           │ (None, 168, 256)          │               0 │ layer_normalization_2[0][… │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ add_2 (Add)                   │ (None, 168, 256)          │               0 │ dropout_2[0][0],           │
│                               │                           │                 │ add_1[0][0]                │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ not_equal_1 (NotEqual)        │ (None, 168)               │               0 │ phoneme_input[0][0]        │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ bidirectional (Bidirectional) │ (None, 168, 256)          │         394,240 │ add_2[0][0],               │
│                               │                           │                 │ not_equal_1[0][0]          │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dropout_3 (Dropout)           │ (None, 168, 256)          │               0 │ bidirectional[0][0]        │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ bidirectional_1               │ (None, 168, 256)          │         394,240 │ dropout_3[0][0],           │
│ (Bidirectional)               │                           │                 │ not_equal_1[0][0]          │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dropout_4 (Dropout)           │ (None, 168, 256)          │               0 │ bidirectional_1[0][0]      │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ bidirectional_2               │ (None, 168, 128)          │         164,352 │ dropout_4[0][0],           │
│ (Bidirectional)               │                           │                 │ not_equal_1[0][0]          │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dropout_5 (Dropout)           │ (None, 168, 128)          │               0 │ bidirectional_2[0][0]      │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ multi_head_attention          │ (None, 168, 128)          │         263,808 │ dropout_5[0][0],           │
│ (MultiHeadAttention)          │                           │                 │ dropout_5[0][0]            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ add_3 (Add)                   │ (None, 168, 128)          │               0 │ dropout_5[0][0],           │
│                               │                           │                 │ multi_head_attention[0][0] │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ layer_normalization_3         │ (None, 168, 128)          │             256 │ add_3[0][0]                │
│ (LayerNormalization)          │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ conv1d_transpose              │ (None, 336, 256)          │         164,096 │ layer_normalization_3[0][… │
│ (Conv1DTranspose)             │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ conv1d_transpose_1            │ (None, 672, 128)          │          98,432 │ conv1d_transpose[0][0]     │
│ (Conv1DTranspose)             │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ conv1d_transpose_2            │ (None, 1344, 128)         │          49,280 │ conv1d_transpose_1[0][0]   │
│ (Conv1DTranspose)             │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ crop_layer (CropLayer)        │ (None, 900, 128)          │               0 │ conv1d_transpose_2[0][0]   │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ mel_output (Conv1D)           │ (None, 900, 80)           │          10,320 │ crop_layer[0][0]           │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ conv1d_3 (Conv1D)             │ (None, 900, 80)           │          32,080 │ mel_output[0][0]           │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ batch_normalization           │ (None, 900, 80)           │             320 │ conv1d_3[0][0]             │
│ (BatchNormalization)          │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ conv1d_4 (Conv1D)             │ (None, 900, 80)           │          32,080 │ batch_normalization[0][0]  │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ batch_normalization_1         │ (None, 900, 80)           │             320 │ conv1d_4[0][0]             │
│ (BatchNormalization)          │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ conv1d_5 (Conv1D)             │ (None, 900, 80)           │          32,080 │ batch_normalization_1[0][… │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ batch_normalization_2         │ (None, 900, 80)           │             320 │ conv1d_5[0][0]             │
│ (BatchNormalization)          │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ conv1d_6 (Conv1D)             │ (None, 900, 80)           │          32,080 │ batch_normalization_2[0][… │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ batch_normalization_3         │ (None, 900, 80)           │             320 │ conv1d_6[0][0]             │
│ (BatchNormalization)          │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ conv1d_7 (Conv1D)             │ (None, 900, 80)           │          32,080 │ batch_normalization_3[0][… │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ batch_normalization_4         │ (None, 900, 80)           │             320 │ conv1d_7[0][0]             │
│ (BatchNormalization)          │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ refined_mel_output (Add)      │ (None, 900, 80)           │               0 │ mel_output[0][0],          │
│                               │                           │                 │ batch_normalization_4[0][… │
└───────────────────────────────┴───────────────────────────┴─────────────────┴────────────────────────────┘
 Total params: 2,697,120 (10.29 MB)
 Trainable params: 2,696,320 (10.29 MB)
 Non-trainable params: 800 (3.12 KB)
Epoch 1/50
328/328 ━━━━━━━━━━━━━━━━━━━━ 0s 1s/step - cosine_similarity: 0.3304 - loss: 61.3631 - mae: 0.8115      
Epoch 1: val_loss improved from inf to 54.08735, saving model to model/2/best_model_cnn.keras

Epoch 1: Learning rate is 0.000999
328/328 ━━━━━━━━━━━━━━━━━━━━ 381s 1s/step - cosine_similarity: 0.3305 - loss: 61.3426 - mae: 0.8112 - val_cosine_similarity: 0.3650 - val_loss: 54.0873 - val_mae: 0.6422
Epoch 2/50
328/328 ━━━━━━━━━━━━━━━━━━━━ 0s 1s/step - cosine_similarity: 0.3704 - loss: 51.6354 - mae: 0.5896  
Epoch 2: val_loss improved from 54.08735 to 51.25095, saving model to model/2/best_model_cnn.keras

Epoch 2: Learning rate is 0.000996
328/328 ━━━━━━━━━━━━━━━━━━━━ 403s 1s/step - cosine_similarity: 0.3704 - loss: 51.6339 - mae: 0.5896 - val_cosine_similarity: 0.3776 - val_loss: 51.2510 - val_mae: 0.6288
Epoch 3/50
328/328 ━━━━━━━━━━━━━━━━━━━━ 0s 1s/step - cosine_similarity: 0.3761 - loss: 50.6657 - mae: 0.5823  
Epoch 3: val_loss improved from 51.25095 to 50.61520, saving model to model/2/best_model_cnn.keras

Epoch 3: Learning rate is 0.000992
328/328 ━━━━━━━━━━━━━━━━━━━━ 408s 1s/step - cosine_similarity: 0.3761 - loss: 50.6646 - mae: 0.5823 - val_cosine_similarity: 0.3823 - val_loss: 50.6152 - val_mae: 0.6418
Epoch 4/50
328/328 ━━━━━━━━━━━━━━━━━━━━ 0s 1s/step - cosine_similarity: 0.3802 - loss: 49.9536 - mae: 0.5805  
Epoch 4: val_loss improved from 50.61520 to 50.07338, saving model to model/2/best_model_cnn.keras

Epoch 4: Learning rate is 0.000986
328/328 ━━━━━━━━━━━━━━━━━━━━ 399s 1s/step - cosine_similarity: 0.3802 - loss: 49.9528 - mae: 0.5805 - val_cosine_similarity: 0.3848 - val_loss: 50.0734 - val_mae: 0.6036
Epoch 5/50
328/328 ━━━━━━━━━━━━━━━━━━━━ 0s 1s/step - cosine_similarity: 0.3826 - loss: 49.5404 - mae: 0.5732  
Epoch 5: val_loss did not improve from 50.07338

Epoch 5: Learning rate is 0.000978
328/328 ━━━━━━━━━━━━━━━━━━━━ 404s 1s/step - cosine_similarity: 0.3826 - loss: 49.5398 - mae: 0.5731 - val_cosine_similarity: 0.3842 - val_loss: 50.3042 - val_mae: 0.6407
Epoch 6/50
328/328 ━━━━━━━━━━━━━━━━━━━━ 0s 1s/step - cosine_similarity: 0.3838 - loss: 49.3441 - mae: 0.5829  
Epoch 6: val_loss improved from 50.07338 to 49.94419, saving model to model/2/best_model_cnn.keras

Epoch 6: Learning rate is 0.000968
328/328 ━━━━━━━━━━━━━━━━━━━━ 398s 1s/step - cosine_similarity: 0.3838 - loss: 49.3435 - mae: 0.5829 - val_cosine_similarity: 0.3865 - val_loss: 49.9442 - val_mae: 0.5780
Epoch 7/50
328/328 ━━━━━━━━━━━━━━━━━━━━ 0s 1s/step - cosine_similarity: 0.3856 - loss: 49.0052 - mae: 0.5728  
Epoch 7: val_loss improved from 49.94419 to 49.68311, saving model to model/2/best_model_cnn.keras

Epoch 7: Learning rate is 0.000957
328/328 ━━━━━━━━━━━━━━━━━━━━ 400s 1s/step - cosine_similarity: 0.3856 - loss: 49.0046 - mae: 0.5728 - val_cosine_similarity: 0.3871 - val_loss: 49.6831 - val_mae: 0.5639
Epoch 8/50
328/328 ━━━━━━━━━━━━━━━━━━━━ 0s 1s/step - cosine_similarity: 0.3873 - loss: 48.7006 - mae: 0.5722  
Epoch 8: val_loss did not improve from 49.68311

Epoch 8: Learning rate is 0.000944
328/328 ━━━━━━━━━━━━━━━━━━━━ 400s 1s/step - cosine_similarity: 0.3873 - loss: 48.7001 - mae: 0.5722 - val_cosine_similarity: 0.3872 - val_loss: 49.7318 - val_mae: 0.5643
Epoch 9/50
328/328 ━━━━━━━━━━━━━━━━━━━━ 0s 1s/step - cosine_similarity: 0.3885 - loss: 48.4918 - mae: 0.5723  
Epoch 9: val_loss improved from 49.68311 to 49.38581, saving model to model/2/best_model_cnn.keras

Epoch 9: Learning rate is 0.000930
328/328 ━━━━━━━━━━━━━━━━━━━━ 396s 1s/step - cosine_similarity: 0.3885 - loss: 48.4913 - mae: 0.5723 - val_cosine_similarity: 0.3889 - val_loss: 49.3858 - val_mae: 0.5637
Epoch 10/50
328/328 ━━━━━━━━━━━━━━━━━━━━ 0s 1s/step - cosine_similarity: 0.3897 - loss: 48.2611 - mae: 0.5698  
Epoch 10: val_loss did not improve from 49.38581

Epoch 10: Learning rate is 0.000914
328/328 ━━━━━━━━━━━━━━━━━━━━ 424s 1s/step - cosine_similarity: 0.3897 - loss: 48.2606 - mae: 0.5698 - val_cosine_similarity: 0.3886 - val_loss: 49.4316 - val_mae: 0.5642
Epoch 11/50
328/328 ━━━━━━━━━━━━━━━━━━━━ 0s 1s/step - cosine_similarity: 0.3908 - loss: 48.0702 - mae: 0.5681  
Epoch 11: val_loss did not improve from 49.38581

Epoch 11: Learning rate is 0.000897
328/328 ━━━━━━━━━━━━━━━━━━━━ 394s 1s/step - cosine_similarity: 0.3908 - loss: 48.0698 - mae: 0.5681 - val_cosine_similarity: 0.3868 - val_loss: 49.7303 - val_mae: 0.5633
Epoch 12/50
328/328 ━━━━━━━━━━━━━━━━━━━━ 0s 1s/step - cosine_similarity: 0.3916 - loss: 47.9305 - mae: 0.5689  
Epoch 12: val_loss improved from 49.38581 to 49.06871, saving model to model/2/best_model_cnn.keras

Epoch 12: Learning rate is 0.000878
328/328 ━━━━━━━━━━━━━━━━━━━━ 397s 1s/step - cosine_similarity: 0.3916 - loss: 47.9301 - mae: 0.5689 - val_cosine_similarity: 0.3904 - val_loss: 49.0687 - val_mae: 0.5657
Epoch 13/50
328/328 ━━━━━━━━━━━━━━━━━━━━ 0s 1s/step - cosine_similarity: 0.3927 - loss: 47.7240 - mae: 0.5685  
Epoch 13: val_loss did not improve from 49.06871

Epoch 13: Learning rate is 0.000858
328/328 ━━━━━━━━━━━━━━━━━━━━ 403s 1s/step - cosine_similarity: 0.3927 - loss: 47.7235 - mae: 0.5685 - val_cosine_similarity: 0.3898 - val_loss: 49.2440 - val_mae: 0.5681
Epoch 14/50
328/328 ━━━━━━━━━━━━━━━━━━━━ 0s 1s/step - cosine_similarity: 0.3941 - loss: 47.4863 - mae: 0.5678  
Epoch 14: val_loss improved from 49.06871 to 49.01123, saving model to model/2/best_model_cnn.keras

Epoch 14: Learning rate is 0.000837
328/328 ━━━━━━━━━━━━━━━━━━━━ 403s 1s/step - cosine_similarity: 0.3941 - loss: 47.4860 - mae: 0.5678 - val_cosine_similarity: 0.3907 - val_loss: 49.0112 - val_mae: 0.5698
Epoch 15/50
328/328 ━━━━━━━━━━━━━━━━━━━━ 0s 1s/step - cosine_similarity: 0.3952 - loss: 47.3022 - mae: 0.5685  
Epoch 15: val_loss did not improve from 49.01123

Epoch 15: Learning rate is 0.000815
328/328 ━━━━━━━━━━━━━━━━━━━━ 398s 1s/step - cosine_similarity: 0.3952 - loss: 47.3020 - mae: 0.5685 - val_cosine_similarity: 0.3906 - val_loss: 49.1106 - val_mae: 0.5667
Epoch 16/50
328/328 ━━━━━━━━━━━━━━━━━━━━ 0s 1s/step - cosine_similarity: 0.3955 - loss: 47.2346 - mae: 0.5671  
Epoch 16: val_loss did not improve from 49.01123

Epoch 16: Learning rate is 0.000791
328/328 ━━━━━━━━━━━━━━━━━━━━ 398s 1s/step - cosine_similarity: 0.3955 - loss: 47.2343 - mae: 0.5671 - val_cosine_similarity: 0.3900 - val_loss: 49.1563 - val_mae: 0.5717
Epoch 17/50
328/328 ━━━━━━━━━━━━━━━━━━━━ 0s 1s/step - cosine_similarity: 0.3970 - loss: 46.9775 - mae: 0.5664  
Epoch 17: val_loss did not improve from 49.01123

Epoch 17: Learning rate is 0.000767
328/328 ━━━━━━━━━━━━━━━━━━━━ 400s 1s/step - cosine_similarity: 0.3970 - loss: 46.9772 - mae: 0.5664 - val_cosine_similarity: 0.3890 - val_loss: 49.4032 - val_mae: 0.5699
Epoch 18/50
328/328 ━━━━━━━━━━━━━━━━━━━━ 0s 1s/step - cosine_similarity: 0.3977 - loss: 46.8504 - mae: 0.5643  
Epoch 18: val_loss did not improve from 49.01123

Epoch 18: Learning rate is 0.000742
328/328 ━━━━━━━━━━━━━━━━━━━━ 400s 1s/step - cosine_similarity: 0.3977 - loss: 46.8501 - mae: 0.5643 - val_cosine_similarity: 0.3893 - val_loss: 49.3584 - val_mae: 0.5809
Epoch 19/50
328/328 ━━━━━━━━━━━━━━━━━━━━ 0s 1s/step - cosine_similarity: 0.3983 - loss: 46.7402 - mae: 0.5653  
Epoch 19: val_loss did not improve from 49.01123

Epoch 19: Learning rate is 0.000716
328/328 ━━━━━━━━━━━━━━━━━━━━ 398s 1s/step - cosine_similarity: 0.3984 - loss: 46.7400 - mae: 0.5653 - val_cosine_similarity: 0.3896 - val_loss: 49.3320 - val_mae: 0.5760
Epoch 19: early stopping
Restoring model weights from the end of the best epoch: 14.
25/25 ━━━━━━━━━━━━━━━━━━━━ 8s 337ms/step - cosine_similarity: 0.3941 - loss: 49.1649 - mae: 0.5700
Test loss: [48.927616119384766, 0.5690494179725647, 0.39150410890579224]


==============

 Total params: 12,707,553 (48.48 MB)
 Trainable params: 12,706,753 (48.47 MB)
 Non-trainable params: 800 (3.12 KB)
Epoch 1/50
328/328 ━━━━━━━━━━━━━━━━━━━━ 0s 4s/step - cosine_similarity: 0.1148 - loss: 0.8618 - mae: 0.3263       
Epoch 1: val_loss improved from inf to 0.69199, saving model to model/2/best_model_cnn_9f.keras

Epoch 1: Learning rate is 0.000999
328/328 ━━━━━━━━━━━━━━━━━━━━ 1360s 4s/step - cosine_similarity: 0.1148 - loss: 0.8613 - mae: 0.3261 - val_cosine_similarity: 0.1278 - val_loss: 0.6920 - val_mae: 0.2804
Epoch 2/50
109/328 ━━━━━━━━━━━━━━━━━━━━ 16:56 5s/step - cosine_similarity: 0.1381 - loss: 0.6540 - mae: 0.2251^CTraceback (most recent call last):


 Total params: 12,707,553 (48.48 MB)
 Trainable params: 12,706,753 (48.47 MB)
 Non-trainable params: 800 (3.12 KB)
Epoch 1/50
328/328 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - cosine_similarity: 0.0707 - loss: 0.9099 - mae: 0.3224       
Epoch 1: val_loss improved from inf to 0.73706, saving model to model/2/best_model_cnn_9f.keras

Epoch 1: Learning rate is 0.000999
328/328 ━━━━━━━━━━━━━━━━━━━━ 743s 2s/step - cosine_similarity: 0.0707 - loss: 0.9094 - mae: 0.3222 - val_cosine_similarity: 0.0807 - val_loss: 0.7371 - val_mae: 0.2411
Epoch 2/50
221/328 ━━━━━━━━━━━━━━━━━━━━ 4:04 2s/step - cosine_similarity: 0.0940 - loss: 0.6970 - mae: 0.2161^CTraceback (most recent call last):