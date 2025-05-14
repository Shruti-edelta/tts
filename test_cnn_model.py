import tensorflow as tf
from tensorflow.keras.metrics import CosineSimilarity
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, Conv1D, Conv1DTranspose, Bidirectional, LSTM, LayerNormalization, Dense, Add, Activation
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ast
from acoustic.text_preprocess import G2PConverter
import soundfile as sf
import librosa
from keras.saving import register_keras_serializable
import keras
keras.config.enable_unsafe_deserialization()

# @tf.keras.utils.register_keras_serializable()
@register_keras_serializable()
class CropLayer(tf.keras.layers.Layer):
    def __init__(self, target_length, **kwargs):
        super().__init__(**kwargs)
        self.length = target_length

    def call(self, inputs):
        return inputs[:, :self.length, :]

    def get_config(self):
        config = super().get_config()
        config.update({'target_length': self.length})
        return config
    
# @tf.keras.utils.register_keras_serializable()
# class ResidualBlock(tf.keras.layers.Layer):
    # def __init__(self, channels, kernel_size, **kwargs):
    #     super().__init__(**kwargs)
    #     self.channels = channels
    #     self.kernel_size = kernel_size
    #     self.conv1 = Conv1D(channels, kernel_size, padding='same', dilation_rate=1)
    #     self.ln1 = LayerNormalization()
    #     self.act1 = Activation('relu')
    #     self.conv2 = Conv1D(channels, kernel_size, padding='same', dilation_rate=1)
    #     self.ln2 = LayerNormalization()

    # def call(self, x, training=False):
    #     residual = x
    #     x = self.conv1(x)
    #     x = self.ln1(x)
    #     x = self.act1(x)
    #     x = self.conv2(x)
    #     x = self.ln2(x)
    #     return Add()([x, residual])

    def get_config(self):
        config = super().get_config()
        config.update({
            "channels": self.channels,
            "kernel_size": self.kernel_size,
        })
        return config


# @tf.keras.saving.register_keras_serializable()
# @tf.keras.utils.register_keras_serializable()
@register_keras_serializable()
# --- Simplified Additive Attention ---
class AdditiveAttention(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_query = tf.keras.layers.Dense(units)
        self.W_values = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values, mask=None):
        query = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W_query(query) + self.W_values(values)))
        score = tf.squeeze(score, axis=-1)

        if mask is not None:
            score += (1.0 - tf.cast(mask, tf.float32)) * -1e9

        attention_weights = tf.nn.softmax(score, axis=-1)
        context = tf.matmul(tf.expand_dims(attention_weights, 1), values)
        context = tf.squeeze(context, axis=1)
        return context, attention_weights

@tf.keras.utils.register_keras_serializable()
# @tf.keras.saving.register_keras_serializable()
class AttentionContextLayer(tf.keras.layers.Layer):
    def __init__(self, units, input_length, **kwargs):
        super(AttentionContextLayer, self).__init__(**kwargs)
        self.units = units
        self.input_length = input_length

    def build(self, input_shape):
        self.attention = AdditiveAttention(units=self.units)
        super(AttentionContextLayer, self).build(input_shape)

    def call(self, inputs):
        query, values, mask = inputs
        context_vector, _ = self.attention(query, values, mask=mask)
        
        # Expand context and concatenate with sequence
        context_expanded = tf.expand_dims(context_vector, axis=1)
        context_expanded = tf.tile(context_expanded, [1, self.input_length, 1])
        return tf.concat([values, context_expanded], axis=-1)
    
@tf.keras.utils.register_keras_serializable()
def positional_encoding(length, depth):
    depth = depth // 2  # keep it tensor-friendly
    positions = tf.cast(tf.range(length)[:, tf.newaxis], dtype=tf.float32)
    depths = tf.cast(tf.range(depth)[tf.newaxis, :], dtype=tf.float32)
    angle_rates = 1 / (10000 ** (depths / tf.cast(depth, tf.float32)))
    angle_rads = positions * angle_rates
    pos_encoding = tf.concat([tf.sin(angle_rads), tf.cos(angle_rads)], axis=-1)
    return pos_encoding

def mel_to_audio(mel_spectrogram, sr=22050):
    mel_inv = librosa.feature.inverse.mel_to_audio(mel_spectrogram, sr=sr)
    return mel_inv

# def compile_model(model):
#     optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
#     def combined_loss(y_true, y_pred):
#         mse = tf.reduce_mean(tf.square(y_true - y_pred))
#         cos_sim = tf.reduce_mean(tf.keras.losses.cosine_similarity(y_true, y_pred, axis=-1))
#         return mse + 0.1 * cos_sim
    
#     model.compile(optimizer=optimizer,
#                   loss=combined_loss,
#                   metrics=['mae', CosineSimilarity(axis=-1)])
#     return model

@register_keras_serializable()
def add_positional_encoding(x):
    seq_len = tf.shape(x)[1]
    depth = tf.shape(x)[2]
    pos_encoding = positional_encoding(seq_len, depth)
    return x + pos_encoding[tf.newaxis, :, :]

# best_model = tf.keras.models.load_model(
#     "model/2/best_model_cnn.keras",
#     compile=False,
#     custom_objects={
#         'CropLayer': CropLayer,
#         'ResidualBlock': ResidualBlock,
#         'AttentionContextLayer': AttentionContextLayer,
#         'AdditiveAttention': AdditiveAttention,
#         'add_positional_encoding': add_positional_encoding,
#         'positional_encoding': positional_encoding,  # if used inside
#     }
# )

# best_model = tf.keras.models.load_model("model/2/best_model_cnn.keras", compile=True,custom_objects={'CropLayer': CropLayer,'ResidualBlock':ResidualBlock,'positional_encoding':positional_encoding})
best_model = tf.keras.models.load_model("model/2/best_model_cnn.keras", compile=False,custom_objects={'CropLayer': CropLayer,'AttentionContextLayer': AttentionContextLayer})
# best_model = compile_model(best_model)

g2p = G2PConverter("model/1/1model_cnn.keras")

text = "This is a test"
text = "However, it now laid down in plain language and with precise details the requirements of a good jail system."
# text = "These bankers wishing for more specific information"
# text="However it now laid down in plain language and with precise details the requirements of a good jail system"
# text = "The first step is to identify the problem and its root cause."
# text = "The second step is to develop a plan to address the problem."
# text = "He likewise indicated he was disenchanted with Russia"
# text="and into his possible motivation for the assassination"
phonemes = g2p.predict(text)  # use your G2P model
print(phonemes)
padded = pad_sequences([phonemes], maxlen=256, padding='post')[0]
# print(padded)
input_tensor = tf.convert_to_tensor([padded], dtype=tf.int32)

predicted_mel = best_model.predict(input_tensor,verbose=0)[0]
print("predicted_mel(input): ",predicted_mel)

# Plot it
plt.imshow(predicted_mel.T, aspect='auto', origin='lower')
plt.title("Predicted Mel for custom sentence")
plt.show()

audio=mel_to_audio(predicted_mel.T)
print("audio: ",audio)
# sf.write('audio/cnn/1best_model_t1.wav', audio, 22050) 

# =================== test_data ====================

def preprocess_testdata(texts, mel_spectrograms, input_length=256,mel_max_len=1024):
    # def pad_or_truncate(mel, max_len=mel_max_len):
    #     T, D = mel.shape
    #     if T > max_len:
    #         print("============== truncate ========")
    #         mel = mel[:max_len, :]
    #     elif T < max_len:
    #         pad_width = max_len - T
    #         mel = np.pad(mel, ((0, pad_width), (0, 0)), mode='constant')
    #     return mel
    print(texts[0])
    # padded_texts = pad_sequences(texts, maxlen=input_length, padding='post')
    # print(padded_texts[0])
    mel_spectrograms = [np.load(mel) for mel in mel_spectrograms]
    mel_spectrograms = np.array(mel_spectrograms)
    return texts, mel_spectrograms

df_test = pd.read_csv('dataset/acoustic_dataset/test.csv')
phoneme_seq = df_test['Phoneme_text'].apply(ast.literal_eval).values
mel_spectrograms_test = df_test['Read_npy'].values

test_text, test_mel = preprocess_testdata(phoneme_seq, mel_spectrograms_test)
x_test = np.expand_dims(test_text[0], axis=0)  # Adding a batch dimension
y_true = np.expand_dims(test_mel[0], axis=0)  # Adding a batch dimension)
# x_test=np.array(ast.literal_eval(x_test.decode()), dtype=np.int32)
# test_loss = model.evaluate(x_test, y_true)
# print(f"Test loss: {test_loss}")
print(x_test)
y_pred = best_model.predict(x_test)

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

are_equal = np.allclose(y_true[0].T, y_pred[0].T)
print("is equal? ", are_equal)
are_equal = np.allclose(predicted_mel.T, y_pred[0].T)
print("is equal? ", are_equal)


'''
[[16  5 11 35 12 41 17 31 41 23  5 41 21 13  9 41  9  5 23 41 17 23 41 27
  21 13 23 41 21  2 24 15 36  3 19 41  2 23  9 41 36 17 32 41 27 28 17 29
   6 38 41  9 17 31 13 21 38 41 10 41 28 17 20 36  6 28 22  3 23 31 29  1
  14 41  3 41 15 33  9 41 19 13 21 41 29 17 29 31  3 22 41  0  0  0  0  0
   0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
   0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
   0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]]

y_true mel : [[[-0.35748488 -0.333911   -0.971453   ... -1.7949594  -1.6517982
   -1.7949594 ]
  [-1.5814261  -1.3502264  -1.1194234  ... -1.6014129  -0.94845766
   -1.3441344 ]
  [-1.74118    -1.2230829  -1.0586957  ... -1.7006984  -1.1716251
   -1.2302768 ]
  ...
  [ 0.          0.          0.         ...  0.          0.
    0.        ]
  [ 0.          0.          0.         ...  0.          0.
    0.        ]
  [ 0.          0.          0.         ...  0.          0.
    0.        ]]]
Audio shape: [-0.02486947 -0.0299985  -0.02318713 ...  0.          0.
  0.        ]

  
Predicted mel spectrograms: [[[-1.0942737e+00 -9.4022268e-01 -6.4864475e-01 ... -1.2555252e+00
   -1.3100214e+00 -1.2505031e+00]
  [-7.6711386e-01 -5.1606470e-01 -9.9363983e-02 ... -1.1566864e+00
   -9.9654776e-01 -8.6031651e-01]
  [-6.6404802e-01 -3.3171162e-01  7.3100731e-02 ... -7.8329396e-01
   -6.7283195e-01 -5.5221170e-01]
  ...
  [-1.3430046e-02 -7.9706665e-03 -7.8488514e-04 ... -7.4825846e-03
   -6.4588552e-03 -7.0636459e-03]
  [-1.3430046e-02 -7.9706665e-03 -7.8488514e-04 ... -7.4825846e-03
   -6.4588552e-03 -7.0636459e-03]
  [-1.3430046e-02 -7.9706665e-03 -7.8488514e-04 ... -7.4825846e-03
   -6.4588552e-03 -7.0636459e-03]]]
Audio shape: [-0.00195196  0.00055073  0.00224646 ... -0.00109117 -0.00098302
 -0.00084891]


 2

 Predicted mel spectrograms: [[[-1.2009586e+00 -8.5913950e-01 -6.1691529e-01 ... -1.3585203e+00
   -1.3233691e+00 -1.2843829e+00]
  [-8.0032873e-01 -4.7144014e-01 -1.4014891e-01 ... -1.0906498e+00
   -1.0195835e+00 -8.5635000e-01]
  [-7.2708976e-01 -2.8212625e-01  1.1320693e-01 ... -8.3038831e-01
   -6.9473577e-01 -5.3810650e-01]
  ...
  [-1.6177472e-02 -9.5593492e-03 -7.3251128e-04 ... -6.4035384e-03
   -4.9288105e-03 -6.5306770e-03]
  [-1.6177472e-02 -9.5593492e-03 -7.3251128e-04 ... -6.4035384e-03
   -4.9288105e-03 -6.5306770e-03]
  [-1.6177472e-02 -9.5593492e-03 -7.3251128e-04 ... -6.4035384e-03
   -4.9288105e-03 -6.5306770e-03]]]
Audio shape: [-0.00134643 -0.00328612 -0.00461446 ... -0.00021091 -0.00019301
 -0.00016783]


3

Predicted mel spectrograms: [[[-0.9686726  -0.8546181  -0.5739881  ... -1.2384353  -1.1967343
   -1.2421371 ]
  [-0.861203   -0.48256212 -0.19080184 ... -1.0651928  -0.8650058
   -0.8312518 ]
  [-0.78879756 -0.32886684 -0.01956961 ... -0.73472375 -0.55934644
   -0.4900487 ]
  ...
  [-0.0158001  -0.01063146 -0.00129643 ... -0.00747786 -0.00575483
   -0.00691452]
  [-0.0158001  -0.01063146 -0.00129643 ... -0.00747786 -0.00575483
   -0.00691452]
  [-0.0158001  -0.01063146 -0.00129643 ... -0.00747786 -0.00575483
   -0.00691452]]]
Audio shape: [0.00092895 0.00221266 0.00343626 ... 0.00130336 0.00106929 0.00084212]

4

Predicted mel spectrograms: [[[-1.2697196e+00 -1.0351181e+00 -7.6236641e-01 ... -1.8035821e+00
   -1.6279957e+00 -1.5927328e+00]
  [-9.5202363e-01 -5.0636697e-01 -2.3445594e-01 ... -1.5623714e+00
   -1.3567277e+00 -1.2428184e+00]
  [-6.6553956e-01 -2.4661896e-01  1.4726552e-01 ... -1.1034539e+00
   -8.2847786e-01 -7.3401594e-01]
  ...
  [-2.6475713e-03 -1.7747730e-03  2.2484651e-03 ... -1.4145132e-03
   -1.4361124e-03 -5.8344752e-04]
  [-3.3930987e-03 -7.2995573e-04  1.9293027e-03 ... -3.9030276e-03
   -3.2363962e-03 -2.4776794e-03]
  [-5.2314475e-03 -3.8587451e-03  1.9132150e-03 ... -1.7991513e-03
   -1.5207268e-03 -5.7959743e-04]]]
Audio shape: [-0.00253677 -0.00122477  0.00018968 ... -0.00138854 -0.00021962
  0.00126286]


5

Predicted mel spectrograms: [[[-8.5966957e-01 -5.2093422e-01 -1.1528850e-02 ... -1.5012910e+00
   -1.4558160e+00 -1.5009389e+00]
  [-1.2920334e+00 -8.2826138e-01 -5.7485932e-01 ... -1.8522505e+00
   -1.5655127e+00 -1.5850322e+00]
  [-1.5673876e+00 -1.0751551e+00 -7.2794306e-01 ... -1.8033912e+00
   -1.3665246e+00 -1.3062834e+00]
  ...
  [-7.4420962e-04  2.8079748e-04 -9.7121205e-03 ...  7.8832433e-03
    9.6754991e-03  8.2132742e-03]
  [-1.0144088e-02 -5.3444440e-03 -8.0314055e-03 ...  5.5557285e-03
    2.7847360e-03  1.8905029e-03]
  [-6.3294386e-03 -9.1997776e-03  1.3563782e-03 ... -9.5414929e-05
   -8.6253919e-03 -1.8696580e-03]]]
Audio shape: [ 0.00204793 -0.00407805 -0.01883351 ...  0.0013487  -0.00233874
  0.00053781]

6

Predicted mel spectrograms: [[[-0.7102686  -0.3694312   0.07390867 ... -1.4608283  -1.4243273
   -1.4648502 ]
  [-1.3074489  -0.8055339  -0.36974433 ... -1.77879    -1.5921308
   -1.3101654 ]
  [-1.440377   -0.81295    -0.5003359  ... -1.7447672  -1.4868308
   -1.1641247 ]
  ...
  [-0.84673333 -0.19151342  0.48570502 ... -0.4250868  -0.39026335
   -0.41393733]
  [-0.8130238  -0.20483732  0.4194604  ... -0.3089788  -0.26853734
   -0.247667  ]
  [-0.7592108  -0.21373747  0.22882824 ... -0.28926045 -0.21911627
   -0.20315792]]]
Audio shape: [-0.02047079 -0.04644066 -0.06607533 ... -0.00173608  0.00049463
  0.00123241]

7

Predicted mel spectrograms: [[[-0.93174404 -0.36247134  0.08564308 ... -1.4199073  -1.2703884
   -1.2984442 ]
  [-1.2053775  -0.7912511  -0.3265379  ... -1.8731132  -1.7248386
   -1.5640322 ]
  [-1.5504576  -1.0215187  -0.49829102 ... -1.8976998  -1.6231242
   -1.4004294 ]
  ...
  [-0.75962687 -0.12373036  0.5294446  ... -0.4474444  -0.41511938
   -0.37434313]
  [-0.85941654 -0.08833456  0.5317384  ... -0.42805913 -0.43755516
   -0.38005462]
  [-1.5704174  -0.12444675  0.52343625 ... -0.42155084 -0.38709697
   -0.346876  ]]]
Audio shape: [-0.01891241 -0.02574675 -0.02302453 ... -0.02481634  0.00624315
 -0.01898777]

# 8

Predicted mel spectrograms: [[[-0.77485865 -0.3067076   0.08863026 ... -1.2574992  -1.2146859
   -1.1646237 ]
  [-1.1192801  -0.75253755 -0.35240936 ... -1.6634479  -1.4593673
   -1.4546597 ]
  [-1.3271168  -0.8241435  -0.29537034 ... -1.7888207  -1.5450888
   -1.2852311 ]
  ...
  [ 0.26042408 -0.17452905 -1.1127672  ...  1.1444936   1.2012577
    1.1150634 ]
  [-1.2173189  -0.3151678  -1.0631738  ...  1.3177809   1.1500137
    1.0967225 ]
  [ 0.07729082 -0.32260653 -1.5151137  ...  1.4721155   1.6426885
    1.5880184 ]]]
Audio shape: [-0.00838447 -0.01665835 -0.02645783 ... -0.0919712  -0.03599525
 -0.00465614]

9

Predicted mel spectrograms: [[[-8.7151414e-01 -5.0691551e-01  8.6330235e-02 ... -1.2747396e+00
   -1.2457026e+00 -1.1284213e+00]
  [-1.2433890e+00 -7.5094330e-01 -5.7447791e-01 ... -1.5560806e+00
   -1.5494072e+00 -1.4262242e+00]
  [-1.4646366e+00 -9.4799143e-01 -4.5513886e-01 ... -1.6990732e+00
   -1.5191672e+00 -1.3990088e+00]
  ...
  [-6.8287328e-02 -1.7769441e-02 -2.5082707e-02 ... -7.2751343e-03
    4.0273927e-04 -1.8968463e-02]
  [-3.5731271e-02 -3.1512305e-02 -3.7183344e-02 ... -3.6923796e-02
   -3.9669964e-03 -9.2172027e-03]
  [-2.1397308e-02  2.1643201e-03 -1.6757131e-02 ... -4.3258965e-03
    1.7440250e-02 -4.4536591e-04]]]
Audio shape: [ 0.01021739  0.02522671  0.03560406 ... -0.00852837 -0.01098674
  0.01808774]

'''





# =============================================

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


# =============================================


# def build_acoustic_model(vocab_size, input_length, mel_dim=80, mel_len=900, embed_dim=256):
#     inputs = layers.Input(shape=(input_length,), name='phoneme_input')
#     x = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)(inputs)

#     # Initial Reshape
#     x = layers.Reshape((input_length, embed_dim))(x)

#     # Conv1D Stack with Residuals
#     for _ in range(3):  # fewer residual blocks = better for CPU training
#         residual = x
#         x = layers.Conv1D(filters=256, kernel_size=5, padding='same', activation='relu')(x)
#         x = layers.BatchNormalization()(x)
#         x = layers.Dropout(0.2)(x)
#         x = layers.Add()([x, residual])  # residual connection

#     # BiLSTM Layer for sequence modeling
#     x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)

#     # Optional attention (lightweight)
#     attention = layers.Dense(1, activation='tanh')(x)
#     attention = layers.Softmax(axis=1)(attention)
#     x = layers.Multiply()([x, attention])

#     # Upsampling stack
#     x = layers.Conv1DTranspose(256, kernel_size=5, strides=2, padding='same', activation='relu')(x)
#     x = layers.Conv1DTranspose(128, kernel_size=3, strides=2, padding='same', activation='relu')(x)
#     x = layers.Conv1DTranspose(128, kernel_size=3, strides=2, padding='same', activation='relu')(x)
#     x = layers.Conv1DTranspose(64, kernel_size=3, strides=2, padding='same', activation='relu')(x)
#     x = layers.Conv1DTranspose(64, kernel_size=3, strides=2, padding='same', activation='relu')(x)

#     # Crop to mel length
#     x = layers.Lambda(lambda t: t[:, :mel_len, :])(x)

#     # Output layer
#     mel_output = layers.Conv1D(mel_dim, kernel_size=1, activation='linear', name="mel_output")(x)

#     model = tf.keras.Model(inputs=inputs, outputs=mel_output)
#     return model


# =============================================


# import tensorflow as tf
# from tensorflow.keras import layers, Model

# @tf.keras.utils.register_keras_serializable()
# class CropLayer(layers.Layer):
#     def __init__(self, target_length, **kwargs):
#         super().__init__(**kwargs)
#         self.target_length = target_length

#     def call(self, inputs):
#         return inputs[:, :self.target_length, :]

#     def get_config(self):
#         config = super().get_config()
#         config.update({'target_length': self.target_length})
#         return config

# def build_acoustic_model(vocab_size, input_length, mel_dim=80, mel_len=900, embed_dim=256):
#     # Input layer
#     inputs = layers.Input(shape=(input_length,), name='phoneme_input')

#     # Embedding with masking
#     embedding = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, mask_zero=True)
#     x = embedding(inputs)
#     mask = embedding.compute_mask(inputs)

#     # Residual CNN blocks
#     for i in range(3):
#         residual = x
#         x = layers.Conv1D(filters=256, kernel_size=5, padding='same', dilation_rate=2, activation='relu')(x)
#         x = layers.LayerNormalization()(x)
#         x = layers.Dropout(0.2)(x)
#         x = layers.Add()([x, residual])

#     # BiLSTM with mask
#     x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x, mask=mask)
#     x = layers.Dropout(0.3)(x)

#     # Attention mechanism
#     score = layers.Dense(128, activation='tanh')(x)
#     attention_weights = layers.Dense(1)(score)
#     attention_weights = tf.nn.softmax(attention_weights, axis=1)
#     context_vector = tf.reduce_sum(x * attention_weights, axis=1, keepdims=True)

#     # Repeat context to match temporal resolution for decoder
#     repeated_context = layers.RepeatVector(input_length)(context_vector)

#     # Upsampling decoder
#     x = repeated_context
#     x = layers.Conv1DTranspose(256, kernel_size=5, strides=2, padding='same', activation='relu')(x)
#     x = layers.Conv1DTranspose(128, kernel_size=3, strides=2, padding='same', activation='relu')(x)
#     x = layers.Conv1DTranspose(128, kernel_size=3, strides=2, padding='same', activation='relu')(x)

#     x = CropLayer(mel_len)(x)

#     # Final mel-spectrogram projection
#     mel_output = layers.Conv1D(mel_dim, kernel_size=1, activation='linear', name="mel_output")(x)

#     # PostNet-like refinement
#     refinement = layers.Conv1D(mel_dim, kernel_size=5, padding='same', activation='tanh')(mel_output)
#     mel_output = layers.Add()([mel_output, refinement])

#     # Build model
#     model = Model(inputs=inputs, outputs=mel_output)
#     return model
