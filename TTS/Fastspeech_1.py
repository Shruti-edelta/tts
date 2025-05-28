import tensorflow as tf
from tensorflow.keras import layers, Model

class PhonemeEmbedding(layers.Layer):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = layers.Embedding(vocab_size, embed_dim)

    def call(self, x):
        return self.embedding(x)

class Encoder(layers.Layer):
    def __init__(self, hidden_dim):
        super().__init__()
        self.conv1 = layers.Conv1D(hidden_dim, 5, padding='same',dilation_rate=2, activation='relu')
        self.norm1 = layers.LayerNormalization()
        self.dp_o1 = layers.Dropout(0.2)
        self.conv2 = layers.Conv1D(hidden_dim, 5, padding='same', dilation_rate=2,activation='relu')
        self.norm2 = layers.LayerNormalization()
        self.dp_o2 = layers.Dropout(0.2)
        self.bi_lstm1 = layers.Bidirectional(layers.LSTM(512, return_sequences=True,dropout=0.3))
        self.bi_lstm2 = layers.Bidirectional(layers.LSTM(512, return_sequences=True,dropout=0.3))
        self.bi_lstm3 = layers.Bidirectional(layers.LSTM(128, return_sequences=True,dropout=0.3))

    def call(self, x):
        x = self.bi_lstm1(x)
        x = self.bi_lstm2(x)
        x = self.bi_lstm3(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.dp_o1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.dp_o2(x)
        # x = self.dp_o1(self.norm1(self.conv1(x)))
        # x = self.dp_o2(self.norm2(self.conv2(x)))
        
        return x

class DurationPredictor(layers.Layer):
    def __init__(self, hidden_dim):
        super().__init__()
        self.conv1 = layers.Conv1D(hidden_dim, 3, padding='same', activation='relu')
        self.norm1 = layers.LayerNormalization()
        self.conv2 = layers.Conv1D(hidden_dim, 3, padding='same', activation='relu')
        self.norm2 = layers.LayerNormalization()
        self.conv3 = layers.Conv1D(hidden_dim, 3, padding='same', activation='relu')
        self.norm3 = layers.LayerNormalization()
        self.out = layers.Dense(1)

    def call(self, x):
        x = self.norm1(self.conv1(x))
        x = self.norm2(self.conv2(x))
        x = self.norm3(self.conv3(x))
        x=self.out(x)
        return x

class LengthRegulator(layers.Layer):
    def call(self, x, durations):
        durations = tf.cast(tf.round(tf.exp(durations)), tf.int32)
        reps = tf.repeat(tf.range(tf.shape(x)[1]), durations[0])
        expanded = tf.gather(tf.repeat(x[0], durations[0], axis=0), reps)
        return tf.expand_dims(expanded, 0)

class Decoder(layers.Layer):
    def __init__(self, hidden_dim):
        super().__init__()
        self.bi_lstm = layers.Bidirectional(layers.LSTM(hidden_dim, return_sequences=True))
        self.dense = layers.Dense(80)  # 80-dim mel spectrogram

    def call(self, x):
        x = self.bi_lstm(x)
        return self.dense(x)


class post_net(layers.Layer):
    def __init__(self,mel_dim):
        super().__init__() 
        self.activation='tanh'
        self.post_conv1=layers.Conv1D(mel_dim, kernel_size=5, padding='same', activation=self.activation)
        self.bnorm1=layers.BatchNormalization()
        self.dpo1=layers.Dropout(0.1)
        self.mel=layers.Add(name="refined_mel_output")

    def call(self,postnet):
        for i in range(5):
            self.activation = 'tanh' if i < 4 else None
            postnet = self.post_conv1(postnet)
            if i < 4:
                postnet = self.bnorm1(postnet)
                postnet = self.dpo1(postnet)
        return postnet       
        

class FastSpeechAcousticModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim=256, mel_dim=80, hidden_dim=256,**kwargs):
        super().__init__(**kwargs)

        # self.inputs = tf.keras.layers.Input(shape=(None,), name='phoneme_input')
        print(vocab_size)
        self.embed = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True)
        self.encoder = Encoder(hidden_dim)  
        self.duration_predictor = DurationPredictor(hidden_dim)
        self.length_regulator = LengthRegulator()
        # self.decoder = Decoder(hidden_dim)
        self.mel_out=layers.Conv1D(mel_dim, kernel_size=1, activation='linear', name="mel_output")
        self.post_net=post_net(mel_dim)
        self.mel=layers.Add(name="refined_mel_output")
        # ✅ THIS WAS MISSING
        # self.mel_linear = tf.keras.layers.Dense(mel_dim)


    def call(self, inputs):
        phoneme_ids, durations = inputs  # unpack tuple
        # input=self.inputs
        x = self.embed(phoneme_ids)  # [B, T, E]
        # Optionally: append or use durations in a duration predictor
        x = self.encoder(x)
        durations_pred = self.duration_predictor(x)
        x = self.length_regulator(x, durations)  # Use provided durations

        mel_output=self.mel_out(x)
        postnet=mel_output
        # mel_out = self.mel_linear(x)
        postnet = self.post_net(postnet)
        mel_out=self.mel([mel_output,postnet])

        return mel_out

    # def call(self, inputs, durations=None, training=False):
    #     x = self.embed(inputs)
    #     x = self.encoder(x)
    #     if training:
    #         x = self.length_regulator(x, durations)
    #     else:
    #         duration_preds = self.duration_predictor(x)
    #         x = self.length_regulator(x, duration_preds)
    #     mel = self.decoder(x)
    #     return mel
    
    # def call(self, inputs):
    #     phoneme_ids, durations = inputs  # unpack tuple
    #     input=self.inputs
    #     x = self.embed(input)  # [B, T, E]
    #     # Optionally: append or use durations in a duration predictor
    #     x = self.encoder(x)
    #     durations_pred = self.duration_predictor(x)
    #     x = self.length_regulator(x, durations)  # Use provided durations
    #     x = self.decoder(x)
    #     mel_out = self.mel_linear(x)

    #     return mel_out






