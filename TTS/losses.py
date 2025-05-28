import tensorflow as tf

class TotalAcousticLoss(tf.keras.losses.Loss):
    def __init__(self, mel_weight=1.0, sc_weight=1.0, log_weight=1.0, name="total_acoustic_loss"):
        super().__init__(name=name)
        self.mel_weight = mel_weight
        self.sc_weight = sc_weight
        self.log_weight = log_weight

    def call(self, y_true, y_pred):
        mel = tf.reduce_mean(tf.square(y_true - y_pred), axis=[1, 2])
        sc = self.spectral_convergence(y_true, y_pred)
        logmel = self.log_mel_loss(y_true, y_pred)
        return self.mel_weight * mel + self.sc_weight * sc + self.log_weight * logmel

    def spectral_convergence(self, y_true, y_pred):
        diff = tf.norm(y_true - y_pred, ord='fro', axis=[1, 2])
        denom = tf.norm(y_true, ord='fro', axis=[1, 2])
        return tf.reduce_mean(diff / (denom + 1e-6))

    def log_mel_loss(self, y_true, y_pred):
        log_y_true = tf.math.log(y_true + 1e-5)
        log_y_pred = tf.math.log(y_pred + 1e-5)
        return tf.reduce_mean(tf.abs(log_y_true - log_y_pred), axis=[1, 2])




# import tensorflow as tf

# def mse_loss(y_true, y_pred):
#     return tf.reduce_mean(tf.square(y_true - y_pred))

# def spectral_convergence(y_true, y_pred):
#     sc = tf.norm(y_true - y_pred, ord='fro', axis=[-2, -1]) / tf.norm(y_true, ord='fro', axis=[-2, -1])
#     return tf.reduce_mean(sc)

# def log_mel_loss(y_true, y_pred):
#     y_true_log = tf.math.log(tf.clip_by_value(y_true, 1e-5, tf.reduce_max(y_true)))
#     y_pred_log = tf.math.log(tf.clip_by_value(y_pred, 1e-5, tf.reduce_max(y_pred)))
#     return tf.reduce_mean(tf.abs(y_true_log - y_pred_log))

# def duration_loss(y_true, y_pred):
#     return tf.reduce_mean(tf.square(y_true - y_pred))

# def total_acoustic_loss(mel_true, mel_pred, duration_true=None, duration_pred=None, mel_weight=1.0, sc_weight=1.0, log_weight=1.0, duration_weight=1.0):
#     print(mel_true,mel_pred)
#     mel = mse_loss(mel_true, mel_pred)
#     sc = spectral_convergence(mel_true, mel_pred)
#     logmel = log_mel_loss(mel_true, mel_pred)
#     loss = mel_weight * mel + sc_weight * sc + log_weight * logmel

#     if duration_true is not None and duration_pred is not None:
#         d_loss = duration_loss(duration_true, duration_pred)
#         loss += duration_weight * d_loss

#     return loss

'''
def compile_model(model):
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=0.001,
            decay_steps=50 * len(train_dataset),
            alpha=0.1
        )
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)    
    # optimizer = tfa.optimizers.AdamW(weight_decay=1e-4, learning_rate=lr_schedule)

    def spectral_convergence_loss(y_true, y_pred):
        return tf.norm(y_true - y_pred, ord='fro', axis=[-2, -1]) / tf.norm(y_true, ord='fro', axis=[-2, -1])

    # def log_mel_loss(y_true, y_pred):
    #     return tf.reduce_mean(tf.math.log(y_true + 1e-6) - tf.math.log(y_pred + 1e-6))

    def log_mel_loss(y_true, y_pred):
        epsilon = 1e-5
        y_true = tf.nn.relu(y_true)
        y_pred = tf.nn.relu(y_pred)
        
        log_true = tf.math.log(y_true + epsilon)
        log_pred = tf.math.log(y_pred + epsilon)
        
        return tf.reduce_mean(tf.abs(log_true - log_pred))  # L1 in log-mel space

    def combined_loss(y_true, y_pred):
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        sc_loss = spectral_convergence_loss(y_true, y_pred)
        log_loss = log_mel_loss(y_true, y_pred)
        return mse + 0.5 * sc_loss + 0.1 * log_loss
    
    model.compile(optimizer=optimizer,
                  loss=combined_loss,
                  metrics=['mae', CosineSimilarity(axis=-1)])
    return model'''