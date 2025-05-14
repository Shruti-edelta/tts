import tensorflow as tf
import matplotlib.pyplot as plt


# model = tf.keras.models.load_model('tts_model_lj_LSTM_attmulti.keras')  # Load your trained model
model = tf.keras.models.load_model('metrics/echo1.h5')  # Load your trained model
# model.load_weights('model_weights.h5')

# Visualize the weights of the embedding layer
embedding_layer = model.get_layer('embedding')  # Get the embedding layer
embedding_weights = embedding_layer.get_weights()[0]  # Get the weight matrix (not the bias)

for layer in model.layers:
    print(f"Layer: {layer.name}")
    weights = layer.get_weights()
    if len(weights) > 0:
        print(f"  Weights: {weights[0].shape}")
        print(f"  Weights Values: {weights[0]}")  # Print the actual weight values
        if len(weights) > 1:
            print(f"  Biases: {weights[1].shape}")
            print(f"  Biases Values: {weights[1]}")  # Print the actual bias values
    else:
        print("  No weights or biases.")
    print("-" * 50)

# Plot the embedding weights
plt.imshow(embedding_weights, aspect='auto', cmap='viridis')
plt.colorbar()
plt.title("Embedding Layer Weights")
# plt.show()

