import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense

def build_autoencoder(input_dim):
    """
    Build an autoencoder for anomaly detection.

    The model learns to reconstruct normal transactions.
    """

    # Input layer
    input_layer = Input(shape=(input_dim,))

    # Encoder
    encoded = Dense(16, activation="relu")(input_layer)
    encoded = Dense(8, activation="relu")(encoded)

    # Decoder
    decoded = Dense(16, activation="relu")(encoded)
    decoded = Dense(input_dim, activation="linear")(decoded)

    # Create model
    autoencoder = Model(input_layer, decoded)

    # Compile model
    autoencoder.compile(
        optimizer="adam",
        loss="mse"
    )

    return autoencoder

if __name__ == "__main__":

    input_dimension = 30

    model = build_autoencoder(input_dimension)

    print("===== AUTOENCODER ARCHITECTURE =====")
    model.summary()