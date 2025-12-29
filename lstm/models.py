"""Model definitions for the LSTM experiments."""
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from keras.models import Sequential


def vanilla_lstm(activation: str, n_steps: int, n_features: int):
    model = Sequential()
    model.add(LSTM(50, activation=activation, input_shape=(n_steps, n_features)))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    return model


def vanillaLSTM(activation: str, n_steps: int, n_features: int):
    return vanilla_lstm(activation, n_steps, n_features)


def stacked_lstm(activation: str, n_steps: int, n_features: int):
    model = Sequential()
    model.add(LSTM(50, activation=activation, return_sequences=True, input_shape=(n_steps, n_features)))
    model.add(LSTM(50, activation=activation))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    return model


def stackedLSTM(activation: str, n_steps: int, n_features: int):
    return stacked_lstm(activation, n_steps, n_features)


def bidirectional_lstm(activation: str, n_steps: int, n_features: int):
    model = Sequential()
    model.add(Bidirectional(LSTM(50, activation=activation), input_shape=(n_steps, n_features)))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    return model


def bidirectionalLSTM(activation: str, n_steps: int, n_features: int):
    return bidirectional_lstm(activation, n_steps, n_features)
