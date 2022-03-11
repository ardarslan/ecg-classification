import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, SimpleRNN, Conv1D
from tensorflow.keras.layers import MaxPool1D, GlobalMaxPool1D


def vanilla_model(type='rnn'):
    if type == 'rnn':

        inputs = Input(shape=(187, 1))
        x = SimpleRNN(units=512, return_sequences=False)(inputs)
        x = Dense(units=128, activation='relu')(x)
        outputs = Dense(units=5, activation=None)(x)

        model = Model(inputs=inputs, outputs=outputs)

        return model

    elif type == 'cnn':

        inputs = Input(shape=(187, 1))  # channels_last
        x = Conv1D(filters=16, kernel_size=3, activation='relu')(inputs)
        x = Conv1D(filters=16, kernel_size=3, activation='relu')(x)
        x = MaxPool1D()(x)
        x = Conv1D(filters=64, kernel_size=3, activation='relu')(x)
        x = MaxPool1D()(x)
        x = Conv1D(filters=128, kernel_size=3, activation='relu')(x)
        x = GlobalMaxPool1D()(x)  # instead of Flatten()
        x = Dense(units=128, activation='relu')(x)
        outputs = Dense(units=5, activation=None)(x)

        model = Model(inputs=inputs, outputs=outputs)

        return model
  
    else:
        raise NotImplementedError('unknown vanilla model type')