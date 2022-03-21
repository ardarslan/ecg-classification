import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, SimpleRNN, Conv1D, LSTM, GRU
from tensorflow.keras.layers import MaxPool1D, GlobalMaxPool1D, Bidirectional
from tensorflow.keras.activations import softmax
from tensorflow.math import reduce_sum 

# TASK 1
def Vanilla_RNN(dataset, rnn_type='lstm'):

    if rnn_type == 'lstm':
        rnn = LSTM
    elif rnn_type == 'simple':
        rnn = SimpleRNN
    elif rnn_type == 'gru':
        rnn = GRU
    else:
        raise Exception()

    inputs = Input(shape=(187, 1))
    x = rnn(units=128, return_sequences=False)(inputs)
    x = Dense(units=128, activation='relu')(x)
    if dataset =='mitbih':
        outputs = Dense(units=5, activation=None)(x)
    elif dataset == 'ptbdb':
        outputs = Dense(units=1, activation=None)(x)
    else:
        raise Exception('wrong dataset type')

    model = Model(inputs=inputs, outputs=outputs)

    return model

# TASK 1
def Vanilla_CNN(dataset):

    inputs = Input(shape=(187, 1))  # channels_last
    x = Conv1D(filters=16, kernel_size=3, activation='relu')(inputs)
    x = Conv1D(filters=16, kernel_size=3, activation='relu')(x)
    x = MaxPool1D()(x)
    x = Conv1D(filters=64, kernel_size=3, activation='relu')(x)
    x = MaxPool1D()(x)
    x = Conv1D(filters=128, kernel_size=3, activation='relu')(x)
    x = GlobalMaxPool1D()(x)  # instead of Flatten()
    x = Dense(units=128, activation='relu')(x)
    if dataset == 'mitbih':
        outputs = Dense(units=5, activation=None)(x)
    elif dataset == 'ptbdb':
        outputs = Dense(units=1, activation=None)(x)
    else:
        raise Exception('wrong dataset type')

    model = Model(inputs=inputs, outputs=outputs)

    return model


# TASK 2
def Bidirectional_RNN(dataset, rnn_type='lstm'):

    if rnn_type == 'lstm':
        rnn = LSTM
    elif rnn_type == 'simple':
        rnn = SimpleRNN
    elif rnn_type == 'gru':
        rnn = GRU
    else:
        raise Exception()

    inputs = Input(shape=(187, 1))
    x = Bidirectional(rnn(units=128, return_sequences=False))(inputs)
    x = Dense(units=128, activation='relu')(x)
    if dataset =='mitbih':
        outputs = Dense(units=5, activation=None)(x)
    elif dataset == 'ptbdb':
        outputs = Dense(units=1, activation=None)(x)
    else:
        raise Exception('wrong dataset type')

    model = Model(inputs=inputs, outputs=outputs)

    return model


# TASK 2
def Attention_RNN(dataset,rnn_type):

    if rnn_type == 'lstm':
        rnn = LSTM
    elif rnn_type == 'simple':
        rnn = SimpleRNN
    elif rnn_type == 'gru':
        rnn = GRU
    else:
        raise Exception()

    inputs = Input(shape=(187, 1))
    hiddens = rnn(units=128, return_sequences=True)(inputs)
    o = Dense(units=1, activation='relu')(hiddens)
    att_weights = softmax(o,axis=1)
    x = reduce_sum(hiddens * att_weights,axis=1)
    x = Dense(units=64, activation='relu')(x)
    if dataset =='mitbih':
        outputs = Dense(units=5, activation=None)(x)
    elif dataset == 'ptbdb':
        outputs = Dense(units=1, activation=None)(x)
    else:
        raise Exception('wrong dataset type')

    model = Model(inputs=inputs, outputs=outputs)

    return model


# TASK 2
class Inception_Block_1D(Model):
    def __init__(self):
        super().__init__()
        self.conv_1by1 = Conv1D(filters=32, kernel_size=1, activation='relu', padding='same')

        self.conv_3by3_1 = Conv1D(filters=48, kernel_size=1, activation='relu', padding='same')
        self.conv_3by3_2 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')

        self.conv_5by5_1 = Conv1D(filters=8, kernel_size=1, activation='relu', padding='same')
        self.conv_5by5_2 = Conv1D(filters=16, kernel_size=5, activation='relu', padding='same')

        self.conv_pool_1 = MaxPool1D(pool_size=3, strides= 1, padding='same')
        self.conv_pool_2 = Conv1D(filters=16, kernel_size=1, activation='relu', padding='same')

    def call(self,x):

        x1 = self.conv_1by1(x)
        x2 = self.conv_3by3_2(self.conv_3by3_1(x))
        x3 = self.conv_5by5_2(self.conv_5by5_1(x))
        x4 = self.conv_pool_2(self.conv_pool_1(x))

        return tf.concat([x1,x2,x3,x4],axis=-1)

def InceptionNet(dataset):

    inputs = Input(shape=(187, 1))
    x = Conv1D(filters=16, kernel_size=7, activation='relu')(inputs)
    x = MaxPool1D()(x)
    x = Inception_Block_1D()(x)
    x = MaxPool1D()(x)
    x = Inception_Block_1D()(x)
    x = MaxPool1D()(x)
    x = Inception_Block_1D()(x)
    x = GlobalMaxPool1D()(x)  # instead of Flatten()
    outputs = Dense(units=256, activation='relu')(x)
    if dataset == 'mitbih':
        outputs = Dense(units=5, activation=None)(x)
    elif dataset == 'ptbdb':
        outputs = Dense(units=1, activation=None)(x)
    else:
        raise Exception('wrong dataset type')

    model = Model(inputs=inputs, outputs=outputs)

    return model


# TASK 4
def TransferLearning_RNN(freeze_base=True):

    model = Bidirectional_RNN(dataset='mitbih',rnn_type='lstm')
    model.load_weights('../checkpoints/mitbih/task2/vanilla_bi_lstm')

    if freeze_base: # freeze the base model
        model.trainable = False 

    x = model.layers[0].output
    x = model.layers[1](x)
    x = Dense(units=128, activation='relu')(x)
    x = Dense(units=64, activation='relu')(x)
    outputs = Dense(units=1, activation=None)(x)

    model2 = Model(inputs= model.layers[0].input, outputs = outputs)

    return model2