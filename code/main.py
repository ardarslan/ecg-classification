import os
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy, \
                                     BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, \
                                        ModelCheckpoint, TensorBoard
from tensorflow.keras.metrics import AUC

from model import Vanilla_CNN, Vanilla_RNN
from model import Attention_RNN, InceptionNet
from model import TransferLearning_RNN

from dataset import load_data

tf.config.set_visible_devices([], 'GPU')

def set_seeds():
    seed = 1337
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


def train():

    X, Y, X_test, Y_test = load_data(data_dir=data_dir, dataset=dataset)

    optimizer = Adam(learning_rate=0.001)

    if dataset == 'mitbih':
        loss = SparseCategoricalCrossentropy(from_logits=True)
        metrics = ['accuracy']

    elif dataset == 'ptbdb':
        loss = BinaryCrossentropy(from_logits=True)
        auroc = AUC(curve='ROC', from_logits=True, name='auroc')
        auprc = AUC(curve='PR', from_logits=True, name='auprc')
        metrics = ['accuracy', auroc, auprc]

    else:
        raise Exception('wrong dataset type')

    early_stopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=2)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1)
    checkpointer = ModelCheckpoint(filepath='../checkpoints/'+dataset+'/'+experiment_name, monitor='val_loss', verbose=1, save_best_only=True)
    tensorboard = TensorBoard(log_dir='../logs/'+dataset+'/'+experiment_name, write_images=True)

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    model.fit(x=X, y=Y, batch_size=64, epochs=100, verbose=2, shuffle=True,
              callbacks=[early_stopper, lr_scheduler, checkpointer,
                         tensorboard], validation_split=0.1)

    model.load_weights('../checkpoints/'+dataset+'/'+experiment_name)  # load the best model
    model.evaluate(x=X_test, y=Y_test)

    return


if __name__ == '__main__':

    set_seeds() # for reproducible results

    experiment_name = 'task2/inception'

    data_dir = os.path.join(os.pardir, "data")
    dataset = 'ptbdb'  # 'mitbih' or 'ptbdb'
    
    #model = Vanilla_CNN(dataset=dataset)
    #model = Vanilla_RNN(dataset=dataset,rnn_type='simple')
    #model = TransferLearning_RNN(freeze_base=False) # no dataset arg needed
    model = InceptionNet(dataset=dataset)

    train()
