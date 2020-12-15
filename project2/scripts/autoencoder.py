import numpy as np
import os
import random

from keras.initializers import VarianceScaling
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import SGD
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def autoencoder_generate(dims, act='relu', init='glorot_uniform'):
    """ Create autoencoder with given dimensions.
    
    Args:
        dims: Array of dimensions the input will go through for encoding and decoding.
        act: Activation function used for each layer. Default: ReLu.
        init: Initializer for the kernel weights matrix. Default: Glorot uniform initializer (Xavier uniform initializer).

    Return:
        autoencoder, encoder models.
    """
    n_stacks = len(dims)-1
    input_img = Input(shape=(dims[0],), name='input')
    x = input_img
    for i in range(n_stacks-1):
        x = Dense(dims[i+1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(x)
    encoded = Dense(dims[-1], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1))(x)

    x = encoded

    for i in range(n_stacks-1, 0, -1):
        x = Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(x)

    x = Dense(dims[0], kernel_initializer=init, name='decoder_0')(x)
    decoded = x
    return Model(inputs=input_img, outputs=decoded, name='AE'), Model(inputs=input_img, outputs=encoded, name='encoder')

def autoencoder_training(img, ae, pretrain_optimizer, pretrain_epochs):
    """ Train autoencoder with a given optimizer and a number of epochs.
    
    Args:
        img: Array of all the data used to train de autoencoder.
        ae: Autoencoder model to train.
        pretrain_optimizer: String (name of optimizer) or optimizer instance to be used.
        pretrain_epochs: Number of epochs to train the model.

    Return:
        Nothing. Saves the weights of the trained autoencoder in the file ae_weights.h5
    """
    train_X, valid_X, train_ground, valid_ground = train_test_split(img, img, test_size=0.2, random_state=13) 
    ae.compile(optimizer=pretrain_optimizer, loss='mse')
    ae_train = ae.fit(train_X, train_X, batch_size=128, epochs=pretrain_epochs, verbose=1, validation_data=(valid_X, valid_ground))
    ae.save_weights('ae_weights.h5')
    loss = ae_train.history['loss']
    val_loss = ae_train.history['val_loss']
    epochs = range(pretrain_epochs)
    plt.figure()
    plt.plot(epochs, loss, 'r', label='Training loss', markersize=3)
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel("Number of Epochs")
    plt.ylabel("Loss")
    plt.legend()
#    plt.show()
    plt.savefig('loss_plot.png')

if __name__== "__main__":
    # Fix random seeds for reproducibility
    np.random.seed(1221)
    random.seed(2703)
    output_dir = os.path.join(os.getcwd(), 'output')
    train_dir = os.path.join(output_dir, 'train')
    if (os.path.isdir(train_dir) == False):
        print("Error no train and test directory found in output/")
        exit()
    train_images = [f for f in os.listdir(train_dir) if os.path.isfile(os.path.join(train_dir, f))]
    x_train = []
    # Setup train data for autoencoder
    for i in tqdm(train_images):
        img = load_img(os.path.join(train_dir, i), color_mode='grayscale')
        img = img_to_array(img)
        img = np.asarray(img)
        img = img.flatten()
        x_train.append(img)
    x_train = np.asarray(x_train)
    x_train = np.divide(x_train, 255.)
    dims = [x_train.shape[-1], 6500, 4000, 10000, 4480]
    init = VarianceScaling(scale=1. / 3., mode='fan_in', distribution='uniform')
    pretrain_optimizer = SGD(lr=1, momentum=0.9)
    pretrain_epochs = 1000
    # Create and train autoencoder
    autoencoder, encoder = autoencoder_generate(dims, init=init)
    print("Training Autoencoder")
    print("Training first model")
    autoencoder_training(x_train, autoencoder, pretrain_optimizer, pretrain_epochs)
    print("Done training first model")
