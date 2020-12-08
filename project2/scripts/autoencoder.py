import numpy as np
import os

from keras.initializers import VarianceScaling
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import SGD
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
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
    ae.compile(optimizer=pretrain_optimizer, loss='mse')
    ae.fit(img, img, batch_size=256, epochs=pretrain_epochs)
    ae.save_weights('ae_weights.h5')

if __name__== "__main__":
    np.random.seed(1221)
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
    dims = [x_train.shape[-1], 500, 500, 2000, 10]
    init = VarianceScaling(scale=1. / 3., mode='fan_in', distribution='uniform')
    pretrain_optimizer = SGD(lr=1, momentum=0.9)
    pretrain_epochs = 300
    batch_size = 256
    # Create and train autoencoder
    autoencoder, encoder = autoencoder_generate(dims, init=init)
    print("Training Autoencoder")
    autoencoder_training(x_train, autoencoder, pretrain_optimizer, pretrain_epochs)