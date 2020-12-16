import numpy as np
import os
import sys

from autoencoder import autoencoder_generate
from keras.initializers import VarianceScaling
from keras.preprocessing.image import img_to_array, load_img
from matplotlib import pyplot as plt
from tqdm import tqdm


def load_autoencoder(dims, init, path):
    """ Load trained autoencoder and encoder.

    Args:
        dims: Dimensions of the autoencoder layers.
        init: Initializer for the kernel weights matrix. Default: Glorot uniform initializer (Xavier uniform initializer).
    Return:
        The trained autoencoder and encoder models.
    """
    # Generate autoencoder and encoder
    ae, encoder = autoencoder_generate(dims, init=init)

    # Load save autoencoder weights
    ae.load_weights(os.path.join(path, 'ae_weights.h5'))

    # Extract encoder weights from layers 1 to 4
    encoder_weights = ae.layers[1].get_weights()
    for e_i in range(2, len(dims)):
        encoder_weights.extend(ae.layers[e_i].get_weights())

    # Set encoder weights
    encoder.set_weights(encoder_weights)

    return ae, encoder


def plot_autoencoder_outputs(autoencoder, n, dims, x_test, path):
    """ Plot a sample of input and output images of the autoencoder.

    Args:
        autoencoder: Trained autoencoder model.
        n: Number of samples to show.
        dims: Dimensions of the reconstructed images, shape=().
        x_test: Original images used as input for the autoencoder.

    Return:
        Nothing.
        Saves the plot containing original and reconstructed images in file autoencoder_results.png
    """
    decoded_imgs = autoencoder.predict(x_test)

    plt.figure(figsize=(10, 4.5))
    for i in range(n):
        # plot original image
        i_r = i*150
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i_r].reshape(*dims))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n/2:
            ax.set_title('Original Images')

        # plot reconstruction 
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i_r].reshape(*dims))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n/2:
            ax.set_title('Reconstructed Images')
    #plt.show()
    plt.savefig(os.path.join(path, 'autoencoder_results.png'))
    # Use this to generate autoencoder model image
    #plot_model(autoencoder, show_shapes=True)


if __name__== "__main__":
    try:
        input_path, output_path = sys.argv[1:]
    except:
        input_path = os.path.join(os.getcwd(), '..', 'data', 'input')
        output_path = os.path.join(os.getcwd(), '..', 'data', 'output')

    test_dir = os.path.join(input_path, 'test')

    if not os.path.isdir(test_dir):
        raise IOError("Error: no test directory found in {}.".format(input_path))

    # Setup train data for autoencoder
    print('Setup testing data')
    dims = [17920, 6500, 4000, 10000, 4480]
    init = VarianceScaling(scale=1. / 3., mode='fan_in', distribution='uniform')
    test_images = [f for f in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, f))]
    x_test = []
    x_test_names = []
    j = 0

    for i in tqdm(test_images):
        img = load_img(os.path.join(test_dir, i), color_mode='grayscale')
        img = img_to_array(img)
        img = np.asarray(img)
        img = img.flatten()
        x_test.append(img)
        x_test_names.append(i)

    x_test = np.asarray(x_test)
    x_test = np.divide(x_test, 255.)

    print('Load Autoencoder from weights.')
    ae, encoder = load_autoencoder(dims, init, output_path)

    plot_autoencoder_outputs(ae, 6, (112, 160), x_test, output_path)