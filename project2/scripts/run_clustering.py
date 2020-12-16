import argparse
import numpy as np
import os
import shutil
import sys

from clustering import extract_clusters, load_encoder, save_clusters
from custom_cluster import ClusteringLayer
from keras.initializers import VarianceScaling
from keras.models import Model
from keras.preprocessing.image import img_to_array, load_img
from tqdm import tqdm


def load_and_run(encoder, n_clusters, x_test, in_path, out_path):
    """ Load trained clustering model using weights saved in file cluster_weights.h5 and run it on given test set.

    Args:
        encoder: Trained encoder model.
        n_clusters: Number of clusters in the trained clustering model to load.
        x_test: Test data set.
        test_dir: Directory containing test images.
    Return:

    """
    clustering_layer = ClusteringLayer(n_clusters, name='clustering')(encoder.output)
    model = Model(inputs=encoder.input, outputs=clustering_layer)
    model.load_weights(os.path.join(out_path, 'cluster_weights.h5'))
    y_pred = model.predict(x_test)
    clusters = extract_clusters(y_pred)
    i = 1
    print("Separating images by clusters")
    for c in clusters:
        save_clusters(in_path, x_test_names, i, c)
        i = i + 1


if __name__== "__main__":
    parser = argparse.ArgumentParser(description='Run clustering model with saved weights')
    parser.add_argument('-tdir', type=str, required=False, help='Directory of test dataset')
    parser.add_argument('-r', '--replace', type=bool, required=False, default=False, help='If this flag is set when running, it will delete existing cluster folder. Default: False.')
    parser.add_argument('-n', '--nclusters', type=int, required=False, default=2, help='Number of clusters for the clutser_weights.h5 model. Default: 2.')
    
    args = parser.parse_args(sys.argv[1:])

    if args.tdir:
        data_path = os.path.join(os.getcwd(), args.tdir)
    else:
        data_path = os.path.join(os.getcwd(), '..', 'data')

    input_path = os.path.join(data_path, 'input')
    output_path = os.path.join(data_path, 'output')

    test_dir = os.path.join(input_path, 'test')

    if not os.path.isdir(test_dir):
        raise IOError("Error: no test or train directory found in {}.".format(input_path))

    replace = args.replace
    n_clusters = args.nclusters

    if replace:
        for idc in range(1, n_clusters+1):
            c_dir = os.path.join(output_path, 'c'+str(idc))
            if os.path.isdir(c_dir):
                shutil.rmtree(c_dir)

    test_images = [f for f in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, f))]
    x_test = []
    x_test_names = []
    for i in tqdm(test_images):
        img = load_img(os.path.join(test_dir, i), color_mode='grayscale')
        img = img_to_array(img)
        img = np.asarray(img)
        img = img.flatten()
        x_test.append(img)
        x_test_names.append(i)
    x_test = np.asarray(x_test)
    x_test = np.divide(x_test, 255.)

    print('Load Autoencoder from weights')
    dims = [17920, 6500, 4000, 10000, 4480]
    init = VarianceScaling(scale=1. / 3., mode='fan_in', distribution='uniform')
    encoder = load_encoder(dims, init, output_path)

    load_and_run(encoder, n_clusters, x_test, test_dir, output_path)