import numpy as np
import os

from keras.initializers import VarianceScaling
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import SGD
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from sklearn.cluster import KMeans
from tqdm import tqdm

from custom_cluster import ClusteringLayer

def autoencoder_generate(dims, act='relu', init='glorot_uniform'):
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
    ae.compile(optimizer=pretrain_optimizer, loss='mse')
    ae.fit(img, img, batch_size=256, epochs=pretrain_epochs)
    ae.save_weights('ae_weights.h5')

def clustering_training(x, n_clusters, encoder):
    clustering_layer = ClusteringLayer(n_clusters, name='clustering')(encoder.output)
    model = Model(inputs=encoder.input, outputs=clustering_layer)
    kmeans = KMeans(n_clusters=n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(encoder.predict(x))
    model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])
    model.compile(optimizer=SGD(0.01, 0.9), loss='kld')
    maxiter = 1000
    update_interval = 140
    for ite in range(int(maxiter)):
        if ite % update_interval == 0:
            q = model.predict(x, verbose=0)
            p = target_distribution(q)
            y_pred = q.argmax(1)
        idx = index_array[index * batch_size: min((index+1) * batch_size, x.shape[0])]
        loss = model.train_on_batch(x=x[idx], y=p[idx])
        index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0
    model.save_weights('cluster_weights.h5')

def target_distribution(q):
    weight = q ** 2 /q.sum(0)
    return (weight.T / weight.sum(1)).T

if __name__== "__main__":
    output_dir = os.path.join(os.getcwd(), 'output')
    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')
    if (os.path.isdir(train_dir) == False or os.path.isdir(test_dir) == False):
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
    autoencoder_training(x_train, autoencoder, pretrain_optimizer, pretrain_epochs)
    # Train clustering model
    clustering_training(x, n_clusters, encoder)
