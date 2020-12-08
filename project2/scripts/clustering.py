import numpy as np
import os

from autoencoder import autoencoder_generate
from keras.initializers import VarianceScaling
from keras.layers import Input
from keras.models import Model
from keras.optimizers import SGD
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import save_img
from keras.preprocessing.image import load_img
from sklearn.cluster import KMeans
from tqdm import tqdm

def clustering_training(x, n_clusters, encoder, x_test, x_test_names, img_dir, batch_size):
    clustering_layer = ClusteringLayer(n_clusters, name='clustering')(encoder.output)
    model = Model(inputs=encoder.input, outputs=clustering_layer)
    kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=1221)
    y_pred = kmeans.fit_predict(encoder.predict(x))
    model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])
    model.compile(optimizer=SGD(0.01, 0.9), loss='kld')
    index_array = np.arange(x.shape[0])
    maxiter = 10000
    update_interval = 70
    loss = 0
    index = 0
    tol = 0.001
    y_pred_last = np.copy(y_pred)
    for ite in range(int(maxiter)):
        if ite % update_interval == 0:
            print('ite : %d, loss : %f'%(ite, loss))
            q = model.predict(x, verbose=0)
            p = target_distribution(q)
            y_pred = q.argmax(1)
            delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
            y_pred_last = np.copy(y_pred)
            if ite > 0 and delta_label < tol:
                print('delta_label ', delta_label, '< tol ', tol)
                print('Reached tolerance threshold. Stopping training.')
                break
        idx = index_array[index * batch_size: min((index+1) * batch_size, x.shape[0])]
        loss = model.train_on_batch(x=x[idx], y=p[idx])
        index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0
    model.save_weights('cluster_weights.h5')
    y_pred = model.predict(x_test)
    clusters = extract_clusters(y_pred)
    i = 1
    print("Separating images by clusters")
    for c in clusters:
        save_clusters(img_dir, x_test_names, i, c)
        i = i + 1

def extract_clusters(y_pred):
    clusters = []
    for i in range(y_pred.shape[0]):
        tmp_max = 0
        tmp_id = 999
        for j in range(y_pred.shape[1]):
            if y_pred[i,j] > tmp_max:
                tmp_max = y_pred[i,j]
                tmp_id = j
        y_pred[i,:] = 0
        y_pred[i,tmp_id] = 1
    for i in range(y_pred.shape[1]):
        clusters.append(y_pred[:,i])
    return clusters

def save_clusters(img_dir, c, idc, chk):
    c_dir = os.path.join(img_dir, 'c'+str(idc))
    if (os.path.isdir(c_dir) == False):
        os.mkdir(c_dir, 0o777)
    cnter = 0
    for i in tqdm(range(len(c))):
        if (chk[i] == 1):
            img_path = os.path.join(img_dir, c[i])
            c_img_path = os.path.join(c_dir, c[i])
            img = load_img(img_path)
            img_array = img_to_array(img)
            save_img(c_img_path, img_array)
            cnter = cnter + 1
    print("cluster %d has %d images"%(idc, cnter))
        

def target_distribution(q):
    weight = q ** 2 /q.sum(0)
    return (weight.T / weight.sum(1)).T

def load_encoder(dims, init):
    # Generate autoencoder and encoder
    ae, encoder = autoencoder_generate(dims, init=init)
    # Load save autoencoder weights
    ae.load_weights('ae_weights.h5')
    # Extract encoder weights from layers 1 to 4
    encoder_weights = ae.layers[1].get_weights()
    for e_i in range(2, len(dims)):
        encoder_weights.extend(ae.layers[e_i].get_weights())
    # Set encoder weights
    encoder.set_weights(encoder_weights)
    return encoder

if __name__== "__main__":
    np.random.seed(1221)
    output_dir = os.path.join(os.getcwd(), 'output')
    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')
    if (os.path.isdir(train_dir) == False or os.path.isdir(test_dir) == False):
        print("Error no train and test directory found in output/")
        exit()
    train_images = [f for f in os.listdir(train_dir) if os.path.isfile(os.path.join(train_dir, f))]
    x_train = []
    # Setup train data for autoencoder
    print('Setup training et testing data')
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
    encoder = load_encoder(dims, init)

    print('Training Clustering model')
    n_clusters = 2
    batch_size = 256
    clustering_training(x_train, n_clusters, encoder, x_test, x_test_names, test_dir, batch_size)