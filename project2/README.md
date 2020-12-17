# Unsupervised classification of video games styles

CS-433 - Machine Learning - EPFL — Martin Jaggi & Nicolas Flammarion

Project by [Camille Montemagni](mailto:camille.montemagni@epfl.ch), [Hugo "Stache" Hueber](mailto:hugo.hueber@epfl.ch), [Luca Joss](mailto:luca.joss@epfl.ch)

Supervised by [Yannick Rochat](mailto:yannick.rochat@epfl.ch), [Selim Krichane](mailto:selim.krichane@epfl.ch)

Available on [GitHub](https://github.com/Amustache/ML-2020/tree/master/project2).

## Introduction

We explored an unsupervised model, using un autoencoder and KMeans clustering layer to check if it was possible to classify the graphic style of video games in order to find similarities. We focused on the classification of games from two different consoles, SNES and SEGA, to verify the power of the classification. We are getting satisfactory results in terms of balancing, and this tool could be used for further research in the future.

Explanatory paper is [available here](./report.pdf).

### Libraries used
The main external libraries used for this project are :

- For Machine Learning training, Keras: version 2.4.3
- For Machine Learning training, Tensorflow: version 2.3.1
- For Machine Learning training, Sklearn: version 0.0
- For the data collection, Pytube: version 10.0.0
- For plotting results, Matplotlib: version 3.3.2
- For image processing, OpenCV: version 4.4.0.46

For the details of all libraries used, the file **`requirements.txt`** contains the full list.

## Basic setup

### Pre-requisites
- On Windows, install [Debian for Windows](https://www.microsoft.com/fr-fr/p/debian/9msvkqc78pk6) and use it for every Unix-related command.
- Make sure `python3`, `git` and `pip` are up and installed.
    - `python --version` should be 3.7. If not, try `python3 --version`. If it works, do `alias python=python3`.
    - `pip --version` should be >= 20.

### Basis
- Create the place where you will work: `mkdir vgstyle; cd vgstyle`.
- Create a virtual environment:
    - [venv](https://docs.python.org/3/tutorial/venv.html): `python -m venv ./vgstyle-env`.
    - [conda](https://docs.conda.io/en/latest/): `conda create -n vgstyle-env python=3.7`
- Activate your environment:
    - venv: `source ./vgstyle-env/bin/activate`.
    - conda: `conda activate vgstyle-env`.
- Upgrade pip to the most recent version: `pip install --upgrade pip`.
- Clone this repo: `git clone git@github.com:Amustache/ML-2020.git`
- Go to the `project2` folder: `cd ML-2020/project2`.
- Install all the requirements: `pip install -r data/requirements.txt`.

### Use of GPU

## How to use

### Folder organisation
```
./
- data/
-- input/
--- test/
--- train/
-- output/
- scripts/
```

### Data and pretrained elements
You can download the following pretrained elements to speed up the reproduction:

- [`test` and `train` folders](https://drive.google.com/file/d/1y-Nq2l2R1mx6Vz7T38hTadxcjKHDlgCA/view?usp=sharing), to extract put in the `` folder.
- [`ae_weights.h5`](https://drive.google.com/file/d/1To7PMAxIHi_i8xN3zEVKYjXhb18zHiWN/view?usp=sharing), to put in the `output` folder.
- [`cluster_weights.h5`](https://drive.google.com/file/d/119go10PsDS5HfGe8jlPN7xWD4i7SmLmW/view?usp=sharing), to put in the `output` folder.

### Reproduction
To reproduce our results, extract the **`output.zip`** folder containing the train and test data set. Run the following scripts:

1. **`python autoencoder.py`**: Generate the **`ae_weights.h5`** file of the trained autoencoder and save the loss and validation loss plot in the file **`loss_plot.png`**.
1. **`python ae_results.py`**: Genrate the plot containing original and reconstructed images in the file **`autoencoder_results.png`**. If line 70 is uncommented will also generate the model of the autoencoder in the file **`model.png`**, but doesn't work well on Windows.
1. **`python clustering.py`**: Generate the **`cluster_weights.h5`** file of the trained clustering model as well as creates the cluster folders with the assigned images. REMOVE ANY CLUSTER FOLDERS IN **`output/test/`** BEFORE RUNNING THIS SCRIPT.
1. **`python run_clustering.py`**: Run the trained clustering model.
1. **`python eval_clusters.py`**: Generate the cluster distribution statistics plots in the file **`cluster_distribution.png`**.

## Details

### 1. Data collection
Data collection is done by using the scripts **`v_parser.py`** and **`youtube_parser.py`**:
- The **`v_parser.py`** script one will parse a video in a local folder and extract the images from it.
- The **`youtube_parser.py`** will download the video of the given youtube url and then extract the images from the video.

By using **`sh dataset_download.sh`**, you can automatically download the whole dataset, and by using **`sh clean.sh`** afterwards, you can cleanup handpicked images as we did for the training.

#### `v_parser.py`
```
v_parser.py [-h] -fname FNAME [-i IMG] [-r RATE] [-f IFORMAT] [-o OUTPUT]

Extract images from videos

optional arguments:
 -h, --help             show this help message and exit
 -fname FNAME           Name of video file to extract
 -i IMG, --img IMG      Names of images saved. (default: snapshot)
 -r RATE, --rate RATE   Image rate to extract. If rate is 3, will save one
                        every 3 images. (default: 10)
 -f IFORMAT, --iformat IFORMAT
                        Format of generated images. (default: png)
 -o OUTPUT, --output OUTPUT
                        Output folder for images. (default: output/)
```
#### `youtube_parser.py`
```
youtube_parser.py [-h] -url URL [-i IMG] [-r RATE] [-f IFORMAT]
                     [-o OUTPUT]

Extract images from a Youtube url

optional arguments:
 -h, --help             show this help message and exit
 -url URL               Youtube url of video to extract
 -i IMG, --img IMG      Names of images saved. (default: snapshot)
 -r RATE, --rate RATE   Image rate to extract. If rate is 3, will save one
                        every 3 images. (default: 10)
 -f IFORMAT, --iformat IFORMAT
                        Format of generated images. (default: png)
 -o OUTPUT, --output OUTPUT
                        Output folder for images. (default: output/)
```

### 2. Model training
The autoencoder and clustering models can be trained separately, but the clustering model requires a trained autoencoder.

Both requires having the training data set in the folder **`<script location>/output/train`** and testing data set in the folder **`<script location>/output/test`**.

#### Train the autoencoder
```
python autoencoder.py
```

#### Train the clustering model
Requires the weight file for the autoencoder, **`ae_weights.h5`**.

```
python clustering.py
```

### 3. Using the model
Use the **`run_clustering.py -tdir output/test`** script to run the trained clustering model, the weight files for the autoencoder, **`ae_weights.h5`**, and for the clustering, **`cluster_weights.h5`**, are required.

```
run_clustering.py [-h] [-tdir TDIR] [-r REPLACE] [-n NCLUSTERS]

Run clustering model with saved weights

optional arguments:
 -h, --help             show this help message and exit
 -tdir TDIR             Directory of test dataset
 -r REPLACE, --replace REPLACE
                        If this flag is set when running, it will delete
                        existing cluster folder. Default: False.
 -n NCLUSTERS, --nclusters NCLUSTERS
                        Number of clusters for the clutser_weights.h5 model.
                        Default: 2.
```

This will create a cluster in `output/test/c1` and `output/test/c2`.

### 4. Generate results
#### Generate autoencoder results
```
python ae_results.py
```

Results are generated in `autoencoder_results.png`.

#### Generate clustering results
Generating the clustering results requires to have a folder for each cluster in **`<path to test data folder>/<cluster folder>`** (e.g. **`output/test/c1`**).
To generate the plots with the clustering results run:
```
python eval_clusters.py
```

## Known bugs / TODO

### Bugs

- Not known bugs for now.

### TODOs

- Nothing is planned beyond the project.

## Acknowledgments

Thank you to Nicolas Flammarion and Martin Jaggi for the teaching. Thank you Yannick Rochat and Selim Krichane for the supervision of our unsupervising. Thank you to our respective roommates for putting up with us during this confinement. A special thanks to Gustave Pistache and Victor Spielberg for the exchanges and help provided.
