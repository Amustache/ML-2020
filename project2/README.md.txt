# Setup

    * Create new environment with python 3.7 (e.g. conda create -n <name> python=3.7)
    * Run : pip install -r requirements.txt

# Data collection

Data collection is done by using the scripts v_parser.py and youtube_parser.py.
The v_parser.py script one will parse a video in a local folder and extract the images from it.
The youtube_parser.py will download the video of the given youtube url and then extract the images from the video.

## v_parser.py :

    v_parser.py [-h] -fname FNAME [-i IMG] [-r RATE] [-f IFORMAT] [-o OUTPUT]

Extract images from videos

optional arguments:
  -h, --help            show this help message and exit
  -fname FNAME          Name of video file to extract
  -i IMG, --img IMG     Names of images saved. (default: snapshot)
  -r RATE, --rate RATE  Image rate to extract. If rate is 3, will save one
                        every 3 images. (default: 10)
  -f IFORMAT, --iformat IFORMAT
                        Format of generated images. (default: png)
  -o OUTPUT, --output OUTPUT
                        Output folder for images. (default: output/)

## youtube_parser.py :

    youtube_parser.py [-h] -url URL [-i IMG] [-r RATE] [-f IFORMAT]
                         [-o OUTPUT]

Extract images from a Youtube url

optional arguments:
  -h, --help            show this help message and exit
  -url URL              Youtube url of video to extract
  -i IMG, --img IMG     Names of images saved. (default: snapshot)
  -r RATE, --rate RATE  Image rate to extract. If rate is 3, will save one
                        every 3 images. (default: 10)
  -f IFORMAT, --iformat IFORMAT
                        Format of generated images. (default: png)
  -o OUTPUT, --output OUTPUT
                        Output folder for images. (default: output/)

# Model training

The autoencoder and clustering models can be trained separately, but the clustering model requires a trained autoencoder.
Both requires having the training data set in the folder <script location>/output/train and testing data set in the folder <script location>/output/test

## Train the autoencoder :

    python autoencoder.py

## Train the clustering model :

Requires the weight file for the autoencoder, ae_weights.h5, and then run :

    python clustering.py

# Running trained models

Use the run_clustering.py script to run the trained clustering model, the weight files for the autoencoder, ae_weights.h5, and for the clustering, cluster_weights.h5, are required.
Run the script :

    run_clustering.py [-h] -tdir TDIR [-r REPLACE] [-n NCLUSTERS]

Run clustering model with saved weights

optional arguments:
  -h, --help            show this help message and exit
  -tdir TDIR            Directory of test dataset
  -r REPLACE, --replace REPLACE
                        If this flag is set when running, it will delete
                        existing cluster folder. Default: False.
  -n NCLUSTERS, --nclusters NCLUSTERS
                        Number of clusters for the clutser_weights.h5 model.
                        Default: 2.

# Generate Results

## Generate autoencoder results :

    python ae_results.py

## Generate clustering results :

Generating the clustering results requires to have a folder for each cluster in the following location : <path to test data folder>/<cluster folder> (e.g. output/test/c1).
To generate the plots with the clustering results run :

    python eval_clusters.py


# Results Reproduction

To reproduce our results, extract the output.zip folder containing the train and test data set. Run the following scripts :

1)    python autoencoder.py
2)    python ae_results.py
3)    python clustering.py
4)    python eval_clusters.py

This will do the following :

1) Generate the ae_weights.h5 file of the trained autoencoder and save the loss and validation loss plot in the file loss_plot.png.
2) Genrate the plot containing original and reconstructed images. If line 70 is uncommented will also generate the model of the autoencoder in the file autoencoder_results.png, but doesn't work well on Windows OS.
3) Generate the cluster_weights.h5 file of the trained clustering model as well as creates the cluster folders with the assigned images. REMOVE ANY CLUSTER FOLDERS IN output/test/ BEFORE RUNNING THIS SCRIPT.
4) Generate the cluster distribution statistics plots in the file cluster_distribution.png.