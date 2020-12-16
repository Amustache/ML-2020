import os
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm


def eval_clusters(cluster_dir):
    """ Extract the number of sega and snes images in a cluster directory.

    Args:
        cluster_dir: Directory of the cluster.
    Return:
        Array containing values [<number of snes images>, <number of sega images>] in the cluster directory.
    """
    img_dir = os.path.join(os.getcwd(), cluster_dir)
    images = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]
    snes = 0
    sega = 0
    for i in tqdm(images):
        if 'snes' in i:
            snes = snes + 1
        if 'sega' in i:
            sega = sega + 1

    return snes, sega


if __name__== "__main__":
    try:
        input_path, output_path = sys.argv[1:]
    except:
        input_path = os.path.join(os.getcwd(), '..', 'data', 'input')
        output_path = os.path.join(os.getcwd(), '..', 'data', 'output')

    # Directories path
    train_dir = os.path.join(input_path, 'train')
    test_dir = os.path.join(input_path, 'test')

    res = []
    res2 = []
    directories=[d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
    fig, axs = plt.subplots(len(directories)+2, 2)
    fig.suptitle('Cluster distribution')
    names = ['SNES', 'SEGA']

    snes, sega = eval_clusters(train_dir)
    res.append([snes, sega])
    p_snes = snes/sum(res[-1])
    p_sega = sega/sum(res[-1])
    res2.append([p_snes, p_sega])
    axs[0, 0].bar(names, res2[-1])
    axs[0, 0].set_ylim(0, 1)
    axs[0, 0].set_title('Training set distribution')

    res = []
    res2 = []
    snes, sega = eval_clusters(test_dir)
    res.append([snes, sega])
    len_test_images = sum(res[-1])
    p_snes = snes/len_test_images
    p_sega = sega/len_test_images
    res2.append([p_snes, p_sega])
    axs[0, 1].bar(names, res2[-1])
    axs[0, 1].set_ylim(0, 1)
    axs[0, 1].set_title('Test set distribution')

    c_names = []
    c_tmp = []
    c_i = 1
    for c in directories:
        c_names.append('Cluster '+str(c_i))
        c_i = c_i + 1
        c_path = os.path.join(test_dir, c)
        c_tmp.append(len([names for name in os.listdir(c_path) if os.path.isfile(os.path.join(c_path, name))]))
    axs[1, 0].bar(c_names, c_tmp)
    axs[1, 0].set_ylim(0, len_test_images)
    axs[1, 0].set_title('Cluster distribution (raw values)')
    axs[1, 1].bar(c_names, np.divide(c_tmp, len_test_images))
    axs[1, 1].set_ylim(0, 1)
    axs[1, 1].set_title('Cluster distribution (percent)')

    i = 2
    res = []
    res2 = []
    for c in directories:
        c_path = os.path.join(test_dir, c)
        snes, sega = eval_clusters(c_path)
        res.append([snes, sega])
        p_snes = snes/sum(res[-1])
        p_sega = sega/sum(res[-1])
        res2.append([p_snes, p_sega])
        axs[i, 0].bar(names, res[-1])
        axs[i, 0].set_ylim(0, sum(res[-1]))
        axs[i, 0].set_title('Cluster '+str(i-1)+' distribution (raw values)')
        axs[i, 1].bar(names, res2[-1])
        axs[i, 1].set_ylim(0, 1)
        axs[i, 1].set_title('Cluster '+str(i-1)+' distribution (percent)')
        i = i+1
    plt.subplots_adjust(hspace=1)
    plt.savefig(os.path.join(output_path, 'cluster_distribution.png'))
    #plt.show()
