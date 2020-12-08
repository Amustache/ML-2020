import os
import math
import matplotlib.pyplot as plt

from tqdm import tqdm

def eval_clusters(cluster_dir):
    img_dir = os.path.join(os.getcwd(), cluster_dir)
    images = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]
    gba = 0
    snes = 0
    sega = 0
    for i in tqdm(images):
        if 'snes' in i:
            snes = snes + 1
        if 'sega' in i:
            sega = sega + 1
        if 'gba' in i:
            gba = gba + 1
    return snes, sega, gba

out_dir = os.path.join(os.getcwd(), 'output')
test_dir = os.path.join(out_dir, 'test')
train_dir = os.path.join(out_dir, 'train')

res = []
res2 = []
directories=[d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
fig, axs = plt.subplots(math.ceil(len(directories)/2)+2, 2)
fig.suptitle('Cluster distribution')
names = ['SNES', 'SEGA', 'GBA']

snes, sega, gba = eval_clusters(train_dir)
res.append([snes, sega, gba])
p_snes = snes/sum(res[-1])
p_sega = sega/sum(res[-1])
p_gba = gba/sum(res[-1])
res2.append([p_snes, p_sega, p_gba])
axs[0, 0].bar(names, res2[-1])
axs[0, 0].set_ylim(0, 1)
axs[0, 0].set_title('Training set distribution')

res = []
res2 = []
snes, sega, gba = eval_clusters(test_dir)
res.append([snes, sega, gba])
p_snes = snes/sum(res[-1])
p_sega = sega/sum(res[-1])
p_gba = gba/sum(res[-1])
res2.append([p_snes, p_sega, p_gba])
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
axs[1, 0].set_title('Cluster distribution')

i = 2
j = 0
res = []
res2 = []
for c in directories:
    c_path = os.path.join(test_dir, c)
    snes, sega, gba = eval_clusters(c_path)
    res.append([snes, sega, gba])
    p_snes = snes/sum(res[-1])
    p_sega = sega/sum(res[-1])
    p_gba = gba/sum(res[-1])
    res2.append([p_snes, p_sega, p_gba])
    axs[i, j].bar(names, res2[-1])
    axs[i, j].set_ylim(0, 1)
    axs[i, j].set_title('Cluster '+str(i+j)+' distribution')
    if j == 1:
        i = i + 1
        j = 0
    else:
        j = j + 1
plt.subplots_adjust(hspace=1)
plt.show()