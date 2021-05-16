import argparse
import sys
import cv2
import numpy as np
from imutils import paths
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

from islib import perceptual_loss

# Local
from islib.hashutils import dhash, convert_hash, hamming


# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-q11", "--query11", required=True, type=str,
                help="Path to input query image 1.1")
ap.add_argument("-q12", "--query12", required=True, type=str,
                help="Path to input query image 1.2")
ap.add_argument("-q21", "--query21", required=True, type=str,
                help="Path to input query image 2.1")
ap.add_argument("-q22", "--query22", required=True, type=str,
                help="Path to input query image 2.2")
ap.add_argument("-b", "--compare", required=True, type=str,
                help="Path to database to compare input image")
args = vars(ap.parse_args())

# Grab the paths to images
print("[INFO] Computing hashes for images...")
images_paths = list(paths.list_images(args["compare"]))

# Remove the `\` character from any filenames containing a space
# (assuming you're executing the code on a Unix machine)
if sys.platform != "win32":
    images_paths = [p.replace("\\", "") for p in images_paths]

# Queries list
queries = [
    [
        {
            'filename': args["query11"],
            'input': cv2.imread(args["query11"]),
            'results': {'hashing': [], 'resnet': []}
        },
        {
            'filename': args["query12"],
            'input': cv2.imread(args["query12"]),
            'results': {'hashing': [], 'resnet': []}
        },
    ],
    [
        {
            'filename': args["query21"],
            'input': cv2.imread(args["query21"]),
            'results': {'hashing': [], 'resnet': []}
        },
        {
            'filename': args["query22"],
            'input': cv2.imread(args["query22"]),
            'results': {'hashing': [], 'resnet': []}
        }
    ]
]

# Loop over queries
for (i, p) in enumerate(queries):
    # Loop over the images paths (first folder)
    query_hash1 = convert_hash(dhash(p[0]['input']))
    query_hash2 = convert_hash(dhash(p[1]['input']))
    query_image1_resized = cv2.resize(p[0]['input'], (160, 160))
    query_image2_resized = cv2.resize(p[1]['input'], (160, 160))
    results1 = p[0]['results']
    results2 = p[1]['results']
    for (j, q) in enumerate(images_paths):
        # Load the image from disk
        print("[INFO] Query #{}: Processing image {}/{}".format(i + 1, j + 1, len(images_paths)))
        image = cv2.imread(q)
        # If the image is None then we could not load it from disk (so skip it)
        if image is None:
            continue
        # Compute the hash and difference hash
        h = convert_hash(dhash(image))
        diff1 = hamming(query_hash1, h)
        diff2 = hamming(query_hash2, h)
        # Computer perceptual loss (VGG)
        pred = cv2.resize(image, (160, 160))
        loss1 = perceptual_loss.perceptual(query_image1_resized, pred)
        loss2 = perceptual_loss.perceptual(query_image2_resized, pred)
        # Update results
        results1['hashing'].insert(len(results1['hashing']), diff1)
        results1['resnet'].insert(len(results1['resnet']), loss1.numpy())
        results2['hashing'].insert(len(results2['hashing']), diff2)
        results2['resnet'].insert(len(results2['resnet']), loss2.numpy())

plt_indexes = ['4.1', '4.2']

# Loop over queries and plot data
for (i, p) in enumerate(queries):
    print("[INFO] Query #{}: Plotting data...".format(i))
    results1 = p[0]['results']
    results2 = p[1]['results']
    filename1 = p[0]['filename'].split('\\')[-1]
    filename2 = p[1]['filename'].split('\\')[-1]
    # Plot histogram (Hamming distance)
    results_hashing1 = np.array(results1['hashing'])
    results_hashing2 = np.array(results2['hashing'])
    bins = np.linspace(0, max(results_hashing1.max(), results_hashing2.max()), 20)
    plt.figure(1)
    n1, _, _ = plt.hist(x=results_hashing1, bins=bins, color='#05ffa1',
                        alpha=0.5, rwidth=0.95, label=filename1,
                        edgecolor='black', linewidth=1,
                        weights=np.ones(len(results_hashing1)) / len(results_hashing1))
    n2, _, _ = plt.hist(x=results_hashing2, bins=bins, color='#f08472',
                        alpha=0.5, rwidth=0.95, label=filename2,
                        edgecolor='black', linewidth=1,
                        weights=np.ones(len(results_hashing2)) / len(results_hashing2))
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Distância Hamming')
    plt.ylabel('% Total de imagens')
    plt.legend(loc='upper left')
    plt.title('Experimento #{} - Utilizando dHash'.format(plt_indexes[i]))
    plt.text(23, 45, r'$\mu=15, b=3$')
    maxfreq = max(n1.max(), n2.max())
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.show()

# Loop over queries and plot data
for (i, p) in enumerate(queries):
    print("[INFO] Query #{}: Plotting data...".format(i))
    results1 = p[0]['results']
    results2 = p[1]['results']
    filename1 = p[0]['filename'].split('\\')[-1]
    filename2 = p[1]['filename'].split('\\')[-1]
    # Plot histogram (Inception ResNet V2)
    results_resnet1 = np.array(results1['resnet'])
    results_resnet2 = np.array(results2['resnet'])
    bins = np.linspace(0, max(results_resnet1.max(), results_resnet2.max()), 20)
    plt.figure(2)
    n1, _, _ = plt.hist(x=results_resnet1, bins=bins, color='#05ffa1',
                        alpha=0.5, rwidth=0.95, label=filename1,
                        edgecolor='black', linewidth=1,
                        weights=np.ones(len(results_resnet1)) / len(results_resnet1))
    n2, _, _ = plt.hist(x=results_resnet2, bins=bins, color='#f08472',
                        alpha=0.5, rwidth=0.95, label=filename2,
                        edgecolor='black', linewidth=1,
                        weights=np.ones(len(results_resnet2)) / len(results_resnet2))
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Erro')
    plt.ylabel('% Total de imagens')
    plt.legend(loc='upper right')
    plt.title('Experimento #{} - Utilizando Inception ResNet V2'.format(plt_indexes[i]))
    plt.text(23, 45, r'$\mu=15, b=3$')
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.show()

print("[INFO] Exiting...")
