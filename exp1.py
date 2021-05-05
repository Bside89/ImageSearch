import argparse
import sys
import cv2
import numpy as np
from imutils import paths
import matplotlib.pyplot as plt

from islib import perceptual_loss


# Local
from islib.hashutils import dhash, convert_hash, hamming

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-q", "--query", required=True, type=str,
                help="Path to input query image")
ap.add_argument("-b1", "--compare1", required=True, type=str,
                help="Path to database to compare input image (folder 1)")
ap.add_argument("-b2", "--compare2", required=True, type=str,
                help="Path to database to compare input image (folder 2)")
args = vars(ap.parse_args())

# Grab the paths to images
print("[INFO] Computing hashes for images...")
images_paths1 = list(paths.list_images(args["compare1"]))
images_paths2 = list(paths.list_images(args["compare2"]))

# Remove the `\` character from any filenames containing a space
# (assuming you're executing the code on a Unix machine)
if sys.platform != "win32":
    images_paths1 = [p.replace("\\", "") for p in images_paths1]
    images_paths2 = [p.replace("\\", "") for p in images_paths2]

# Load the input query image and compute hash
query_image = cv2.imread(args["query"])
query_hash = convert_hash(dhash(query_image))
query_image_resized = cv2.resize(query_image, (160, 160))

# Initialize the dictionary that will map the image to corresponding hash
hash_dict = {}

# Results
results1 = {'hashing': [], 'resnet': []}
results2 = {'hashing': [], 'resnet': []}

# Loop over the images paths (first folder)
for (i, p) in enumerate(images_paths1):
    # Load the image from disk
    print("[INFO] Folder #1: Processing image {}/{}".format(i + 1, len(images_paths1)))
    image = cv2.imread(p)
    # If the image is None then we could not load it from disk (so skip it)
    if image is None:
        continue
    # Compute the hash and difference hash
    h = convert_hash(dhash(image))
    diff = hamming(query_hash, h)
    # Computer perceptual loss (VGG)
    pred = cv2.resize(image, (160, 160))
    loss = perceptual_loss.perceptual(query_image_resized, pred)
    # Update the dictionary
    hash_dict[p] = {'hash': h, 'difference': diff}
    # Update results
    results1['hashing'].insert(len(results1['hashing']), diff)
    results1['resnet'].insert(len(results1['resnet']), loss.numpy())

# Plot histogram (Hamming distance)
print("[INFO] Folder #1: Plotting data...")
results_hashing1 = np.array(results1['hashing'])
results_hashing2 = np.array(results2['hashing'])
bins = np.linspace(0, max(results_hashing1.max(), results_hashing2.max()), 20)
plt.figure(1)
n1, bins1, patches1 = plt.hist(x=results_hashing1, bins=bins, color='#05ffa1',
                               alpha=0.5, rwidth=0.95, label='Mesmo carro',
                               edgecolor='black', linewidth=1)
n2, bins2, patches2 = plt.hist(x=results_hashing2, bins=bins, color='#f08472',
                               alpha=0.5, rwidth=0.95, label='Carros diferentes',
                               edgecolor='black', linewidth=1)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Distância Hamming')
plt.ylabel('Nº de imagens')
plt.legend(loc='upper right')
plt.title('Experimento #1 - Utilizando Hashing e VPTree')
plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = max(n1.max(), n2.max())
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
plt.show()

# Loop over the images paths (second folder)
for (i, p) in enumerate(images_paths2):
    # Load the image from disk
    print("[INFO] Folder #2: Processing image {}/{}".format(i + 1, len(images_paths2)))
    image = cv2.imread(p)
    # If the image is None then we could not load it from disk (so skip it)
    if image is None:
        continue
    # Compute the hash and difference hash
    h = convert_hash(dhash(image))
    diff = hamming(query_hash, h)
    # Computer perceptual loss (VGG)
    pred = cv2.resize(image, (160, 160))
    loss = perceptual_loss.perceptual(query_image_resized, pred)
    # Update results
    results2['hashing'].insert(len(results2['hashing']), diff)
    results2['resnet'].insert(len(results2['resnet']), loss.numpy())

# Plot histogram (Inception ResNet V2)
print("[INFO] Folder #2: Plotting data...")
results_resnet1 = np.array(results1['resnet'])
results_resnet2 = np.array(results2['resnet'])
bins = np.linspace(0, max(results_resnet1.max(), results_resnet2.max()), 20)
plt.figure(2)
n1, bins1, patches1 = plt.hist(x=results_resnet1, bins=bins, color='#05ffa1',
                               alpha=0.5, rwidth=0.95, label='Mesmo carro',
                               edgecolor='black', linewidth=1)
n2, bins2, patches2 = plt.hist(x=results_resnet2, bins=bins, color='#f08472',
                               alpha=0.5, rwidth=0.95, label='Carros diferentes',
                               edgecolor='black', linewidth=1)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Erro')
plt.ylabel('Nº de imagens')
plt.legend(loc='upper right')
plt.title('Experimento #1 - Utilizando Inception ResNet V2')
plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = max(n1.max(), n2.max())
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
plt.show()

print("[INFO] Exiting...")
