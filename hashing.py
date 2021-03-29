from imutils import paths
import argparse
import time
import sys
import cv2
import os


def dhash(input_image, hash_size=8):
    # Resize the input image, adding a single column (width) so we
    # can compute the horizontal gradient
    resized = cv2.resize(input_image, (hash_size + 1, hash_size))
    # Compute the (relative) horizontal gradient between adjacent column pixels
    diff = resized[:, 1:] > resized[:, :-1]
    # Convert the difference image to a hash
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])


# Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-a", "--haystack", required=True,
                help="Dataset of images to search through (i.e., the haystack)")
ap.add_argument("-n", "--needles", required=True,
                help="Set of images we are searching for (i.e., needles)")
args = vars(ap.parse_args())

# Grab the paths to both the haystack and needle images
print("[INFO] computing hashes for haystack...")
haystackPaths = list(paths.list_images(args["haystack"]))
needlePaths = list(paths.list_images(args["needles"]))

# Remove the `\` character from any filenames containing a space
# (assuming you're executing the code on a Unix machine)
if sys.platform != "win32":
    haystackPaths = [p.replace("\\", "") for p in haystackPaths]
    needlePaths = [p.replace("\\", "") for p in needlePaths]

# Grab the base subdirectories for the needle paths, initialize the
# dictionary that will map the image hash to corresponding image,
# hashes, then start the timer
BASE_PATHS = set([p.split(os.path.sep)[-2] for p in needlePaths])
haystack = {}
start = time.time()

# Loop over the haystack paths
for p in haystackPaths:
    # Load the image from disk
    image = cv2.imread(p)
    # If the image is None then we could not load it from disk (so skip it)
    if image is None:
        continue
    # Convert the image to grayscale and compute the hash
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imageHash = dhash(image)
    # Update the haystack dictionary
    haystackImages = haystack.get(imageHash, [])
    haystackImages.append(p)
    haystack[imageHash] = haystackImages

# Show timing for hashing haystack images, then start computing the hashes for needle images
print("[INFO] processed {} images in {:.2f} seconds".format(len(haystack), time.time() - start))
print("[INFO] computing hashes for needles...")

# Dictionary containing needle images and their correspondent haystack image path
results = {}

# Loop over the needle paths
for p in needlePaths:
    # Load the image from disk
    image = cv2.imread(p)
    # If the image is None then we could not load it from disk (so skip it)
    if image is None:
        continue
    # Convert the image to grayscale and compute the hash
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imageHash = dhash(image)
    # Grab all image paths that match the hash
    matchedPaths = haystack.get(imageHash, [])
    # Save the results
    results[p] = matchedPaths


# Display results found
print("[INFO] input images and their correspondents:")
# Loop over each subdirectory and display it
for key in results:
    print('Image: {}'.format(key))
    if len(results[key]) > 0:
        for img in results[key]:
            print('-> {}'.format(img))
    else:
        print('-> No similar images found.')
