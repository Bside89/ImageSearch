from imutils import paths
import argparse
import pickle
import time
import sys
import cv2
import os
import json


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
haystack_paths = list(paths.list_images(args["haystack"]))
needle_paths = list(paths.list_images(args["needles"]))

# Remove the `\` character from any filenames containing a space
# (assuming you're executing the code on a Unix machine)
if sys.platform != "win32":
    haystack_paths = [p.replace("\\", "") for p in haystack_paths]
    needle_paths = [p.replace("\\", "") for p in needle_paths]

# Initialize the dictionary that will map the image hash to
# corresponding image, hashes, then start the timer
haystack = {}
start = time.time()

# Loop over the haystack paths
for p in haystack_paths:
    # Load the image from disk
    image = cv2.imread(p)
    # If the image is None then we could not load it from disk (so skip it)
    if image is None:
        continue
    # Convert the image to grayscale and compute the hash
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imageHash = dhash(image)
    # Update the haystack dictionary
    haystack_images = haystack.get(imageHash, [])
    haystack_images.append(p)
    haystack[imageHash] = haystack_images

# Show timing for hashing haystack images, then start computing the hashes for needle images
print("[INFO] processed {} images in {:.2f} seconds".format(len(haystack), time.time() - start))
print("[INFO] computing hashes for needles...")

# Dictionary containing needle images and their correspondent haystack image path
results = {}

# Loop over the needle paths
for p in needle_paths:
    # Load the image from disk
    image = cv2.imread(p)
    # If the image is None then we could not load it from disk (so skip it)
    if image is None:
        continue
    # Convert the image to grayscale and compute the hash
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imageHash = dhash(image)
    # Grab all image paths that match the hash
    matched_paths = haystack.get(imageHash, [])
    # Save the results
    results[p] = matched_paths

# Save dictionary to database
hash_data = json.dumps(haystack)
f = open(args["haystack"] + ".json", "w")
f.write(hash_data)
f.close()

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
