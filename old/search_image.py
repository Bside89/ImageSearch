import argparse
import pickle
import cv2
from imutils import paths

# Local
from islib.hashutils import dhash

# Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-a", "--haystack", required=True,
                help="Dataset of images to search through (i.e., the haystack)")
ap.add_argument("-n", "--needles", required=True,
                help="Set of images we are searching for (i.e., needles)")
args = vars(ap.parse_args())


# Load hashes
database_path = args["haystack"]
with open(database_path + ".pickle", "rb") as f:
    hash_dict = pickle.load(f)

# Grab the paths to both the haystack and needle images
print("[INFO] computing hashes for needles...")
needle_paths = list(paths.list_images(args["needles"]))

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
    matched_paths = hash_dict.get(imageHash, [])
    # Save the results
    results[p] = matched_paths

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
