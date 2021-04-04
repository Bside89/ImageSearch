import argparse
import sys
import cv2
import pickle
from imutils import paths

# Local
from hashutils import dhash


# Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-a", "--path", required=True,
                help="Dataset of images to search through.")
args = vars(ap.parse_args())

# Grab the paths to images
print("[INFO] computing hashes for images...")
database_path = args["path"]
images_paths = list(paths.list_images(database_path))

# Remove the `\` character from any filenames containing a space
# (assuming you're executing the code on a Unix machine)
if sys.platform != "win32":
    images_paths = [p.replace("\\", "") for p in images_paths]

# Initialize the dictionary that will map the image hash to
# corresponding image
hash_dict = {}

# Loop over the images paths
for p in images_paths:
    # Load the image from disk
    image = cv2.imread(p)
    # If the image is None then we could not load it from disk (so skip it)
    if image is None:
        continue
    # Convert the image to grayscale and compute the hash
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_hash = dhash(image)
    # Update the dictionary
    list_images = hash_dict.get(image_hash, [])
    list_images.append(p)
    hash_dict[image_hash] = list_images

# Save dictionary to database
hash_filename = database_path + ".pkl"
with open(hash_filename, "wb") as f:
    pickle.dump(hash_dict, f)
    print("Hash data saved to " + hash_filename)
