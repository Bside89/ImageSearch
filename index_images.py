import argparse
import sys
import cv2
import pickle
import vptree
from imutils import paths

# Local
from islib.hashutils import dhash, hamming, convert_hash


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
for (i, p) in enumerate(images_paths):
    # Load the image from disk
    print("[INFO] processing image {}/{}".format(i + 1, len(images_paths)))
    image = cv2.imread(p)
    # If the image is None then we could not load it from disk (so skip it)
    if image is None:
        continue
    # Compute the hash
    h = convert_hash(dhash(image))
    # Update the dictionary
    list_images = hash_dict.get(h, [])
    list_images.append(p)
    hash_dict[h] = list_images

# Save dictionary to database
hash_filename = database_path + ".dict.pickle"
with open(hash_filename, "wb") as f:
    pickle.dump(hash_dict, f)
    print("[INFO] Hash data saved: " + hash_filename)

print("[INFO] Generating VPTree...")
tree = vptree.VPTree(list(hash_dict.keys()), hamming)

tree_filename = database_path + ".tree.pickle"
with open(tree_filename, "wb") as f:
    pickle.dump(tree, f)
    print("[INFO] VPTree generated and saved: " + tree_filename)
