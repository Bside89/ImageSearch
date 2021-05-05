import argparse
import pickle
import time
import cv2

# Local
from islib.hashutils import dhash, convert_hash

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-q", "--query", required=True, type=str,
                help="Path to input query image")
ap.add_argument("-b", "--data", required=True, type=str,
                help="Path to database to search input image")
ap.add_argument("-d", "--distance", type=int, default=10,
                help="Maximum hamming distance")
args = vars(ap.parse_args())

# Load the VP-Tree and hashes dictionary
print("[INFO] Loading VP-Tree and hashes...")

tree = pickle.loads(open(args["data"] + ".tree.pickle", "rb").read())
hashes = pickle.loads(open(args["data"] + ".dict.pickle", "rb").read())

# Load the input query image
image = cv2.imread(args["query"])
cv2.imshow("Query image", image)

# Compute the hash for the query image, then convert it
queryHash = convert_hash(dhash(image))

# Perform the search
print("[INFO] Performing search...")
start = time.time()
results = tree.get_all_in_range(queryHash, args["distance"])
results = sorted(results)
end = time.time()
print("[INFO] Search took {} seconds".format(end - start))

# Loop over the results
for (d, h) in results:
    # Grab all image paths in our dataset with the same hash
    result_paths = hashes.get(h, [])
    print("[INFO] {} total image(s) with d: {}, h: {}".format(len(result_paths), d, h))
    # Loop over the result paths
    for p in result_paths:
        # Load the result image and display it to our screen
        result = cv2.imread(p)
        cv2.imshow(p, result)
        cv2.waitKey(0)  # Esc
