import cv2
import numpy as np


def dhash(input_image, hash_size=8):
    # convert the image to grayscale
    # gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    # Resize the input image, adding a single column (width) so we
    # can compute the horizontal gradient
    resized = cv2.resize(input_image, (hash_size + 1, hash_size))
    # Compute the (relative) horizontal gradient between adjacent column pixels
    diff = resized[:, 1:] > resized[:, :-1]
    # Convert the difference image to a hash
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])


def convert_hash(h):
    # Convert the hash to NumPy's 64-bit float and then back to
    # Python's built in int
    return int(np.array(h, dtype="float64"))


def hamming(a, b):
    # Compute and return the Hamming distance between the integers
    return bin(int(a) ^ int(b)).count("1")
