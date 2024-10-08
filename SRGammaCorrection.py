from __future__ import print_function, division
import cv2 as cv
import numpy as np
import argparse

# Initial parameters
alpha = 1.0
alpha_max = 500
beta = 0
beta_max = 200
gamma = 1.0
gamma_max = 300
spectral_ratio = 0.8  # An example

# Function to add labels under each image
def add_label(img, text):
    font, font_scale, color, thickness = cv.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 3
    label_height = 150

    # Create a new image with extra space for the label
    img_with_label = np.zeros((img.shape[0] + label_height, img.shape[1], 3), dtype=np.uint8)
    img_with_label[:img.shape[0]] = img  

    # Center the text and add the label
    text_x = (img.shape[1] - cv.getTextSize(text, font, font_scale, thickness)[0][0]) // 2
    text_y = img.shape[0] + label_height // 2 + cv.getTextSize(text, font, font_scale, thickness)[0][1] // 2
    cv.putText(img_with_label, text, (text_x, text_y), font, font_scale, color, thickness)
    
    return img_with_label

def gammaCorrection():
    global gamma
    
    # Create a lookup table for gamma correction
    lookUpTable = np.empty((1, 256), np.uint8)
    for i in range(256):
        lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)

    res_gamma = cv.LUT(img_original, lookUpTable)

    # Apply spectral ratio to gamma correction
    adjusted_gamma = gamma * ( 1 + spectral_ratio)
    lookUpTable_spectral = np.empty((1, 256), np.uint8)
    for i in range(256):
        lookUpTable_spectral[0, i] = np.clip(pow(i / 255.0, adjusted_gamma) * 255.0, 0, 255)

    res_spectral_gamma = cv.LUT(img_original, lookUpTable_spectral)

    # Add labels to each image
    labeled_original = add_label(img_original, "Original")
    labeled_gamma = add_label(res_gamma, "Gamma Correction")
    labeled_spectral = add_label(res_spectral_gamma, "SR-guided Gamma Correction")

    # Concatenate all three labeled images
    img_combined = cv.hconcat([labeled_original, labeled_gamma, labeled_spectral])
    cv.imshow("Gamma Correction and Spectral Ratio", img_combined)

def on_gamma_correction_trackbar(val):
    global gamma
    gamma = val / 100.0  # Convert slider value to float
    gammaCorrection()

# Argument parser to load the image
parser = argparse.ArgumentParser(description='Code for Gamma correction and Spectral Ratio adjustment.')
parser.add_argument('--input', help='Path to input image.', default='skier.jpg')
args = parser.parse_args()

# Load the input image
img_original = cv.imread(cv.samples.findFile(args.input))
if img_original is None:
    print('Could not open or find the image: ', args.input)
    exit(0)

# Create a window to display images
cv.namedWindow('Gamma Correction and Spectral Ratio', cv.WINDOW_NORMAL)

# Initialize gamma trackbar
gamma_init = int(gamma * 100)
cv.createTrackbar('Gamma Correction', 'Gamma Correction and Spectral Ratio', gamma_init, gamma_max, on_gamma_correction_trackbar)

# Apply initial gamma correction
on_gamma_correction_trackbar(gamma_init)

cv.waitKey(0)
cv.destroyAllWindows()