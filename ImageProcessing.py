#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import math
import cv2
import os
from FindLaneLines_Library import *

# Reading in an image
dir_path = os.path.dirname(os.path.realpath(__file__))
image = mpimg.imread(dir_path+'/test_images/whiteCarLaneSwitch.jpg')

# Grayscaling the image
gray_image = grayscale(image)

# Defining kernal and apllying gaussian filtering
kernel_size = 5
blur_image = gaussian_blur(gray_image, kernel_size)

# Applying canny to detect edges
low_threshold = 50
high_threshold = 200
edges_image = canny(blur_image, low_threshold, high_threshold)

# Masking the region of interest
region_vertices = np.array([[(60,540),(460, 310), (490, 310),(930,540)]], dtype=np.int32)
mask_image = region_of_interest(edges_image, region_vertices)

# Running hough transform to get detected arrays
rho = 1 # distance resolution in pixels of the Hough grid
theta = np.pi/180 # angular resolution in radians of the Hough grid
threshold = 25     # minimum number of votes (intersections in Hough grid cell)
min_line_len = 5 #minimum number of pixels making up a line
max_line_gap = 1    # maximum gap in pixels between connectable line segments

Lane_lines = hough_lines(mask_image, rho, theta, threshold, min_line_len, max_line_gap)

# Visualizing the lane lines by weighing original and image with hough lines 
LaneLines_image = weighted_img(Lane_lines, image, α=0.8, β=1., γ=0.)

plt.imshow(LaneLines_image)




