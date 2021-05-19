#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import math
import cv2
import os
from FindLaneLines_Library import *
from moviepy.editor import VideoFileClip
from IPython.display import HTML

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # Grayscaling the image
    gray_image = grayscale(image)

    #Defining kernal and apllying gaussian filtering
    kernel_size = 5
    blur_image = gaussian_blur(gray_image, kernel_size)

    #Applying canny to detect edges
    low_threshold = 50
    high_threshold = 200
    edges_image = canny(blur_image, low_threshold, high_threshold)

    #Masking the region of interest
    region_vertices = np.array([[(60,540),(460, 310), (490, 310),(930,540)]], dtype=np.int32)
    mask_image = region_of_interest(edges_image, region_vertices)

    #Running hough transform to get detected arrays
    rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 25     # minimum number of votes (intersections in Hough grid cell)
    min_line_len = 5 #minimum number of pixels making up a line
    max_line_gap = 1    # maximum gap in pixels between connectable line segments

    Lane_lines = hough_lines(mask_image, rho, theta, threshold, min_line_len, max_line_gap)

    LaneLines_image = weighted_img(Lane_lines, image, α=0.8, β=1., γ=0.)

    return LaneLines_image

# Defining current directory as working path
dir_path = os.path.dirname(os.path.realpath(__file__))

# Processing frames of video for lane line detection
video_output = dir_path+'/test_videos_output/solidWhiteRight.mp4'
clip = VideoFileClip(dir_path+"/test_videos/solidWhiteRight.mp4")
video_clip = clip.fl_image(process_image)
video_clip.write_videofile(video_output, audio=False)
