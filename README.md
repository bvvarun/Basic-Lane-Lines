# FindingLaneLines
---

## Overview

When we drive, we use our eyes to decide where to go.  The lines on the road that show us where the lanes are act as our constant reference for where to steer the vehicle.  Naturally, one of the first things we would like to do in developing a self-driving car is to automatically detect lane lines using an algorithm.
The goal of this project is to write a simple software pipeline to detect lane lines in images using Python and OpenCV.

---

## Folder structure

* The file `FindLaneLines_Library.py` contains the functions that are used to convert image to gray scale, find edges using canny edge transform, mask region of interest, find lane lines using hough transform and finally viusalize lane lines on original image.
* The file `ImageProcessing.py` contains the SW pipeline to identify lane lines on single images and draw the lines back on the image.
* The file `VideoProcessing.py` contains SW pipeline to process the frames of video to identify lane lines and display them back on the written video.
* There are few single images and videos present in folders `./test_images/test*.jpg` and `./test_video/*.mp4` that are used to test the pipeline.
* The folder `./test_ videos_output`contains the results of an example video tested on the pipeline.
* The file `writeup.pdf` file is the final report explaining briefly the steps taken in the pipeline.
---
