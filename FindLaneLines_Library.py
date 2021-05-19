#importing some useful packages
import numpy as np
import cv2
import math

#------------------------------------ Fucntion to convert image to grayscale-----------------------------------------------------------
def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#------------------------------------- Fucntion to apply canny transform---------------------------------------------------------------
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

#------------------------------------ Fucntion to apply gaussian blur------------------------------------------------------------------
def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

#------------------------------------ Fucntion to apply mask identifying region of interest-----------------------------------------------
def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

#------------------------------------- Fucntion to extrapolate lane lines from hough transform ---------------------------------------------------------------
def draw_lines(img, lines, color=[255, 0, 0], thickness=12):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    #Defining arrays to sort slopes and intercepts of left and right lane lines
    LeftLineSlopes = []
    LeftLineIntercepts = []
    RightLineSlopes = []
    RightLineIntercepts = []
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            
            #Calculating slopes and intercepts
            slope = (y2-y1)/(x2-x1)
            intercept = y1-x1*slope
            
            #sorting positive and negative slopes to identify left and right lane lines
            if slope > 0:
                LeftLineSlopes.append(slope)
                LeftLineIntercepts.append(intercept)
            else:
                RightLineSlopes.append(slope)
                RightLineIntercepts.append(intercept)
    
    #Calculating average of slopes and intercepts
    AvgLeftLineSlope = np.mean(LeftLineSlopes)
    AvgLeftLineIntercept = np.mean(LeftLineIntercepts)
    AvgRightLineSlope = np.mean(RightLineSlopes)
    AvgRightLineIntercept = np.mean(RightLineIntercepts)
    
    #Defining line coordinates using average slopes and intercepts
    if AvgLeftLineSlope!=0 and AvgRightLineSlope!=0:
        x1_LeftLine = int((540 - AvgLeftLineIntercept)/AvgLeftLineSlope)
        y1_LeftLine = 540
        x2_LeftLine = int((320 - AvgLeftLineIntercept)/AvgLeftLineSlope)
        y2_LeftLine = 320

        x1_RightLine = int((540 - AvgRightLineIntercept)/AvgRightLineSlope)
        y1_RightLine = 540
        x2_RightLine = int((320 - AvgRightLineIntercept)/AvgRightLineSlope)
        y2_RightLine = 320
        
    #Defining left and right lane lines
    cv2.line(img, (x1_LeftLine,y1_LeftLine),(x2_LeftLine,y2_LeftLine), color, thickness)
    cv2.line(img, (x1_RightLine,y1_RightLine),(x2_RightLine,y2_RightLine), color, thickness)
        
#------------------------------------- Fucntion to apply hough transform---------------------------------------------------------------
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

#------------------------------------- Fucntion to visualize lane lines on original image---------------------------------------------------------------
def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)




