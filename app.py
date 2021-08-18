# import relevant packages
import streamlit as st
from streamlit.hashing import _CodeHasher
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
from time import sleep
import math
import os
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from sklearn.linear_model import LinearRegression


st.set_page_config(
    page_title="Self Driving Car ND", # String or None. Strings get appended with "• Streamlit". 
    page_icon=":blue_car:", # String, anything supported by st.image, or None.
    layout="centered", # Can be "centered" or "wide". In the future also "dashboard", etc.
    initial_sidebar_state="expanded", # Can be "auto", "expanded", "collapsed"
)

try:
    # Before Streamlit 0.65
    from streamlit.ReportThread import get_report_ctx
    from streamlit.server.Server import Server
except ModuleNotFoundError:
    # After Streamlit 0.65
    from streamlit.report_thread import get_report_ctx
    from streamlit.server.server import Server


# initiate linear regression
reg = LinearRegression()

# Initiate global variables
r_x1 = 0
r_x2 = 0
r_x3 = 0
l_x1 = 0
l_x2 = 0
l_x3 = 0


# initialize the default values of the parameters
param_kernel_size = 5
param_low_threshold = 50
param_high_threshold = 150
param_rho = 1
param_theta = np.pi/180
param_threshold = 40
param_min_line_length = 10
param_max_line_gap = 70


# Helper Functions
def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

 
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


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


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
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
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


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


def process_image(img):
    """Note: The output you return should be a color image (3 channel) for processing video below."""
    
    global param_kernel_size
    global param_low_threshold
    global param_high_threshold
    global param_rho
    global param_theta
    global param_threshold
    global param_min_line_length
    global param_max_line_gap
    
    # read and make a copy of the image
    image = np.copy(img)
    
    # convert the image to grayscale
    gray_image = grayscale(image)
    
    # Define a kernel size and apply Gaussian smoothing
    blur_gray = gaussian_blur(gray_image, kernel_size = param_kernel_size)

    # Define our parameters for Canny and apply
    edges = canny(blur_gray, low_threshold = param_low_threshold, high_threshold = param_high_threshold)

    # This time we are defining a four sided polygon to mask
    imshape = image.shape
    vertices = np.array([[(.55*imshape[1], .60*imshape[0]), 
                        (.45*imshape[1], .60*imshape[0]),  
                        (.15*imshape[1], .90*imshape[0]), 
                        (.30*imshape[1], .90*imshape[0]), 
                        (.50*imshape[1], .60*imshape[0]),
                        (.70*imshape[1], .90*imshape[0]),
                        (.85*imshape[1], .90*imshape[0]),
                        (.55*imshape[1], .60*imshape[0])]], dtype=np.int32)
    
    masked_edges = region_of_interest(edges, vertices)

    # draw lines using hough transform
    line_image = hough_lines(masked_edges, rho = param_rho, theta = param_theta, threshold = param_threshold, 
                             min_line_len = param_min_line_length, max_line_gap = param_max_line_gap)
 
    # perform weighted addition of line_image and original image to potray lane markings
    image_lanes = weighted_img(line_image, image, α=1.0, β=1.0, γ=0.)
    result = image_lanes
    
    # return the final image where lines are drawn on lanes
    return result


def add_image_line(img, x1, y1, x2, y2, x3, color, thickness):
    imshape = img.shape
    x1 = int(np.round(x1, 0))
    x2 = int(np.round(x2, 0))
    x3 = int(np.round(x3, 0))
    dx_r = int(np.round(x1 + (imshape[1]/100), 0))
    dx_l = int(np.round(x1 - (imshape[1]/100), 0))
    dy = int(np.round(y2 + (imshape[0]/50), 0))
    y1 = int(np.round(y1, 0))
    y2 = int(np.round(y2, 0))
    # draw the lines
    cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    cv2.line(img, (dx_r, y1), (x3, dy), color, thickness)
    cv2.line(img, (dx_l, y1), (x3, dy), color, thickness)


def draw_lines_extrapolated(img, lines, color=[9, 219, 44], thickness=5):
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
    
    global r_x1
    global r_x2
    global r_x3
    global l_x1
    global l_x2
    global l_x3
    
    rc = np.array([])
    lc = np.array([])
    rx = np.array([])
    ry = np.array([])
    lx = np.array([])
    ly = np.array([])
    
    imshape = img.shape
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            center = [(x1+x2)/2, (y1+y2)/2]
    
            try:
                slope = (y2-y1)/(x2-x1)
            except ZeroDivisionError:
                slope = np.inf            
            
            if slope > 0.5 and slope < 10 and x1 > (.50*imshape[1]) and x2 > (.50*imshape[1]):
                rc = np.append(rc, center)
                rx = np.append(rx, [x1,x2])
                ry = np.append(ry, [y1,y2])
                
            elif slope < -0.5 and slope > -10 and x1 < (.50*imshape[1]) and x2 < (.50*imshape[1]):
                lc = np.append(lc, center)
                lx = np.append(lx, [x1,x2])
                ly = np.append(ly, [y1,y2])
    

    r_center = np.mean(rc, axis = 0)
    l_center = np.mean(lc, axis = 0)
    y1 = imshape[0]
    y2 = imshape[0]*0.62
    
    if not np.isnan(r_center).all():
        rx = rx.reshape(-1,1)
        reg.fit(rx, ry)
        r_slope, r_intercept = reg.coef_[0], reg.intercept_
        if r_slope > 0.5 and r_slope < 10:
            r_x1 = (y1 - r_intercept) / r_slope
            r_x2 = (y2 - r_intercept) / r_slope
            r_x3 = ((y2 + (imshape[0]/50))- r_intercept) / r_slope
            add_image_line(img, r_x1, y1, r_x2, y2, r_x3, color, thickness)
        else:
            add_image_line(img, r_x1, y1, r_x2, y2, r_x3, color, thickness)
        
    elif np.isnan(r_center).all():
        add_image_line(img, r_x1, y1, r_x2, y2, r_x3, color, thickness)
    
    if not np.isnan(l_center).all():
        lx = lx.reshape(-1,1)
        reg.fit(lx, ly)
        l_slope, l_intercept = reg.coef_[0], reg.intercept_
        if l_slope < -0.5 and l_slope > -10:
            l_x1 = (y1 - l_intercept) / l_slope
            l_x2 = (y2 - l_intercept) / l_slope
            l_x3 = ((y2 + (imshape[0]/50))- l_intercept) / l_slope
            add_image_line(img, l_x1, y1, l_x2, y2, l_x3, color, thickness)
        else:
            add_image_line(img, l_x1, y1, l_x2, y2, l_x3, color, thickness)
    
    elif np.isnan(l_center).all():
        add_image_line(img, l_x1, y1, l_x2, y2, l_x3, color, thickness)


def hough_lines_extrapolated(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), 
                            minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines_extrapolated(line_img, lines)
    return line_img


def process_image_extrapolated(img):
    """Note: The output you return should be a color image (3 channel) for processing video below."""
    
    global param_kernel_size
    global param_low_threshold
    global param_high_threshold
    global param_rho
    global param_theta
    global param_threshold
    global param_min_line_length
    global param_max_line_gap
    
    # read and make a copy of the image
    image = np.copy(img)
    
    # convert the image to grayscale
    gray_image = grayscale(image)
    
    # Define a kernel size and apply Gaussian smoothing
    blur_gray = gaussian_blur(gray_image, kernel_size = param_kernel_size)

    # Define our parameters for Canny and apply
    edges = canny(blur_gray, low_threshold = param_low_threshold, high_threshold = param_high_threshold)

    # This time we are defining a four sided polygon to mask
    imshape = image.shape
    vertices = np.array([[(.55*imshape[1], .60*imshape[0]), 
                        (.45*imshape[1], .60*imshape[0]),  
                        (.15*imshape[1], .90*imshape[0]), 
                        (.30*imshape[1], .90*imshape[0]), 
                        (.50*imshape[1], .60*imshape[0]),
                        (.70*imshape[1], .90*imshape[0]),
                        (.85*imshape[1], .90*imshape[0]),
                        (.55*imshape[1], .60*imshape[0])]], dtype=np.int32)
    
    masked_edges = region_of_interest(edges, vertices)

    # draw lines using hough transform
    line_image = hough_lines_extrapolated(masked_edges, rho = param_rho, theta = param_theta, threshold = param_threshold, 
                                          min_line_len = param_min_line_length, max_line_gap = param_max_line_gap)
 
    # perform weighted addition of line_image and original image to potray lane markings
    image_lanes = weighted_img(line_image, image, α=1.0, β=1.0, γ=0.)
    result = image_lanes
    
    # return the final image where lines are drawn on lanes
    return result



def draw_lines_extrapolated_stable(img, lines, color=[9, 219, 44], thickness=5):
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
    
    global r_x1
    global r_x2
    global r_x3
    global l_x1
    global l_x2
    global l_x3
    
    rc = np.array([])
    lc = np.array([])
    rx = np.array([])
    ry = np.array([])
    lx = np.array([])
    ly = np.array([])
    
    imshape = img.shape
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            center = [(x1+x2)/2, (y1+y2)/2]
    
            try:
                slope = (y2-y1)/(x2-x1)
            except ZeroDivisionError:
                slope = np.inf            
            
            if slope > 0.5 and slope < 10 and x1 > (.50*imshape[1]) and x2 > (.50*imshape[1]):
                rc = np.append(rc, center)
                rx = np.append(rx, [x1,x2])
                ry = np.append(ry, [y1,y2])
                
            elif slope < -0.5 and slope > -10 and x1 < (.50*imshape[1]) and x2 < (.50*imshape[1]):
                lc = np.append(lc, center)
                lx = np.append(lx, [x1,x2])
                ly = np.append(ly, [y1,y2])
    

    r_center = np.mean(rc, axis = 0)
    l_center = np.mean(lc, axis = 0)
    y1 = imshape[0]
    y2 = imshape[0]*0.65
    
    if not np.isnan(r_center).all():
        rx = rx.reshape(-1,1)
        reg.fit(rx, ry)
        r_slope, r_intercept = reg.coef_[0], reg.intercept_
        if r_slope > 0.5 and r_slope < 10:
            r_x1_new = (y1 - r_intercept) / r_slope
            r_x2_new = (y2 - r_intercept) / r_slope
            r_x3_new = ((y2 + (imshape[0]/50)) - r_intercept) / r_slope
            
            if r_x1 == 0:
                learning_rate = 1
            else:
                learning_rate = 0.2
                
            r_x1 = (learning_rate * r_x1_new) + ((1 - learning_rate) * r_x1)
            r_x2 = (learning_rate * r_x2_new) + ((1 - learning_rate) * r_x2)
            r_x3 = (learning_rate * r_x3_new) + ((1 - learning_rate) * r_x3)
            add_image_line(img, r_x1, y1, r_x2, y2, r_x3, color, thickness)
        else:
            add_image_line(img, r_x1, y1, r_x2, y2, r_x3, color, thickness)
        
    elif np.isnan(r_center).all():
        add_image_line(img, r_x1, y1, r_x2, y2, r_x3, color, thickness)
    
    if not np.isnan(l_center).all():
        lx = lx.reshape(-1,1)
        reg.fit(lx, ly)
        l_slope, l_intercept = reg.coef_[0], reg.intercept_
        if l_slope < -0.5 and l_slope > -10:
            l_x1_new = (y1 - l_intercept) / l_slope
            l_x2_new = (y2 - l_intercept) / l_slope
            l_x3_new = ((y2 + (imshape[0]/50)) - l_intercept) / l_slope
            
            if l_x1 == 0:
                learning_rate = 1
            else:
                learning_rate = 0.2
                
            l_x1 = (learning_rate * l_x1_new) + ((1 - learning_rate) * l_x1)
            l_x2 = (learning_rate * l_x2_new) + ((1 - learning_rate) * l_x2)
            l_x3 = (learning_rate * l_x3_new) + ((1 - learning_rate) * l_x3)
            add_image_line(img, l_x1, y1, l_x2, y2, l_x3, color, thickness)
        else:
            add_image_line(img, l_x1, y1, l_x2, y2, l_x3, color, thickness)
    
    elif np.isnan(l_center).all():
        add_image_line(img, l_x1, y1, l_x2, y2, l_x3, color, thickness)



def hough_lines_extrapolated_stable(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), 
                            minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines_extrapolated_stable(line_img, lines)
    return line_img



def process_image_extrapolated_stable(img):
    """Note: The output you return should be a color image (3 channel) for processing video below."""
    
    global param_kernel_size
    global param_low_threshold
    global param_high_threshold
    global param_rho
    global param_theta
    global param_threshold
    global param_min_line_length
    global param_max_line_gap
    
    # read and make a copy of the image
    image = np.copy(img)
    
    # convert the image to grayscale
    gray_image = grayscale(image)
    
    # Define a kernel size and apply Gaussian smoothing
    blur_gray = gaussian_blur(gray_image, kernel_size = param_kernel_size)

    # Define our parameters for Canny and apply
    edges = canny(blur_gray, low_threshold = param_low_threshold, high_threshold = param_high_threshold)

    # This time we are defining a four sided polygon to mask
    imshape = image.shape
    vertices = np.array([[(.55*imshape[1], .60*imshape[0]), 
                        (.45*imshape[1], .60*imshape[0]),  
                        (.15*imshape[1], .90*imshape[0]), 
                        (.30*imshape[1], .90*imshape[0]), 
                        (.50*imshape[1], .60*imshape[0]),
                        (.70*imshape[1], .90*imshape[0]),
                        (.85*imshape[1], .90*imshape[0]),
                        (.55*imshape[1], .60*imshape[0])]], dtype=np.int32)
    
    masked_edges = region_of_interest(edges, vertices)

    # draw lines using hough transform
    line_image = hough_lines_extrapolated_stable(masked_edges, rho = param_rho, theta = param_theta, threshold = param_threshold, 
                                                 min_line_len = param_min_line_length, max_line_gap = param_max_line_gap)
 
    # perform weighted addition of line_image and original image to potray lane markings
    image_lanes = weighted_img(line_image, image, α=1.0, β=1.0, γ=0.)
    result = image_lanes
    
    # return the final image where lines are drawn on lanes
    return result


def add_image_fill(img, r_x1, r_x2, l_x1, l_x2, y1, y2, ignore_mask_color):
    vertices = np.array([[(r_x1, y1), 
                          (l_x1, y1), 
                          (l_x2, y2), 
                          (r_x2, y2)]], dtype = np.int32)
    cv2.fillPoly(img, vertices, ignore_mask_color)


def draw_lines_fill(img, lines, color=[9, 219, 44], thickness=5):
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
    
    global r_x1
    global r_x2
    global r_x3
    global l_x1
    global l_x2
    global l_x3
    
    rc = np.array([])
    lc = np.array([])
    rx = np.array([])
    ry = np.array([])
    lx = np.array([])
    ly = np.array([])
    
    imshape = img.shape
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            center = [(x1+x2)/2, (y1+y2)/2]
    
            try:
                slope = (y2-y1)/(x2-x1)
            except ZeroDivisionError:
                slope = np.inf            
            
            if slope > 0.5 and slope < 10 and x1 > (.50*imshape[1]) and x2 > (.50*imshape[1]):
                rc = np.append(rc, center)
                rx = np.append(rx, [x1,x2])
                ry = np.append(ry, [y1,y2])
                
            elif slope < -0.5 and slope > -10 and x1 < (.50*imshape[1]) and x2 < (.50*imshape[1]):
                lc = np.append(lc, center)
                lx = np.append(lx, [x1,x2])
                ly = np.append(ly, [y1,y2])
    

    r_center = np.mean(rc, axis = 0)
    l_center = np.mean(lc, axis = 0)
    y1 = imshape[0]
    y2 = imshape[0]*0.65
    
    if not np.isnan(r_center).all():
        rx = rx.reshape(-1,1)
        reg.fit(rx, ry)
        r_slope, r_intercept = reg.coef_[0], reg.intercept_
        if r_slope > 0.5 and r_slope < 10:
            r_x1_new = (y1 - r_intercept) / r_slope
            r_x2_new = (y2 - r_intercept) / r_slope
            r_x3_new = ((y2 + (imshape[0]/50)) - r_intercept) / r_slope
            
            if r_x1 == 0:
                learning_rate = 1
            else:
                learning_rate = 0.2
                
            r_x1 = (learning_rate * r_x1_new) + ((1 - learning_rate) * r_x1)
            r_x2 = (learning_rate * r_x2_new) + ((1 - learning_rate) * r_x2)
            r_x3 = (learning_rate * r_x3_new) + ((1 - learning_rate) * r_x3)
        
    
    if not np.isnan(l_center).all():
        lx = lx.reshape(-1,1)
        reg.fit(lx, ly)
        l_slope, l_intercept = reg.coef_[0], reg.intercept_
        if l_slope < -0.5 and l_slope > -10:
            l_x1_new = (y1 - l_intercept) / l_slope
            l_x2_new = (y2 - l_intercept) / l_slope
            l_x3_new = ((y2 + (imshape[0]/50)) - l_intercept) / l_slope
            
            if l_x1 == 0:
                learning_rate = 1
            else:
                learning_rate = 0.2
                
            l_x1 = (learning_rate * l_x1_new) + ((1 - learning_rate) * l_x1)
            l_x2 = (learning_rate * l_x2_new) + ((1 - learning_rate) * l_x2)
            l_x3 = (learning_rate * l_x3_new) + ((1 - learning_rate) * l_x3)
    
    # fill the image
    add_image_fill(img, r_x1, r_x2, l_x1, l_x2, y1, y2, color)


def hough_lines_fill(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), 
                            minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines_fill(line_img, lines)
    return line_img



def process_image_fill(img):
    """Note: The output you return should be a color image (3 channel) for processing video below."""
    
    global param_kernel_size
    global param_low_threshold
    global param_high_threshold
    global param_rho
    global param_theta
    global param_threshold
    global param_min_line_length
    global param_max_line_gap
    
    # read and make a copy of the image
    image = np.copy(img)
    
    # convert the image to grayscale
    gray_image = grayscale(image)
    
    # Define a kernel size and apply Gaussian smoothing
    blur_gray = gaussian_blur(gray_image, kernel_size = param_kernel_size)

    # Define our parameters for Canny and apply
    edges = canny(blur_gray, low_threshold = param_low_threshold, high_threshold = param_high_threshold)

    # This time we are defining a four sided polygon to mask
    imshape = image.shape
    vertices = np.array([[(.55*imshape[1], .60*imshape[0]), 
                        (.45*imshape[1], .60*imshape[0]),  
                        (.15*imshape[1], .90*imshape[0]), 
                        (.30*imshape[1], .90*imshape[0]), 
                        (.50*imshape[1], .60*imshape[0]),
                        (.70*imshape[1], .90*imshape[0]),
                        (.85*imshape[1], .90*imshape[0]),
                        (.55*imshape[1], .60*imshape[0])]], dtype=np.int32)
    
    masked_edges = region_of_interest(edges, vertices)

    # draw lines using hough transform
    line_image = hough_lines_fill(masked_edges, rho = param_rho, theta = param_theta, threshold = param_threshold, 
                                  min_line_len = param_min_line_length, max_line_gap = param_max_line_gap)
 
    # perform weighted addition of line_image and original image to potray lane markings
    image_lanes = weighted_img(line_image, image, α=1.0, β=0.4, γ=0.)
    result = image_lanes
    
    # return the final image where lines are drawn on lanes
    return result



def main():
    state = _get_state()
    pages = {
        "Home": project_explanation,
        "Parameters": parameter_experiment,
        "Hough Lines": hough_lines_page,
        "Extrapolate Lines": extrapolate_lines,
        "Stabilize Lines": stabilize_lines,
        "Fill Lines": fill_lines
    }

    st.sidebar.title(":bookmark_tabs: Navigation")
    
    # Display the selected page with the session state
    # if state.clicked1 or state.clicked2 or state.clicked3:
        # page = "Your Strategy"
    # else:
        # page = st.sidebar.radio("Select your page", tuple(pages.keys()),)

    page = st.sidebar.radio("Select your page", tuple(pages.keys()),)
    pages[page](state)

    # Mandatory to avoid rollbacks with widgets, must be called at the end of your app
    state.sync()


def project_explanation(state):
    st.title("Finding Lane Lines on the Road")
    st.markdown("[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)")
    st.write('----')
    image1 = Image.open("images/image1.jpg")
    st.image(image1, use_column_width=True)
    

    st.markdown('`Welcome to the web application to detect lane lines on the road using Computer Vision.`')
    st.markdown('<p style="text-align: justify;">When we drive, we use our eyes to decide where to go. \
                 The lines on the road that show us where the lanes are act as our constant reference \
                 for where to steer the vehicle. Naturally, one of the first things we would like to do \
                 in developing a self-driving car is to automatically detect lane lines using an algorithm. \
                 In this project you will detect lane lines in images using Python and OpenCV. OpenCV means \
                 "Open-Source Computer Vision", which is a package that has many useful tools for analyzing \
                 images.</p>', unsafe_allow_html=True)

    image3 = Image.open("images/image5.jpg")
    st.image(image3, use_column_width=True)

    col5, col6 = st.beta_columns((1,1.5))
    col5.write('----')

    col3, col4 = st.beta_columns((1,1.4))
    col4.markdown('<p style="text-align: justify;">To complete the project, two files will be submitted: \
                   a file containing project code and a file containing a brief write up explaining your \
                   solution. We have included template files to be used both for the code and the writeup. \
                   The code file is called P1.ipynb and the writeup template is writeup_template.md</p>', unsafe_allow_html=True)
    # col4.markdown("`Go to the Get Started page to enter details ->`")
    image2 = Image.open("examples/laneLines_thirdPass.jpg")
    col3.image(image2, use_column_width=True)

    col7, col8 = st.beta_columns((1,1.5))
    col8.write('----')

    



def parameter_experiment(state):
    # st.title("Experiment with parameters values")
    url = "Experiment with Parameters values"
    st.markdown(f'<p style="background-color:#0686c2;color:#2b2b2b;font-size:34px;font-weight:bold;font-family:sans-serif;border-radius:2%;text-align:center">{url}</p>', unsafe_allow_html=True)
    # st.write('------')

    col1, col2 = st.beta_columns(2)

    # select image
    test_images = os.listdir("test_images/")
    state.test_image = col1.selectbox('select image', test_images)
    root_directory = "test_images/"
    image_path = state.test_image
    path = os.path.join(root_directory, image_path)
    image1 = Image.open(path)
    col1.image(image1, use_column_width=True)

    # select filter
    filters = ['Original', 'Grayscale', 'Blur', 'Canny Edge Detection', 'Masked Edges']
    state.filter = col2.selectbox('select filter', filters, 0)
    image1 = np.array(image1.convert('RGB'))
    if state.filter == 'Original':
        image2 = image1.copy()
        col2.image(image2, use_column_width=True, caption=None, clamp = True)
    elif state.filter == 'Grayscale':
        image2 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
        col2.image(image2, channels='GRAY', use_column_width=True)
    elif state.filter == 'Blur':
        image2 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
        state.kernel_size = st.slider("Select blur level (recomemded: 5)", 1, 11, 5, 2)
        st.text('Kernel size: {}'.format(state.kernel_size))
        image2 = cv2.GaussianBlur(image2, (state.kernel_size, state.kernel_size), 0)
        col2.image(image2, use_column_width=True)
    elif state.filter == 'Canny Edge Detection':
        image2 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
        state.kernel_size = st.slider("Select blur level (recomemded: 5)", 1, 11, 5, 2)
        image2 = cv2.GaussianBlur(image2, (state.kernel_size, state.kernel_size), 0)
        state.threshold_values = st.slider('Select the threshold range for canny edge \
                                            detection: (recomended: 50, 150)', 0, 255, (50, 150))
        state.low_threshold = state.threshold_values[0]
        state.high_threshold = state.threshold_values[1]
        image2 = cv2.Canny(image2, state.low_threshold, state.high_threshold)
        col2.image(image2, use_column_width=True)
    elif state.filter == 'Masked Edges':
        image2 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
        state.kernel_size = st.slider("Select blur level (recomemded: 5)", 1, 11, 5, 2)
        image2 = cv2.GaussianBlur(image2, (state.kernel_size, state.kernel_size), 0)
        state.threshold_values = st.slider('Select the threshold range for canny edge \
                                            detection: (recomended: 50, 150)', 0, 255, (50, 150))
        state.low_threshold = state.threshold_values[0]
        state.high_threshold = state.threshold_values[1]
        image2 = cv2.Canny(image2, state.low_threshold, state.high_threshold)
        # Next we'll create a masked edges image using cv2.fillPoly()
        mask = np.zeros_like(image2)   
        #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(image2.shape) > 2:
            channel_count = image2.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255
        # This time we are defining a four sided polygon to mask
        imshape = image1.shape
        vertices = np.array([[(.55*imshape[1], .60*imshape[0]), 
                              (.45*imshape[1], .60*imshape[0]),  
                              (.15*imshape[1], .90*imshape[0]), 
                              (.30*imshape[1], .90*imshape[0]), 
                              (.50*imshape[1], .60*imshape[0]),
                              (.70*imshape[1], .90*imshape[0]),
                              (.85*imshape[1], .90*imshape[0]),
                              (.55*imshape[1], .60*imshape[0])]], dtype=np.int32)
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        image2 = cv2.bitwise_and(image2, mask)
        image2 = np.dstack((image2, image2, image2)) 
        # identify the region of interest
        isClosed = True
        color = (0, 195, 255)
        thickness = 2
        vertices = vertices.reshape((-1, 1, 2))
        cv2.polylines(image2, [vertices], isClosed, color, thickness)
        col2.image(image2, use_column_width=True)



def hough_lines_page(state):
    # st.title("Draw Hough lines on the image")
    text1 = "Hough Lines Pipeline"
    st.markdown(f'<p style="background-color:#0686c2;color:#2b2b2b;font-size:34px;font-weight:bold;font-family:sans-serif;border-radius:2%;text-align:center">{text1}</p>', unsafe_allow_html=True)
    st.write('----')
    st.markdown('<p style="text-align: justify;">The goal is to piece together a pipeline \
                 to detect the line segments in the image, then average/extrapolate them \
                 and draw them onto the image for display (as displayed below). Once we have a working \
                 pipeline, then we can move on to work on the video stream.</p>', unsafe_allow_html=True)                         

    col1, col2, col3 = st.beta_columns([6,1,6])
    image1 = Image.open("test_images/solidWhiteRight.jpg")
    image2 = Image.open("examples/line-segments-example.jpg")
    col1.image(image1, use_column_width=True, caption="The orginal image used to detect land lines on the road")
    col2.markdown(' ')
    col2.markdown(' ')
    col2.markdown(' ')
    col2.markdown(' ')
    col2.markdown('<p style="font-size:30px;"><b>&#8594;</b></p>', unsafe_allow_html=True)
    col3.image(image2, use_column_width=True, caption="Output detecting line segments using helper functions")
    
    st.write(' ')
    st.write(' ')
    st.write(' ')
    st.write(' ')
    
    col4, col5, col6 = st.beta_columns([5,4,5])
    # col4.markdown('----')
    # col5.write(':: Hough Lines Pipeline ::')
    # col6.markdown('----')

    text2 = ':: Hough Lines Pipeline on Images ::'
    st.markdown(f'<p style="background-color:#32a894;color:#2b2b2b;font-size:24px;font-weight:bold;font-family:sans-serif;border-radius:2%;text-align:center">{text2}</p>', unsafe_allow_html=True)
    st.write('----')

    # select image
    test_images = os.listdir("test_images/")
    test_images_challenge = os.listdir("test_images_challenge/")
    test_images.extend(test_images_challenge)
    state.test_image = st.selectbox('select image', test_images)
    root_directory = "test_images"
    root_directory_challenge = "test_images_challenge"
    image_path = state.test_image
    path = os.path.join(root_directory, image_path)
    path_challenge = os.path.join(root_directory_challenge, image_path)
    try:
        image1 = Image.open(path)
    except FileNotFoundError:
        image1 = Image.open(path_challenge)
    original_image = np.array(image1.convert('RGB'))
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
    state.kernel_size = st.slider("Select blur level: (recomemded: 5)", 1, 11, 5, 2)
    global param_kernel_size
    param_kernel_size = state.kernel_size
    blur_image = cv2.GaussianBlur(gray_image, (state.kernel_size, state.kernel_size), 0)
    state.threshold_values = st.slider('Select the threshold range for canny edge \
                                        detection: (recomended: 50, 150)', 0, 255, (50, 150))
    state.low_threshold = state.threshold_values[0]
    state.high_threshold = state.threshold_values[1]
    global param_low_threshold
    global param_high_threshold
    param_low_threshold = state.low_threshold
    param_high_threshold = state.high_threshold
    canny_image = cv2.Canny(blur_image, state.low_threshold, state.high_threshold)
    # Next we'll create a masked edges image using cv2.fillPoly()
    mask = np.zeros_like(canny_image)   
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(canny_image.shape) > 2:
        channel_count = canny_image.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    # This time we are defining a four sided polygon to mask
    imshape = original_image.shape
    vertices = np.array([[(.55*imshape[1], .60*imshape[0]), 
                          (.45*imshape[1], .60*imshape[0]),  
                          (.15*imshape[1], .90*imshape[0]), 
                          (.30*imshape[1], .90*imshape[0]), 
                          (.50*imshape[1], .60*imshape[0]),
                          (.70*imshape[1], .90*imshape[0]),
                          (.85*imshape[1], .90*imshape[0]),
                          (.55*imshape[1], .60*imshape[0])]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    mask_image = cv2.bitwise_and(canny_image, mask)
    stacked_canny = np.dstack((canny_image, canny_image, canny_image)) 
    # identify the region of interest
    isClosed = True
    color = (0, 195, 255)
    thickness = 2
    vertices = vertices.reshape((-1, 1, 2))
    cv2.polylines(stacked_canny, [vertices], isClosed, color, thickness)
    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    # distance resolution in pixels of the Hough grid
    state.rho = st.slider("Select rho value for resolution: (recomemded: 1 pixel)", 1, 5, 1, 1)
    # angular resolution in radians of the Hough grid
    state.angle = st.slider("Select angle for theta: (recomemded: 180)", 1, 360, 180, 1)
    state.theta = np.pi/state.angle
    # minimum number of votes (intersections in Hough grid cell)
    state.threshold = st.slider("Select minimum number of votes to form a line: (recomemded: 40)", 1, 100, 40, 1)
    # minimum number of pixels making up a line
    state.min_line_length = st.slider("Select minimum number of pixels making up a line: (recomemded: 10)", 1, 100, 10, 1)
    # maximum gap in pixels between connectable line segments
    state.max_line_gap = st.slider("Select maximum gap in pixels between connectable line segments: (recomemded: 70)", 1, 100, 70, 1)
    
    global param_rho
    global param_theta
    global param_threshold
    global param_min_line_length
    global param_max_line_gap

    param_rho = state.rho
    param_theta = state.theta
    param_threshold = state.threshold
    param_min_line_length = state.min_line_length
    param_max_line_gap = state.max_line_gap

    line_image = np.copy(original_image)*0 # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(mask_image, state.rho, state.theta, state.threshold, np.array([]),
                            state.min_line_length, state.max_line_gap)
    
    # Iterate over the output "lines" and draw lines on a blank image
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)

    # Create a "color" binary image to combine with line image
    stacked_mask = np.dstack((mask_image, mask_image, mask_image)) 

    # Draw the lines on the edge image
    hough_canny = cv2.addWeighted(stacked_mask, 0.8, line_image, 1, 0)
    hough_original = cv2.addWeighted(original_image, 0.8, line_image, 1, 0)

    if st.button("Reset Parameters"):
        state.clear()
        # reset the default values of the parameters
        param_kernel_size = 5
        param_low_threshold = 50
        param_high_threshold = 150
        param_rho = 1
        param_theta = np.pi/180
        param_threshold = 40
        param_min_line_length = 10
        param_max_line_gap = 70

        

    st.markdown(' ')
    st.markdown(' ')
    st.markdown('> `The below images in the pipeline changes with respect to the parameters selected above`')
    st.markdown(' ')
    st.markdown(' ')
    
    col7, col8, col9, col10, col11 = st.beta_columns([4,1,4,1,4])
    # col7.markdown('<p style="text-align: center; color: gray">Original Image</p>', unsafe_allow_html=True)
    col7.markdown(f'<p style="background-color:#0686c2;color:#2b2b2b;font-weight:bold;font-family:sans-serif;border-radius:2%;text-align:center">Original Image</p>', unsafe_allow_html=True)
    col7.image(original_image, use_column_width=True)
    col8.markdown(' ')
    col8.markdown(' ')
    col8.markdown(' ')
    col8.markdown(' ')
    col8.markdown('<p style="font-size:30px;"><b>&#8594;</b></p>', unsafe_allow_html=True)
    # col9.markdown('<p style="text-align: center;">Gray Image</p>', unsafe_allow_html=True)
    col9.markdown(f'<p style="background-color:#0686c2;color:#2b2b2b;font-weight:bold;font-family:sans-serif;border-radius:2%;text-align:center">Gray Image</p>', unsafe_allow_html=True)
    col9.image(gray_image, use_column_width=True)
    col10.markdown(' ')
    col10.markdown(' ')
    col10.markdown(' ')
    col10.markdown(' ')
    col10.markdown('<p style="font-size:30px;"><b>&#8594;</b></p>', unsafe_allow_html=True)
    # col11.markdown('<p style="text-align: center;">Blur Image</p>', unsafe_allow_html=True)
    col11.markdown(f'<p style="background-color:#0686c2;color:#2b2b2b;font-weight:bold;font-family:sans-serif;border-radius:2%;text-align:center">Blur Image</p>', unsafe_allow_html=True)
    col11.image(blur_image, use_column_width=True)

    col12, col13, col14, col15, col16 = st.beta_columns([4,1,4,1,4])
    col16.markdown('<p style="font-size:30px; text-align: center;"><b>&#8595;</b></p>', unsafe_allow_html=True)
    
    col17, col18, col19, col20, col21 = st.beta_columns([4,1,4,1,4])
    # col17.markdown('<p style="text-align: center;">Masked Image</p>', unsafe_allow_html=True)
    col17.markdown(f'<p style="background-color:#0686c2;color:#2b2b2b;font-weight:bold;font-family:sans-serif;border-radius:2%;text-align:center">Masked Image</p>', unsafe_allow_html=True)
    col17.image(mask_image, use_column_width=True)
    col18.markdown(' ')
    col18.markdown(' ')
    col18.markdown(' ')
    col18.markdown(' ')
    col18.markdown('<p style="font-size:30px;"><b>&#8592;</b></p>', unsafe_allow_html=True)
    # col19.markdown('<p style="text-align: center;">Area of Interest</p>', unsafe_allow_html=True)
    col19.markdown(f'<p style="background-color:#0686c2;color:#2b2b2b;font-weight:bold;font-family:sans-serif;border-radius:2%;text-align:center">Area of Interest</p>', unsafe_allow_html=True)
    col19.image(stacked_canny, use_column_width=True)
    col20.markdown(' ')
    col20.markdown(' ')
    col20.markdown(' ')
    col20.markdown(' ')
    col20.markdown('<p style="font-size:30px;"><b>&#8592;</b></p>', unsafe_allow_html=True)
    # col21.markdown('<p style="text-align: center;">Canny Edges</p>', unsafe_allow_html=True)
    col21.markdown(f'<p style="background-color:#0686c2;color:#2b2b2b;font-weight:bold;font-family:sans-serif;border-radius:2%;text-align:center">Canny Edges</p>', unsafe_allow_html=True)
    col21.image(canny_image, use_column_width=True)

    col22, col23, col24, col25, col26 = st.beta_columns([4,1,4,1,4])
    col22.markdown('<p style="font-size:30px; text-align: center;"><b>&#8595;</b></p>', unsafe_allow_html=True)
    
    col27, col28, col29, col30, col31 = st.beta_columns([4,1,4,1,4])
    # col27.markdown('<p style="text-align: center;">Hough Lines Masked</p>', unsafe_allow_html=True)
    col27.markdown(f'<p style="background-color:#0686c2;color:#2b2b2b;font-weight:bold;font-family:sans-serif;border-radius:2%;text-align:center">Hough Lines Masked</p>', unsafe_allow_html=True)
    col27.image(hough_canny, use_column_width=True)
    col28.markdown(' ')
    col28.markdown(' ')
    col28.markdown(' ')
    col28.markdown(' ')
    col28.markdown('<p style="font-size:30px;"><b>&#8594;</b></p>', unsafe_allow_html=True)
    # col29.markdown('<p style="text-align: center;">Hough Lines Original</p>', unsafe_allow_html=True)
    col29.markdown(f'<p style="background-color:#0686c2;color:#2b2b2b;font-weight:bold;font-family:sans-serif;border-radius:2%;text-align:center">Hough Lines Original</p>', unsafe_allow_html=True)
    col29.image(hough_original, use_column_width=True)
        
    st.write(' ')
    st.write(' ')
    st.write(' ')
    st.write(' ')
    st.write(' ')
    st.write(' ')

    col32, col33, col34 = st.beta_columns([4.5,5,4.5])
    # col32.markdown('----')
    # col33.write(':: Hough lines on video stream ::')
    # col34.markdown('----')

    text3 = ':: Hough Lines Pipeline on Video Stream ::'
    st.markdown(f'<p style="background-color:#32a894;color:#2b2b2b;font-size:24px;font-weight:bold;font-family:sans-serif;border-radius:2%;text-align:center">{text3}</p>', unsafe_allow_html=True)
    st.write('----')
    
    st.write(' ')
    st.write(' ')

    st.markdown('<p style="text-align: justify;">You know what\'s cooler than drawing lanes over images? \
                Drawing lanes over video! We can test our solution on two provided videos and one optional \
                challenge video: <mark>solidWhiteRight.mp4</mark>, <mark>solidYellowLeft.mp4</mark>, and \
                <mark>challenge.mp4</mark></p>', unsafe_allow_html=True)                         

    col35, col36, col37 = st.beta_columns([6,1,6])
    video_file1 = open('test_videos/solidYellowLeft.mp4', 'rb')
    video_file2 = open('examples/raw-lines-example.mp4', 'rb')
    video_bytes1 = video_file1.read()
    video_bytes2 = video_file2.read()
    col35.video(video_bytes1)
    col36.markdown(' ')
    col36.markdown(' ')
    col36.markdown(' ')
    col36.markdown(' ')
    col36.markdown('<p style="font-size:30px;"><b>&#8594;</b></p>', unsafe_allow_html=True)
    col37.video(video_bytes2)
    
    # select video
    test_video = os.listdir("test_videos/")
    state.test_video = st.selectbox('select video', test_video, 2)
    video_root_directory = "test_videos"
    video_name = state.test_video
    video_path = os.path.join(video_root_directory, video_name)
    # st.write(video_name)
    # st.write(video_path)

    # test process pipeline on an image
    # image1 = Image.open("test_images/solidYellowCurve.jpg")
    # original_image = np.array(image1.convert('RGB'))
    # original_image = mpimg.imread('test_images/solidWhiteRight.jpg')
    # process = process_image(original_image)
    # st.image(process)

    # video_root_directory = "test_videos"
    # video_name = "solidYellowLeft.mp4"
    # video_path = os.path.join(video_root_directory, video_name)

    if video_name == "solidWhiteRight.mp4":
        prefix = "1_1_"
    elif video_name == "solidYellowLeft.mp4":
        prefix = "2_1_"
    elif video_name == "challenge.mp4":
        prefix = "3_1_"
    postfix = "_houghlines"
    video_name_new = prefix + video_name.split('.')[0] + postfix + "." + video_name.split('.')[1]
    # st.write(video_name_new)

    # Import everything needed to edit/save/watch video clips
    from moviepy.editor import VideoFileClip

    video_output = os.path.join("test_videos_app", video_name_new)

    # st.write(video_name_new)
    # st.write(video_output)

    button_processVid = st.button("Process Hough Lines on the Video")
    
    if button_processVid:
       
        errorholder = st.empty()
        errorholder.markdown('> `Wait for the process to finish. Refresh the web application if there is an error!`')

        placeholder = st.empty()
        gif_path = 'images/3.gif'
        placeholder.image(gif_path)
        placeholder.image('images/34.gif', use_column_width=True)

        ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
        ## To do so add .subclip(start_second,end_second) to the end of the line below
        ## Where start_second and end_second are integer values representing the start and end of the subclip
        ## You may also uncomment the following line for a subclip of the first 5 seconds
        # video_clip = VideoFileClip('test_videos/solidYellowLeft.mp4').subclip(0,5)
        video_clip = VideoFileClip(video_path)
        # st.write('ok step 1')
        # st.write(clip2)
        # video_clip.close()

        processed_clip = video_clip.fl_image(process_image)
        
        # temp_clip = clip2.rotate(180)
        # temp_clip.show()
        # temp_clip.preview(fps=15, audio=False)
        # st.write('ok step 2')
        
        processed_clip.write_videofile(str(video_output), audio=False)
        
        # my_bar = st.progress(0)
        # for percent_complete in range(100):
            # time.sleep(0.1)
            # my_bar.progress(percent_complete + 1)

        # temp_clip.write_videofile(yellow_output, audio=False)

        # video_clip.close()

        # st.write('ok step 3')
        # get_ipython().magic('time yellow_clip.write_videofile(yellow_output, audio=False)')

        errorholder.empty()
        placeholder.empty()

        video_file = open(str(video_output), 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes)
    






def extrapolate_lines(state):
    # st.title("Extrapolate Hough lines on the image")
    text1 = "Extraplated Hough Lines Pipeline"
    st.markdown(f'<p style="background-color:#0686c2;color:#2b2b2b;font-size:34px;font-weight:bold;font-family:sans-serif;border-radius:2%;text-align:center">{text1}</p>', unsafe_allow_html=True)
    st.write('----')
    st.markdown('<p style="text-align: justify;">At this point, we are successful with making the pipeline and tuning the parameters, \
                 and have the Hough line segments drawn onto the road, but what about identifying the full extent of the lane and \
                 marking it clearly as in the example video (P1_example.mp4)? Think about defining a line to run the full length of \
                 the visible lane based on the line segments you identified with the Hough Transform. The goal is to improvize the \
                 pipeline to detect the line segments in the image, then average/extrapolate them and draw them onto the image for \
                 display (as below).</p>', unsafe_allow_html=True)                         

    col1, col2, col3 = st.beta_columns([6,1,6])
    image1 = Image.open("examples/line-segments-example.jpg")
    image2 = Image.open("examples/laneLines_thirdPass.jpg")
    col1.image(image1, use_column_width=True, caption="The orginal image used to detect land lines on the road")
    col2.markdown(' ')
    col2.markdown(' ')
    col2.markdown(' ')
    col2.markdown(' ')
    col2.markdown('<p style="font-size:30px;"><b>&#8594;</b></p>', unsafe_allow_html=True)
    col3.image(image2, use_column_width=True, caption="Output detecting line segments using helper functions")
    
    st.write(' ')
    st.write(' ')
    st.write(' ')
    st.write(' ')
    
    col4, col5, col6 = st.beta_columns([5,4,5])
    # col4.markdown('----')
    # col5.write(':: Hough Lines Pipeline ::')
    # col6.markdown('----')

    text2 = ':: Extrapolated Hough Lines Pipeline on Images ::'
    st.markdown(f'<p style="background-color:#32a894;color:#2b2b2b;font-size:24px;font-weight:bold;font-family:sans-serif;border-radius:2%;text-align:center">{text2}</p>', unsafe_allow_html=True)
    st.write('----')

    # select image
    test_images = os.listdir("test_images/")
    test_images_challenge = os.listdir("test_images_challenge/")
    test_images.extend(test_images_challenge)
    state.test_image = st.selectbox('select image', test_images, 6)
    root_directory = "test_images"
    root_directory_challenge = "test_images_challenge"
    image_path = state.test_image
    path = os.path.join(root_directory, image_path)
    path_challenge = os.path.join(root_directory_challenge, image_path)
    try:
        image1 = Image.open(path)
    except FileNotFoundError:
        image1 = Image.open(path_challenge)
    original_image = np.array(image1.convert('RGB'))
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
    state.kernel_size = st.slider("Select blur level: (recomemded: 5)", 1, 11, 5, 2)
    global param_kernel_size
    param_kernel_size = state.kernel_size
    blur_image = cv2.GaussianBlur(gray_image, (state.kernel_size, state.kernel_size), 0)
    state.threshold_values = st.slider('Select the threshold range for canny edge \
                                        detection: (recomended: 50, 150)', 0, 255, (50, 150))
    state.low_threshold = state.threshold_values[0]
    state.high_threshold = state.threshold_values[1]
    global param_low_threshold
    global param_high_threshold
    param_low_threshold = state.low_threshold
    param_high_threshold = state.high_threshold
    canny_image = cv2.Canny(blur_image, state.low_threshold, state.high_threshold)
    # Next we'll create a masked edges image using cv2.fillPoly()
    mask = np.zeros_like(canny_image)
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(canny_image.shape) > 2:
        channel_count = canny_image.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    # This time we are defining a four sided polygon to mask
    imshape = original_image.shape
    vertices = np.array([[(.55*imshape[1], .60*imshape[0]),
                          (.45*imshape[1], .60*imshape[0]),  
                          (.15*imshape[1], .90*imshape[0]), 
                          (.30*imshape[1], .90*imshape[0]), 
                          (.50*imshape[1], .60*imshape[0]),
                          (.70*imshape[1], .90*imshape[0]),
                          (.85*imshape[1], .90*imshape[0]),
                          (.55*imshape[1], .60*imshape[0])]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    mask_image = cv2.bitwise_and(canny_image, mask)
    stacked_canny = np.dstack((canny_image, canny_image, canny_image)) 
    # identify the region of interest
    isClosed = True
    color = (0, 195, 255)
    thickness = 2
    vertices = vertices.reshape((-1, 1, 2))
    cv2.polylines(stacked_canny, [vertices], isClosed, color, thickness)
    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    # distance resolution in pixels of the Hough grid
    state.rho = st.slider("Select rho value for resolution: (recomemded: 1 pixel)", 1, 5, 1, 1)
    # angular resolution in radians of the Hough grid
    state.angle = st.slider("Select angle for theta: (recomemded: 180)", 1, 360, 180, 1)
    state.theta = np.pi/state.angle
    # minimum number of votes (intersections in Hough grid cell)
    state.threshold = st.slider("Select minimum number of votes to form a line: (recomemded: 40)", 1, 100, 40, 1)
    # minimum number of pixels making up a line
    state.min_line_length = st.slider("Select minimum number of pixels making up a line: (recomemded: 10)", 1, 100, 10, 1)
    # maximum gap in pixels between connectable line segments
    state.max_line_gap = st.slider("Select maximum gap in pixels between connectable line segments: (recomemded: 70)", 1, 100, 70, 1)
    
    global param_rho
    global param_theta
    global param_threshold
    global param_min_line_length
    global param_max_line_gap

    param_rho = state.rho
    param_theta = state.theta
    param_threshold = state.threshold
    param_min_line_length = state.min_line_length
    param_max_line_gap = state.max_line_gap

    line_image = np.copy(original_image)*0 # creating a blank to draw lines on
    line_image_reduced = np.copy(original_image)*0 # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(mask_image, state.rho, state.theta, state.threshold, np.array([]),
                            state.min_line_length, state.max_line_gap)

    # Iterate over the output "lines" and draw lines on a blank image
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
    
    # Iterate over the output "lines" and draw lines on a blank image
    for line in lines:
        for x1,y1,x2,y2 in line:
            try:
                slope = (y2-y1)/(x2-x1)
            except ZeroDivisionError:
                slope = np.inf            
            
            if slope > 0.5 and slope < 10 and x1 > (.50*imshape[1]) and x2 > (.50*imshape[1]):
                print('Right Lane with slope:  ',slope)
                cv2.line(line_image_reduced,(x1,y1),(x2,y2),(255,0,0),10)
            elif slope < -0.5 and slope > -10 and x1 < (.50*imshape[1]) and x2 < (.50*imshape[1]):
                print('Left Lane with slope : ',slope)
                cv2.line(line_image_reduced,(x1,y1),(x2,y2),(255,0,0),10)
    
    # Create a "color" binary image to combine with line image
    stacked_mask = np.dstack((mask_image, mask_image, mask_image)) 

    # Draw the lines on the edge image
    hough_canny = cv2.addWeighted(stacked_mask, 0.8, line_image, 1, 0)
    hough_original = cv2.addWeighted(original_image, 0.8, line_image, 1, 0)
    hough_canny_reduced = cv2.addWeighted(stacked_mask, 0.8, line_image_reduced, 1, 0)
    hough_original_reduced = cv2.addWeighted(original_image, 0.8, line_image_reduced, 1, 0)
    
    # draw extrapolated lines using hough transform
    hough_extrapolated_image = hough_lines_extrapolated(mask_image, state.rho, state.theta, state.threshold,
                                                        state.min_line_length, state.max_line_gap)
    # perform weighted addition of line_image and original image to potray lane markings
    extrapolated_image = weighted_img(hough_extrapolated_image, original_image, α=1, β=0.6, γ=0.)


    if st.button("Reset Parameters"):
        state.clear()
        # reset the default values of the parameters
        param_kernel_size = 5
        param_low_threshold = 50
        param_high_threshold = 150
        param_rho = 1
        param_theta = np.pi/180
        param_threshold = 40
        param_min_line_length = 10
        param_max_line_gap = 70

    st.markdown(' ')
    st.markdown(' ')
    st.markdown('> `The below images in the pipeline changes with respect to the parameters selected above`')
    st.markdown(' ')
    st.markdown(' ')
    
    col7, col8, col9, col10, col11 = st.beta_columns([4,1,4,1,4])
    # col7.markdown('<p style="text-align: center; color: gray">Original Image</p>', unsafe_allow_html=True)
    col7.markdown(f'<p style="background-color:#0686c2;color:#2b2b2b;font-weight:bold;font-family:sans-serif;border-radius:2%;text-align:center">Original Image</p>', unsafe_allow_html=True)
    col7.image(original_image, use_column_width=True)
    col8.markdown(' ')
    col8.markdown(' ')
    col8.markdown(' ')
    col8.markdown(' ')
    col8.markdown('<p style="font-size:30px;"><b>&#8594;</b></p>', unsafe_allow_html=True)
    # col9.markdown('<p style="text-align: center;">Gray Image</p>', unsafe_allow_html=True)
    col9.markdown(f'<p style="background-color:#0686c2;color:#2b2b2b;font-weight:bold;font-family:sans-serif;border-radius:2%;text-align:center">Gray Image</p>', unsafe_allow_html=True)
    col9.image(gray_image, use_column_width=True)
    col10.markdown(' ')
    col10.markdown(' ')
    col10.markdown(' ')
    col10.markdown(' ')
    col10.markdown('<p style="font-size:30px;"><b>&#8594;</b></p>', unsafe_allow_html=True)
    # col11.markdown('<p style="text-align: center;">Blur Image</p>', unsafe_allow_html=True)
    col11.markdown(f'<p style="background-color:#0686c2;color:#2b2b2b;font-weight:bold;font-family:sans-serif;border-radius:2%;text-align:center">Blur Image</p>', unsafe_allow_html=True)
    col11.image(blur_image, use_column_width=True)

    col12, col13, col14, col15, col16 = st.beta_columns([4,1,4,1,4])
    col16.markdown('<p style="font-size:30px; text-align: center;"><b>&#8595;</b></p>', unsafe_allow_html=True)
    
    col17, col18, col19, col20, col21 = st.beta_columns([4,1,4,1,4])
    # col17.markdown('<p style="text-align: center;">Masked Image</p>', unsafe_allow_html=True)
    col17.markdown(f'<p style="background-color:#0686c2;color:#2b2b2b;font-weight:bold;font-family:sans-serif;border-radius:2%;text-align:center">Masked Image</p>', unsafe_allow_html=True)
    col17.image(mask_image, use_column_width=True)
    col18.markdown(' ')
    col18.markdown(' ')
    col18.markdown(' ')
    col18.markdown(' ')
    col18.markdown('<p style="font-size:30px;"><b>&#8592;</b></p>', unsafe_allow_html=True)
    # col19.markdown('<p style="text-align: center;">Area of Interest</p>', unsafe_allow_html=True)
    col19.markdown(f'<p style="background-color:#0686c2;color:#2b2b2b;font-weight:bold;font-family:sans-serif;border-radius:2%;text-align:center">Area of Interest</p>', unsafe_allow_html=True)
    col19.image(stacked_canny, use_column_width=True)
    col20.markdown(' ')
    col20.markdown(' ')
    col20.markdown(' ')
    col20.markdown(' ')
    col20.markdown('<p style="font-size:30px;"><b>&#8592;</b></p>', unsafe_allow_html=True)
    # col21.markdown('<p style="text-align: center;">Canny Edges</p>', unsafe_allow_html=True)
    col21.markdown(f'<p style="background-color:#0686c2;color:#2b2b2b;font-weight:bold;font-family:sans-serif;border-radius:2%;text-align:center">Canny Edges</p>', unsafe_allow_html=True)
    col21.image(canny_image, use_column_width=True)

    col22, col23, col24, col25, col26 = st.beta_columns([4,1,4,1,4])
    col22.markdown('<p style="font-size:30px; text-align: center;"><b>&#8595;</b></p>', unsafe_allow_html=True)
    
    col27, col28, col29, col30, col31 = st.beta_columns([4,1,4,1,4])
    # col27.markdown('<p style="text-align: center;">Hough Lines Masked</p>', unsafe_allow_html=True)
    col27.markdown(f'<p style="background-color:#0686c2;color:#2b2b2b;font-weight:bold;font-family:sans-serif;border-radius:2%;text-align:center">Hough Lines Masked</p>', unsafe_allow_html=True)
    col27.image(hough_canny, use_column_width=True)
    col28.markdown(' ')
    col28.markdown(' ')
    col28.markdown(' ')
    col28.markdown(' ')
    col28.markdown('<p style="font-size:30px;"><b>&#8594;</b></p>', unsafe_allow_html=True)
    # col29.markdown('<p style="text-align: center;">Clear redundant lines</p>', unsafe_allow_html=True)
    col29.markdown(f'<p style="background-color:#0686c2;color:#2b2b2b;font-weight:bold;font-family:sans-serif;border-radius:2%;text-align:center">Clear redundant lines</p>', unsafe_allow_html=True)
    col29.image(hough_canny_reduced, use_column_width=True)
    col30.markdown(' ')
    col30.markdown(' ')
    col30.markdown(' ')
    col30.markdown(' ')
    col30.markdown('<p style="font-size:30px;"><b>&#8594;</b></p>', unsafe_allow_html=True)
    # col31.markdown('<p style="text-align: center;">Extrapolated Lines</p>', unsafe_allow_html=True)
    col31.markdown(f'<p style="background-color:#0686c2;color:#2b2b2b;font-weight:bold;font-family:sans-serif;border-radius:2%;text-align:center">Extrapolated Lines</p>', unsafe_allow_html=True)
    col31.image(extrapolated_image, use_column_width=True)
        
    st.write(' ')
    st.write(' ')
    st.write(' ')
    st.write(' ')
    st.write(' ')
    st.write(' ')

    col32, col33, col34 = st.beta_columns([4.5,5,4.5])
    # col32.markdown('----')
    # col33.write(':: Hough lines on video stream ::')
    # col34.markdown('----')

    text3 = ':: Extrapolated Hough Lines Pipeline on Video Stream ::'
    st.markdown(f'<p style="background-color:#32a894;color:#2b2b2b;font-size:24px;font-weight:bold;font-family:sans-serif;border-radius:2%;text-align:center">{text3}</p>', unsafe_allow_html=True)
    st.write('----')
    
    st.write(' ')
    st.write(' ')

    st.markdown('<p style="text-align: justify;">You know what\'s cooler than drawing lanes over images? \
                Drawing lanes over video! We can test our solution on two provided videos and one optional \
                challenge video: <mark>solidWhiteRight.mp4</mark>, <mark>solidYellowLeft.mp4</mark>, and \
                <mark>challenge.mp4</mark></p>', unsafe_allow_html=True)                         

    col35, col36, col37 = st.beta_columns([6,1,6])
    video_file1 = open('examples/raw-lines-example.mp4', 'rb')
    video_file2 = open('examples/P1_example.mp4', 'rb')
    video_bytes1 = video_file1.read()
    video_bytes2 = video_file2.read()
    col35.video(video_bytes1)
    col36.markdown(' ')
    col36.markdown(' ')
    col36.markdown(' ')
    col36.markdown(' ')
    col36.markdown('<p style="font-size:30px;"><b>&#8594;</b></p>', unsafe_allow_html=True)
    col37.video(video_bytes2)
    
    # select video
    test_video = os.listdir("test_videos/")
    state.test_video = st.selectbox('select video', test_video, 2)
    video_root_directory = "test_videos"
    video_name = state.test_video
    video_path = os.path.join(video_root_directory, video_name)
    # st.write(video_name)
    # st.write(video_path)

    # test process pipeline on an image
    # image1 = Image.open("test_images/solidYellowCurve.jpg")
    # original_image = np.array(image1.convert('RGB'))
    # original_image = mpimg.imread('test_images/solidWhiteRight.jpg')
    # process = process_image(original_image)
    # st.image(process)

    # video_root_directory = "test_videos"
    # video_name = "solidYellowLeft.mp4"
    # video_path = os.path.join(video_root_directory, video_name)

    if video_name == "solidWhiteRight.mp4":
        prefix = "1_2_"
    elif video_name == "solidYellowLeft.mp4":
        prefix = "2_2_"
    elif video_name == "challenge.mp4":
        prefix = "3_2_"
    postfix = "_extrapolated"
    video_name_new = prefix + video_name.split('.')[0] + postfix + "." + video_name.split('.')[1]
    # st.write(video_name_new)

    # Import everything needed to edit/save/watch video clips
    from moviepy.editor import VideoFileClip

    video_output = os.path.join("test_videos_app", video_name_new)

    # st.write(video_name_new)
    # st.write(video_output)

    button_processVid = st.button("Extrapolate Hough Lines on the Video")
    
    if button_processVid:

        global r_x1
        global r_x2
        global r_x3
        global l_x1
        global l_x2
        global l_x3

        # reset the previous coordinate values to avoid carry forward from previous clip
        r_x1 = 0
        r_x2 = 0
        r_x3 = 0
        l_x1 = 0
        l_x2 = 0
        l_x3 = 0
       
        errorholder = st.empty()
        errorholder.markdown('> `Wait for the process to finish. Refresh the web application if there is an error!`')

        placeholder = st.empty()
        gif_path = 'images/26.gif'
        placeholder.image(gif_path)

        ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
        ## To do so add .subclip(start_second,end_second) to the end of the line below
        ## Where start_second and end_second are integer values representing the start and end of the subclip
        ## You may also uncomment the following line for a subclip of the first 5 seconds
        # video_clip = VideoFileClip('test_videos/solidYellowLeft.mp4').subclip(0,5)
        video_clip = VideoFileClip(video_path)
        # st.write('ok step 1')
        # st.write(clip2)
        # video_clip.close()

        processed_clip = video_clip.fl_image(process_image_extrapolated)
        
        # temp_clip = clip2.rotate(180)
        # temp_clip.show()
        # temp_clip.preview(fps=15, audio=False)
        # st.write('ok step 2')
        
        processed_clip.write_videofile(str(video_output), audio=False)
        
        # my_bar = st.progress(0)
        # for percent_complete in range(100):
            # time.sleep(0.1)
            # my_bar.progress(percent_complete + 1)

        # temp_clip.write_videofile(yellow_output, audio=False)

        # video_clip.close()

        # st.write('ok step 3')
        # get_ipython().magic('time yellow_clip.write_videofile(yellow_output, audio=False)')

        errorholder.empty()
        placeholder.empty()

        video_file = open(str(video_output), 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes)
    





def stabilize_lines(state):
    # st.title("Stabilize Hough Lines Pipeline")
    text1 = "Stabilize Hough Lines Pipeline"
    st.markdown(f'<p style="background-color:#0686c2;color:#2b2b2b;font-size:34px;font-weight:bold;font-family:sans-serif;border-radius:2%;text-align:center">{text1}</p>', unsafe_allow_html=True)
    st.write('----')
    st.markdown('<p style="text-align: justify;">Stabilize the extrapolated lines by defining a learning rate \
                with the previously detected lines.</p>', unsafe_allow_html=True)                         

    
    global param_kernel_size
    global param_low_threshold
    global param_high_threshold
    global param_rho
    global param_theta
    global param_threshold
    global param_min_line_length
    global param_max_line_gap



    state_dict = {
        'kernel_size' : state.kernel_size,
        'low_threshold' : state.low_threshold,
        'high_threshold' : state.high_threshold,
        'rho' : state.rho,
        'theta' : state.theta,
        'threshold' : state.threshold,
        'min_line_length' : state.min_line_length,
        'max_line_gap' : state.max_line_gap

    }


    state_df = pd.DataFrame(state_dict.items(), columns=['parameter', 'value'])


    if state_df['value'].isnull().sum():
        # st.write('Message: NULL values detected ')
        # initialize the default values of the parameters
        param_kernel_size = 5
        param_low_threshold = 50
        param_high_threshold = 150
        param_rho = 1
        param_theta = np.pi/180
        param_threshold = 40
        param_min_line_length = 10
        param_max_line_gap = 70
    else:
        # st.write('Message: No NULL values detected')
        param_kernel_size = state.kernel_size
        param_low_threshold = state.low_threshold
        param_high_threshold = state.high_threshold
        param_rho = state.rho
        param_theta = state.theta
        param_threshold = state.threshold
        param_min_line_length = state.min_line_length
        param_max_line_gap = state.max_line_gap

    state_df['selected_value'] = pd.Series([param_kernel_size,
                                            param_low_threshold,
                                            param_high_threshold,
                                            param_rho,
                                            param_theta,
                                            param_threshold,
                                            param_min_line_length,
                                            param_max_line_gap])

    state_df['recommended_value'] = pd.Series([5, 50, 150, 1, np.pi/180, 40, 10, 70])
    state_df.drop(columns = ['value'], inplace=True)

    st.write(state_df)

    if st.button("Reset parameters to default values"):
        state.clear()

            
    st.write(' ')
    st.write(' ')
    st.write(' ')
    st.write(' ')
    st.write(' ')
    st.write(' ')

    text3 = ':: Stabilize Hough Lines Pipeline on Video Stream ::'
    st.markdown(f'<p style="background-color:#32a894;color:#2b2b2b;font-size:24px;font-weight:bold;font-family:sans-serif;border-radius:2%;text-align:center">{text3}</p>', unsafe_allow_html=True)
    st.write('----')

    # select video
    test_video = os.listdir("test_videos/")
    state.test_video = st.selectbox('select video', test_video, 2)
    video_root_directory = "test_videos"
    video_name = state.test_video
    video_path = os.path.join(video_root_directory, video_name)
    # st.write(video_name)
    # st.write(video_path)

    # test process pipeline on an image
    # image1 = Image.open("test_images/solidYellowCurve.jpg")
    # original_image = np.array(image1.convert('RGB'))
    # original_image = mpimg.imread('test_images/solidWhiteRight.jpg')
    # process = process_image(original_image)
    # st.image(process)

    # video_root_directory = "test_videos"
    # video_name = "solidYellowLeft.mp4"
    # video_path = os.path.join(video_root_directory, video_name)

    if video_name == "solidWhiteRight.mp4":
        prefix = "1_3_"
    elif video_name == "solidYellowLeft.mp4":
        prefix = "2_3_"
    elif video_name == "challenge.mp4":
        prefix = "3_3_"
    postfix = "_extrapolated_stable"
    video_name_new = prefix + video_name.split('.')[0] + postfix + "." + video_name.split('.')[1]
    # st.write(video_name_new)

    # Import everything needed to edit/save/watch video clips
    from moviepy.editor import VideoFileClip

    video_output = os.path.join("test_videos_app", video_name_new)

    # st.write(video_name_new)
    # st.write(video_output)

    button_processVid = st.button("Stabilize Hough Lines on the Video")
    
    if button_processVid:
       
        errorholder = st.empty()
        errorholder.markdown('> `Wait for the process to finish. Refresh the web application if there is an error!`')

        placeholder = st.empty()
        gif_path = 'images/23.gif'
        placeholder.image(gif_path)

        ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
        ## To do so add .subclip(start_second,end_second) to the end of the line below
        ## Where start_second and end_second are integer values representing the start and end of the subclip
        ## You may also uncomment the following line for a subclip of the first 5 seconds
        # video_clip = VideoFileClip('test_videos/solidYellowLeft.mp4').subclip(0,5)
        video_clip = VideoFileClip(video_path)
        # st.write('ok step 1')
        # st.write(clip2)
        # video_clip.close()

        processed_clip = video_clip.fl_image(process_image_extrapolated_stable)
        
        # temp_clip = clip2.rotate(180)
        # temp_clip.show()
        # temp_clip.preview(fps=15, audio=False)
        # st.write('ok step 2')
        
        processed_clip.write_videofile(str(video_output), audio=False)
        
        # my_bar = st.progress(0)
        # for percent_complete in range(100):
            # time.sleep(0.1)
            # my_bar.progress(percent_complete + 1)

        # temp_clip.write_videofile(yellow_output, audio=False)

        # video_clip.close()

        # st.write('ok step 3')
        # get_ipython().magic('time yellow_clip.write_videofile(yellow_output, audio=False)')

        errorholder.empty()
        placeholder.empty()

        video_file = open(str(video_output), 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes)
    






def fill_lines(state):
    # st.title("Fill Hough Lines Pipeline")
    text1 = "Fill Area inside Hough Lines Pipeline"
    st.markdown(f'<p style="background-color:#0686c2;color:#2b2b2b;font-size:34px;font-weight:bold;font-family:sans-serif;border-radius:2%;text-align:center">{text1}</p>', unsafe_allow_html=True)
    st.write('----')
    st.markdown('<p style="text-align: justify;">Stabilize the extrapolated lines by defining a learning rate \
                with the previously detected lines.</p>', unsafe_allow_html=True)                         

    
    global param_kernel_size
    global param_low_threshold
    global param_high_threshold
    global param_rho
    global param_theta
    global param_threshold
    global param_min_line_length
    global param_max_line_gap



    state_dict = {
        'kernel_size' : state.kernel_size,
        'low_threshold' : state.low_threshold,
        'high_threshold' : state.high_threshold,
        'rho' : state.rho,
        'theta' : state.theta,
        'threshold' : state.threshold,
        'min_line_length' : state.min_line_length,
        'max_line_gap' : state.max_line_gap

    }


    state_df = pd.DataFrame(state_dict.items(), columns=['parameter', 'value'])


    if state_df['value'].isnull().sum():
        # st.write('Message: NULL values detected ')
        # initialize the default values of the parameters
        param_kernel_size = 5
        param_low_threshold = 50
        param_high_threshold = 150
        param_rho = 1
        param_theta = np.pi/180
        param_threshold = 40
        param_min_line_length = 10
        param_max_line_gap = 70
    else:
        # st.write('Message: No NULL values detected')
        param_kernel_size = state.kernel_size
        param_low_threshold = state.low_threshold
        param_high_threshold = state.high_threshold
        param_rho = state.rho
        param_theta = state.theta
        param_threshold = state.threshold
        param_min_line_length = state.min_line_length
        param_max_line_gap = state.max_line_gap

    state_df['selected_value'] = pd.Series([param_kernel_size,
                                            param_low_threshold,
                                            param_high_threshold,
                                            param_rho,
                                            param_theta,
                                            param_threshold,
                                            param_min_line_length,
                                            param_max_line_gap])

    state_df['recommended_value'] = pd.Series([5, 50, 150, 1, np.pi/180, 40, 10, 70])
    state_df.drop(columns = ['value'], inplace=True)

    st.write(state_df)

    if st.button("Reset parameters to default values"):
        state.clear()

            
    st.write(' ')
    st.write(' ')
    st.write(' ')
    st.write(' ')
    st.write(' ')
    st.write(' ')

    text3 = ':: Fill Area inside Hough Lines Pipeline on Video Stream ::'
    st.markdown(f'<p style="background-color:#32a894;color:#2b2b2b;font-size:24px;font-weight:bold;font-family:sans-serif;border-radius:2%;text-align:center">{text3}</p>', unsafe_allow_html=True)
    st.write('----')

    # select video
    test_video = os.listdir("test_videos/")
    state.test_video = st.selectbox('select video', test_video, 2)
    video_root_directory = "test_videos"
    video_name = state.test_video
    video_path = os.path.join(video_root_directory, video_name)
    # st.write(video_name)
    # st.write(video_path)

    # test process pipeline on an image
    # image1 = Image.open("test_images/solidYellowCurve.jpg")
    # original_image = np.array(image1.convert('RGB'))
    # original_image = mpimg.imread('test_images/solidWhiteRight.jpg')
    # process = process_image(original_image)
    # st.image(process)

    # video_root_directory = "test_videos"
    # video_name = "solidYellowLeft.mp4"
    # video_path = os.path.join(video_root_directory, video_name)

    if video_name == "solidWhiteRight.mp4":
        prefix = "1_4_"
    elif video_name == "solidYellowLeft.mp4":
        prefix = "2_4_"
    elif video_name == "challenge.mp4":
        prefix = "3_4_"
    postfix = "_fill_area"
    video_name_new = prefix + video_name.split('.')[0] + postfix + "." + video_name.split('.')[1]
    # st.write(video_name_new)

    # Import everything needed to edit/save/watch video clips
    from moviepy.editor import VideoFileClip

    video_output = os.path.join("test_videos_app", video_name_new)

    # st.write(video_name_new)
    # st.write(video_output)

    button_processVid = st.button("Fill Area inside Hough Lines on the Video")
    
    if button_processVid:
       
        errorholder = st.empty()
        errorholder.markdown('> `Wait for the process to finish. Refresh the web application if there is an error!`')

        placeholder = st.empty()
        gif_path = 'images/19.gif'
        placeholder.image(gif_path)

        ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
        ## To do so add .subclip(start_second,end_second) to the end of the line below
        ## Where start_second and end_second are integer values representing the start and end of the subclip
        ## You may also uncomment the following line for a subclip of the first 5 seconds
        # video_clip = VideoFileClip('test_videos/solidYellowLeft.mp4').subclip(0,5)
        video_clip = VideoFileClip(video_path)
        # st.write('ok step 1')
        # st.write(clip2)
        # video_clip.close()

        processed_clip = video_clip.fl_image(process_image_fill)
        
        # temp_clip = clip2.rotate(180)
        # temp_clip.show()
        # temp_clip.preview(fps=15, audio=False)
        # st.write('ok step 2')
        
        processed_clip.write_videofile(str(video_output), audio=False)
        
        # my_bar = st.progress(0)
        # for percent_complete in range(100):
            # time.sleep(0.1)
            # my_bar.progress(percent_complete + 1)

        # temp_clip.write_videofile(yellow_output, audio=False)

        # video_clip.close()

        # st.write('ok step 3')
        # get_ipython().magic('time yellow_clip.write_videofile(yellow_output, audio=False)')

        errorholder.empty()
        placeholder.empty()

        video_file = open(str(video_output), 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes)
    




def display_state_values(state):
    df_filters = pd.DataFrame()
    df_filters['Filter'] = []
    df_filters['Value'] = []
    # st.write(df_filters)

    if st.button("Reset Search"):
        state.clear()



class _SessionState:

    def __init__(self, session, hash_funcs):
        """Initialize SessionState instance."""
        self.__dict__["_state"] = {
            "data": {},
            "hash": None,
            "hasher": _CodeHasher(hash_funcs),
            "is_rerun": False,
            "session": session,
        }

    def __call__(self, **kwargs):
        """Initialize state data once."""
        for item, value in kwargs.items():
            if item not in self._state["data"]:
                self._state["data"][item] = value

    def __getitem__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)
        
    def __getattr__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)

    def __setitem__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value

    def __setattr__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value
    
    def clear(self):
        """Clear session state and request a rerun."""
        self._state["data"].clear()
        self._state["session"].request_rerun()
    
    def sync(self):
        """Rerun the app with all state values up to date from the beginning to fix rollbacks."""

        # Ensure to rerun only once to avoid infinite loops
        # caused by a constantly changing state value at each run.
        #
        # Example: state.value += 1
        if self._state["is_rerun"]:
            self._state["is_rerun"] = False
        
        elif self._state["hash"] is not None:
            if self._state["hash"] != self._state["hasher"].to_bytes(self._state["data"], None):
                self._state["is_rerun"] = True
                self._state["session"].request_rerun()

        self._state["hash"] = self._state["hasher"].to_bytes(self._state["data"], None)


def _get_session():
    session_id = get_report_ctx().session_id
    session_info = Server.get_current()._get_session_info(session_id)

    if session_info is None:
        raise RuntimeError("Couldn't get your Streamlit Session object.")
    
    return session_info.session


def _get_state(hash_funcs=None):
    session = _get_session()

    if not hasattr(session, "_custom_session_state"):
        session._custom_session_state = _SessionState(session, hash_funcs)

    return session._custom_session_state


if __name__ == "__main__":
    main()