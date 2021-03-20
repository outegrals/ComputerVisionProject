# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 08:59:46 2021

@author: jesse
"""

import numpy as np
import sys
import cv2 as cv




# Load the image
img = cv.imread('C:\\Users\\jesse\\Google Drive\\School\\usf\\ComputerVision\\Group\\archive\\images\\images\\train\\clean\\31.png')
# Check if image is loaded fine
if img is None:
    print ('Error opening image: ')
else:
    print("Image loaded successfully")
    
# Show source image
cv.imshow("img", img)

# Transform source image to gray if it is not already

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("gray", gray)
# Show gray image
#show_wait_destroy("gray", gray)


# Apply adaptiveThreshold at the bitwise_not of gray, notice the ~ symbol
gray = cv.bitwise_not(gray)


# (inputImage, maxValue, adaptiveMethod, thresholdType, blockSize/kernel size, Constant subtracted from the mean or weighted mean)
bw = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 15, -2)      

cv.imshow('bw', bw)

# Create the images that will use to extract the horizontal and vertical lines
horizontal = np.copy(bw)
vertical = np.copy(bw)

print('111    horizontal.shape ', horizontal.shape)
print('111    vertical.shape ', vertical.shape)


# Specify size on horizontal axis
cols = horizontal.shape[1]
horizontal_size = cols // 30

print('222    cols ', cols)
print('222    horizontal_size ', horizontal_size)

# Create structure element for extracting horizontal lines through morphology operations
horizontalStructure = cv.getStructuringElement(cv.MORPH_RECT, (horizontal_size, 1))

print('222   type horizontalStructure ', type(horizontalStructure))
print('222    horizontalStructure ', horizontalStructure)

# Apply morphology operations
horizontal = cv.erode(horizontal, horizontalStructure)
horizontal = cv.dilate(horizontal, horizontalStructure)
print('222   type horizontal ', type(horizontal))
print('222    horizontal ', horizontal)
print('222    horizontal shape ', horizontal.shape)
cv.imshow('horizontal', horizontal)


 # Specify size on vertical axis
rows = vertical.shape[0]
verticalsize = rows // 30
print('333    rows ', rows)
print('333    verticalsize ', verticalsize)


# Create structure element for extracting vertical lines through morphology operations
verticalStructure = cv.getStructuringElement(cv.MORPH_RECT, (1, verticalsize))
print('333   type verticalStructure ', type(verticalStructure))
print('333    verticalStructure ', verticalStructure)

# Apply morphology operations
vertical = cv.erode(vertical, verticalStructure)
vertical = cv.dilate(vertical, verticalStructure)
print('333   type vertical ', type(vertical))
print('333    vertical ', vertical)
print('333    vertical shape ', vertical.shape)
cv.imshow('vertical', vertical)


# CURRENTLY JUST VERTICAL
# refine the edges in order to obtain a smoother result
# Inverse vertical image
vertical = cv.bitwise_not(vertical)
cv.imshow('vertical_bit', vertical)
'''
Extract edges and smooth image according to the logic
1. extract edges
2. dilate(edges)
3. src.copyTo(smooth)
4. blur smooth img
5. smooth.copyTo(src, edges)
'''
# Step 1
edges = cv.adaptiveThreshold(vertical, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 3, -2)
cv.imshow('edges', edges)

# Step 2
kernel = np.ones((2, 2), np.uint8)
print('444     kernel.shape  ', kernel.shape)
print('444     kernel  ', kernel)

edges = cv.dilate(edges, kernel)

cv.imshow('dilate', edges)

# Step 3
smooth = np.copy(vertical)
# Step 4
smooth = cv.blur(smooth, (2, 2))
# Step 5
(rows, cols) = np.where(edges != 0)
vertical[rows, cols] = smooth[rows, cols]
# Show final result
cv.imshow('smooth - final', vertical)









