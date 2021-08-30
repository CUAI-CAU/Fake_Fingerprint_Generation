import cv2
import glob
import os
import numpy as np
from random import shuffle
from tqdm import tqdm
import random
import shutil
import math

def get_thresh(img, ratio=1):
    w, h, _ = img.shape
    cx, cy = int(w/2), int(h/2)
    center = img[cx - 15:cx+15, cy-15: cy+15, :]

    upper = np.array(np.clip(center.max(axis=0).max(axis=0) * (2-ratio), 0, 255), dtype=np.uint8)
    lower = np.clip(np.array(center.min(axis=0).min(axis=0) * ratio, dtype=np.uint8), 0, 255)
    
    return lower, upper

def get_mask(img, ratio = 0.9):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    rgb = img

    rgb_lower, rgb_upper = get_thresh(rgb, np.array([1, 1, 1])*ratio)
    rgb_mask = cv2.inRange(rgb, rgb_lower, rgb_upper)

    ycrcb_lower, ycrcb_upper = get_thresh(ycrcb, np.array([1, 1, 1])*ratio)
    ycrcb_mask = cv2.inRange(ycrcb, ycrcb_lower, ycrcb_upper)

    hsv_lower, hsv_upper = get_thresh(hsv, np.array([1, 1, 1])*ratio)
    hsv_mask = cv2.inRange(hsv, hsv_lower, hsv_upper)

    hsv_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,20))
    hsv_finger = cv2.morphologyEx(hsv_mask, cv2.MORPH_OPEN, hsv_kernel)

    rgb_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,20))
    rgb_finger = cv2.morphologyEx(rgb_mask, cv2.MORPH_OPEN, rgb_kernel)
    
    ycrcb_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,20))
    ycrcb_finger = cv2.morphologyEx(ycrcb_mask, cv2.MORPH_OPEN, ycrcb_kernel)
    
    mask = cv2.bitwise_and(rgb_finger, hsv_finger)
    mask = cv2.bitwise_and(mask, ycrcb_finger)

    finger = cv2.bitwise_and(img, img, mask = mask)
    
    return finger, mask


# sigma is adjusted according to the ridge period, so that the filter does not contain more than three effective peaks 
def _gabor_sigma(ridge_period):
    _sigma_conv = (3.0/2.0)/((6*math.log(10))**0.5)
    return _sigma_conv * ridge_period

def _gabor_size(ridge_period):
    p = int(round(ridge_period * 2 + 1))
    if p % 2 == 0:
        p += 1
    return (p, p)

def gabor_kernel(period, orientation):
    f = cv2.getGaborKernel(_gabor_size(period), _gabor_sigma(period), np.pi/2 - orientation, period, gamma = 1, psi = 0)
    f /= f.sum()
    f -= f.mean()
    return f

def Enhancement(fingerprint, mask):
    # Calculate the local gradient (using Sobel filters)
    gx, gy = cv2.Sobel(fingerprint, cv2.CV_32F, 1, 0), cv2.Sobel(fingerprint, cv2.CV_32F, 0, 1)
    
    # Calculate the magnitude of the gradient for each pixel
    gx2, gy2 = gx**2, gy**2

    
    W = (23, 23)
    gxx = cv2.boxFilter(gx2, -1, W, normalize = False)
    gyy = cv2.boxFilter(gy2, -1, W, normalize = False)
    gxy = cv2.boxFilter(gx * gy, -1, W, normalize = False)
    gxx_gyy = gxx - gyy
    gxy2 = 2 * gxy

    orientations = (cv2.phase(gxx_gyy, -gxy2) + np.pi) / 2 # '-' to adjust for y axis direction
    

    region = fingerprint[80:180,80:180]

    # before computing the x-signature, the region is smoothed to reduce noise
    smoothed = cv2.blur(region, (5,5), -1)
    xs = np.sum(smoothed, 1) # the x-signature of the region

    # Find the indices of the x-signature local maxima
    local_maxima = np.nonzero(np.r_[False, xs[1:] > xs[:-1]] & np.r_[xs[:-1] >= xs[1:], False])[0]

    # Calculate all the distances between consecutive peaks
    distances = local_maxima[1:] - local_maxima[:-1]

    # Estimate the ridge line period as the average of the above distances
    ridge_period = np.average(distances)

    # Create the filter bank
    or_count = 8
    gabor_bank = [gabor_kernel(ridge_period, o) for o in np.arange(0, np.pi, np.pi/or_count)]

    # Filter the whole image with each filter
    # Note that the negative image is actually used, to have white ridges on a black background as a result
    nf = 255-fingerprint
    all_filtered = np.array([cv2.filter2D(nf, cv2.CV_32F, f) for f in gabor_bank])

    y_coords, x_coords = np.indices(fingerprint.shape)
    # For each pixel, find the index of the closest orientation in the gabor bank
    orientation_idx = np.round(((orientations % np.pi) / np.pi) * or_count).astype(np.int32) % or_count
    # Take the corresponding convolution result for each pixel, to assemble the final result
    filtered = all_filtered[orientation_idx, y_coords, x_coords]
    # Convert to gray scale and apply the mask
    enhanced = mask & np.clip(filtered, 0, 255).astype(np.uint8)

    return enhanced

def extract_edge(img):
    copy = img.copy()
    effect = cv2.convertScaleAbs(copy, alpha=0.8, beta=25)
    gray = cv2.cvtColor(effect, cv2.COLOR_RGB2GRAY)
    equ = cv2.equalizeHist(gray)
    norm = cv2.normalize(equ, None, 0, 256, cv2.NORM_MINMAX)
    result = cv2.adaptiveThreshold(norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2)

    return result

