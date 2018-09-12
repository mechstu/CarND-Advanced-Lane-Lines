# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 01:20:37 2018

@author: sugho
"""


import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
from moviepy.editor import VideoFileClip

# Calibrating the Camera
images = glob.glob('./camera_cal/calibration*.jpg')
nx = 9 # the number of inside corners in x
ny = 6 # the number of inside corners in y

# Initialize arrays to hold obj points and image points
objpoints = []
imgpoints = []

# Default object points grid
objp = np.zeros((nx*ny,3),np.float32)
objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

# Camera Calibration from the 20 images
for fname in images:
    img = cv2.imread(fname)
    # convert to gray scale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Find chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
    # if conrners are found append obj and img points
    if ret == True:
        imgpoints.append(corners)
        objpoints.append(objp)
    
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints,imgpoints,gray.shape[::-1],None,None)   

# make example undistorted image for writeup
#img = cv2.imread('./camera_cal/calibration1.jpg')
#cv2.imwrite('./output_images/Distorted_image.jpg',img)
#cv2.imwrite('./output_images/Undistorted_image.jpg',cv2.undistort(img,mtx,dist,None,mtx))

#%% Perspective Transform matrix and inverse matrix is calculated
img = mpimg.imread('./test_images/test1.jpg')
img_undist = cv2.undistort(img, mtx, dist, None, mtx)
cv2.imwrite('./output_images/test1_undist.jpg',img_undist,)
img_size = (gray.shape[1], gray.shape[0])
# these points were measured using an image processing software
src = np.float32([[562,470],[720,470],[1105,685],[279,685]])
# rectangle to transform it into:
dst = np.float32([[250,0], [img_size[0]-250, 0], [img_size[0]-250, img_size[1]], [250, img_size[1]]])
M = cv2.getPerspectiveTransform(src,dst)
Minv = cv2.getPerspectiveTransform(dst,src)
warped = cv2.warpPerspective(img_undist,M,img_size,flags=cv2.INTER_LINEAR)

#top_down, perspective_M, perspective_Minv = corners_unwarp(img, nx, ny, mtx, dist)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(warped)
ax2.set_title('Undistorted and Warped Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

#%% Function to find lane lines from a warped binary image

def find_lane_pixels(binary_warped, brute_force=True):
    global left_fit_hist
    global right_fit_hist
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)

    histogram = np.sum(histogram,1)
    # Create an output image to draw on and visualize the result
    out_img = np.copy(binary_warped) #np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 8
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    if brute_force == True:
        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),
            (win_xleft_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),
            (win_xright_high,win_y_high),(0,255,0), 2) 
            
            # Identify the nonzero pixels in x and y within the window #
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    
        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass
    
    else:
        left_lane_inds = ((nonzerox > (left_fit_hist[0]*(nonzeroy**2) + left_fit_hist[1]*nonzeroy + 
                        left_fit_hist[2] - margin)) & (nonzerox < (left_fit_hist[0]*(nonzeroy**2) + 
                        left_fit_hist[1]*nonzeroy + left_fit_hist[2] + margin)))
        right_lane_inds = ((nonzerox > (right_fit_hist[0]*(nonzeroy**2) + right_fit_hist[1]*nonzeroy + 
                        right_fit_hist[2] - margin)) & (nonzerox < (right_fit_hist[0]*(nonzeroy**2) + 
                        right_fit_hist[1]*nonzeroy + right_fit_hist[2] + margin)))



    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img

#%%  Radius of Curvature function
def curvature(xdata, ploty):
    ym_per_pix = 5/143 # m per pixel in the y direction
    xm_per_pix = 3.7/370  # m per pixel in the x direction
    fit_cr = np.polyfit(ploty*ym_per_pix, xdata*xm_per_pix, 2)
    y_eval = np.max(ploty)
    
    # Calculation of R_curve (radius of curvature)
    curverad = ((1 + (2*fit_cr[0]*y_eval*ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])
    return curverad

#%%

global brute_force
global left_fit_hist   #store weighted historic Left lane line fit coefficients
global right_fit_hist  #store weighted historic Right lane line fit coefficients


#initialize global vars
brute_force = True
left_fit_hist = np.array([0, 0, 0],dtype = float)
right_fit_hist = np.array([0, 0, 0],dtype = float)

def pipeline(img, s_thresh=(190, 245), sx_thresh=(50, 170)):
    global brute_force
    global left_fit_hist
    global right_fit_hist
    
    # un distort the image
    img_undist = cv2.undistort(img, mtx, dist, None, mtx)
    # perspective transform the image
    img_warped = cv2.warpPerspective(img_undist,M,img_size,flags=cv2.INTER_LINEAR)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img_warped, cv2.COLOR_BGR2HLS)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    #cv2.imwrite('color_binary.jpg',color_binary)
    
    # Detect lane lines using sliding window
    leftx, lefty, rightx, righty, out_img1 = find_lane_pixels(color_binary, brute_force)
    
    ######### Polynomial fit
    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    ###### Using a forgetting factor to average over old lane line fits:
    ff = 0.7 #Forgetting factor
    

    # Generate x and y values for plotting
    ploty = np.linspace(0, color_binary.shape[0]-1, color_binary.shape[0] )
    try:
        # Check if left and right fit are none or incorrect
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        # If left and right fit have worked, then set brute_force flag to false
        brute_force = False
       
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        # if lane lines weren't detected, set the lane lines to historic fit data for smooth results
        left_fitx = left_fit_hist[0]*ploty**2 + left_fit_hist[1]*ploty + left_fit_hist[2]
        right_fitx = right_fit_hist[0]*ploty**2 + right_fit_hist[1]*ploty + right_fit_hist[2]
        # set brute force flag to true for next lane line detection
        brute_force = True
     
    #### Sanity check if new lines found can be tursted.
    if brute_force == False:  #new line has been found, lets check for validity of new line found
        #check if lines are parallel by comparing the diff between the constant term
        if (right_fit[2]-left_fit[2]) < 550 or (right_fit[2]-left_fit[2]) > 850:    #Not parallel
            left_fitx = left_fit_hist[0]*ploty**2 + left_fit_hist[1]*ploty + left_fit_hist[2]
            right_fitx = right_fit_hist[0]*ploty**2 + right_fit_hist[1]*ploty + right_fit_hist[2]
            brute_force = True
        else:  #Lane line fit are parallel
             # Average the left and right fit using weighted avg with historic line data to smooth results
            if left_fit_hist.all() != 0 and right_fit_hist.all() != 0:
                left_fit = (left_fit + left_fit_hist*ff)/(1+ff)
                right_fit = (right_fit + right_fit_hist*ff)/(1+ff)
        
            left_fit_hist = left_fit
            right_fit_hist = right_fit
            
            #Calcualte the smoothed left_fitx and right fitx
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
            
    # Calculate radius of curvature    
    left_curverad = curvature(left_fitx,ploty)
    right_curverad = curvature(right_fitx,ploty)
    
    curverad = (left_curverad+right_curverad)/2
    
    #Calculate lane position
    xm_per_pix = 3.7/370  # m per pixel in the x direction
    lane_pos = ((right_fitx[-1]+left_fitx[-1])/2-color_binary.shape[1]/2)*xm_per_pix

    
    # Create an image to draw the lines on
    color_warp = np.zeros_like(color_binary).astype(np.uint8)
    #color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
    cv2.putText(result, 'Radius of Curvature: {:.0f} m'.format(curverad), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(result, 'Lane Position- off center: {:.2f} m'.format(lane_pos), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)    
    #plt.imshow(result)    
    
    return result
    
    
#%% Process the video

#vid_output = 'project_video_out.mp4'
#clip = VideoFileClip('project_video.mp4')

vid_output = 'harder_challenge_video_out.mp4'
clip = VideoFileClip('harder_challenge_video.mp4')

vid_clip = clip.fl_image(pipeline)
vid_clip.write_videofile(vid_output, audio=False)




#%%

# 371 pixels for 3.7 m in the x direction
# 143 pixels for a 5m in y direction