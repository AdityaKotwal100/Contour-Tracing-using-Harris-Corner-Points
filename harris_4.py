import numpy as np
import cv2
def HarrisCorners(image, window_size, k, threshold):

    #Finding Gradients in x and y direction.
    dy, dx = np.gradient(image)
    Image_xx = dx**2
    Image_xy = dy*dx
    Image_yy = dy**2
    height = image.shape[0]
    width = image.shape[1]

    corners = []
    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    delta = window_size//2
    
    #Sliding window over entire image
    for i in range(delta, height-delta):
      
        for j in range(delta, width-delta):
            
            #Finding sum of all Intensities in each window.
            window_Ixx = Image_xx[i-delta:i+delta+1, j-delta:j+delta+1]
            window_Ixy = Image_xy[i-delta:i+delta+1, j-delta:j+delta+1]
            window_Iyy = Image_yy[i-delta:i+delta+1, j-delta:j+delta+1]
            Sum_xx = window_Ixx.sum()
            Sum_xy = window_Ixy.sum()
            Sum_yy = window_Iyy.sum()

            #Calculating determinant and trace from each window
            determinant = (Sum_xx * Sum_yy) - (Sum_xy**2)
            trace = Sum_xx + Sum_yy
            #Find 't' value, this is the score for corner points.
            t = determinant - k*(trace**2)
            #Create a mask by only having intensity values at corners.
            if t < threshold:
                color_image.itemset((i, j, 0), 0)
                color_image.itemset((i, j, 1), 0)
                color_image.itemset((i, j, 2), 0)
            
    return color_image, corners