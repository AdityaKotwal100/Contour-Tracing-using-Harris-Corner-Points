from harris_4 import HarrisCorners
import cv2
import argparse
import numpy as np
import sys
import getopt
import operator
import matplotlib.pyplot as plt

#Open CV's inbuilt Saliency detection method.
def saliencyDetect(image):
  saliency = cv2.saliency.StaticSaliencyFineGrained_create()
  (success, saliencyMap) = saliency.computeSaliency(image)
  threshMap = cv2.threshold((saliencyMap*255).astype("uint8"), 0, 255,
	  cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

  return image, saliencyMap, threshMap


def main():
    """
    window_size = 2
    k = 0.04
    thresh = 10000000
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help='Path to input image')
    parser.add_argument('--k', type=float, help='Value of constant "k" for Harris corner detection')
    parser.add_argument('--window_size', type=int, help='Window Size')
    parser.add_argument('--thresh', type=float, help='Threshold value')
    args = parser.parse_args()

    k = args.k
    thresh = args.thresh
    window_size = args.window_size
    filepath = args.path
    color_image = cv2.imread(filepath, 1)
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    
    #Smoothening image
    kernel = np.ones((5,5),np.float32)/25
    dst = cv2.filter2D(gray_image,-1,kernel)

    #Finding Harris corner points, returns corner mask and list of all corner points
    corner_image, cornerList = HarrisCorners(dst, int(window_size), float(k), int(thresh))
    ret,thresh1 = cv2.threshold(corner_image,127,255,cv2.THRESH_BINARY)

    #Reducing dimension of image for matching dimensionality while adding
    thresh1 = thresh1[:,:, 0]
    thresh2 = thresh1
    cv2.imshow("thresh",thresh1)
    cv2.waitKey(0)
    cv2.imshow("harris",corner_image)
    cv2.waitKey(0)
    edges = cv2.Canny(dst,100,200)
    cv2.imshow("edges",edges)
    cv2.waitKey(0)
    edges_plus_corners = thresh1 + edges
    cv2.imshow("edges_plus_corners",edges_plus_corners)
    cv2.waitKey(0)
    
    #Drawing contours over original image
    image1 = color_image.copy()
    for i in range(color_image.shape[0]):
        for j in range(color_image.shape[1]):
            if(edges_plus_corners[i,j] == 255):
                color_image.itemset((i,j,0), 255)
                color_image.itemset((i,j,1), 0)
                color_image.itemset((i,j,2), 0)

    cv2.imshow("image",color_image)
    cv2.waitKey(0)
   
    #testing cv2's inbuilt contour detection method
    contours, hierarchy = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image1, contours, -1, (0,255,0), 3)
    cv2.imshow("image 1",image1)
    
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
