from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os
from PIL import Image
from skimage.morphology import erosion, dilation, opening, closing
from skimage.morphology import disk
from skimage.feature import canny

in_dir = 'test_images'
out_dir = 'output_images'

# read the input image file names with paths into a list
infiles = in_dir + '/*.png'
img_names = glob(infiles)

# loop over each input image in a for loop

for fn in img_names:
    print('processing %s...' % fn)

    # read an input image as gray
    image = cv.imread(fn, 0)
    # apply median filter for edge preservartion
    gray = cv.medianBlur(image, 35)

    # ----------------------------------
    # Remove noise
    # Gaussian
    """Image blurring is achieved by convolving the image with a low-pass filter kernel.
     It is useful for removing noise. It actually removes high frequency content (e.g: noise, edges)
      from the image resulting in edges being blurred when this is filter is applied."""
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    #
    # # -------------------------------------------------------------
    # Threshold segmentation
    ret, thresh = cv.threshold(blur, 127, 255, cv.THRESH_TRUNC)

    # # ---------------------------------------------------------------

    """morphological operations"""
    selem = disk(10)
    # opened=opening(thresh,selem)
    # eroded = erosion(thresh, selem)
    # dilated = dilation(thresh, selem)
    closed = closing(thresh, selem)

    # titles = ['Original Image', 'eroded', 'dilated','closed']
    # images = [thresh, eroded, dilated,closed]
    # for i in range(4):
    #     plt.subplot(3, 3, i + 1), plt.imshow(images[i], 'gray')
    #     plt.title(titles[i])
    #     plt.xticks([]), plt.yticks([])
    # plt.show()
    # # ---------------------------------------------------------------
    # canny edge detection

    def modify_img(img, sigma):
        array = np.array(img)
        out = np.uint8(canny(array, sigma, ) * 255)
        edges = Image.fromarray(out, mode='L')
        edges_img = np.array(edges)
        # create a binary thresholded image
        _, binary = cv.threshold(edges_img, 127, 255, cv.THRESH_BINARY_INV)
        # plt.imshow(binary, cmap="gray")
        # plt.show()
        return binary
    binary = modify_img(closed, 2.5)

    # find the contours from the thresholded image
    contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # remove border contour of image
    largest_areas = sorted(contours, key=cv.contourArea)

    out = cv.drawContours(image.copy(), largest_areas, -1, (0, 255, 0), -1)
    largest_areas = sorted(contours, key=cv.contourArea)
    del largest_areas[-1]
    # Try to join the contour using convex hull
    hull = []
    # calculate points for each contour
    for i in range(len(contours)):
        # creating convex hull object for each contour
        hull.append(cv.convexHull(contours[i], False))
        # create an empty black image
        drawing = np.zeros((thresh.shape[0], thresh.shape[1], 3), np.uint8)
        # draw contours and hull points
    for i in range(len(contours)):
        color = (255, 255, 255)  # blue - color for convex hull
        cv.drawContours(drawing, hull, i, color, 1, 1)
    # plt.imshow(drawing)
    # plt.show()
    titles = ['Original image','Segmented image', 'Edge detection']
    images = [image,thresh,drawing]
    for i in range(3):
        plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()
    # save
    # write the result to disk in the previously created output directory
    name = os.path.basename(fn)
    outfile = out_dir + '/' + name
    cv.imwrite(outfile, drawing)
    # # ---------------------------------------------------------------