# %% IMPORTS
import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage.morphology as morph

# %% COUNT PEARS
def countPears(img):
    # apply edge preserving smoothing filter
    bil_img = cv2.bilateralFilter(img, 25, 75, 75)

    # histogram plot of img to determine threshold
    # plt.hist(img.ravel(), bins = img.max())
    # plt.title('Histogram of the input image')
    # plt.show()

    # binary thresholding based on histogram
    th_img = np.zeros_like(bil_img)
    th_img[bil_img <= 120.0] = 0
    th_img[bil_img > 120.0] = 1

    # split the image into four regions
    strel = morph.disk(3) # using disk strel to better structure the circular pears

    #...upper left...
    th_img[:int(img.shape[0]/2), :int(img.shape[1]/2)] = cv2.dilate(cv2.erode(th_img[:int(img.shape[0]/2), :int(img.shape[1]/2)], strel, iterations=12), strel, iterations=10) # seperating pears

    #...lower left...
    th_img[int(img.shape[0]/2):, :int(img.shape[1]/2)] = cv2.erode(cv2.dilate(th_img[int(img.shape[0]/2):, :int(img.shape[1]/2)], morph.disk(2), iterations=6), morph.disk(3), iterations=7) # filling holes
    th_img[int(img.shape[0]/2):, :int(img.shape[1]/2)] = cv2.dilate(cv2.erode(th_img[int(img.shape[0]/2):, :int(img.shape[1]/2)], strel, iterations=10), strel, iterations=10) # seperating pears

    #...upper right...
    th_img[:int(img.shape[0]/2), int(img.shape[1]/2):] = cv2.dilate(cv2.erode(th_img[:int(img.shape[0]/2), int(img.shape[1]/2):], strel, iterations=15), strel, iterations=10) # seperating pears

    #...lower right...
    th_img[int(img.shape[0]/2):, int(img.shape[1]/2):] = cv2.erode(cv2.dilate(th_img[int(img.shape[0]/2):, int(img.shape[1]/2):], morph.disk(2), iterations=6), morph.disk(3), iterations=6) # filling holes
    th_img[int(img.shape[0]/2):, int(img.shape[1]/2):] = cv2.dilate(cv2.erode(th_img[int(img.shape[0]/2):, int(img.shape[1]/2):], strel, iterations=10), strel, iterations=10) # seperating pears

    # apply final segmentation to entire img for touch-ups
    th_img = cv2.dilate(cv2.erode(th_img, strel, iterations=15), strel, iterations=14) # seperating pears
    # plt.imshow(th_img, cmap = 'gray')
    # plt.show()

    # get stats of segmented img
    num_objects, objects, stats, centers = cv2.connectedComponentsWithStats(th_img)

    # find the (x,y) coordinates of pears and their areas
    pear_x = [center[0] for center in centers]
    pear_y = [center[1] for center in centers]
    pear_size = [stat[-1] for stat in stats]

    return pear_x, pear_y, pear_size
#

# %% MAIN
# read in img
pears = cv2.imread('pears.png', cv2.IMREAD_GRAYSCALE)
plt.imshow(pears, cmap = 'gray')
plt.show()

# call pear function
x, y, area = countPears(pears)
print(f'There are {len(x)} pears with x coordinates {x} and y coordinates {y}\n')
print(f'The pears have areas {area}\n')