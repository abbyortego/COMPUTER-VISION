import math as m
import numpy as np
from skimage import data, color
from skimage.util import random_noise
import matplotlib.pyplot as plt

''' find spatial matrix given sample '''
def spatial(winSize, sigmaS):
    mtrx = np.zeros([winSize, winSize], dtype = float) # initiaties spat_mtrx

    # find center
    center = [int(winSize / 2), int(winSize / 2)] 

    # calculate mtrx using the center as a basis for coordinates
    for x,y in np.ndindex(mtrx.shape):
        mtrx[x,y] = m.exp(-((x-center[0])**2 + (y-center[1])**2) / (2 * sigmaS**2))
    #

    return mtrx
#

''' find intensity matrix given sample '''
def intensity(smpl, sigmaR):
    mtrx = np.zeros([len(smpl), len(smpl)], dtype = float) # initiates inten_mtrx

    # find center value
    center = smpl[int(smpl.shape[0] / 2), int(smpl.shape[0] / 2)] 

    # calculate intensity compared to center
    for x,y in np.ndindex(mtrx.shape):
        mtrx[x,y] = m.exp(-(center-smpl[x, y])**2 / (2 * sigmaR**2))
    #

    return mtrx
#

''' bilateral filter '''
def bilateralFilter(img, sigmaR, sigmaS, winSize):
    # pad the img
    padded_img = np.pad(img, int(winSize / 2))

    # find spatial matrix first >> x = x coord, y = y coord
    spat_mtrx = spatial(winSize, sigmaS)

    # for each point in the img matrix...
    for x,y in np.ndindex(img.shape):

        # ...grab smpl matrix...
        smpl_mtrx = padded_img[x:winSize+x, y:winSize+y]

        # ...find its intensity matrix >> y = center, x = all others...
        inten_mtrx = intensity(smpl_mtrx, sigmaR)

        # ...pnt-wise mult. with spatial matrix and normalization...
        b_filter = (spat_mtrx * inten_mtrx)  / np.sum((spat_mtrx * inten_mtrx) )

        # ...pnt-wise mult. with sample and summation to replace point...
        img[x,y] = np.sum(smpl_mtrx * b_filter)
        
    # ...fin

    return img
#

''' testing the filter '''
# create fig
fig = plt.figure(figsize = (16,6))
fig.suptitle('Bilateral Filter')

# grab test img from sk image
fig.add_subplot(1,3, 1)
img = color.rgb2gray(data.astronaut())
plt.imshow(img, cmap='gray')
plt.title('Original Image')

# add some noise
fig.add_subplot(1,3, 2)
noise_img = random_noise(img, mode='gaussian')
plt.imshow(noise_img, cmap='gray')
plt.title('Noisey Image')

# filter the img
fig.add_subplot(1,3, 3)
filtered_img = bilateralFilter(noise_img, 15, 1, 9)
plt.imshow(filtered_img, cmap='gray')
plt.title('Filtered Noisey Image')

# show results
plt.show()