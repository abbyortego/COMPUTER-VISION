# %% IMPORTS
import matplotlib.pyplot as plt
import random
import numpy as np
import cv2

# %% DISTANCE
def euclideanDist(coord1, coord2):
    return sum(np.sqrt((np.array(coord1).astype(int) - np.array(coord2).astype(int))**2))
#

# %% KMEANS FUNCTIONS
def kMeansColor(r, g, b, k, its): # (r,g,b) k means
    centroids_r = [] # r,g,b coordinates for centers
    centroids_g = []
    centroids_b = []
    # choose k random points >> new centers
    for i in range(k):
        rndm_indx = random.randint(0, len(r)-1)
        centroids_r.append(r[rndm_indx])
        centroids_g.append(g[rndm_indx])
        centroids_b.append(b[rndm_indx])
    #

    clusters = {} # indx of data that belongs to each cluster
    # while old centers != new centers...
    for i in range(its): # loop until centers stop moving
        clusters = {}
        # assign data to clusters
        for centroid_indx in range(len(centroids_r)):
            pnts = [] # holds the list of points (by indx) that match the current cluster

            # loop through the points
            for pnt_indx in range(len(r)):
                # find dist from centers                
                dist = [euclideanDist([centroids_r[j], centroids_g[j], centroids_b[j]], [r[pnt_indx], g[pnt_indx], b[pnt_indx]]) for j in range(len(centroids_r))]

                # if the min(dist) is this centroid assign it to that cluster
                if centroid_indx == dist.index(min(dist)):
                    pnts.append(pnt_indx)
                #
                
                # add list of cluster pnts to the clusters dict
                clusters[centroid_indx] = pnts
            #
        #

        for key in clusters: # loop through clusters
            avg_r = np.mean([r[indx] for indx in clusters[key]])
            avg_g = np.mean([g[indx] for indx in clusters[key]])
            avg_b = np.mean([b[indx] for indx in clusters[key]]) # find avg for r,g,b
            
            centroids_r[key] = avg_r
            centroids_g[key] = avg_g
            centroids_b[key] = avg_b # set it to the centroids
        #
    #

    return centroids_r, centroids_g, centroids_b, clusters
#

# %% SAMPLE 1
sample1 = cv2.cvtColor(cv2.imread('SAMPLE IMAGES/sample1.jpg'), cv2.COLOR_BGR2RGB)
plt.imshow(sample1)
plt.show()

tst = plt.axes(projection = '3d')
sample1_2 = sample1.reshape((-1,3)) # (x,y) x (r,g,b)
print(sample1_2.shape)
tst.scatter3D(xs=sample1_2[:,0], ys=sample1_2[:,1], zs=sample1_2[:,2])
plt.show()

r_cntr, g_cntr, b_cntr, rgb_clusts = kMeansColor(sample1_2[:,0], sample1_2[:,1], sample1_2[:,2], 5, 50)

flowers = plt.axes(projection = '3d') # plotted rgb clusters
flowers.scatter3D(xs=sample1_2[rgb_clusts[0]][:,0], ys=sample1_2[rgb_clusts[0]][:,1], zs=sample1_2[rgb_clusts[0]][:,2])
flowers.scatter3D(xs=sample1_2[rgb_clusts[1]][:,0], ys=sample1_2[rgb_clusts[1]][:,1], zs=sample1_2[rgb_clusts[1]][:,2])
flowers.scatter3D(xs=sample1_2[rgb_clusts[2]][:,0], ys=sample1_2[rgb_clusts[2]][:,1], zs=sample1_2[rgb_clusts[2]][:,2])
flowers.scatter3D(xs=sample1_2[rgb_clusts[3]][:,0], ys=sample1_2[rgb_clusts[3]][:,1], zs=sample1_2[rgb_clusts[3]][:,2])
flowers.scatter3D(xs=sample1_2[rgb_clusts[4]][:,0], ys=sample1_2[rgb_clusts[4]][:,1], zs=sample1_2[rgb_clusts[4]][:,2])
plt.show()

for key in rgb_clusts: # img[clust_indx] = centers
    sample1_2[rgb_clusts[key]] = [r_cntr[key], g_cntr[key], b_cntr[key]]
#
sample1_3 = sample1_2.reshape((sample1.shape[0], sample1.shape[1], 3)) # shape back and plot
plt.imshow(sample1_3)
plt.show()

clust = {} # holds a dictionary of the segmented flowers
for key in rgb_clusts: # segmenting out similar flowers
    clust[key] = np.zeros_like(sample1_2)
    clust[key][rgb_clusts[key]] = [r_cntr[key], g_cntr[key], b_cntr[key]]
    clust[key] = clust[key].reshape((sample1.shape[0], sample1.shape[1], 3))
    plt.imshow(clust[key])
    plt.show()
#

# %% SAMPLE 3
sample3 = cv2.cvtColor(cv2.imread('SAMPLE IMAGES/sample3.jpg'), cv2.COLOR_BGR2RGB)
plt.imshow(sample3)
plt.show()

tst = plt.axes(projection = '3d')
sample3_2 = sample3.reshape((-1,3)) # (x,y) x (r,g,b)
tst.scatter3D(xs=sample3_2[:,0], ys=sample3_2[:,1], zs=sample3_2[:,2])
plt.show()

r_cntr, g_cntr, b_cntr, rgb_clusts = kMeansColor(sample3_2[:,0], sample3_2[:,1], sample3_2[:,2], 5, 50)

flowers = plt.axes(projection = '3d') # plotted rgb clusters
flowers.scatter3D(xs=sample3_2[rgb_clusts[0]][:,0], ys=sample3_2[rgb_clusts[0]][:,1], zs=sample3_2[rgb_clusts[0]][:,2])
flowers.scatter3D(xs=sample3_2[rgb_clusts[1]][:,0], ys=sample3_2[rgb_clusts[1]][:,1], zs=sample3_2[rgb_clusts[1]][:,2])
flowers.scatter3D(xs=sample3_2[rgb_clusts[2]][:,0], ys=sample3_2[rgb_clusts[2]][:,1], zs=sample3_2[rgb_clusts[2]][:,2])
flowers.scatter3D(xs=sample3_2[rgb_clusts[3]][:,0], ys=sample3_2[rgb_clusts[3]][:,1], zs=sample3_2[rgb_clusts[3]][:,2])
flowers.scatter3D(xs=sample3_2[rgb_clusts[4]][:,0], ys=sample3_2[rgb_clusts[4]][:,1], zs=sample3_2[rgb_clusts[4]][:,2])
plt.show()

for key in rgb_clusts: # img[clust_indx] = centers
    sample3_2[rgb_clusts[key]] = [r_cntr[key], g_cntr[key], b_cntr[key]]
#
sample3_3 = sample3_2.reshape((sample3.shape[0], sample3.shape[1], 3)) # shape back and plot
plt.imshow(sample3_3)
plt.show()

clust = {} # holds a dictionary of the segmented flowers
for key in rgb_clusts: # segmenting out similar flowers
    clust[key] = np.zeros_like(sample3_2)
    clust[key][rgb_clusts[key]] = [r_cntr[key], g_cntr[key], b_cntr[key]]
    clust[key] = clust[key].reshape((sample3.shape[0], sample3.shape[1], 3))
    plt.imshow(clust[key])
    plt.show()
#

# %% SAMPLE 4
sample4 = cv2.cvtColor(cv2.imread('SAMPLE IMAGES/sample4.jpg'), cv2.COLOR_BGR2RGB)
plt.imshow(sample4)
plt.show()

tst = plt.axes(projection = '3d')
sample4_2 = sample4.reshape((-1,3)) # (x,y) x (r,g,b)
tst.scatter3D(xs=sample4_2[:,0], ys=sample4_2[:,1], zs=sample4_2[:,2])
plt.show()

r_cntr, g_cntr, b_cntr, rgb_clusts = kMeansColor(sample4_2[:,0], sample4_2[:,1], sample4_2[:,2], 5, 50)

flowers = plt.axes(projection = '3d') # plotted rgb clusters
flowers.scatter3D(xs=sample4_2[rgb_clusts[0]][:,0], ys=sample4_2[rgb_clusts[0]][:,1], zs=sample4_2[rgb_clusts[0]][:,2])
flowers.scatter3D(xs=sample4_2[rgb_clusts[1]][:,0], ys=sample4_2[rgb_clusts[1]][:,1], zs=sample4_2[rgb_clusts[1]][:,2])
flowers.scatter3D(xs=sample4_2[rgb_clusts[2]][:,0], ys=sample4_2[rgb_clusts[2]][:,1], zs=sample4_2[rgb_clusts[2]][:,2])
flowers.scatter3D(xs=sample4_2[rgb_clusts[3]][:,0], ys=sample4_2[rgb_clusts[3]][:,1], zs=sample4_2[rgb_clusts[3]][:,2])
flowers.scatter3D(xs=sample4_2[rgb_clusts[4]][:,0], ys=sample4_2[rgb_clusts[4]][:,1], zs=sample4_2[rgb_clusts[4]][:,2])
plt.show()

for key in rgb_clusts: # img[clust_indx] = centers
    sample4_2[rgb_clusts[key]] = [r_cntr[key], g_cntr[key], b_cntr[key]]
#
sample4_3 = sample4_2.reshape((sample4.shape[0], sample4.shape[1], 3)) # shape back and plot
plt.imshow(sample4_3)
plt.show()

clust = {} # holds a dictionary of the segmented flowers
for key in rgb_clusts: # segmenting out similar flowers
    clust[key] = np.zeros_like(sample4_2)
    clust[key][rgb_clusts[key]] = [r_cntr[key], g_cntr[key], b_cntr[key]]
    clust[key] = clust[key].reshape((sample4.shape[0], sample4.shape[1], 3))
    plt.imshow(clust[key])
    plt.show()
#