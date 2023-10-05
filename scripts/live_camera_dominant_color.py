'''
Sources: 
https://code.likeagirl.io/finding-dominant-colour-on-an-image-b4e075f98097
https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html

Changes:
I combined parts of the above two sources and added the find dominant color function as well
as cropping the region of interest of the image for finding the dominant color to just the center.
'''
import numpy as np
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

cap = cv.VideoCapture(0)
lower_blue = np.array([110, 50, 50]) 
upper_blue = np.array([130, 255, 255]) 

def find_histogram(clt):
    """
    create a histogram with k clusters
    :param: clt
    :return:hist
    """
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()

    return hist

def plot_colors2(hist, centroids):
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    # return the bar chart
    return bar

def find_dominant_color(clt):
    # Count the number of pixels assigned to each cluster
    num_pixels = np.bincount(clt.labels_)

    # Find the index of the cluster with the most pixels
    dominant_cluster = np.argmax(num_pixels)

    # Get the RGB values of the cluster center of the dominant cluster
    dominant_color_rgb = clt.cluster_centers_[dominant_cluster]

    return dominant_color_rgb.astype(int)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame_RGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    height, width, _ = frame_RGB.shape
    center_x = width // 2
    center_y = height // 2
    crop_size = 100

    center_crop = frame_RGB[center_y - crop_size // 2:center_y + crop_size // 2,
                            center_x - crop_size // 2:center_x + crop_size // 2]

    # Reshape the cropped image for K-means clustering
    img = center_crop.reshape((crop_size * crop_size, 3))

    #Threshold the blue values
    clt = KMeans(n_clusters=3) #cluster number
    clt.fit(img)

    dom_color = find_dominant_color(clt)

    print("Dominant Color in (RGB):", dom_color)

    cv.imshow('frame',frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()