import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import watershed
import imageio
from scipy import ndimage as ndi
from tqdm import tqdm
import scipy.ndimage.filters as filters
import matplotlib.patches as patches
import cv2


def mark_local_maxima(image, neighborhood_size=25, threshold=1500):
    data_max = filters.maximum_filter(image, neighborhood_size)
    maxima = (image == data_max)
    data_min = filters.minimum_filter(image, neighborhood_size)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0

    labeled, num_objects = ndi.label(maxima)
    slices = ndi.find_objects(labeled)
    x, y = [], []
    for dy, dx in slices:
        x_center = (dx.start + dx.stop - 1) / 2
        x.append(x_center)
        y_center = (dy.start + dy.stop - 1) / 2
        y.append(y_center)

    return x, y


def dist2(p1, p2):
    return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2


def fuse(x, y, d):
    points = np.array([x, y]).T
    ret = []
    d2 = d * d
    n = len(points)
    taken = [False] * n
    for i in range(n):
        if not taken[i]:
            count = 1
            point = [points[i][0], points[i][1]]
            taken[i] = True
            for j in range(i + 1, n):
                if dist2(points[i], points[j]) < d2:
                    point[0] += points[j][0]
                    point[1] += points[j][1]
                    count += 1
                    taken[j] = True

            point[0] /= count
            point[1] /= count
            ret.append((point[0], point[1]))
    if len(ret) != 0:
        ret = np.array(ret)
        return ret[:, 0], ret[:, 1]
    else:
        return [], []
    

def distance_transform(pic):
    dt = cv2.distanceTransform(pic, cv2.DIST_L2, 5)
    return dt


def find_max(dt):
    x, y = mark_local_maxima(dt, 30, 5)  #окрестность поиск # порог
    x, y = fuse(x, y, 30) # unite maxima
    return x, y


def process_watershed(dt, mask):
    x, y = find_max(dt)
    if len(x)==0 and len(y)==0:
        return []

    local_maxi = np.zeros(dt.shape)
    for i, x_ in enumerate(tqdm(x)):

        if (x_ < 1) or (y[i] < 1) or ((-x_ + dt.shape[1]) < 1) or ((-y[i] + dt.shape[0]) < 1):
            continue

        local_maxi[int(y[i]), int(x_)] = True

    markers = ndi.label(local_maxi)[0]
    labels = watershed(-dt, markers, mask=mask)

    return labels


def process_frame(frame):
    mask_frame = frame.astype(np.uint8)
    dt = distance_transform(mask_frame)

    labels = process_watershed(dt, mask_frame)
    if len(labels)==0:
        return [], []
    uniq = np.unique(labels)

    output = np.zeros((len(uniq), 5))

    for i, label in enumerate(tqdm(np.unique(labels)[1:])):
        temp_image = np.zeros(labels.shape)
        temp_image[labels == label] = 1

        ccws = cv2.connectedComponentsWithStats(temp_image.astype(np.uint8), 4, cv2.CV_32S)
        output[i] = ccws[2][1]

    return output, labels


def detect_cells(total_image, x_1, x_2, y_1, y_2, cell_mask, figsize=(10, 60), path=None):
    
    exp_image = total_image[y_1:y_2, x_1:x_2]
    img_blur = cv2.GaussianBlur(exp_image, (3,3), 0)
    img_blur = np.uint8(img_blur*255)

    edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200) # Canny Edge Detection

    kernel = np.ones((5,5),np.uint8)
    dilation = cv2.dilate(edges,kernel,iterations = 1)

    inverted_mask = cv2.bitwise_not(dilation)
    inverted_mask[cell_mask[y_1:y_2, x_1:x_2] == 0] = 0 
    
    coords, labels = process_frame(inverted_mask)
    if (len(coords)==0) and (len(labels)==0):
        return [], []
    
    if figsize:
        plt.figure(figsize=figsize)
        ax = plt.gca()
        # Display the image
        ax.imshow(exp_image)

        # Create a Rectangle patch

        for coord in coords:
            rect = patches.Rectangle((coord[0], coord[1]), coord[2], coord[3], linewidth=1, edgecolor='#FBBA0E', facecolor='none')

            # Add the patch to the Axes
            ax.add_patch(rect)
        plt.axis('off')
        if path:
            plt.savefig(path, dpi=300, bbox_inches='tight')
        
    return coords, labels
