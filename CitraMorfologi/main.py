import cv2
import argparse
import numpy as np
import os
import sys
from matplotlib import pyplot as plt
import skimage.io as io
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu

## For good sake :D we have to separate the def and main code
## Def Section ##
def switch(case):
    options={
        'erosi':erosi,
        'dilasi':dilasi,
        'opening':opening,
        'closing':closing,
        'thinning':thinning,
        'thickening':thickening
    }
    if case not in options:
        nothing()
    else:
        return options.get(case)

def nothing():
    print('You have wrong input the options')
    sys.exit()

def erosi():
    global morphOptions,morphTitle
    morphOptions = cv2.erode(thresh,kernel,iterations = 2)
    morphTitle = 'Erosi'

def dilasi():
    global morphOptions,morphTitle
    morphOptions = cv2.dilate(thresh,kernel,iterations = 2)
    morphTitle = 'Dilasi'

def opening():
    global morphOptions,morphTitle
    morphOptions = cv2.morphologyEx(img_org,cv2.MORPH_OPEN,kernel,iterations = 1)
    morphTitle = 'Opening'

def closing():
    global morphOptions,morphTitle
    morphOptions = cv2.morphologyEx(img_org,cv2.MORPH_CLOSE,kernel,iterations = 1)
    morphTitle = 'Closing'

def thinning():
    global morphOptions,morphTitle
    morphOptions = zhangSuen(img_org)
    morphTitle = 'Thinning'

def thickening():
    print('thickening')
    sys.exit()

def neighbours(x,y,image):
    "Return 8-neighbours of image point P1(x,y), in a clockwise order"
    img = image
    x_1, y_1, x1, y1 = x-1, y-1, x+1, y+1
    return [ img[x_1][y], img[x_1][y1], img[x][y1], img[x1][y1],     # P2,P3,P4,P5
                img[x1][y], img[x1][y_1], img[x][y_1], img[x_1][y_1] ]    # P6,P7,P8,P9

def transitions(neighbours):
    "No. of 0,1 patterns (transitions from 0 to 1) in the ordered sequence"
    n = neighbours + neighbours[0:1]      # P2, P3, ... , P8, P9, P2
    return sum( (n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]) )  # (P2,P3), (P3,P4), ... , (P8,P9), (P9,P2)

def zhangSuen(image):
    "the Zhang-Suen Thinning Algorithm"
    Image_Thinned = image.copy()  # deepcopy to protect the original image
    changing1 = changing2 = 1        #  the points to be removed (set as 0)
    while changing1 or changing2:   #  iterates until no further changes occur in the image
        # Step 1
        changing1 = []
        rows, columns = Image_Thinned.shape               # x for rows, y for columns
        for x in range(1, rows - 1):                     # No. of  rows
            for y in range(1, columns - 1):            # No. of columns
                P2,P3,P4,P5,P6,P7,P8,P9 = n = neighbours(x, y, Image_Thinned)
                if (Image_Thinned[x][y] == 1     and    # Condition 0: Point P1 in the object regions 
                    2 <= sum(n) <= 6   and    # Condition 1: 2<= N(P1) <= 6
                    transitions(n) == 1 and    # Condition 2: S(P1)=1  
                    P2 * P4 * P6 == 0  and    # Condition 3   
                    P4 * P6 * P8 == 0):         # Condition 4
                    changing1.append((x,y))
        for x, y in changing1: 
            Image_Thinned[x][y] = 0
        # Step 2
        changing2 = []
        for x in range(1, rows - 1):
            for y in range(1, columns - 1):
                P2,P3,P4,P5,P6,P7,P8,P9 = n = neighbours(x, y, Image_Thinned)
                if (Image_Thinned[x][y] == 1   and        # Condition 0
                    2 <= sum(n) <= 6  and       # Condition 1
                    transitions(n) == 1 and      # Condition 2
                    P2 * P4 * P8 == 0 and       # Condition 3
                    P2 * P6 * P8 == 0):            # Condition 4
                    changing2.append((x,y))    
        for x, y in changing2: 
            Image_Thinned[x][y] = 0
    return Image_Thinned
## End Def Section ##

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to images")
ap.add_argument("-o", "--options", required=True, type=str,
	help="options to process the morph")
args = vars(ap.parse_args())

if os.path.isfile(args['image']):
    if args['options'] not in ('thinning','thickening'):
        img_org = cv2.imread(args['image'],cv2.IMREAD_GRAYSCALE)
        ret, thresh = cv2.threshold(img_org, 250, 225, cv2.THRESH_BINARY)
    else:
        img_read = rgb2gray(io.imread(args['image']))
        thresh = threshold_otsu(img_read)
        img_org = img_read < thresh    # must set object region as 1, background region as 0 !
else:
    print('Must be a valid image or path')
    sys.exit()


kernel = np.ones((5,5),np.uint8)
switch(args['options'])()

if args['options'] not in ('thinning','thickening'):
    plt.subplot(131),plt.imshow(img_org,cmap = 'gray')
    plt.title('Citra Awal'), plt.xticks([]), plt.yticks([])
    plt.subplot(132),plt.imshow(thresh,cmap = 'gray')
    plt.title('Citra Biner'), plt.xticks([]), plt.yticks([])
    plt.subplot(133),plt.imshow(morphOptions,cmap = 'gray')
    plt.title('Hasil {}'.format(morphTitle)), plt.xticks([]), plt.yticks([])
    plt.show()
else:
    fig, ax = plt.subplots(1, 2)
    ax1, ax2 = ax.ravel()
    ax1.imshow(img_org, cmap=plt.cm.gray)
    ax1.set_title('Original binary image')
    ax1.axis('off')
    ax2.imshow(morphOptions, cmap=plt.cm.gray)
    ax2.set_title('Skeleton of the image')
    ax2.axis('off')
    plt.show()

