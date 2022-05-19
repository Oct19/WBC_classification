import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import hsv_to_rgb

import argparse
import pandas as pd

# titles and images for segmentation plots
titles = []
images = []
draw = False

# Load image and convert to RGB, Grayscale, HSV color space

JPEG_directory = 'blood-cell-wt\JPEGImages'
Seg_directory = 'blood-cell-wt\Segmentation'
Aug_directory = 'aug-images\TRAIN'
Aug_Seg_directory = 'aug-images\TRAIN-Seg'

df=pd.read_csv('blood-cell-wt\labels-clean.csv', sep=',',header=None)

label_dict = dict(zip(df.values[:,0], df.values[:,1]))


img = cv.imread("blood-cell-wt\JPEGImages\BloodImage_00030.jpg")

def rgb_mask(img):
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    if draw:
        titles.append('Original')
        images.append(img_rgb)
    # Set min and max value of RGB
    rgb_min = np.array([0,0,195])
    rgb_max = np.array([205,190,255])
    rgb_mask = cv.inRange(img_rgb, rgb_min, rgb_max)
    if draw:
        titles.append('RGB Threshold')
        images.append(rgb_mask)

    return rgb_mask



# img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
# # Set min and max value of Grayscale
# gray_min = np.array([100])
# gray_max = np.array([155])
# gray_mask = cv.inRange(img_gray, gray_min, gray_max)
# gray_result = cv.bitwise_and(img, img, mask=gray_mask)

# # Set min and max value of HSV
# hsv_min = np.array([113,35,157])
# hsv_max = np.array([180,255,255])
# hsv_mask = cv.inRange(img_hsv, hsv_min, hsv_max)
# hsv_result = cv.bitwise_and(img, img, mask=hsv_mask)


# def ColorSpacePlot():
#     cv.imshow("Original RGB", img)
#     cv.imshow("RGB mask", rgb_result)
#     cv.imshow("Grayscale", img_gray)
#     cv.imshow("Gray mask", gray_result)
#     cv.imshow("HSV", img_hsv)
#     cv.imshow("HSV mask", hsv_result)
#     cv.waitKey(0)
#     cv.destroyAllWindows()

# ColorSpacePlot()

def dilation(mask):
    # disk shape kernal (R=3)
    y,x = np.ogrid[-3: 3+1, -3: 3+1]
    kernal = x**2+y**2 <= 3**2
    kernal = kernal.astype(np.uint8)
    dilated_mask = cv.dilate(mask, kernal, iterations=1)

    if draw:
        titles.append('Dilation')
        images.append(dilated_mask)

    return dilated_mask

# dilated_mask = dilation(rgb_mask)

def findLargestblob(mask):
    # If all four neighbors are 0, assign a new label
    connectivity = 4
    connected = cv.connectedComponentsWithStats(mask, connectivity, cv.CV_32S)
    (numLabels, labels, stats, _) = connected

    # Largest area is background, second largest is cell
    areas = stats[:,cv.CC_STAT_AREA]
    i = areas.argsort()[-2]
    area = areas[i]
    blobMask = (labels).astype("uint8") * 0

    # if blob area is decent, consider as correct detection
    if (area > 1000 and area < 100000):
        blobMask = (labels == i).astype("uint8") * 255
        BlobFound = True
    else:
        BlobFound = False

    if draw:
        titles.append('Largest blob')
        images.append(blobMask)
    return blobMask, BlobFound

# largest_blob = findLargestblob(dilated_mask)

def boundingRect(mask):

    # Finding contour for the thresholded image
    contours, hierarchy = cv.findContours(
        mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # Find longest contour
    contour = np.array(sorted(contours, key=len, reverse=True))[0]

    [x,y,w,h] = cv.boundingRect(contour)
    
    # check good aspect ratio
    good_ratio = True
    if w/h>1.5 or h/w>1.5:
        good_ratio = False

    # if draw:
    #     # draw contour
    #     drawing_contour = cv.bitwise_and(img_rgb, img_rgb, mask=largest_blob)
    #     cv.drawContours(drawing_contour, contour, -1, (0,255,0), 5, 8)
    #     titles.append('Contour')
    #     images.append(drawing_contour)
    #     # draw boundingRect
    #     drawing_bounding_rect = img_rgb.copy()
    #     cv.rectangle(drawing_bounding_rect, (x,y), (x+w,y+h), (0, 255, 0), 5, 8)
    #     titles.append('Bounding Box')
    #     images.append(drawing_bounding_rect)

    return x,y,w,h,good_ratio

# boundingRect(largest_blob)



if draw:
    for i in range(len(titles)):
        plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])

    plt.show()


def segmentation(import_dir, export_dir):
    for file in os.listdir(import_dir):
        filename = os.fsdecode(file)
        file_number = int(filename[-9:-4])
        if filename.endswith(".jpg") and file_number in label_dict:
            img = cv.imread(os.path.join(import_dir, filename))

            # RGB mask
            img_mask = rgb_mask(img)
            # Dilation
            img_mask = dilation(img_mask)
            # Largest blob
            img_mask = findLargestblob(img_mask)
            # Bounding box
            x,y,w,h = boundingRect(img_mask)
            # Crop image
            crop_img = img[y:y+h, x:x+w]
            # resize
            resized = cv.resize(crop_img, (128,128), 
                                interpolation=cv.INTER_LINEAR)

            label = label_dict[file_number]

            savename = os.path.join(export_dir, 'Seg_' + 
                                str(file_number).zfill(5) + '_' + label + '.jpg')
            cv.imwrite(savename, resized)

    return 

# segmentation(JPEG_directory, Seg_directory)

def augSegmentation(import_dir, export_dir):
    # number of img
    i = 0
    for subdir, dirs, files in os.walk(import_dir):
        label = os.path.basename(subdir)
        for file in files:
            print(os.path.join(subdir, file))
            img = cv.imread(os.path.join(subdir, file))

            # RGB mask
            img_mask = rgb_mask(img)
            # Dilation
            img_mask = dilation(img_mask)
            # Largest blob
            img_mask,BlobFound = findLargestblob(img_mask)
            # Skip if on blob found
            if not BlobFound:
                continue
            # Bounding box
            x,y,w,h,good_ratio = boundingRect(img_mask)

            if not good_ratio:
                continue
            # Crop image
            crop_img = img[y:y+h, x:x+w]
            # resize
            resized = cv.resize(crop_img, (128,128), 
                                interpolation=cv.INTER_LINEAR)

            savename = os.path.join(export_dir, 'Seg_' + 
                                str(i).zfill(5) + '_' + label[0:3] + '.jpg')
            print(savename)
            cv.imwrite(savename, resized)
            i += 1

    return 

augSegmentation(Aug_directory, Aug_Seg_directory)