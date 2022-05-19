import cv2 as cv
import numpy as np
import os

Seg_directory = 'aug-images\TEST-Seg'
hog = []
# NEU = 1
# EOS = 2
# MON = 3
# LYM = 4

class ImgClass:
  def __init__(self, img, num, label):
    self.img = img
    self.num = num
    self.label = label
    self

  def myfunc(self):
    print(self.label)
    print(self.num)
    cv.imshow("Image {num}, {label}".format(
        num = self.num, label = self.label), self.img)

def extractFeatures(self):

    img = self.img

    # define HOG parameters
    cell_size = (32, 32)  # h x w in pixels
    block_size = (2, 2)  # h x w in cells
    nbins = 9  # number of orientation bins
    # winSize is the size of the image cropped to an multiple of the cell size
    # cell_size is the size of the cells of the img patch over which to calculate the histograms
    # block_size is the number of cells which fit in the patch
    hog = cv.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1],
                                        img.shape[0] // cell_size[0] * cell_size[0]),
                            _blockSize=(block_size[1] * cell_size[1],
                                        block_size[0] * cell_size[0]),
                            _blockStride=(cell_size[1], cell_size[0]),
                            _cellSize=(cell_size[1], cell_size[0]),
                            _nbins=nbins)
    self.hog = hog.compute(img)

    return


for file in os.listdir(Seg_directory):

    filename = os.fsdecode(file)
    file_number = int(filename[-13:-8])
    file_label = filename[-7:-4]
    if file_label == 'NEU':
        file_label = [1]
    elif file_label == 'EOS':
        file_label = [2]
    elif file_label == 'MON':
        file_label = [3]
    elif file_label == 'LYM':
        file_label = [4]

    dir = os.path.join(Seg_directory, filename)
    img = cv.imread(dir)
    image_data = ImgClass(img, file_number, file_label)
    extractFeatures(image_data)
    hog.append(np.concatenate((image_data.label,image_data.hog)))
    print(dir)


np.savetxt("aug-test-hog.csv", hog, delimiter=",")


# cv.waitKey(0)
# cv.destroyAllWindows()