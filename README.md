# WBC_classification

BCCD Dataset is a small-scale dataset for blood cells detection.

Thanks the original data and annotations from [cosmicad](https://github.com/cosmicad/dataset) and [akshaylamba](https://github.com/akshaylamba/all_CELL_data). The original dataset is re-organized into VOC format. BCCD Dataset is under *[MIT licence](./LICENSE)*.


## Instructions

1. [Download](https://www.kaggle.com/paultimothymooney/blood-cells) the augmented dataset

2. Copy `dataset2-master\images` to root directory and rename folder as `aug-image`

3. **[Optional]** Install packages from `requirements.txt`

4. Run `segmentation.py` to generate segmented images, saved in `aug-image`

5. Run `hog.py` for segmented train and test images seperately, saved in `aug-train-hog.csv` and `aug-test-hog.csv`

6. Run `pca.py` for feature extraction. Traditional machine learing methods are also included. Uncomment to run the following methods:

    - Logistic regression
    - Decision trees
    - RandomForest
    - KNN
    - SVM


