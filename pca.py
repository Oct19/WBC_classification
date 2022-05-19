from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA

import time
start_time = time.time()

# importing or loading the dataset
train_dataset = pd.read_csv('aug-train-hog.csv')
test_dataset = pd.read_csv('aug-test-hog.csv')
 
# distributing the dataset into two components X and Y
# X = dataset.iloc[:, 1:].values
# y = dataset.iloc[:, 0].values
# X_train, X_test, y_train, y_test = train_test_split(
#                 X, y, test_size = 0.2, random_state = 1)

X_train = train_dataset.iloc[:, 1:].values
y_train = train_dataset.iloc[:, 0].values

X_test = test_dataset.iloc[:, 1:].values
y_test = test_dataset.iloc[:,0].values



# performing preprocessing part
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Applying PCA function on training
# and testing set of X component

pca = PCA(n_components = 100)

X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

explained_variance = pca.explained_variance_ratio_

print("PCA--- %s seconds ---" % (time.time() - start_time))


# # Fitting Logistic Regression To the training set
# from sklearn.linear_model import LogisticRegression 
# classifier = LogisticRegression(random_state = 0)
# classifier.fit(X_train, y_train)
# # Predicting the test set result using
# # predict function under LogisticRegression
# y_pred = classifier.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print("Logistic Regression: " + str(accuracy))

# # Decision Tree
# from sklearn.tree import DecisionTreeClassifier
# model = DecisionTreeClassifier(min_samples_split=5)

# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print("Decision Tree: " + str(accuracy))

# # RandomForest
# from sklearn.ensemble import RandomForestClassifier
# clf = RandomForestClassifier(max_depth=100, random_state=0)
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print("Random Forest: " + str(accuracy))

# # SVM linear
# from sklearn.svm import SVC
# svclassifier = SVC(kernel='linear')
# svclassifier.fit(X_train, y_train)
# y_pred = svclassifier.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print("SVM Linear: " + str(accuracy))

# # SVM cubic
# from sklearn.svm import SVC
# svclassifier = SVC(kernel='poly',degree=3)
# svclassifier.fit(X_train, y_train)
# y_pred = svclassifier.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print("SVM cubic: " + str(accuracy))

# K nearest neighbors
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(X_train, y_train)
y_pred = neigh.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("KNN: " + str(accuracy))

print("PCA + Classification--- %s seconds ---" % (time.time() - start_time))

# making confusion matrix of test set of Y and predicted value
cm = confusion_matrix(y_test, y_pred)
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()