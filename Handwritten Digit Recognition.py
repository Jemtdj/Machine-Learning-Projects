###### Basic Machine Learning Example: Handwritten Digit Recognition ######
'''
Author: Jeremy Tan
Description: Utilizes several classifier techniques to train and predict
handwritten digits
Datasets: 
	1) Sklearn digit database
		- 1,797 samples, 8x8 pixel dimension
	2) MNIST digit database
		- 70,000 samples, 28x28 pixel dimension
Note: the digit database for the MNIST takes significantly longer to process
 	because the pixel dimension is larger as compared to the Sklearn dataset.
'''


##### Importing the required packages ####
import numpy as np
from sklearn.datasets import fetch_mldata, load_digits
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt



### Fetching the data from respective database ###
# fetching Sklearn data of handwritten digits
digits = load_digits()
# fetching MNIST data of handwritten digits from mldata.org
mnist = fetch_mldata('MNIST original', data_home='custom_data_home')
# Splitting data into training and test sample as predefined by MNIST database
# also reshaping the MNIST bitmap data from 2d to 3d to generate image
dataset = int(input('Which dataset would you like to use, Sklearn (1) or MNIST (2)? Please enter 1 or 2: '))
if dataset == 1:
	X_train, y_train = digits.data[:1500], digits.target[:1500]
	X_test, y_test = digits.data[1500:], digits.target[1500:]
elif dataset == 2:
	X_train, y_train = mnist.data[:60000], mnist.target[:60000]
	X_test, y_test = mnist.data[60000:], mnist.target[60000:]
	X_train_image = np.reshape(X_train, (60000,28,28))
	X_test_image = np.reshape(X_test, (10000,28,28))
else:
	raise ValueError('Please enter the number 1 or 2: ')


### Classification Models ### 
if dataset == 1:
	num = int(input('Choose a number from 1 to 297: '))
elif dataset == 2:
	num = int(input('Choose a number from 1 to 10000: '))

# Simple linear regression
# Training model
lm = LinearRegression()
lm.fit(X_train, y_train)
# Overall accuracy
# predict = lm.predict(X_test)
# print('Linear Regression - Accuracy: ', accuracy_score(y_test, predict))
# Using trained model to predict single number
actual = y_test[num]
lm_predicted = lm.predict(X_test[num])
# Returning results
print('Linear Regression - Actual: ', actual)
print('Linear Regression - Predicted: ', lm_predicted)

# Decision Trees
# Training model
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
# Overall accuracy
predict = dtc.predict(X_test)
print('Decision Tree - Accuracy: ', accuracy_score(y_test, predict))
# Using trained model to predict
actual = y_test[num]
dtc_predicted = dtc.predict(X_test[num])
# Returning results
print('Decision Tree - Actual: ', actual)
print('Decision Tree - Predicted: ', dtc_predicted)

# Using K-Nearest Neighbours
# Training model
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)
# Overall accuracy
predict = knn.predict(X_test)
print('KNN - Accuracy: ', accuracy_score(y_test, predict))
# Using trained model to predict
actual = y_test[num]
knn_predicted = knn.predict(X_test[num])
# Returning results
print('KNN (10) - Actual: ', actual)
print('KNN (10) - Predicted: ', knn_predicted)
# print(accuracy_score(actual, predicted))

# Support Vector Machines
# Training model
clf = svm.SVC(gamma=0.01)
clf.fit(X_train, y_train)
# Overall accuracy
predict = clf.predict(X_test)
print('SVM - Accuracy: ', accuracy_score(y_test, predict))
# Using trained model to predict
actual = y_test[num]
svm_predicted = clf.predict(X_test[num])
# Returning results
print('SVM - Actual: ', actual)
print('SVM - Predicted: ', svm_predicted)


### Generating image results for various classifiers ###
if dataset == 1:
	plt.imshow(digits.images[1500+num], cmap=plt.cm.gray_r, interpolation='nearest')
elif dataset == 2:
	plt.imshow(X_test_image[num], cmap=plt.cm.gray_r, interpolation='nearest')
plt.axis('off')
plt.title('Prediction: \n Linear Regression: %i  Decision Tree: %i  KNN: %i  SVM %i' % (lm_predicted, dtc_predicted, knn_predicted, svm_predicted))
plt.show()




