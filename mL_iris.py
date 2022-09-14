from ast import increment_lineno
import numpy as np
import pandas as pd 
from pandas import Series, DataFrame

import seaborn as sns 
import matplotlib.pyplot as plt




iris = pd.read_csv(r'C:\Users\dozie\Downloads\iris.csv') 
print(iris.head())
iris.info()


#Some EDA with Iris
fig = iris[iris.variety == 'Setosa'].plot(kind='scatter', x='sepal.length', y='sepal.width', color='orange', label='Setosa')
iris[iris.variety == 'Versicolor'].plot(kind='scatter', x='sepal.length', y='sepal.width', color='blue', label='Versicolor', ax=fig)
iris[iris.variety == 'Virginica'].plot(kind='scatter', x='sepal.length', y='sepal.width', color='green', label='Virginica', ax=fig)

fig.set_xlabel('Sepal Length')
fig.set_ylabel('Sepal Width')
fig.set_title('Sepal Length Vs Width')

fig=plt.gcf()
fig.set_size_inches(10, 7)
#plt.show()

fig = iris[iris.variety == 'Setosa'].plot(kind='scatter', x='petal.length', y='petal.width', color='orange', label='Setosa')
iris[iris.variety == 'Versicolor'].plot(kind='scatter', x='petal.length', y='petal.width', color='blue', label='Versicolor', ax=fig)
iris[iris.variety == 'Virginica'].plot(kind='scatter', x='petal.length', y='petal.width', color='green', label='Virginica', ax=fig)

fig.set_xlabel('Petal Length')
fig.set_ylabel('Petal Width')
fig.set_title('Petal Length Vs Width')

fig=plt.gcf()
fig.set_size_inches(10, 7)
#plt.show()

# importing alll the necessary packages to use the various classification algorithms
from sklearn.linear_model import LogisticRegression # for Logistic Regression Algorithm
from sklearn.model_selection import train_test_split # to split the dataset for training and testing 
from sklearn.neighbors import KNeighborsClassifier # KNN classifier
from sklearn import svm # for suport vector machine algorithm
from sklearn import metrics # for checking the model accuracy
from sklearn.tree import DecisionTreeClassifier # for using DTA

print(iris.shape)

plt.figure(figsize=(8,4))
sns.heatmap(iris.corr(), annot=True, cmap='cubehelix_r') # draws heatmap with input as correlation matrix calculated by iris.corr() 
plt.show()

train, test = train_test_split(iris, test_size=0.3) # our main data split into train and test
# the attribute test_size=0.3 splits the data into 70% and 30% ratio. train=70% and test=30%
print(train.shape)
print(test.shape)

train_X = train[['sepal.length','sepal.width','petal.length','petal.width']] # taking the training data features
train_y = train.variety # output of the training data

test_X = test[['sepal.length','sepal.width','petal.length','petal.width']] # taking test data feature
test_y = test.variety # output value of the test data

print(train_X.head())

print(test_X.head())

print(train_y.head())

#Support Vector Machine 
model = svm.SVC() # select the svm algorithm

# we train the algorithm with training data and training output
model.fit(train_X, train_y)

# we pass the testing data to the stored algorithm to predict the outcome
prediction = model.predict(test_X)
print('The accuracy of the SVM is: ', metrics.accuracy_score(prediction, test_y)) # we check the accuracy of the algorithm
#we pass the predicted output by the model and the actual output

#Logistic Regresssion 
model = LogisticRegression()
model.fit(train_X, train_y)
prediction = model.predict(test_X)
print('The accuracy of Logistic Regression is: ', metrics.accuracy_score(prediction, test_y))

#Decision Tree
model = DecisionTreeClassifier()
model.fit(train_X, train_y)
prediction = model.predict(test_X)
print('The accuracy of Decision Tree is: ', metrics.accuracy_score(prediction, test_y))

#K Nearest Neighbor
model = KNeighborsClassifier(n_neighbors=3) # this examines 3 neighbors for putting the data into class
model.fit(train_X, train_y)
prediction = model.predict(test_X)
print('The accuracy of KNN is: ', metrics.accuracy_score(prediction, test_y))

#Let's check the accuracy for various values of n for K-Nearest nerighbours
a_index = list(range(1,11))
a = pd.Series(dtype='float64')
for i in list(range(1,11)):
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(train_X, train_y)
    prediction = model.predict(test_X)
    a = a.append(pd.Series(metrics.accuracy_score(prediction, test_y)))
plt.plot(a_index, a)
x = [1,2,3,4,5,6,7,8,9,10]
plt.xticks(x)

# Creating separate training data for petals and sepals
petal = iris[['petal.length','petal.width','variety']]
sepal = iris[['sepal.length','sepal.width','variety']]

# For iris petal
train_p,test_p = train_test_split(petal, test_size=0.3, random_state=0) #petals
train_x_p = train_p[['petal.width','petal.length']]
train_y_p = train_p.variety

test_x_p = test_p[['petal.width','petal.length']]
test_y_p = test_p.variety

# For iris sepal 
train_s,test_s = train_test_split(sepal, test_size=0.3, random_state=0) #sepals
train_x_s = train_s[['sepal.width','sepal.length']]
train_y_s = train_s.variety

test_x_s = test_s[['sepal.width','sepal.length']]
test_y_s = test_s.variety

#SVM algorithm
model=svm.SVC()
model.fit(train_x_p,train_y_p) 
prediction=model.predict(test_x_p) 
print('The accuracy of the SVM using Petals is:',metrics.accuracy_score(prediction,test_y_p))

model=svm.SVC()
model.fit(train_x_s,train_y_s) 
prediction=model.predict(test_x_s) 
print('The accuracy of the SVM using Sepals is:',metrics.accuracy_score(prediction,test_y_s))

#Logistic Regression
model = LogisticRegression()
model.fit(train_x_p,train_y_p) 
prediction=model.predict(test_x_p) 
print('The accuracy of the Logistic Regression using Petals is:',metrics.accuracy_score(prediction,test_y_p))

model.fit(train_x_s,train_y_s) 
prediction=model.predict(test_x_s) 
print('The accuracy of the Logistic Regression using Sepals is:',metrics.accuracy_score(prediction,test_y_s))

#Decision Tree 
model=DecisionTreeClassifier()
model.fit(train_x_p,train_y_p) 
prediction=model.predict(test_x_p) 
print('The accuracy of the Decision Tree using Petals is:',metrics.accuracy_score(prediction,test_y_p))

model.fit(train_x_s,train_y_s) 
prediction=model.predict(test_x_s) 
print('The accuracy of the Decision Tree using Sepals is:',metrics.accuracy_score(prediction,test_y_s))

#K Nearesr Neighbor
model=KNeighborsClassifier(n_neighbors=3) 
model.fit(train_x_p,train_y_p) 
prediction=model.predict(test_x_p) 
print('The accuracy of the KNN using Petals is:',metrics.accuracy_score(prediction,test_y_p))

model.fit(train_x_s,train_y_s) 
prediction=model.predict(test_x_s) 
print('The accuracy of the KNN using Sepals is:',metrics.accuracy_score(prediction,test_y_s))