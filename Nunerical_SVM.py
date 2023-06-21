# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets, metrics, model_selection, svm
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

dataset = pd.read_csv('colors.csv')
x = dataset.iloc[:,1:-1].values
y = dataset.iloc[:, -1].values

x[:, 1] = le.fit_transform(x[:, 1])
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])])
x = np.array(ct.fit_transform(x))


dataset['id']=le.fit_transform(dataset['id'])
dataset['name']=le.fit_transform(dataset['name'])
dataset['rgb']=le.fit_transform(dataset['rgb'])
dataset['is_trans']=le.fit_transform(dataset['is_trans'])

from sklearn.model_selection import train_test_split

a = dataset.iloc[:,3].values

b= dataset.iloc[:, -1].values

a=np.reshape(a,(-1,1))
a_train, a_test, b_train, b_test = train_test_split(a,b, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
a_train = sc.fit_transform(a_train)
a_test = sc.transform(a_test)

from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(a_train, b_train)

y_pred = classifier.predict(a_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), b_test.reshape(len(b_test),1)),1),'\n')

classifier.score(a_test,b_test)
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(b_test, y_pred)
print('confusion_matrix\n',cm)
print('Accuracy\n',(accuracy_score(b_test, y_pred)*100))

metrics.plot_roc_curve(classifier, a_test, b_test)
plt.show()
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve

a,b= load_digits(return_a_b=True)
estimator = SVC(gamma=0.001)
train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(estimator, a, b, cv=30,return_times=True)
plt.plot(train_sizes,np.mean(train_scores,axis=1))
