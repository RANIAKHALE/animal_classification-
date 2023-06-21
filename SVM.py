# -*- coding: utf-8 -*-
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
tf.__version__
dataset = pd.read_csv('student-mat.csv')
x = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values
print(dataset.head)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x[:, 2] = le.fit_transform(x[:, 2])
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
x = np.array(ct.fit_transform(x))


dataset['guardian']=le.fit_transform(dataset['guardian'])
dataset['age']=le.fit_transform(dataset['age'])
dataset['Pstatus']=le.fit_transform(dataset['Pstatus'])
dataset['address']=le.fit_transform(dataset['address'])
dataset['famsize']=le.fit_transform(dataset['famsize'])
dataset['reason']=le.fit_transform(dataset['reason'])
dataset['school']=le.fit_transform(dataset['school'])
dataset['Mjob']=le.fit_transform(dataset['Mjob'])
dataset['Fjob']=le.fit_transform(dataset['Fjob'])

dataset = dataset.replace({'sex': {'F': 1,'M': 0}})
dataset = dataset.replace({'romantic': {'yes': 1,'no': 0}})
dataset = dataset.replace({'internet': {'yes': 1,'no': 0}})
dataset = dataset.replace({'higher': {'yes': 1,'no': 0}})
dataset = dataset.replace({'nursery': {'yes': 1,'no': 0}})
dataset = dataset.replace({'activities': {'yes': 1,'no': 0}})
dataset = dataset.replace({'paid': {'yes': 1,'no': 0}})
dataset = dataset.replace({'famsup': {'yes': 1,'no': 0}})
dataset = dataset.replace({'schoolsup': {'yes': 1,'no': 0}})

from sklearn.model_selection import train_test_split
a = dataset.iloc[:, 29].values
b = dataset.iloc[:, -1].values
a=np.reshape(a,(-1,1))
a_train, a_test, b_train, b_test = train_test_split(a, b, test_size = 0.25, random_state = 0)
# from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer
# tfidf_vect=TfidfVectorizer(stop_words='english',max_df=0.25)
# tfidf_train=tfidf_vect.fit_transform(a_train)
# tfidf_test=tfidf_vect.transform(a_test)
# from sklearn.linear_model import SGDClassifier
# fake_detector_svc = SGDClassifier().fit(tfidf_train, b_train)
# predictions = fake_detector_svc.predict(tfidf_test)
# predictions
# from sklearn.metrics import classification_report
# print(classification_report(b_test,predictions))
print(a_train)
print(b_train)
print(a_test)
print(b_test)
print('............................')
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
a_train = sc.fit_transform(a_train)
a_test = sc.transform(a_test)

print(a_train)
print('............................')
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(a_train, b_train)

print(classifier.predict(sc.transform([[10000]])))

b_pred = classifier.predict(a_test)
print(np.concatenate((b_pred.reshape(len(b_pred),1), b_test.reshape(len(b_test),1)),1))

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(b_test, b_pred)
print('Confusion Matrix:')
print(cm)

accuracy_score(b_test, b_pred)
print('............................')
from sklearn import svm
from sklearn import metrics
clf = svm.SVC()
clf.fit(a_train, b_train)
b_pred = clf.predict(a_test)
acc = metrics.accuracy_score(b_test, b_pred)
print('accuaracy:')
print(acc)
print('.......................')
metrics.plot_roc_curve(classifier, a_test, b_test)
plt.show()
