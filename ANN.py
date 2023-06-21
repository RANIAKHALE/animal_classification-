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

print(dataset)
print(".....................")
print(".....................")
dataset = dataset.astype(str).astype(float)
print(dataset.dtypes)
from sklearn.model_selection import train_test_split
a = dataset.iloc[:, 2].values
b = dataset.iloc[:, -1].values
# a_train=dataset[0:316:,0:33];
# b_train=dataset[0:316,33];

# a_test=dataset[316::,0:33];
# b_test=dataset[316:,33];
data = dataset[:,0:2]
labels = dataset[:,2]

a_train, a_test, b_train, b_test = train_test_split(data, labels,test_size=0.2)

plt.scatter(a_train[:,0], a_train[:,1], s=40, c=b_train, cmap=plt.cm.Spectral)
plt.show()


