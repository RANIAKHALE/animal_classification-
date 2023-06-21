# -*- coding: utf-8 -*-
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
tf.__version__

le=LabelEncoder()
dataset=pd.read_csv('colors.csv')
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

from sklearn.preprocessing import MinMaxScaler
ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
checkpoint = keras.callbacks.ModelCheckpoint(filepath="weights.h5", verbose=1, save_best_only=True)

history=ann.fit(a_train, b_train,learning_rate=0.001, batch_size = 32,  verbose=1, epochs = 100,validation_split=0.2,callbacks=[checkpoint])
import matplotlib.pyplot as plt
ann.summary()
print(history.history.keys())
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['test','Validition'], loc='upper right')
plt.show()
print(ann.evaluate(a_test, b_test))
print(ann.metrics_names)
print(ann.predict(sc.transform([[7]])) > .5)
y_pred = ann.predict(b_test)
y_pred = (y_pred > 0.5)

print(np.concatenate((y_pred.reshape(len(y_pred),1), b_test.reshape(len(b_test),1)),1))
# ####Make Confusion Matrix#####################
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(b_test, y_pred)
print('confusion_matrix\n',cm)
print('Accuracy\n',(accuracy_score(b_test, y_pred)*100))

from sklearn.metrics import roc_curve, auc 
y_pred = ann.predict(a_test).ravel()
nn_fpr_keras, nn_tpr_keras, nn_thresholds_keras = roc_curve(b_test , y_pred)
auc_keras = auc(nn_fpr_keras, nn_tpr_keras)
plt.plot(nn_fpr_keras, nn_tpr_keras, marker='.', label='Neural Network (auc = %0.3f)' % auc_keras)


