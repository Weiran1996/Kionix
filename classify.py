import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

norm = np.load('norm.npy')
abn = np.load('abn.npy')
fast = np.load('fast.npy')
slow = np.load('slow.npy')
single = np.load('single.npy')

acc_data = np.empty((100, 15))
label = np.empty(100, dtype=object)

acc_data[0:20, 0:15] = norm
acc_data[20:40, 0:15] = abn
acc_data[40:60, 0:15] = fast
acc_data[60:80, 0:15] = slow
acc_data[80:100, 0:15] = single
for i in range(20):
	label[i] = 'norm'
	label[20+i] = 'abn'
	label[40+i] = 'fast'
	label[60+i] = 'slow'
	label[80+i] = 'single'

X_train, X_test, y_train, y_test = train_test_split(acc_data, label, test_size=0.2, random_state=0)

#Gaussian Naive Bayes

#Polynomial SVM
clf = SVC(degree=2, gamma='auto', kernel='poly').fit(X_train, y_train)
y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)
acc_train = accuracy_score(y_train, y_train_pred)
acc_test = accuracy_score(y_test, y_test_pred)
print("SVM with Polynomial kernel, degree 2")
print("training accuracy: ", acc_train)
print("testing accuracy: ", acc_test)
#acc_train = []
#acc_test = []
#n = np.array(range(20))
#c = 2**n
#for i in range(20):
#	clf = SVC(C=c[i], degree=2, gamma='auto', kernel='poly').fit(X_train, y_train)
#	y_train_pred = clf.predict(X_train)
#	y_test_pred = clf.predict(X_test)
#	acc_train.append(accuracy_score(y_train, y_train_pred))
#	acc_test.append(accuracy_score(y_test, y_test_pred))

#plt.figure()
#plt.semilogx(c, acc_train)
#plt.semilogx(c, acc_test)
#plt.title("SVM with Polynomial kernel, degree 2")
#plt.show()

#

#Gaussian SVM
#acc_train = []
#acc_test = []
#n = np.array(range(20))
#c = 2**n
#for i in range(20):
#	clf = SVC(C=c[i], gamma='auto', kernel='rbf').fit(X_train, y_train)
#	y_train_pred = clf.predict(X_train)
#	y_test_pred = clf.predict(X_test)
#	acc_train.append(accuracy_score(y_train, y_train_pred))
#	acc_test.append(accuracy_score(y_test, y_test_pred))

#plt.figure()
#plt.semilogx(c, acc_train)
#plt.semilogx(c, acc_test)
#plt.title("SVM with Gaussian kernel")
#plt.show()
