import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import neighbors
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import learning_curve

a = np.loadtxt('skin.txt', delimiter=',')
n=0
d_tra_errors = []
d_te_errors = []

length = a.shape[0]
for x in range(0,length):
    if a[x,4]==0:
        n=n+1

test0 = pd.DataFrame({
    'variance': a[0:200,0],
    'skewness': a[0:200,1],
    'curtosis': a[0:200,2],
    'entropy': a[0:200,3],
    'class_': a[0:200,4],
})

test1 = pd.DataFrame({
    'variance': a[n:n+200,0],
    'skewness': a[n:n+200,1],
    'curtosis': a[n:n+200,2],
    'entropy': a[n:n+200,3],
    'class_': a[n:n+200,4],
})

test = test0.merge(test1, how='outer')

train0 = pd.DataFrame({
    'variance': a[200:n,0],
    'skewness': a[200:n,1],
    'curtosis': a[200:n,2],
    'entropy': a[200:n,3],
    'class_': a[200:n,4],
})
# tra0_X = train0[['variance', 'skewness', 'curtosis', 'entropy']]
# tra0_Y = train0[['class_']]

train1 = pd.DataFrame({
    'variance': a[n+200:length,0],
    'skewness': a[n+200:length,1],
    'curtosis': a[n+200:length,2],
    'entropy': a[n+200:length,3],
    'class_': a[n+200:length,4],
})
# tra1_X = train1[['variance', 'skewness', 'curtosis', 'entropy']]
# tra1_Y = train1[['class_']]

train = train0.merge(train1, how='outer')

tra_X = train[['variance', 'skewness', 'curtosis', 'entropy']]
tra_Y = train[['class_']]
te_X = test[['variance', 'skewness', 'curtosis', 'entropy']]
te_Y = test[['class_']]

# knn = neighbors.KNeighborsClassifier(n_neighbors=20)
# knn.fit(tra_X, tra_Y.values.ravel())
# pre_Y = knn.predict(te_X)
# print(confusion_matrix(te_Y, pre_Y))

# N_list = list(range(50,901,50))
# for N in N_list:
N = 800
d_train0 = train0.loc[0:N/2]
d_train = d_train0.merge(train1.loc[0:N/2], how='outer')
d_tra_X = d_train[['variance', 'skewness', 'curtosis', 'entropy']]
d_tra_Y = d_train[['class_']]
k_list = list(range(1,N,40))
for d_k in k_list:
    knn = neighbors.KNeighborsClassifier(n_neighbors=d_k)
    # print(len(d_tra_Y.values.ravel()))
    d_tra_err = knn.fit(d_tra_X, d_tra_Y.values.ravel()).score(d_tra_X, d_tra_Y.values.ravel())
    knn.fit(d_tra_X, d_tra_Y.values.ravel())
    pred_Y = knn.predict(te_X)
    d_te_err = accuracy_score(te_Y, pred_Y)
    d_tra_errors.append(d_tra_err)
    d_te_errors.append(d_te_err)
# plt.plot(k_list, d_tra_errors, 'r', label='training error')
# plt.plot(k_list, d_te_errors, 'b', label='test error')

# plt.xlabel('value of k')
# plt.ylabel('accuracy')
# plt.legend()
# plt.show()


N_list = list(range(50,901,50))
k_values = list(1 for i in range(1,19))
plt.plot(N_list, k_values)
plt.xlabel('value of k')
plt.ylabel('value of N')
plt.show()