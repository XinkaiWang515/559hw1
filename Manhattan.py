import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import neighbors
from sklearn.metrics import accuracy_score

a = np.loadtxt('skin.txt',delimiter=',')
n=0
tra_errors = []
te_errors = []
tra_ks=[]
p_tra_errors = []
p_te_errors = []

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

train1 = pd.DataFrame({
    'variance': a[n+200:length,0],
    'skewness': a[n+200:length,1],
    'curtosis': a[n+200:length,2],
    'entropy': a[n+200:length,3],
    'class_': a[n+200:length,4],
})

train = train0.merge(train1, how='outer')

tra_X = train[['variance', 'skewness', 'curtosis', 'entropy']]
tra_Y = train[['class_']]
te_X = test[['variance', 'skewness', 'curtosis', 'entropy']]
te_Y = test[['class_']]
cov = np.cov(np.transpose(tra_X))
i_cov = np.linalg.pinv(cov)
list_k = list(range(1,902,10))
for k in list_k:
    # knn = neighbors.KNeighborsClassifier(n_neighbors=k, metric='manhattan', weights='distance')
    # knn = neighbors.KNeighborsClassifier(n_neighbors=k, metric='chebyshev', weights='distance')
    knn = neighbors.KNeighborsClassifier(n_neighbors=k, metric='mahalanobis', metric_params={'V': cov, 'VI': i_cov})
    knn.fit(tra_X, tra_Y.values.ravel())
    pred_tra_Y = knn.predict(tra_X)
    tra_err = 1-accuracy_score(tra_Y, pred_tra_Y)
    pred_te_Y = knn.predict(te_X)
    te_err = 1-accuracy_score(te_Y, pred_te_Y)
    tra_ks.append(1/k)
    tra_errors.append(tra_err)
    te_errors.append(te_err)


print(tra_errors)
print(te_errors)
plt.plot(tra_ks, tra_errors, 'r', label='training error')
plt.plot(tra_ks, te_errors, 'b', label='test error')
plt.xlabel('1/k')
plt.ylabel('error')
plt.legend()
plt.show()

# m = np.arange(0.1,1.1,0.1)
# p_list = 10**m
# for p in p_list:
#     knn = neighbors.KNeighborsClassifier(n_neighbors=2, metric='manhattan', p=p)
#     knn.fit(tra_X, tra_Y.values.ravel())
#     p_Y = knn.predict(tra_X)
#     p_tra_err = 1-accuracy_score(tra_Y, p_Y)
#     p_te_Y = knn.predict(te_X)
#     p_te_err = 1-accuracy_score(te_Y, p_te_Y)
#     p_tra_errors.append(p_tra_err)
#     p_te_errors.append(p_te_err)
#
# plt.figure()
# plt.plot(p_list, p_tra_errors, 'r', label='training error')
# plt.plot(p_list, p_te_errors, 'b', label='test error')
# plt.xlabel('value of p')
# plt.ylabel('error')
# plt.legend()
# plt.show()