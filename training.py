import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

a = np.loadtxt('skin.txt',delimiter=',')
n=0
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

print(test)
print(train)