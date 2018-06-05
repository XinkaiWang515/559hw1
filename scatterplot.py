import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

a=np.loadtxt('skin.txt',delimiter=',')
ds = pd.DataFrame({
    'variance': a[:,0],
    'skewness': a[:,1],
    'curtosis': a[:,2],
    'entropy': a[:,3],
    'class_': a[:,4],
})

# fig = sns.FacetGrid(data=ds, hue='class_')
# fig.map(plt.scatter, 'variance', 'skewness').add_legend()
# plt.show()
#
# fig1 = sns.FacetGrid(data=ds, hue='class_')
# fig1.map(plt.scatter, 'variance', 'curtosis').add_legend()
# plt.show()
#
# fig2 = sns.FacetGrid(data=ds, hue='class_')
# fig2.map(plt.scatter, 'variance', 'entropy').add_legend()
# plt.show()
#
# fig3 = sns.FacetGrid(data=ds, hue='class_')
# fig3.map(plt.scatter, 'skewness', 'curtosis').add_legend()
# plt.show()
#
# fig4 = sns.FacetGrid(data=ds, hue='class_')
# fig4.map(plt.scatter, 'skewness', 'entropy').add_legend()
# plt.show()
#
# fig5 = sns.FacetGrid(data=ds, hue='class_')
# fig5.map(plt.scatter, 'curtosis', 'entropy').add_legend()
# plt.show()

plt.figure()
with sns.color_palette("husl", 2):
    sns.boxplot(data=ds, hue='class_', x='class_', y='variance')
plt.show()

plt.figure()
with sns.color_palette("husl", 3):
    sns.boxplot(data=ds, hue='class_', x='class_', y='skewness')
plt.show()

plt.figure()
with sns.color_palette("husl", 3):
    sns.boxplot(data=ds, hue='class_', x='class_', y='curtosis')
plt.show()

plt.figure()
with sns.color_palette("husl", 3):
    sns.boxplot(data=ds, hue='class_', x='class_', y='entropy')
plt.show()