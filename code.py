import pandas as pd
import numpy as np
#import holoviews as hv
import seaborn as sns
#hv.extension('bokeh', 'matplotlib')

iris_data = pd.read_csv('assets/iris.csv')

iris_data.columns = ['sepal_length', 'sepal_width' , 'petal_length', 'petal_width', 'species']

#you can specific the number to show here
iris_data.head(10)

#Discovering the Shape of the table
iris_data.shape

#Find out unique classification/type of iris flower and the amount
iris_data['species'].unique()

print(iris_data.groupby('species').size())

#Investigating the data: Min, Max, Mean, Median and Standard Deviation
#Get the minimum value of all the column in python pandas

iris_data.min()

#Get the maximum value of all the column in python pandas

iris_data.max()

#Get the mean value of all the column in python pandas

iris_data.mean()

#Get the median value of all the column in python pandas

iris_data.median()

#Get the standard deviation value of all the column in python pandas

iris_data.std()

#Presenting the Summary Statistics a more readable way
#DataFrame.describe: Generates descriptive statistics that summarize the central tendency, dispersion and shape of a dataset’s distribution, excluding NaN values.

#R has a faster way of getting the data with summary

summary = iris_data.describe()
summary = summary.transpose()
summary.head()

#From the above summary, we can see there is huge range in the size of the Sepal Length and Petal Length. We will use a scatter plot to see if the size is related to the species of Iris.

#Let's investigate the various species and see if there are any obvious differences.

#Boxplot
#The boxplot is a quick way of visually summarizing one or more groups of numerical data through their quartiles. Comparing the distributions of:

import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="whitegrid", palette="GnBu_d", rc={'figure.figsize':(11.7,8.27)})

title="Compare the Distributions of Sepal Length"

sns.boxplot(x="species", y="sepal_length", data=iris_data)

# increasing font size
plt.title(title, fontsize=26)
# Show the plot
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="whitegrid", palette="GnBu_d", rc={'figure.figsize':(11.7,8.27)})

title="Compare the Distributions of Sepal Width"

sns.boxplot(x="species", y="sepal_width", data=iris_data)

# increasing font size
plt.title(title, fontsize=26)
# Show the plot
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="whitegrid", palette="GnBu_d", rc={'figure.figsize':(11.7,8.27)})

title="Compare the Distributions of Petal Length"

sns.boxplot(x="species", y="petal_length", data=iris_data)

# increasing font size
plt.title(title, fontsize=26)
# Show the plot
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="whitegrid", palette="GnBu_d", rc={'figure.figsize':(11.7,8.27)})

title="Compare the distributions of Petal Width"

sns.boxplot(x="species", y="petal_width", data=iris_data)

# increasing font size
plt.title(title, fontsize=26)
# Show the plot
plt.show()

#From the boxplot chart analysis, there are clear differences in the size of the Sepal Length, Petal Length and Petal Width among the different species.

#Comparing the Petal Width and Petal Length across the different Species
#Using different colours it is clear that the three species have very different petal sizes.

from bokeh.plotting import figure, output_file, show
output_file("test1.html")

color1 = '#fcc5c0'
color2 = '#f768a1'
color3 = '#7a0177'
    
#adding colors
colormap = {'Iris-setosa': color1, 'Iris-versicolor': color2, 'Iris-virginica': color3}
colors = [colormap[x] for x in iris_data['species']]

#adding labels
p = figure(title = "Petal Width and Petal Length")
p.xaxis.axis_label = 'Petal Length'
p.yaxis.axis_label = 'Petal Width'
p.legend.location = "top_left"

p.diamond(iris_data["petal_length"], iris_data["petal_width"],color=colors, fill_alpha=0.2, size=10)


show(p)
#Comparing the Petal Width and Sepal Length across the different Species
from bokeh.plotting import figure, output_file, show
output_file("test2.html")

#adding colors
colormap = {'Iris-setosa': color1, 'Iris-versicolor': color2, 'Iris-virginica': color3}
colors = [colormap[x] for x in iris_data['species']]

#adding labels
p = figure(title = "Petal Width and Sepal Length")
p.xaxis.axis_label = 'Sepal Length'
p.yaxis.axis_label = 'Petal Width'


p.circle(iris_data["sepal_length"], iris_data["petal_width"],
         color=colors, fill_alpha=0.2, size=10)


show(p)
#Looking for relationships between variables across multiple dimensions

# My favourite colors = palettes GnBu_d
import seaborn as sns
# Scatter plots for the features and histograms
# custom markers also applied
sns.pairplot(iris_data, hue="species", palette="GnBu_d", markers=["o", "s", "D"])

#Remove the top and right spines from plot
sns.despine()

#show plot
import matplotlib.pyplot as plt
plt.show()

#setting the background color
import seaborn as sns
sns.set(style="whitegrid")

sns.pairplot(iris_data, hue="species", palette="GnBu_d", diag_kind="kde", markers=["o", "s", "D"])

#Remove the top and right spines from plot
sns.despine()

import matplotlib.pyplot as plt
plt.show()

# plotting regression and confidence intervals
import seaborn as sns
sns.pairplot(iris_data,hue="species", kind='reg', palette="GnBu_d")

#Remove the top and right spines from plot
sns.despine()

#show plot
import matplotlib.pyplot as plt
plt.show()

#strong relationship between petal length and petal width and petal length and sepal length

import seaborn as sns
#setting the background color and size of graph
sns.set(style="whitegrid", palette="GnBu_d", rc={'figure.figsize':(11.7,8.27)})

# "Melt" the dataset
iris2 = pd.melt(iris_data, "species", var_name="measurement")

# Draw a categorical scatterplot
sns.swarmplot(x="measurement", y="value", hue="species",palette="GnBu_d", data=iris2)

#Remove the top and right spines from plot
sns.despine()

#show plot
import matplotlib.pyplot as plt
plt.show()

import seaborn as sns
#setting the background color and size of graph
sns.set(style="whitegrid", palette="GnBu_d", rc={'figure.figsize':(11.7,8.27)})


sns.violinplot(x="species", y="petal_length", palette="GnBu_d", data=iris_data)

#Remove the top and right spines from plot
sns.despine()

#show plot
import matplotlib.pyplot as plt
plt.show()

import seaborn as sns
#setting the background color and size of graph
sns.set(style="whitegrid", palette="GnBu_d", rc={'figure.figsize':(11.7,8.27)})

sns.violinplot(x="species", y="petal_width", palette="GnBu_d", data=iris_data)
#Remove the top and right spines from plot
sns.despine()


#show plot
import matplotlib.pyplot as plt
plt.show()

import matplotlib.pyplot as plt
iris_data.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()


# import load_iris function from datasets module
from sklearn.datasets import load_iris
In [110]:
# save "bunch" object containing iris dataset and its attributes
iris = load_iris()
type(iris)
Out[110]:
sklearn.utils.Bunch
In [111]:
# print the iris data
print(iris.data)
[[5.1 3.5 1.4 0.2]
 [4.9 3.  1.4 0.2]
 [4.7 3.2 1.3 0.2]
 [4.6 3.1 1.5 0.2]
 [5.  3.6 1.4 0.2]
 [5.4 3.9 1.7 0.4]
 [4.6 3.4 1.4 0.3]
 [5.  3.4 1.5 0.2]
 [4.4 2.9 1.4 0.2]
 [4.9 3.1 1.5 0.1]
 [5.4 3.7 1.5 0.2]
 [4.8 3.4 1.6 0.2]
 [4.8 3.  1.4 0.1]
 [4.3 3.  1.1 0.1]
 [5.8 4.  1.2 0.2]
 [5.7 4.4 1.5 0.4]
 [5.4 3.9 1.3 0.4]
 [5.1 3.5 1.4 0.3]
 [5.7 3.8 1.7 0.3]
 [5.1 3.8 1.5 0.3]
 [5.4 3.4 1.7 0.2]
 [5.1 3.7 1.5 0.4]
 [4.6 3.6 1.  0.2]
 [5.1 3.3 1.7 0.5]
 [4.8 3.4 1.9 0.2]
 [5.  3.  1.6 0.2]
 [5.  3.4 1.6 0.4]
 [5.2 3.5 1.5 0.2]
 [5.2 3.4 1.4 0.2]
 [4.7 3.2 1.6 0.2]
 [4.8 3.1 1.6 0.2]
 [5.4 3.4 1.5 0.4]
 [5.2 4.1 1.5 0.1]
 [5.5 4.2 1.4 0.2]
 [4.9 3.1 1.5 0.1]
 [5.  3.2 1.2 0.2]
 [5.5 3.5 1.3 0.2]
 [4.9 3.1 1.5 0.1]
 [4.4 3.  1.3 0.2]
 [5.1 3.4 1.5 0.2]
 [5.  3.5 1.3 0.3]
 [4.5 2.3 1.3 0.3]
 [4.4 3.2 1.3 0.2]
 [5.  3.5 1.6 0.6]
 [5.1 3.8 1.9 0.4]
 [4.8 3.  1.4 0.3]
 [5.1 3.8 1.6 0.2]
 [4.6 3.2 1.4 0.2]
 [5.3 3.7 1.5 0.2]
 [5.  3.3 1.4 0.2]
 [7.  3.2 4.7 1.4]
 [6.4 3.2 4.5 1.5]
 [6.9 3.1 4.9 1.5]
 [5.5 2.3 4.  1.3]
 [6.5 2.8 4.6 1.5]
 [5.7 2.8 4.5 1.3]
 [6.3 3.3 4.7 1.6]
 [4.9 2.4 3.3 1. ]
 [6.6 2.9 4.6 1.3]
 [5.2 2.7 3.9 1.4]
 [5.  2.  3.5 1. ]
 [5.9 3.  4.2 1.5]
 [6.  2.2 4.  1. ]
 [6.1 2.9 4.7 1.4]
 [5.6 2.9 3.6 1.3]
 [6.7 3.1 4.4 1.4]
 [5.6 3.  4.5 1.5]
 [5.8 2.7 4.1 1. ]
 [6.2 2.2 4.5 1.5]
 [5.6 2.5 3.9 1.1]
 [5.9 3.2 4.8 1.8]
 [6.1 2.8 4.  1.3]
 [6.3 2.5 4.9 1.5]
 [6.1 2.8 4.7 1.2]
 [6.4 2.9 4.3 1.3]
 [6.6 3.  4.4 1.4]
 [6.8 2.8 4.8 1.4]
 [6.7 3.  5.  1.7]
 [6.  2.9 4.5 1.5]
 [5.7 2.6 3.5 1. ]
 [5.5 2.4 3.8 1.1]
 [5.5 2.4 3.7 1. ]
 [5.8 2.7 3.9 1.2]
 [6.  2.7 5.1 1.6]
 [5.4 3.  4.5 1.5]
 [6.  3.4 4.5 1.6]
 [6.7 3.1 4.7 1.5]
 [6.3 2.3 4.4 1.3]
 [5.6 3.  4.1 1.3]
 [5.5 2.5 4.  1.3]
 [5.5 2.6 4.4 1.2]
 [6.1 3.  4.6 1.4]
 [5.8 2.6 4.  1.2]
 [5.  2.3 3.3 1. ]
 [5.6 2.7 4.2 1.3]
 [5.7 3.  4.2 1.2]
 [5.7 2.9 4.2 1.3]
 [6.2 2.9 4.3 1.3]
 [5.1 2.5 3.  1.1]
 [5.7 2.8 4.1 1.3]
 [6.3 3.3 6.  2.5]
 [5.8 2.7 5.1 1.9]
 [7.1 3.  5.9 2.1]
 [6.3 2.9 5.6 1.8]
 [6.5 3.  5.8 2.2]
 [7.6 3.  6.6 2.1]
 [4.9 2.5 4.5 1.7]
 [7.3 2.9 6.3 1.8]
 [6.7 2.5 5.8 1.8]
 [7.2 3.6 6.1 2.5]
 [6.5 3.2 5.1 2. ]
 [6.4 2.7 5.3 1.9]
 [6.8 3.  5.5 2.1]
 [5.7 2.5 5.  2. ]
 [5.8 2.8 5.1 2.4]
 [6.4 3.2 5.3 2.3]
 [6.5 3.  5.5 1.8]
 [7.7 3.8 6.7 2.2]
 [7.7 2.6 6.9 2.3]
 [6.  2.2 5.  1.5]
 [6.9 3.2 5.7 2.3]
 [5.6 2.8 4.9 2. ]
 [7.7 2.8 6.7 2. ]
 [6.3 2.7 4.9 1.8]
 [6.7 3.3 5.7 2.1]
 [7.2 3.2 6.  1.8]
 [6.2 2.8 4.8 1.8]
 [6.1 3.  4.9 1.8]
 [6.4 2.8 5.6 2.1]
 [7.2 3.  5.8 1.6]
 [7.4 2.8 6.1 1.9]
 [7.9 3.8 6.4 2. ]
 [6.4 2.8 5.6 2.2]
 [6.3 2.8 5.1 1.5]
 [6.1 2.6 5.6 1.4]
 [7.7 3.  6.1 2.3]
 [6.3 3.4 5.6 2.4]
 [6.4 3.1 5.5 1.8]
 [6.  3.  4.8 1.8]
 [6.9 3.1 5.4 2.1]
 [6.7 3.1 5.6 2.4]
 [6.9 3.1 5.1 2.3]
 [5.8 2.7 5.1 1.9]
 [6.8 3.2 5.9 2.3]
 [6.7 3.3 5.7 2.5]
 [6.7 3.  5.2 2.3]
 [6.3 2.5 5.  1.9]
 [6.5 3.  5.2 2. ]
 [6.2 3.4 5.4 2.3]
 [5.9 3.  5.1 1.8]]

# print integers representing the species of each observation
print(iris.target)

#print the names of the targets
print(iris.target_names)

# print the names of the four features
print(iris.feature_names)
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2]
['setosa' 'versicolor' 'virginica']
['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

from sklearn.datasets import load_iris
iris = load_iris()

from matplotlib import pyplot as plt

# The indices of the features that we are plotting
x_index = 0
y_index = 1

# this formatter will label the colorbar with the correct target names
formatter = plt.FuncFormatter(lambda i, *args: iris.target_names[int(i)])

plt.figure(figsize=(5, 4))
plt.scatter(iris.data[:, x_index], iris.data[:, y_index], c=iris.target)
plt.colorbar(ticks=[0, 1, 2], format=formatter)
plt.xlabel(iris.feature_names[x_index])
plt.ylabel(iris.feature_names[y_index])

plt.tight_layout()
plt.show()

# print the names of the four features
print(iris.target.shape)
(150,)

# store feature matrix in "X"
x = iris.data

# store response vector in "y"
y = iris.target

print(x.shape)

print(y.shape)
print(np.unique(y))
(150, 4)
(150,)
[0 1 2]
#Using the powerful classification algorithm called K-Nearest-Neighbors (KNN). KNN falls in the supervised learning family of algorithms.

# loading library
from sklearn.neighbors import KNeighborsClassifier

# instantiate learning model (k = 1)
knn = KNeighborsClassifier(n_neighbors=1)

print(knn)
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=1, p=2,
           weights='uniform')

# fitting the model
knn.fit(x, y)
Out[117]:
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=1, p=2,
           weights='uniform')

# What kind of iris has 3cm x 5cm sepal and 4cm x 2cm petal?

result = knn.predict([[3, 5, 4, 2]])

print(iris.target_names[result])
['virginica']
In [119]:
X_new = [[3, 5, 4, 2], [5, 4, 3, 2]]

# predict the response for new
knn.predict(X_new)
Out[119]:
array([2, 1])
In [88]:
import numpy as np
from matplotlib import pyplot as plt
from sklearn import neighbors, datasets
from matplotlib.colors import ListedColormap

# Create color maps for 3-class classification problem, as with iris
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features. We could
                    # avoid this ugly slicing by using a two-dim dataset
y = iris.target

knn = neighbors.KNeighborsClassifier(n_neighbors=1)
knn.fit(X, y)

x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.axis('tight')
plt.show()
