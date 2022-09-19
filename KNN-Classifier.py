from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

# Laoding Dataset
iris = datasets.load_iris()

# print(iris.DESCR)
features = iris.data
labels = iris.target
# print(features[10],labels[26])

# Training the classifier
clf = KNeighborsClassifier()
clf.fit(features,labels)

preds = clf.predict([[1.3,8.6,8.7,7.6]])
if(preds==0):
  print("Setosa")
if(preds==1):
  print("Versicolor")
if(preds==2):
  print("Verginica")
