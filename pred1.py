"""import pandas as pd

data = pd.read_csv('weather.csv')

features = data.columns

x = data[data.columns[:-1]]
y = data[data.columns[-1]]

print(x)
print(y)
#print(data.readline())
#print(data)
"""

import sklearn
import joblib
from sklearn.datasets import load_iris
iris = load_iris()

x = iris.data
y = iris.target

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=1)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(x_train,y_train)

y_pred = knn.predict(x_test)

from sklearn import metrics
print("KNN Model Accuracy: ",metrics.accuracy_score(y_test,y_pred))

sample = [[4,6,3,2],[4,3,2,4]]
preds = knn.predict(sample)
pred_species = [iris.target_names[p] for p in preds]
print("Predictions: ",pred_species)

from sklearn.externals import joblib
joblib.dump(knn, 'iris_knn.pkl')
