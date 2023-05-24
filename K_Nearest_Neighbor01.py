# Implimentation of KNN
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report 

data = pd.read_csv('C:/Users/tiwar/OneDrive/Desktop/Spyder_projects/Machine_Learning/K_Nearest_Neighbour/diabetes.csv')

print(data.head())

print(data.info())

print(data.isnull().sum())

print(data.columns)

X = data.drop(['Outcome'],axis=1)
y = data['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size= 0.25)

scaler = StandardScaler()

scaler.fit_transform(X_train)
scaler.transform(X_test)

clf = KNeighborsClassifier(metric='euclidean')

parameters = {'n_neighbors':[2,4,6,8,10,12,14,16,18,20], 'p':[2,4,6]}
clfcv = GridSearchCV(clf, parameters, scoring='accuracy', cv=5)

clfcv.fit(X_train, y_train)

print(clfcv.best_params_)
print(clfcv.best_score_)

y_pred = clfcv.predict(X_test)
print('Accuracy Score:',accuracy_score(y_test, y_pred))
print('Classification Report:\n',classification_report(y_test, y_pred))
