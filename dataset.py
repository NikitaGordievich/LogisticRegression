import numpy as np
import sklearn 

titanic=sklearn.datasets.fetch_openml('titanic', version=1)

for i in titanic.frame.columns.tolist():
    print(i,titanic.frame[i].nunique())

data=titanic['data']
data=data.drop(['body','boat','name'],axis=1)

for i in data:
    print(i)
