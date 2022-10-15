import pandas as pd
import seaborn as sns

data=sns.load_dataset('iris')
data.head()

x=data.loc[:,data.columns!='species']
y=data['species'].map({'setosa':1, 'versicolor' : 2 ,'virginica':3 })

from sklearn.ensemble import RandomForestClassifier
model= RandomForestClassifier(n_estimators=100, criterion='gini')
model.fit(x,y)

import joblib
joblib.dump(model, 'model.pkl')
