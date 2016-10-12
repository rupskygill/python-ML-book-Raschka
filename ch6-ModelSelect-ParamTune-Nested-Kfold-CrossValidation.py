from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

import pandas as pd
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', header=None)

from sklearn.preprocessing import LabelEncoder
X = df.loc[:, 2:].values
y = df.loc[:, 1].values
le = LabelEncoder()
y = le.fit_transform(y)

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)


from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
pipe_svc = Pipeline([('scl', StandardScaler()),  ('clf', SVC(random_state=1))])
param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid = [{'clf__C': param_range,
                'clf__kernel': ['linear']},
                {'clf__C': param_range,
                'clf__gamma': param_range,
                'clf__kernel': ['rbf']}]

from sklearn.cross_validation import cross_val_score
# Inner Loop for parameter tuning
gs = GridSearchCV(estimator=pipe_svc,
                param_grid=param_grid,
                scoring='accuracy',
                cv=2,
                n_jobs=-1)

#Outer Loop for model selection
scores = cross_val_score(gs, X_train, y_train, scoring='accuracy', cv=5)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))


from sklearn.tree import DecisionTreeClassifier
gs = GridSearchCV(
        estimator=DecisionTreeClassifier(random_state=0),
        param_grid=[{'max_depth': [1, 2, 3, 4, 5, 6, 7, None]}],
        scoring='accuracy',
        cv=2)

scores = cross_val_score(gs,
            X_train,
            y_train,
            scoring='accuracy',
            cv=5)

print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
