import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # Grab wines data set
    df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)

    df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
    'Alcalinity of ash', 'Magnesium', 'Total phenols',
    'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
    'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']

    from sklearn.cross_validation import train_test_split
    X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    from sklearn.preprocessing import StandardScaler
    stdsc = StandardScaler()
    X_train_std = stdsc.fit_transform(X_train)
    X_test_std = stdsc.transform(X_test)

    from sklearn.ensemble import RandomForestClassifier
    feat_labels = df_wine.columns[1:]
    forest = RandomForestClassifier(n_estimators=10000,
                                    random_state=0,
                                    n_jobs=-1)

    forest.fit(X_train, y_train)
    importances = forest.feature_importances_

    indices = np.argsort(importances)[::-1]

    for f in range(X_train.shape[1]):
        print("%2d) %-*s %f" % (f + 1, 30,
                                feat_labels[indices[f]],
                                importances[indices[f]]))

    plt.title('Feature Importances')
    plt.bar(range(X_train.shape[1]),
            importances[indices],
            color='lightblue',
            align='center')

    plt.xticks(range(X_train.shape[1]),
               feat_labels[indices], rotation=90)
    plt.xlim([-1, X_train.shape[1]])
    plt.tight_layout()
    #plt.savefig('./random_forest.png', dpi=300)
    plt.show()
