from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np
import pandas as pd

def classify(X_train, y_train, X_test, y_test):
    clf = LogisticRegression(C=2)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print('{:<10} {}'.format('Accuracy:', metrics.accuracy_score(y_test, y_pred)))
    print('{:<10} {}'.format('Precision:', metrics.precision_score(y_test, y_pred)))
    print('{:<10} {}'.format('Recall:', metrics.recall_score(y_test, y_pred)))
    print('{:<10} {}'.format('F-1:', metrics.fbeta_score(y_test, y_pred, beta=1)))
    tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_pred).ravel()
    d = {'Predicted 0' : pd.Series([tn, fn], index=['Actual 0', 'Actual 1']),
         'Predicted 1' : pd.Series([fp, tp], index=['Actual 0', 'Actual 1'])}
    print('Confusion Matrix:\n', pd.DataFrame(d))
