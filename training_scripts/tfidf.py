from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd

data_train = pd.read_csv('train_trans_lem.csv')
data_test = pd.read_csv('val_trans_lem.csv')
X_train = data_train.text
X_test = data_test.text
y_train = data_train.target
y_test = data_test.target

text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', RandomForestClassifier(n_estimators=200, verbose=5, n_jobs=-1)),
                     ])

text_clf.fit(X_train, y_train)

predicted = text_clf.predict(X_test)
predicted_proba = text_clf.predict_proba(X_test)

print(metrics.classification_report(y_test, predicted, digits=5))

n_objs = len(y_test)
classes = set(predicted)
mapping = {k: v for k, v in zip(classes, range(len(classes)))}
encoded = [mapping[t] for t in predicted]
pd.DataFrame({'text': X_test, 'target': y_test, 'predict': predicted,
              'proba': predicted_proba.max(axis=1)}).to_csv('rf_predicts.csv')