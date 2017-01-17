import pandas as pd
import numpy as np

from sklearn.cross_validation import cross_val_score

train_df = pd.read_csv('./data/processed/train.csv')
test_df = pd.read_csv('./data/processed/test.csv')

X_train = train_df.drop(['Survived', 'PassengerId'], axis=1).values
y_train = train_df['Survived']


from sklearn.ensemble import RandomForestClassifier
clf1 = RandomForestClassifier(n_estimators=300)
scores = cross_val_score(clf1, X_train, y_train, cv=5)
print scores

from sklearn.neighbors import KNeighborsClassifier
clf2 = KNeighborsClassifier(n_neighbors=5)
scores = cross_val_score(clf2, X_train, y_train, cv=5)
print scores

from sklearn.svm import SVC
clf3 = SVC(kernel='rbf', probability=True)
scores = cross_val_score(clf3, X_train, y_train, cv=5)
print scores

from sklearn.ensemble import VotingClassifier
clf = VotingClassifier(estimators=[('rf', clf1), ('knn', clf2), ('svc', clf3)], voting='soft', weights=[2,1,1])
clf = VotingClassifier(estimators=[('rf', clf1), ('knn', clf2), ('svc', clf3)], voting='hard')
scores = cross_val_score(clf, X_train, y_train, cv=5)
print scores


forest = RandomForestClassifier(n_estimators=300)
forest.fit(X_train, y_train)

X_test = test_df.drop(['PassengerId'], axis=1).values
preds = forest.predict(X_test)

test_df['Survived'] = preds

prediction_df = test_df[['PassengerId', 'Survived']]
prediction_df = prediction_df.astype('int')
prediction_df.to_csv('./models/predictions.csv', index=False)
