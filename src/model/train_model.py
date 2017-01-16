import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier

train_df = pd.read_csv('./data/processed/train.csv')
test_df = pd.read_csv('./data/processed/test.csv')

forest = RandomForestClassifier(n_estimators=100)

X_train = train_df.drop(['Survived', 'PassengerId'], axis=1).values
y_train = train_df['Survived']
forest.fit(X_train, y_train)

X_test = test_df.drop(['PassengerId'], axis=1).values
preds = forest.predict(X_test)

test_df['Survived'] = preds

prediction_df = test_df[['PassengerId', 'Survived']]
prediction_df = prediction_df.astype('int')
prediction_df.to_csv('./models/predictions.csv', index=False)
