import pandas as pd
import numpy as np
import csv as csv
import pickle

from collections import defaultdict, Counter

def select_dummy_values(dataset, features, limit=100):
  dummy_values = {}
  for feature in features:
      values = [
          value
          for (value, _) in Counter(dataset[feature]).most_common(limit)
      ]
      dummy_values[feature] = values

  return dummy_values

def dummy_encode_dataframe(dataset, dummies):
  for (feature, dummy_values) in dummies.items():
      for dummy_value in dummy_values:
          dummy_name = u'%s_value_%s' % (feature, unicode(dummy_value))
          dataset[dummy_name] = (dataset[feature] == dummy_value).astype('double')
      del dataset[feature]

  return dataset

def load_data(file, dummies=None):

  df = pd.read_csv(file, header=0)

  from sklearn.feature_extraction.text import CountVectorizer
  vectorizer = CountVectorizer()
  X = vectorizer.fit_transform(df['Name'])

  words = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names())
  names = words.drop(['mr', 'mrs', 'miss', 'master'], axis=1).sum(axis=0).sort_values(ascending=False)[0:10].index.tolist()

  df['TitleMiss'] = words['miss']
  df['TitleMaster'] = words['master']
  df['PopularName'] = words[names].idxmax(axis=1)

  df['Gender'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

  df['Embarked'] = df['Embarked'].fillna('?')

  median_age = df['Age'].median()
  df['Age'] = df['Age'].fillna(median_age)

  median_fare = df['Fare'].median()
  df['Fare'] = df['Fare'].fillna(median_fare)

  df['CabinCat'] = df['Cabin'].fillna('').apply(lambda cabin: '?' if not cabin else cabin[0:1])

  if not dummies:
    dummies = select_dummy_values(df, ['Embarked', 'CabinCat', 'PopularName'])

  dummy_encode_dataframe(df, dummies)

  df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin'], axis=1)

  df = df.astype('float32')

  return df, dummies

train_df, dummies = load_data('./data/raw/train.csv')
test_df, dummies = load_data('./data/raw/test.csv', dummies)

f = open('./data/processed/dummies.pkl', 'w')
pickle.dump(dummies, f)
f.close()

train_df.to_csv('./data/processed/train.csv', index=False)
test_df.to_csv('./data/processed/test.csv', index=False)
