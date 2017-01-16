import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

pd.set_option('expand_frame_repr', False)

df = pd.read_csv('./data/raw/train.csv', header=0)

df['Sex'] = df['Sex'].astype('category')
df['CabinCat'] = df['Cabin'].fillna('').apply(c)

df[0:10]

# age distribution of victims and survivors
ax = sns.distplot(df[df['Survived']==1]['Age'].dropna(), label='Survivor', color='g', hist=False);
sns.distplot(df[df['Survived']==0]['Age'].dropna(), label='Victim', ax=ax, color='r', hist=False);
plt.show()

# age distribution of women and men
ax = sns.distplot(df[df['Sex']=='male']['Age'].dropna(), label='Males', color='b', hist=False);
sns.distplot(df[df['Sex']=='female']['Age'].dropna(), label='Females', ax=ax, color='m', hist=False);
plt.show()

# fare distribution of victims and survivors
ax = sns.distplot(df[df['Survived']==1]['Fare'].dropna(), label='Survivor', color='g', hist=False);
sns.distplot(df[df['Survived']==0]['Fare'].dropna(), label='Victim', ax=ax, color='r', hist=False);
plt.show()

# survival rate and socio-economic class / sex
sns.countplot(y="Pclass", data=df, color="c", hue="Sex")
plt.show()
sns.barplot(x='Sex', y='Survived', hue='Pclass', data=df)
plt.show()

# survival rate and number of parents / sex
sns.boxplot(x='Parch', y='Age', hue='Sex', data=df[df['Survived']==1])
plt.show()

# survival rate and number of siblings / sex
sns.boxplot(x='SibSp', y='Age', hue='Sex', data=df[df['Survived']==1])
plt.show()

# survival rate by embarcation port (C = Cherbourg; Q = Queenstown; S = Southampton)
sns.countplot(y='Embarked', hue='Survived', data=df)
plt.show()

sns.countplot(y='Embarked', hue='Pclass', data=df)
plt.show()

# survival rate by cabin category
sns.countplot(y='CabinCat', hue='Survived', data=df)
plt.show()

sns.countplot(y='CabinCat', hue='Pclass', data=df)
plt.show()

# extract titles and common names
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['Name'])

words = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names())
words.sum(axis=0).sort_values(ascending=False)[0:10]
