import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

pd.set_option('expand_frame_repr', False)

df = pd.read_csv('./data/raw/train.csv', header=0)

df['Sex'] = df['Sex'].astype('category')
df['CabinCat'] = df['Cabin'].fillna('').apply(lambda cabin: '?' if not cabin else cabin[0:1])

df[0:10]

# age distribution of victims and survivors
ax = sns.distplot(df[df['Survived']==1]['Age'].dropna(), label='Survivor', color='g', hist=False);
sns.distplot(df[df['Survived']==0]['Age'].dropna(), label='Victim', ax=ax, color='r', hist=False);
plt.savefig('./reports/figures/age_survived_dist.png')
plt.show()

# age distribution of women and men
ax = sns.distplot(df[df['Sex']=='male']['Age'].dropna(), label='Males', color='b', hist=False);
sns.distplot(df[df['Sex']=='female']['Age'].dropna(), label='Females', ax=ax, color='m', hist=False);
plt.savefig('./reports/figures/age_sex_dist.png')
plt.show()

# fare distribution of victims and survivors
ax = sns.distplot(df[df['Survived']==1]['Fare'].dropna(), label='Survivor', color='g', hist=False);
sns.distplot(df[df['Survived']==0]['Fare'].dropna(), label='Victim', ax=ax, color='r', hist=False);
plt.savefig('./reports/figures/fare_survived_dist.png')
plt.show()

# survival rate and socio-economic class / sex
sns.countplot(y="Pclass", data=df, color="c", hue="Sex")
plt.show()
sns.barplot(x='Sex', y='Survived', hue='Pclass', data=df)
plt.savefig('./reports/figures/SEC_sex_count.png')
plt.show()

# survival rate and number of parents / sex
sns.boxplot(x='Parch', y='Age', hue='Sex', data=df[df['Survived']==1])
plt.savefig('./reports/figures/parents_age_survived.png')
plt.show()

# survival rate and number of siblings / sex
sns.boxplot(x='SibSp', y='Age', hue='Sex', data=df[df['Survived']==1])
plt.savefig('./reports/figures/siblings_age_survived.png')
plt.show()

# survival rate by embarcation port (C = Cherbourg; Q = Queenstown; S = Southampton)
sns.countplot(y='Embarked', hue='Survived', data=df)
plt.savefig('./reports/figures/embarcation_survived_count.png')
plt.show()

sns.countplot(y='Embarked', hue='Pclass', data=df)
plt.savefig('./reports/figures/embarcation_SEC_count.png')
plt.show()

# survival rate by cabin category
sns.countplot(y='CabinCat', hue='Survived', data=df)
plt.savefig('./reports/figures/cabin_survived_count.png')
plt.show()

sns.countplot(y='CabinCat', hue='Pclass', data=df)
plt.savefig('./reports/figures/cabin_SEC_count.png')
plt.show()

# extract titles and common names
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['Name'])

words = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names())
names = words.drop(['mr', 'mrs', 'miss'], axis=1).sum(axis=0).sort_values(ascending=False)[0:20].index.tolist()

df['title'] = words[['miss', 'mr', 'mrs']].idxmax(axis=1)
df['popularName'] = words[names].idxmax(axis=1)
    
# survival rate for mrs and miss
sns.countplot(y='title', hue='Survived', data=df[df['Sex']=='female'])
plt.show()

# survival rate for popular names
sns.countplot(y='popularName', hue='Survived', data=df)
plt.savefig('./reports/figures/names_count.png')
plt.show()