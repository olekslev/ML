import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import re

sns.set(context="paper", font="monospace")

#Step1: Load data into dataframe
df_test = pd.read_csv("test.csv")
df_train = pd.read_csv("train.csv")

#Combine two datasets to proceed with preprocessing step
df = df_test.append(df_train, ignore_index=True, verify_integrity=True) # to separete them again use df[df.Survived.isnull()]

#Analyze data
df.head()

#Count nan values
df.count().sort_values()

#Transform categorical field Sex and Embarked
df = pd.concat([df, pd.get_dummies(pd.get_dummies(df[['Sex', 'Embarked']]))], axis=1)
del df['Sex']


#Fare vs Pclass
df.Fare.value_counts().hist()
sns.boxplot(x="Pclass", y="Fare", data=df, whis=np.inf)
sns.boxplot(x="Survived", y="Fare", data=df, whis=np.inf)
# Fill missing Fare with mean from the same Pclass
df.loc[152, 'Fare'] = df.groupby('Pclass').mean()['Fare'][3]
df = pd.concat([df, pd.get_dummies(pd.get_dummies(df['Pclass']))],axis=1)
df.Fare = pd.qcut(df.Fare, 4, labels=[0, 1, 2, 3]).astype(int)

#Age
tmp = df.groupby(['Pclass', 'Embarked'])['Age'].mean()
for i in df.index:
    if pd.np.isnan(df.loc[i, 'Age']):
        df.loc[i, 'Age'] = tmp[df.loc[i, 'Pclass']][df.loc[i, 'Embarked']]

df.Age = pd.qcut(df.Age, 5, labels=[0, 1, 2, 3, 4]).astype(int)
del df['Pclass']
del df['Embarked']

#Family
df['Family'] = df.SibSp + df.Parch
sns.boxplot(y="Family", x="Survived", data=df, whis=np.inf)
#df['Family_NO'] = np.where(df['Family'] == 0, 1, 0)
#df['Family_Small'] = np.where((df['Family'] > 0) & (df['Family'] <= 3), 1, 0)
#df['Family_Big'] = np.where(df['Family'] > 3, 1, 0)

#del df['Family']
del df['Parch']
del df['SibSp']

#Ticket
df['Sub'] = df.Ticket.str.split(pat=' ')
df.Sub = df.Sub.apply(len)
df.Sub = df.Sub.apply(str)
df = pd.concat([df, pd.get_dummies(pd.get_dummies(df['Sub'], prefix='Special'))], axis=1)

del df['Ticket']
del df['Sub']

#Name
df['Lastname'] = df.Name.str.split(pat=',').str.get(0)
df['Surname'] = df.Name.str.split(pat=',').str.get(1)
df['Titel'] = df.Surname.str.split(pat='.').str.get(0)
df.Titel.replace(' Miss', ' Ms', inplace=True)
df.Titel.replace(' Mne', ' Mrs', inplace=True)
df.loc[~df.Titel.isin([' Mr', ' Ms', ' Mrs', ' Master']), 'Titel'] = 'Other'

df = pd.concat([df, pd.get_dummies(pd.get_dummies(df['Titel']))], axis=1)
del df['Name']
del df['Lastname']
del df['Surname']
del df['Titel']

#cabin:
df['Cabin_NO'] = np.where(df['Cabin'].isnull(), 1, 0)
df['Cabin_YES'] = np.where(df['Cabin'].isnull()==False, 1, 0)
del df['Cabin']

#Models
X_train = df[df.Survived.isnull()==False]
Y_train = X_train.Survived

del X_train['Survived']
del X_train['PassengerId']

X_test = df[df.Survived.isnull()]
del X_test['Survived']
ids = pd.DataFrame(X_test['PassengerId'], columns=['PassengerId'])
del X_test['PassengerId']

X_train.shape, X_test.shape, Y_train.shape
df_models = pd.DataFrame(index=[0, 1, 2, 3, 4, 5], columns=['Name', 'Coeff'])

#Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc = round(logreg.score(X_train, Y_train) * 100, 2)
df_models.loc[0, 'Coeff'] = acc
df_models.loc[0, 'Name'] = 'Logistic Regression'

# Support Vector Machines
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc = round(svc.score(X_train, Y_train) * 100, 2)
df_models.loc[1, 'Coeff'] = acc
df_models.loc[1, 'Name'] = 'SVM'

#KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc = round(knn.score(X_train, Y_train) * 100, 2)
df_models.loc[2, 'Coeff'] = acc
df_models.loc[2, 'Name'] = 'KNN'

# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc = round(decision_tree.score(X_train, Y_train) * 100, 2)
df_models.loc[3, 'Coeff'] = acc
df_models.loc[3, 'Name'] = 'Decision Tree'



#export_graphviz(decision_tree, out_file='tree.dot', feature_names=X_train.columns,
#                impurity=False, filled=True, class_names=['Survived', 'Not Survived'])
#with open('tree.dot') as f:
#    dot = f.read()
#graphviz.Source(dot)

#GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

param_grid = [
  {'max_depth': [1, 2, 3, 4, 5, 6], 'learning_rate': [0.01, 0.02, 0.03, 0.05, 0.06]}
 ]

#gb = GradientBoostingClassifier()
#clf = GridSearchCV(gb, param_grid)
clf = GradientBoostingClassifier(learning_rate=0.01)
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)
acc = round(clf.score(X_train, Y_train) * 100, 2)
print (acc)
df_models.loc[4, 'Coeff'] = acc
df_models.loc[4, 'Name'] = ' GradientBoost'




#XGBoost
from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
probab = model.predict_proba(X_test)
acc = round(model.score(X_train, Y_train) * 100, 2)
df_models.ix[2] = ['XGBoost', acc]
df_importance = pd.DataFrame(columns=['Factor', 'Coeff'])
df_importance['Factor'] = X_train.columns
df_importance['Coeff'] = model.feature_importances_
df_importance = df_importance.sort_values('Coeff')

df_models.sort_values(by='Coeff', ascending=False)
results_tmp = pd.DataFrame(Y_pred, index=ids.index, columns=['Survived'])
results_tmp = pd.concat([ids, results_tmp], axis=1)
results_tmp.Survived = results_tmp.Survived.apply(int)
results_tmp.to_csv('gender_submission.csv', index=False)
