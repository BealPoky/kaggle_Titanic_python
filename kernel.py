# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
#Question and problem definition:





# Step1:Aquire Data
train_df = pd.read_csv('/Users/wzj/Documents/Study/Data Ming Project/kaggle_Titanic Problem/Data/train.csv')
test_df = pd.read_csv('/Users/wzj/Documents/Study/Data Ming Project/kaggle_Titanic Problem/Data/test.csv')
combine = [train_df, test_df]

# Step2:Get some basic info about Features (Objective)
print(train_df.columns.values)
print(train_df.head())

'''
  Analysis:
  We get the whole features of the dataset:
  1:total feature number : 12
  2:we classify these features by:
  (1)categorical(be equivalent to discrete numerical type) :  Survived , Pclass , Sex , Embarked
  
  (2)numerical (Continous of course, otherwise belong to categorical) : PassengerId , Age , SibSp , Parch , Fare  
  
  (Since the boarder of the features above are not definitely clear , so we must confirm it by statistic data)
  
  (3)mix data type(String & numerical) : Ticket , Cabin
  
  (4)may contain error or typos (Usually should be drooped first) : Name
'''
train_df.info()
print('_'*40)
test_df.info()
'''
  Analysis:
  1: we know that the NULL value number the Features held :
     Cabin(77.1%) > Age(19.9%) > Embarked(<1%)
'''
print(train_df.describe())
'''
  Analysis:
1:Total samples are 891 or 40% of the actual number of passengers on board the Titanic (2,224).notice:2224 is given in the problem description.
2:Survived is a categorical feature with 0 or 1 values.
3:Around 38% samples survived representative of the actual survival rate at 32%.
4:Most passengers (> 75%) did not travel with parents or children.
5:Nearly 30% of the passengers had siblings and/or spouse aboard.
6:Fares varied significantly with few passengers (<1%) paying as high as $512.
7:Few elderly passengers (<1%) within age range 65-80.
'''

print(train_df.describe(include=['O']))
'''
  Analysis:
1:Names are unique across the dataset (count=unique=891)
2:Sex variable as two possible values with 65% male (top=male, freq=577/count=891).
3;Cabin values have several dupicates across samples. Alternatively several passengers shared a cabin.
4;Embarked takes three possible values. S port used by most passengers (top=S)
5:Ticket feature has high ratio (22%) of duplicate values (unique=681).
'''

#Step3: put some Assumtions based on the analysis (Subjective)

'''
Note :(1) 7c principle
'''
'''
Assumtions:
1 Correlating:(1)(major) Goal Solution : Feature Survived ; So, any other features correlating to Survived should take 
                                         into account.
                 Assumtion1:=#1  Pclass , Sex , Age , SibSp , Parch  (level 1 : most likely)
                             #2  Embarked , Fare , Cabin , Ticket (level 2 : uncertain)
                             #3  Name , PassengerId , Survived (level 3 : most unlikely)
              (2)（Secondary）The link among other features : one feature may influence another feature , to make this 
                              clear is a very complicated and challenging task !
                             #4  Pclass->Name-Ticket-Fare-Carbin
2 Completing : #5 Age , Embarked have NaN Value ,Once decision that they have correlation have been pulled out, they 
                  should be completing.
3 Correcting : #6 Since the feature Ticket have 681 unique value (total :891) as a nonnumerical type , Drop it off is a 
                  wiser choice.
               #7 Canbin have too many null value to consider it ?(or change the Carbin to bool type 'HaveCarbin')
                  (argument Assumption)
4 Creating :   #8 Since the number of person who have childern & parents & Sib &Sp is too small ,draw temp together, 
                  Creating a new feature : FamilySize (total # of the family members)
               #9 By ob the name ,we can get something like "Mr , Mis ... ", so we can creat a new feature called Title.
               #10 continue type -> discrete type : Create a new "AgeBand" substitute feature "Age".
               #11 Create a "FareBand" substitute "Fare" if the feature help our solution goals.
5 Classifying :#12 Women (Sex=female) were more likely to have survived.
               #13 Children (Age<?) were more likely to have survived.
               #14 The upper-class passengers (Pclass=1) were more likely to have survived.
'''

#Step4: verifying Assumption & Deciding

print(train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print(train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print(train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print(train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).size())
print(train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False))

'''
Analysis by pivoting features
(1) Pclass :  #1 Pass
(2) Sex :  #1 Pass
(3) SibSp & Parch : #8 Pass 
(4) Age: 
'''

g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=30)
plt.show()
'''
Visualisation Analysis Age by histogram 
Observations.
Infants (Age <=4) had high survival rate.
Oldest passengers (Age = 80) survived.
Large number of 15-25 year olds did not survive.
Most passengers are in 15-35 age range.

Desicion:
(1) #1 Pass -> (2) #5 Pass ->(3)#10 Pass
'''

grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()
plt.show()


grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()
plt.show()

g = sns.FacetGrid(train_df, col='Embarked')
g.map(plt.hist, 'Pclass', bins=30)
plt.show()


grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()
'''
Observation:
ect.

Desicion:
#1 : Add Sex feature to model training.
#2&#5 : Complete and add Embarked feature to model training.
#11 : Consider banding Fare feature.
'''

#Step5 : Wrangle data (execute our decisions made above) : Transform the row dataset to the dataset to be trained.

print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)

train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]

print("After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)
'''
Creating : Droping features : 'Ticket' , 'Cabin' 
'''

for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

print(pd.crosstab(train_df['Title'], train_df['Sex']))
'''
We decide to retain the new Title feature for model training.
'''

for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', \
                                                 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

print(train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())
'''
We can replace many titles with a more common name or classify them as Rare.
'''


title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

print(train_df.head())
'''
We can convert the categorical titles to ordinal.
'''



train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]
train_df.shape, test_df.shape
'''
Now we can safely drop the Name feature from training and testing datasets. We also do not need the PassengerId feature in the training dataset.
'''


for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

train_df.head()
'''
 converting Sex feature to a new feature called Gender where female=1 and male=0
'''

guess_ages = np.zeros((2,3))
for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & \
                               (dataset['Pclass'] == j + 1)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i, j] = int(age_guess / 0.5 + 0.5) * 0.5

    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j + 1), \
                        'Age'] = guess_ages[i, j]

    dataset['Age'] = dataset['Age'].astype(int)

train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
for dataset in combine:
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']
print(train_df.head())

train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]
train_df.head()

for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_df, test_df]

print(train_df.head())

for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

print(train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10))

freq_port = train_df.Embarked.dropna().mode()[0]
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived',
                                                                                            ascending=False)

for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)

train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)

for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]



#Step6 : Model, predict and solve(the data has been processed.)

X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()



# Logistic Regression

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)


coeff_df = pd.DataFrame(train_df.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)

# Support Vector Machines

svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)


knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)


# Gaussian Naive Bayes

gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)


# Perceptron

perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)


# Linear SVC

linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)


# Stochastic Gradient Descent

sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)


# Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)


# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)


models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression',
              'Random Forest', 'Naive Bayes', 'Perceptron',
              'Stochastic Gradient Decent', 'Linear SVC',
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log,
              acc_random_forest, acc_gaussian, acc_perceptron,
              acc_sgd, acc_linear_svc, acc_decision_tree]})
print(models.sort_values(by='Score', ascending=False))

submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('/Users/wzj/Documents/Study/Data Ming Project/kaggle_Titanic Problem/Data/submission.csv', index=False)