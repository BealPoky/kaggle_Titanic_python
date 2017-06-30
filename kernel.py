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
Analyze by pivoting features
(1) Pclass : approval #1
(2) Sex : approval #1
(3) SibSp & Parch 
(4) Age: 
'''

