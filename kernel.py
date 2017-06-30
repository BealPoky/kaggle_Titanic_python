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

# Step2:Get some basic info about Features
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

#Step3: put some Assumtions based on the analysis

'''
Assumtions:



'''
