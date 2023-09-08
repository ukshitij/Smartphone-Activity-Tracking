

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

from collections import Counter

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

"""## Importing the data"""

train = pd.read_csv('/content/train.csv')
test = pd.read_csv('/content/test.csv')

##train.head()

##test.head()

##train.describe()

##test.describe()

"""## Data preprocessing

#### Check for duplicates
"""

print('Number of duplicate values in train :', sum(train.duplicated()))
print('Number of duplicate values in test :', sum(test.duplicated()))

"""#### Check for null values"""

train.isnull().sum()

test.isnull().sum()

print("Shape of train dataset:", train.shape)
print("Shape of test dataset:", test.shape)

traincopy = train.copy()
traincopy.dropna(inplace=True)
traincopy.shape

testcopy = test.copy()
testcopy.dropna(inplace=True)
testcopy.shape

train = traincopy
test = testcopy

##train['Activity'].dtypes

"""## Checking for class imbalance"""

sns.countplot(x = 'Activity', data = train)
plt.xticks(rotation = 45)
plt.show()

"""## Exploratory Data Analysis

### There are 563 parameters to analyse which is not possible
### we have mean, std dev, abs mean, on x, y, z axes for tbodyacc
### and same for various other parameters
### we will reduce them into one parameter like tbodyacc, angle rather than having mean,stdev of x,y,z axes
"""

##train.head()

train.columns[0].split('-')[0]

"""## use list comprehension to loop through all columns

## Counter gets the count of each column split occurring how many times and stores it in dictionary Format
## Convert the dictionary to dataframe
"""

pd.DataFrame.from_dict(Counter([col.split('-')[0].split('(')[0] for col in train.columns]), orient = 'index').rename(columns = {0 : 'count'}).sort_values('count', ascending = False)

"""# Tbodyacc mean plot wrt to activity
### it is uderstood that performing static activities acceleration would be less and dynamic activities acceleration would be more
"""

facetgrid = sns.FacetGrid(train, hue = 'Activity', height=5, aspect=2)
facetgrid.map(sns.distplot, 'tBodyAccMag-mean()', hist=False)

## visualization techniques
plt.annotate('Static activities', xy=(-1,6), xytext=(-0.6,20), arrowprops={'arrowstyle' : '-', 'ls' : 'dashed'})
plt.annotate('Static activities', xy=(-1,1.5), xytext=(-0.6,20), arrowprops={'arrowstyle' : '-', 'ls' : 'dashed'})
plt.annotate('Static activities', xy=(-1,29), xytext=(-0.6,20), arrowprops={'arrowstyle' : '-', 'ls' : 'dashed'})

plt.annotate('Dynamic activities', xy=(0,4), xytext=(0.5,11), arrowprops={'arrowstyle' : '-', 'ls' : 'dashed'})
plt.annotate('Dynamic activities', xy=(-.1,5.9), xytext=(0.5,11), arrowprops={'arrowstyle' : '-', 'ls' : 'dashed'})
plt.annotate('Dynamic activities', xy=(-.15,8.8), xytext=(0.5,11), arrowprops={'arrowstyle' : '-', 'ls' : 'dashed'})

sns.boxplot(x = 'Activity', y = 'tBodyAccMag-mean()', data = train, showfliers = False)   ## showfliers = false means outliers will not be plotted
plt.xticks(rotation = 15)
plt.show()

"""## Relation between activities and X axis gravity"""

sns.boxplot(x = 'Activity', y = 'angle(X,gravityMean)', data = train, showfliers = False)   ## showfliers = false means outliers will not be plotted
plt.xticks(rotation = 15)
plt.show()

"""### Relation between activities and gravity along the y axis"""

sns.boxplot(x = 'Activity', y = 'angle(Y,gravityMean)', data = train, showfliers = False)   ## showfliers = false means outliers will not be plotted
plt.xticks(rotation = 15)
plt.show()

"""## Dimensionality reduction using PCA"""

x_pca = train.drop(['subject', 'Activity'], axis = 1)   ## as subject is of no use and activity is output parameter

x_pca = PCA(n_components=2, random_state=0).fit_transform(x_pca)

plt.figure(figsize=(6,4))
sns.scatterplot(x = x_pca[:,0], y = x_pca[:,1], hue = train['Activity'])
plt.show()

"""## As we can see Static tasks like standing, sitting, laying are grouped and dynamic activities like walking, upstairs, downstairs are grouped separately

## Dimensionality reduction using tsne
"""

x_tsne = train.drop(['subject', 'Activity'], axis = 1)   ## as subject is of no use and activity is output parameter

x_tsne = TSNE(n_components=2, random_state=0, n_iter=1000).fit_transform(x_tsne)

plt.figure(figsize=(6,4))
sns.scatterplot(x = x_tsne[:,0], y = x_tsne[:,1], hue = train['Activity'])
plt.show()

"""## ML Models

## Preparing training and testing data
"""

x_train = train.drop(['Activity','subject'], axis = 1)
y_train = train['Activity']

x_test = test.drop(['Activity','subject'], axis = 1)
y_test = test['Activity']

print('Training data shape:', x_train.shape)
print('Testing data shape:', x_test.shape)

"""## Using logistic regression model with hyperparameter tuning and cross validation"""

lr_classifier = LogisticRegression()

## lr_classifier.fit(x_train, y_train)   ### normal way
## but we have to add cross validation

parameters_lr = {'max_iter' : [100, 200, 250]}             ### setting parameters iteration for hyperparameter tuning

## Randomized Search CV for K fold cross validation
lr_classifier_cv = RandomizedSearchCV(lr_classifier, param_distributions = parameters_lr, cv = 5, random_state = 42)   ### rus train for 5 iterations and 3 parameters to select the best out of 15 possibilities
lr_classifier_cv.fit(x_train, y_train)

## predict the data
y_pred_lr = lr_classifier_cv.predict(x_test)

lr_accuracy = accuracy_score(y_true = y_test, y_pred = y_pred_lr)
print("Accuracy of Logistic Regression model : ", round(lr_accuracy*100,3))

"""## to find best random search model"""

def best_randomsearch_results(model):
  print("Best estimator        : ", model.best_estimator_)
  print("Best set of paramters : ", model.best_params_)


best_randomsearch_results(lr_classifier_cv)

## confusion matrix

cm_lr = confusion_matrix(y_test.values, y_pred_lr)
cm_lr

"""## Using SVM for classifier

#### defining parameters for svm
"""



parameters_svc = {
    'kernel' :['linear', 'rbf', 'poly', 'sigmoid'],
    'C' : [150, 100, 50]
}

svm_classifier = SVC()

svm_classifier_cv = RandomizedSearchCV(svm_classifier, param_distributions=parameters_svc, cv = 3, random_state=42)
svm_classifier_cv.fit(x_train, y_train)

y_pred_svc = svm_classifier_cv.predict(x_test)

## accuracy
svc_accuracy = accuracy_score(y_true = y_test, y_pred = y_pred_svc)
print("Accuracy of Support Vector Classifier model : ", round(svc_accuracy*100,3))

cm_svc = confusion_matrix(y_test.values, y_pred_svc)
cm_svc

best_randomsearch_results(svm_classifier_cv)

"""# Decision tree and Random Forest

## Decision Tree
"""

paramters_dt = {'max_depth' : np.arange(1,20,8)}

dt_classifier = DecisionTreeClassifier()
dt_classifier_cv = RandomizedSearchCV(dt_classifier, param_distributions= paramters_dt, random_state = 42)
dt_classifier_cv.fit(x_train,y_train)

y_pred_dtc = dt_classifier_cv.predict(x_test)

## accuracy
dtc_accuracy = accuracy_score(y_true = y_test, y_pred = y_pred_dtc)
print("Accuracy of Decision Tree Classifier model : ", round(dtc_accuracy*100,3))

## confusion matrix

cm_dtc = confusion_matrix(y_test.values, y_pred_dtc)
cm_dtc

"""## Random Forest"""

parameters_rf = {
    'n_estimators' : np.arange(20, 101, 10),
    'max_depth' : np.arange(2,17, 2)
}

rf_classifier = RandomForestClassifier()
rf_classifier_cv = RandomizedSearchCV(rf_classifier, param_distributions=parameters_rf, random_state = 42)
rf_classifier_cv.fit(x_train, y_train)

y_pred_rfc = rf_classifier_cv.predict(x_test)

## accuracy
rfc_accuracy = accuracy_score(y_true = y_test, y_pred = y_pred_rfc)
print("Accuracy of Random Forest Classifier model : ", round(rfc_accuracy*100,3))

## confusion matrix

cm_rfc = confusion_matrix(y_test.values, y_pred_dtc)
cm_rfc