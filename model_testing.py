import warnings
warnings.filterwarnings('ignore')
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from matplotlib import colors as mcolors
from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence
import pandas as pd
import sklearn as sk
from xgboost import XGBClassifier
from sklearn.preprocessing.imputation import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
import plotly.offline as py
from sklearn.preprocessing import Imputer
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
%matplotlib inline

# load data ... (to be described)
dataset = pd.read_csv("C:\\Users\\Amadeusz\\Downloads\\all\\training_set.csv")

# dataset.head()

# load metadata, ... (to be described)
meta_dataset = pd.read_csv('C:\\Users\\Amadeusz\\Downloads\\all\\training_set_metadata.csv')
column_names = {6: "class_6", 15: "class_15", 16: "class_16", 42: "class_42", 52: "class_52", 53: "class_53",
                62: "class_62", 64: "class_64", 65: "class_65", 67: "class_67", 88: "class_88", 90: "class_90",
                92: "class_92", 95: "class_95"}
# change labels according to sample submission example
meta_dataset["target"] = list(map(lambda name: column_names[name], meta_dataset["target"]))

# meta_dataset.head()

# use imputer to search for NaN values and compute mean() values instead of them (column vise)
mean_imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
mean_imputer = mean_imputer.fit(meta_dataset.iloc[:,:-1]) # without last target column (imputer does not recognize string data)
imputed_meta_dataset = mean_imputer.transform(meta_dataset.iloc[:,:-1].values)
imputed_meta_dataset = pd.DataFrame(data=imputed_meta_dataset, columns=meta_dataset.iloc[:,:-1].columns.values)
imputed_meta_dataset["target"] = meta_dataset.iloc[:,-1] # add last target column to imputed meta dataset
# change object_id and ddf values back to int() ... imputer interprets all values as float()
imputed_meta_dataset["object_id"] = list(map(lambda val: int(val), imputed_meta_dataset["object_id"]))
imputed_meta_dataset["ddf"] = list(map(lambda val: int(val), imputed_meta_dataset["ddf"]))

# imputed_meta_dataset.head()

# merges two datasets, merge by group_id -> common key in both DataFrames
training_dataset = pd.merge(dataset, imputed_meta_dataset)

# training_dataset.head()

# check if training_dataset consists of any empty values
columns_missing = [col for col in training_dataset.columns if training_dataset[col].isnull().any()]
if columns_missing:
    print("Dataset has missing values in the following columns:\n{}".format(columns_missing))
else:
    print("Dataset do not has any column with empty value.")

# split data into training and test datasets (X and Y) ... data is randomly chosen
test_size = 0.2
seed = 7
# description
# train_test_split(X,Y,test_size,random_state)
X_train, X_test, Y_train, Y_test = train_test_split(training_dataset.iloc[:,:-1], training_dataset.iloc[:,-1], 
                                                    test_size=test_size, random_state=seed)
X_train = pd.DataFrame(data=X_train, columns=training_dataset.columns.values.tolist()[:-1])
X_test = pd.DataFrame(data=X_test, columns=training_dataset.columns.values.tolist()[:-1])


import time
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.datasets import make_multilabel_classification
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score


start = time.time()
et = ExtraTreesClassifier(n_estimators=100)
# .iloc[:,1:].values (removes object_id zolumn from computing)
et.fit(X_train.iloc[:,1:].values, Y_train)
end = time.time()
# accuracy score is not valid here ... accuracy means if we predict label correctly it is very good,
# but not enough big predictions for other labels are ignored
print("Extra Trees Elapsed Training time: {}   ---   Training accuracy score: {}".format(end - start, et.score(X_train.iloc[:,1:].values, Y_train)))
start = time.time()
predictions = et.predict(X_test.iloc[:,1:].values)
end = time.time()
print("Extra Trees Elapsed Test prediction time: {}   ---   Testing accuracy score: {}".format(end - start, accuracy_score(Y_test, predictions)))
# compute predictions for each class for each row (sample)
start = time.time()
predicted_et = pd.DataFrame(et.predict_proba(X_test.iloc[:,1:].values), columns=et.classes_)
end = time.time()
print("Predict_proba computed in: {}".format(end-start))

# predicted_et.head()

#log loss function example of perfect prediction
                                                 #0,1,2,3,4,5
# log_loss(np.array([1,0,2,3,3,4,5,5]),np.array([[0,1,0,0,0,0],
#                                                [1,0,0,0,0,0],
#                                                [0,0,1,0,0,0],
#                                                [0,0,0,1,0,0],
#                                                [0,0,0,1,0,0],
#                                                [0,0,0,0,1,0],
#                                                [0,0,0,0,0,1],
#                                                [0,0,0,0,0,1]]))
#First argument is list of known labels, there will be comparison between them and our prediction
#Second argument is matrix of our predictions (rows - each sumple) (columns - each label class alphabeticall sorted)
#The return value of log loss is the measure how big distribution crossentropy is between perfect prediciton and the case,
#smaller value means better result

#To ilustrate this value more clear, exp(-log_loss) can be computed
#This means how close we are to the perfect prediction in percentage 0-1 range

from sklearn.metrics import log_loss
loss = log_loss(y_true=Y_test, y_pred=predicted_et.values, labels=predicted_et.columns.values.tolist())
print("Log_loss: {}".format(loss))
print("Precision: {}".format(np.exp(-loss)))
