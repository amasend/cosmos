import warnings
import time
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
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
%matplotlib inline

# dataset = pd.read_csv("C:\\Users\\Amadeusz\\Downloads\\all\\test_set.csv", 
#                       dtype={"object_id":np.int64, "mjd":np.float64, "passband":np.int64,
#                              "flux":np.float64, "flux_err":np.float64, "detected":np.int64})
# print("Dataset loaded")

# x = list(set(dataset["object_id"]))
# n_parts = 10
# dlugosc =  np.int(np.ceil(len(x)/n_parts))
# for i, n in zip(range(0, len(x), dlugosc), range(1, 11)):
#     start = time.time()
#     dataset[dataset["object_id"].isin(x[i:i+dlugosc])].to_csv("..\\test_{}.csv".format(n))
#     end = time.time()
#     print("test_{}.csv saved - elapsed time: {}".format(n, end-start))
# %xdel dataset
# %xdel x

import pickle
from sklearn.metrics import log_loss
try:
    # load the model from disk
    filename = 'ET_model_log_loss.sav'
    et = pickle.load(open(filename, 'rb'))
    
except FileNotFoundError as e:
    # load data ... (to be described)
    dataset = pd.read_csv("C:\\Users\\Amadeusz\\Downloads\\all\\training_set.csv")
    # load metadata, ... (to be described)
    meta_dataset = pd.read_csv('C:\\Users\\Amadeusz\\Downloads\\all\\training_set_metadata.csv')
    column_names = {6: "class_6", 15: "class_15", 16: "class_16", 42: "class_42", 52: "class_52", 53: "class_53",
                    62: "class_62", 64: "class_64", 65: "class_65", 67: "class_67", 88: "class_88", 90: "class_90",
                    92: "class_92", 95: "class_95"}
    # change labels according to sample submission example
    meta_dataset["target"] = list(map(lambda name: column_names[name], meta_dataset["target"]))
    # use imputer to search for NaN values and compute mean() values instead of them (column vise)
    mean_imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
    mean_imputer = mean_imputer.fit(meta_dataset.iloc[:,:-1]) # without last target column (imputer does not recognize string data)
    imputed_meta_dataset = mean_imputer.transform(meta_dataset.iloc[:,:-1].values)
    imputed_meta_dataset = pd.DataFrame(data=imputed_meta_dataset, columns=meta_dataset.iloc[:,:-1].columns.values)
    imputed_meta_dataset["target"] = meta_dataset.iloc[:,-1] # add last target column to imputed meta dataset
    # change object_id and ddf values back to int() ... imputer interprets all values as float()
    imputed_meta_dataset["object_id"] = list(map(lambda val: int(val), imputed_meta_dataset["object_id"]))
    imputed_meta_dataset["ddf"] = list(map(lambda val: int(val), imputed_meta_dataset["ddf"]))
    # merges two datasets, merge by group_id -> common key in both DataFrames
    training_dataset = pd.merge(dataset, imputed_meta_dataset)
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
    from sklearn.metrics import log_loss
    loss = log_loss(y_true=Y_test, y_pred=predicted_et.values, labels=predicted_et.columns.values.tolist())
    print("Log_loss: {}".format(loss))
    print("Precision: {}".format(np.exp(-loss)))

    # save model
    filename = 'ET_model_log_loss.sav'
    pickle.dump(et, open(filename, 'wb'))
    
    def predictions(predicted_dataframe, object_id):
#     print(predicted_dataframe.columns.values)
#     columns = ["class_6","class_15","class_16","class_42","class_52","class_53",
#            "class_62","class_64","class_65","class_67","class_88","class_90",
#            "class_92","class_95"]
#     predicted_dataframe.columns = columns
    columns = predicted_dataframe.columns.values.tolist()
    start = time.time()
    class_99 = np.any(predicted_dataframe[predicted_dataframe[columns] <= 0.25].apply(np.isnan), axis=1).apply(np.logical_not).apply(np.int)
    predicted_dataframe["class_99"] = list(map(lambda x: x/2, class_99))
    end = time.time()
    print("After search, elapsed time: {}".format(end-start))
    predicted_dataframe.insert(0,"object_id",object_id)
    
%xdel dataset
%xdel meta_dataset
%xdel imputed_meta_dataset
%xdel training_dataset
%xdel X_train
%xdel X_test
%xdel Y_train
%xdel Y_test
print("After memory relocation")

meta_dataset = pd.read_csv('C:\\Users\\Amadeusz\\Downloads\\all\\test_set_metadata.csv')
mean_imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
mean_imputer = mean_imputer.fit(meta_dataset)
imputed_meta_dataset = mean_imputer.transform(meta_dataset.values)
imputed_meta_dataset = pd.DataFrame(data=imputed_meta_dataset, columns=meta_dataset.columns.values)
# change object_id and ddf values back to int() ... imputer interprets all values as float()
imputed_meta_dataset["object_id"] = list(map(lambda val: int(val), imputed_meta_dataset["object_id"]))
imputed_meta_dataset["ddf"] = list(map(lambda val: int(val), imputed_meta_dataset["ddf"]))

for i in range(1,11):
    dataset = pd.read_csv("..\\test_{}.csv".format(i))
    dataset_merged = pd.merge(dataset, imputed_meta_dataset)
    %xdel dataset
    object_id = dataset_merged["object_id"]
    #dataset_merged = dataset_merged.drop(columns=["Unnamed: 0", "object_id"])
    start = time.time()
    predicted_et = pd.DataFrame(et.predict_proba(dataset_merged.iloc[:,1:].values), columns=et.classes_)
    end = time.time()
    print("Part {} Predict Elapsed time: {}".format(i, end-start))
    predictions(predicted_et, object_id)
    start = time.time()
    predicted_et.to_csv("..\\predicted_et_{}.csv".format(i))
    end = time.time()
    print("Part {} Save to CSV Elapsed time: {}".format(i, end-start))
    %xdel dataset_merged
    %xdel predicted_et
    
    
    
    for i in range(1,11):
    start = time.time()
    dataset = pd.read_csv("..\\predicted_et_{}.csv".format(i)).drop(columns=["Unnamed: 0"])
    grouped = dataset.groupby(by="object_id").mean()
    grouped.to_csv("..\\grouped_{}.csv".format(i))
    %xdel grouped
    %xdel dataset
    end = time.time()
    print("Group {} saved. Elapsed time: {}".format(i, end-start))

for i in range(1,11):
    start = time.time()
    if i == 1:
        dataset = pd.read_csv("..\\grouped_{}.csv".format(i))
    else:
        dataset = pd.concat([dataset, pd.read_csv("..\\grouped_{}.csv".format(i))], ignore_index=True)
    end = time.time()
    print("Group {} added. Elapsed time: {}".format(i, end-start))

dataset.to_csv("..\\LSST_project_prediction_05.csv", index=False)
%xdel dataset
