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



dataset = pd.read_csv("C:\\Users\\Amadeusz\\Downloads\\all\\test_set.csv", 
                      dtype={"object_id":np.int64, "mjd":np.float64, "passband":np.int64,
                             "flux":np.float64, "flux_err":np.float64, "detected":np.int64})
print("Dataset loaded")

x = list(set(dataset["object_id"]))
n_parts = 10
dlugosc =  np.int(np.ceil(len(x)/n_parts))
for i, n in zip(range(0, len(x), dlugosc), range(1, 11)):
    start = time.time()
    dataset[dataset["object_id"].isin(x[i:i+dlugosc])].to_csv("..\\test_{}.csv".format(n))
    end = time.time()
    print("test_{}.csv saved - elapsed time: {}".format(n, end-start))
%xdel dataset
%xdel x

import pickle
try:
    # load the model from disk
    filename = 'ET_model.sav'
    et = pickle.load(open(filename, 'rb'))
    
except FileNotFoundError as e:
    # train and save model to file
    dataset = pd.read_csv("C:\\Users\\Amadeusz\\Downloads\\all\\training_set.csv")
    meta_dataset = pd.read_csv('C:\\Users\\Amadeusz\\Downloads\\all\\training_set_metadata.csv')
    from sklearn.preprocessing import Imputer
    mean_imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
    mean_imputer = mean_imputer.fit(meta_dataset)
    imputed_meta_dataset = mean_imputer.transform(meta_dataset.values)
    imputed_meta_dataset = pd.DataFrame(data=imputed_meta_dataset, columns=meta_dataset.columns.values)
    training_dataset = pd.merge(dataset, imputed_meta_dataset)
    # how many rows entire dataset has
    n_rows, n_cols = training_dataset.shape
    # get column names and change order
    columns = training_dataset.columns.tolist()
    # check if dataset consists of empty values
    columns_missing = [col for col in dataset.columns if dataset[col].isnull().any()]
    if columns_missing:
        print("Dataset has missing values in the following columns:\n{}".format(columns_missing))
    else:
        print("Dataset do not has any column with empty value.")
    # split data accordingly:
    # split rest of dataset into training and validation datasets (80% to 20%)
    array = training_dataset.values
    X = array[:, 1:-1].astype(float)
    Y = array[:, -1]
    test_size = 0.2
    seed = 7
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
    X_train = pd.DataFrame(data=X_train, columns=columns[1:-1])
    X_test = pd.DataFrame(data=X_test, columns=columns[1:-1])
    import time
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.datasets import make_multilabel_classification
    from sklearn.ensemble import BaggingClassifier
    from sklearn.ensemble import ExtraTreesClassifier
    start = time.time()
    et = ExtraTreesClassifier(n_estimators=100)
    et.fit(X_train, Y_train)
    end = time.time()
    print("Extra Trees Elapsed Training time: {}   ---   {}".format(end - start, et.score(X_train, Y_train)))
    # https://stackoverflow.com/questions/16858652/how-to-find-the-corresponding-class-in-clf-predict-proba
    #predicted_et = pd.DataFrame(et.predict_proba(X_test), columns=et.classes_)


    # save model
    filename = 'ET_model.sav'
    pickle.dump(et, open(filename, 'wb'))
    
def predictions(predicted_dataframe, object_id):
    print(predicted_dataframe.columns.values)
    columns = ["class_6","class_15","class_16","class_42","class_52","class_53",
           "class_62","class_64","class_65","class_67","class_88","class_90",
           "class_92","class_95"]
    predicted_dataframe.columns = columns
    start = time.time()
    predicted_dataframe["class_99"] = np.any(predicted_dataframe[predicted_dataframe[columns] <= 0.4].apply(np.isnan), axis=1).apply(np.logical_not).apply(np.int)
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

for i in range(1,11):
    dataset = pd.read_csv("..\\test_{}.csv".format(i))
    dataset_merged = pd.merge(dataset, imputed_meta_dataset)
    %xdel dataset
    object_id = dataset_merged["object_id"]
    dataset_merged = dataset_merged.drop(columns=["Unnamed: 0", "object_id"])
    start = time.time()
    predicted_et = pd.DataFrame(et.predict_proba(dataset_merged), columns=et.classes_)
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

dataset.to_csv("..\\LSST_project_prediction.csv", index=False)
%xdel dataset
