import warnings
warnings.filterwarnings('ignore')
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from matplotlib import colors as mcolors
import pandas as pd
import gc
import plotly.offline as py
py.init_notebook_mode(connected=True)
%matplotlib inline

dataset = pd.read_csv("C:\\Users\\Amadeusz\\Downloads\\all\\training_set.csv")
dataset.head()
#
# mjd - the time in Modified Julian Date (MJD) of the observation. 
#       Can be read as days since November 17, 1858. Can be converted to Unix epoch time with the formula 
#       unix_time = (MJD−40587)×86400. Float64
#
# passband - The specific LSST passband integer, such that u, g, r, i, z, Y = 0, 1, 2, 3, 4, 5 in which it was viewed. Int8
# 
# flux - the measured flux (brightness) in the passband of observation as listed in the passband column. 
#        These values have already been corrected for dust extinction (mwebv), though heavily extincted objects 
#        will have larger uncertainties (flux_err) in spite of the correction. Float32
#
# flux_err -  the uncertainty on the measurement of the flux listed above. Float32
#
# detected - If 1, the object's brightness is significantly different at the 3-sigma level 
#           relative to the reference template. Only objects with at least 2 detections are included in the dataset. Boolean

dataset.describe()
# "detected" column is binary (contains 0 or 1 values)

# extract each passband
for key, val in {'u': 0, 'g': 1, 'r': 2, 'i': 3, 'z': 4, 'y': 5}.items():
    dataset[key] = (dataset['passband']-val).apply(np.bool).apply(np.logical_not).apply(np.int)
dataset.head()

# convert mjd to unix time
unix_time = (dataset['mjd']-40587)*86400
#convert unix time to readable format
time_ = pd.to_datetime(unix_time, unit='ms')

# information extraction from modified julian date
# dataset["year"] = time_.dt.year
# dataset["month"] = time_.dt.month
# dataset["day"] = time_.dt.day
# dataset["week"] = time_.dt.week
# dataset["dayofweek"] = time_.dt.dayofweek
dataset["hour"] = time_.dt.hour
dataset["minute"] = time_.dt.minute
dataset["second"] = time_.dt.second
dataset["ms"] = time_.dt.microsecond
# dataset = dataset.drop(["mjd"], axis=1)
dataset.head()

# check if there is any null value
dataset.isnull().sum()

gc.enable()
dataset['flux_ratio_sq'] = np.power(dataset['flux'] / dataset['flux_err'], 2.0)
dataset['flux_by_flux_ratio_sq'] = dataset['flux'] * dataset['flux_ratio_sq']

aggs = {
    'mjd': ['min', 'max', 'size'],
    'passband': ['min', 'max', 'mean', 'median', 'std'],
    'flux': ['min', 'max', 'mean', 'median', 'std','skew'],
    'flux_err': ['min', 'max', 'mean', 'median', 'std','skew'],
    'detected': ['mean'],
    'flux_ratio_sq':['sum','skew'],
    'flux_by_flux_ratio_sq':['sum','skew'],
    'r': ['sum'],
    'g': ['sum'],
    'i': ['sum'],
    'u': ['sum'],
    'z': ['sum'],
    'y': ['sum'],
#     'month': ['min', 'max', 'median'],
#     'day': ['min', 'max', 'median'],
#     'week': ['min', 'max', 'median'],
#     'dayofweek': ['min', 'max', 'median'],
    'hour': ['min', 'max', 'median'],
    'minute': ['min', 'max', 'median'],
    'second': ['min', 'max', 'median'],
    'ms': ['min', 'max', 'median'],
}

agg_train = dataset.groupby('object_id').agg(aggs)
new_columns = [k + '_' + agg for k in aggs.keys() for agg in aggs[k]]
agg_train.columns = new_columns
# agg_train['month_diff'] = agg_train['month_max'] - agg_train['month_min']
# agg_train['day_diff'] = agg_train['day_max'] - agg_train['day_min']
# agg_train['week_diff'] = agg_train['week_max'] - agg_train['week_min']
# agg_train['dayofweek_diff'] = agg_train['dayofweek_max'] - agg_train['dayofweek_min']
agg_train['hour_diff'] = agg_train['hour_max'] - agg_train['hour_min']
agg_train['minute_diff'] = agg_train['minute_max'] - agg_train['minute_min']
agg_train['second_diff'] = agg_train['second_max'] - agg_train['second_min']
agg_train['ms_diff'] = agg_train['ms_max'] - agg_train['ms_min']
agg_train['mjd_diff'] = agg_train['mjd_max'] - agg_train['mjd_min']
agg_train['flux_diff'] = agg_train['flux_max'] - agg_train['flux_min']
agg_train['flux_dif2'] = (agg_train['flux_max'] - agg_train['flux_min']) / agg_train['flux_mean']
agg_train['flux_w_mean'] = agg_train['flux_by_flux_ratio_sq_sum'] / agg_train['flux_ratio_sq_sum']
agg_train['flux_dif3'] = (agg_train['flux_max'] - agg_train['flux_min']) / agg_train['flux_w_mean']

del agg_train['mjd_max'], agg_train['mjd_min']
# del agg_train['month_max'], agg_train['month_min']
# del agg_train['day_max'], agg_train['day_min']
# del agg_train['week_max'], agg_train['week_min']
# del agg_train['dayofweek_max'], agg_train['dayofweek_min']
del agg_train['hour_max'], agg_train['hour_min']
del agg_train['minute_max'], agg_train['minute_min']
del agg_train['second_max'], agg_train['second_min']
del agg_train['ms_max'], agg_train['ms_min']
agg_train.head()

del dataset
gc.collect()

agg_train.head()

meta_dataset = pd.read_csv('C:\\Users\\Amadeusz\\Downloads\\all\\training_set_metadata.csv')
column_names = {6: "class_6", 15: "class_15", 16: "class_16", 42: "class_42", 52: "class_52", 53: "class_53",
                62: "class_62", 64: "class_64", 65: "class_65", 67: "class_67", 88: "class_88", 90: "class_90",
                92: "class_92", 95: "class_95"}
meta_dataset["target"] = list(map(lambda name: column_names[name], meta_dataset["target"]))
meta_dataset.head()

# ra - right ascension, sky coordinate: co-longitude in degrees. Float32
#
# decl - declination, sky coordinate: co-latitude in degrees. Float32
#
# gal_l - galactic longitude in degrees. Float32
#
# gal_b - galactic latitude in degrees. Float32
#
# ddf - A flag to identify the object as coming from the DDF survey area 
#       (with value DDF = 1 for the DDF, DDF = 0 for the WFD survey). 
#       Note that while the DDF fields are contained within the full WFD survey area, the DDF fluxes have 
#       significantly smaller uncertainties. Boolean
#
# hostgal_specz - the spectroscopic redshift of the source. This is an extremely accurate measure of redshift, 
#                 available for the training set and a small fraction of the test set. Float32
#
# hostgal_photoz - The photometric redshift of the host galaxy of the astronomical source. 
#                  While this is meant to be a proxy for hostgal_specz, there can be large differences between 
#                  the two and should be regarded as a far less accurate version of hostgal_specz. Float32
#
# hostgal_photoz_err - The uncertainty on the hostgal_photoz based on LSST survey projections. Float32
# 
# distmod - The distance to the source calculated from hostgal_photoz and using general relativity. Float32
# 
# mwebv - MW E(B-V). this ‘extinction’ of light is a property of the Milky Way (MW) dust along the line of sight 
#         to the astronomical source, and is thus a function of the sky coordinates of the source ra, decl. 
#       This is used to determine a passband dependent dimming and redenning of light from astronomical sources 
#       as described in subsection 2.1, and based on the Schlafly et al. (2011) and Schlegel et al. (1998) dust models. Float32

# from description -> NaN values in "distmod" teels us that this object is in our galaxy (no redshift provided/hostgal),
# distmod is calculated from redshift via general relativity
# create new features (in_our_galaxy)
meta_dataset['in_our_galaxy'] = meta_dataset['distmod'].apply(np.isnan).astype(int)
meta_dataset.head()

meta_dataset.isnull().sum()

print("distmod missing data in %: {}".format(meta_dataset.isnull().sum()[-4]*100/meta_dataset.shape[0]))
# put 0 instead of NaN in distmod (in that way we can store distance for objects out of our galaxy)
meta_dataset = meta_dataset.fillna(0)
meta_dataset.head()

meta_dataset.isnull().sum()

# check test dataset
meta_test = pd.read_csv('C:\\Users\\Amadeusz\\Downloads\\all\\test_set_metadata.csv')
n_samples = meta_test.shape[0]
missing_values = meta_test.isnull().sum()

for val, ind in zip(missing_values, missing_values.index):
    print("{} has {:2.2f}%  missing values.".format(ind, val*100/n_samples))
# hostgal_specz - missing 96%+ samples -> insufficient data for analysis and ML algorithms, need to be dropped
%xdel meta_test

# delete not valuable column
meta_dataset = meta_dataset.drop(['hostgal_specz'], axis=1)
meta_dataset.head()

# training_dataset = pd.merge(dataset, meta_dataset) # without additional computed features
training_dataset = pd.merge(agg_train.reset_index(), meta_dataset) # with additional computed features
training_dataset.head()
