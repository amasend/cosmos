from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold


# definition and declaration of ML model hyper-parameters, try to use different values and gridsearch between them
# WARNING! if you use a lot of values at once, probably tour computer will be computing forever...
# https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
# This is a nice guide how to tune hyperparameters for XGB, you can use it on different algorithms (need to look at the specific algorithm doc)

param_grid = {
    'XGB__objective': ['multi:softprob'],
    'XGB__max_depth': [5],
    'XGB__learning_rate': [0.1],
    'XGB__n_estimators': [10, 100, 500, 1000],
    'XGB__max_delta_step': [0, 3, 7, 15],
#     'XGB__reg_lambda': [1, 4, 8, 15],
#     'XGB__reg_alpha': [0, 4, 8, 15],
#     'n_jobs': [4],
#     'XGB__gamma': [0, 5, 10, 50, 100],
#     'XGB__min_child_weight': [0, 4, 8, 10, 15]
#     'subsample': [0.8],
#     'colsample_bytree': [0.8],
#     'seed': [7],
#     'scale_pos_weight': [1]
}
    
num_fold = 5
seed = 7
scoring = 'neg_log_loss'
clf = Pipeline([('Scaler', StandardScaler()), ('XGB', XGBClassifier())])
kfold = StratifiedKFold(n_splits=num_fold, random_state=seed)
grid = GridSearchCV(estimator=clf, param_grid=param_grid, scoring=scoring, cv=kfold, n_jobs=-1)
grid_result = grid.fit(X, Y)
print("Best: {} using {}".format(grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("{} ({}) with: {}".format(mean, stdev, param))
