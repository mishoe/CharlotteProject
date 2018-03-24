import pickle
import xgboost as xgb
import gc
import pandas as pd
import numpy as np
from TextMiningMachine.splitting import TrainTestDateSplitter
#from TextMiningMachine.xgboost_optimizer import XgbRegressionBuilder
import math
import datetime
import sklearn
import random
gc.collect()

# load preproccessed features
features = xgb.DMatrix('data/xgb.features.data')

# load raw data
data = pd.read_pickle('data/raw_data.p')
all_ind = random.sample(range(data.shape[0]),data.shape[0])
train_ind = all_ind[0:80000]
test_ind = all_ind[80001:]


all_target = [1 if data.loc[i,'race']=='Black' else 0 for i in range(data.shape[0])]
train_target = [all_target[i] for i in train_ind]
test_target = [all_target[i] for i in test_ind]
train=features.slice(train_ind)
train.set_label(train_target)
test=features.slice(test_ind)
test.set_label(test_target)

# load transform
with open( 'data/text_cat_transformer.p', 'rb') as f:
    trans = pickle.load(f)


# set target and date col
target_col = 'race'

# set up params for xgboost to
params = {'booster': 'gbtree',
          'max_depth': 3,
          #'colsample_bytree': .8,
          'colsample_bylevel':1,
          #'base_score': 0.1,
          'eta':0.2,
          'objective': 'binary:logistic',
          'eval_metric': "auc",
          'gamma':0,
          #'alpha':1000,
          'lambda':1}



gc.collect()

num_boost_round = 500

# train a model with initial parameters to estimate errors
model = xgb.train(
    params,
    train,
    num_boost_round=num_boost_round,
    evals=[(train, 'Train'), (test, "Test")],
    early_stopping_rounds=15,
    maximize=True,
)

# get predictions
train_preds = model.predict(train)
test_preds = model.predict(test)

# get R2 vals
train_r2 = sklearn.metrics.r2_score(y_train, train_preds)
test_r2 = sklearn.metrics.r2_score(y_test, test_preds)
print("The train R_2 value for predictions on {} is {}".format(target_col, train_r2))
print("The test R_2 value for predictions on {} is {}".format(target_col, test_r2))

print("Best RMSE: {:.2f} in {} rounds".format(model.best_score, model.best_iteration+1))

# grid search parameters regularization
# find current best score and best params
best_score = model.best_score
best_params = params
new_params = params

# set up grid
reg_alpha = np.arange(.1, .5, .05)

# execute the model building loop for L1 Reg Alpha
for r in reg_alpha:
    gc.collect()
    print("building with with alpha={}, ".format(r))
    # Update our parameters
    new_params['alpha'] = r
    model = xgb.train(
        new_params,
        train,
        num_boost_round=num_boost_round,
        evals=[(train, 'Train'), (test, "Test")],
        early_stopping_rounds=8
    )
    new_score = model.best_score
    # get predictions
    train_preds = model.predict(train)
    test_preds = model.predict(test)

    # get R2 vals
    train_r2 = sklearn.metrics.r2_score(y_train, train_preds)
    test_r2 = sklearn.metrics.r2_score(y_test, test_preds)
    print("The train R_2 value for predictions on {} is {}".format(target_col, train_r2))
    print("The test R_2 value for predictions on {} is {}".format(target_col, test_r2))

    if new_score > best_score:
        print('Score Beaten! New Best', new_score)
        best_params = new_params
        best_score = new_score
    else:
        print('failed to beat best score', best_score)

# train the model with the best parameters
best_model = xgb.train(
    best_params,
    train,
    num_boost_round=1000,
    evals=[(train, 'Train'), (test, "Test")],
    early_stopping_rounds=10)

# get the best iteration round
num_boost_round = model.best_iteration + 1

# re train the model to only the optimum iteration (it usually has been over trained
best_model = xgb.train(
    best_params,
    train,
    num_boost_round=num_boost_round,
    evals=[(train, 'Train'), (test, "Test")])

# get predictions
train_preds = best_model.predict(train)
test_preds = best_model.predict(test)

# get R2 vals
train_r2 = sklearn.metrics.r2_score(y_train, train_preds)
test_r2 = sklearn.metrics.r2_score(y_test, test_preds)
print("The train R_2 value for predictions on {} is {}".format(target_col, train_r2))
print("The test R_2 value for predictions on {} is {}".format(target_col, test_r2))

# # save the best model
model_file_name = 'data/GunViolence_xgb.model'
best_model.save_model(model_file_name)
