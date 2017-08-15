import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.metrics import classification_report, precision_score, recall_score, auc, roc_curve
import pandas as pd
import xgboost as xgb
from preprocessing.data_utils import *
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error, accuracy_score
from statsmodels.tsa.ar_model import AR

from preprocessing.data_utils import load_data, DATA_CLUSTERED_EXCEPT_PHARMFORM_FINAL_PICKLE

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# Train XGBoost for order
data = load_data('../data/{filename}'.format(filename=DATA_CLUSTERED_EXCEPT_PHARMFORM_FINAL_PICKLE),
                 drop_cols=['basket', 'click',
                            'content_1', 'content_2', 'content_3',
                            'CM', 'G', 'ML', 'P',
                            'omitted_group'
                            ], mode='pkl')

train_df, val_df, test_df = split_train_val_test(data)

train_df = train_df.drop('day', axis=1)
val_df = val_df.drop('day', axis=1)
test_df = test_df.drop('day', axis=1)

train_X = train_df.drop(['revenue', 'order', 'count'], axis=1)
train_Y = train_df['order']
T_train_xgb = xgb.DMatrix(train_X, train_Y)

params = {"objective": "reg:linear", "booster": "gblinear", "eval_metric": "rmse",
          # "eta": 0.6,
          # "gamma": 100,
          # "max_depth": 18,
          }
gbm = xgb.train(dtrain=T_train_xgb, params=params)
val_pred_order = gbm.predict(xgb.DMatrix(val_df.drop(['order', 'revenue', 'count'], axis=1)))
val_pred_order = pd.Series(val_pred_order)
val_pred_order[val_pred_order < 0.2] = 0

# Train XGBoost for count
train_X = train_df.drop(['revenue', 'order', 'count'], axis=1)
train_Y = train_df['count']
T_train_xgb = xgb.DMatrix(train_X, train_Y)

params = {"objective": "reg:linear", "booster": "gblinear", "eval_metric": "rmse",
          # "eta": 0.6,
          # "gamma": 100,
          # "max_depth": 18,
          }
gbm = xgb.train(dtrain=T_train_xgb, params=params)
val_pred_count = gbm.predict(xgb.DMatrix(val_df.drop(['order', 'revenue', 'count']
                                                     , axis=1)))
val_pred_count = pd.Series(val_pred_count)

print(1 / len(val_pred_count) *
      sum(((val_pred_count.values * val_df['price'] * val_pred_order.values)
           - (val_df['revenue'])) ** 2))
# Random Forest on order Classification
# forest = RandomForestClassifier(n_jobs=-1,
#                                 n_estimators=500,
#                                 max_features=None)
# train_X = data_val.drop(['order'], axis=1)
# train_Y = data_val['order']
#
# forest.fit(train_X, train_Y)
# predictions = forest.predict(data_train.drop(['order'], axis=1))
# report = classification_report(data_train['order'], predictions)
# print(report)
# joblib.dump(forest, 'forest_model_sajad.pkl')
