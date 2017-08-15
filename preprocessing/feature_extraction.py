import numpy as np
import xgboost as xgb

from preprocessing.data_utils import *
from preprocessing.data_utils import load_data

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

data = load_data('../data/{filename}'.format(filename=DATA_FINAL_PICKLE),
                 drop_cols=['basket', 'click',
                            'content_1', 'content_2', 'content_3',
                            'CM', 'G', 'ML', 'P',
                            'omitted_group'
                            ], mode='pkl')

train_df, val_df, test_df = split_train_val_test(data)

data['discount'] = (data['rrp'] - data['price']) / data['rrp']
data['compete'] = (data['price'] - data['competitorPrice']) / data['price']
data['f1'] = (data['rrp'] - data['price'])
data['f3'] = (data['rrp'] - data['competitorPrice'])
data['f4'] = (data['price'] - data['competitorPrice'])

records = []
first_month = train_df
for i in data['pid'].unique():
    records.append(sum(first_month[first_month['pid'] == i]['revenue']))

sales_record = pd.DataFrame({'pid': data['pid'].unique(), 'record': records})
data = data.merge(sales_record, how='left', on='pid', sort=False)
print(data)
# print(data.describe())
# Train XGBoost for order
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
