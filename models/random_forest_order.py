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

from preprocessing.data_utils import load_data, DATA_FINAL_PICKLE

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# Train XGBoost for order
data = load_data('../data/{filename}'.format(filename=DATA_FINAL_PICKLE),
                 drop_cols=['basket', 'click',
                            'competitorPrice',
                            'rrp',
                            'content_1', 'content_2', 'content_3',
                            'CM', 'G', 'ML', 'P',
                            'omitted_group'
                            ], mode='pkl')

train_df, val_df, test_df = split_train_val_test(data)

train_df = train_df.drop('day', axis=1)
val_df = val_df.drop('day', axis=1)
test_df = test_df.drop('day', axis=1)

# Random Forest on order Classification
forest = RandomForestClassifier(n_jobs=-1,
                                n_estimators=200,
                                max_features=None)
train_X = train_df.drop(['revenue', 'order', 'count'], axis=1)
train_Y = train_df['order']

forest.fit(train_X, train_Y)
predictions = forest.predict(val_df.drop(['order'], axis=1))
report = classification_report(val_df['order'], predictions)
print(report)
joblib.dump(forest, 'forest_model_sajad.pkl')
