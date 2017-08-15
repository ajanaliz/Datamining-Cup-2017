import pandas as pd
from preprocessing.data_utils import *
from statsmodels.tsa.ar_model import AR
import numpy as np

data = load_data('../data/{filename}'.format(filename=DATA_FINAL_PICKLE),
                 drop_cols=['basket', 'click',
                            'content_1', 'content_2', 'content_3',
                            'CM', 'G', 'ML', 'P',
                            'omitted_group'
                            ], mode='pkl')
train_df, val_df, test_df = split_train_val_test(data)

# train_days_sum = []
# for i in train_df['day'].unique():
#     train_days_sum.append(sum(train_df[train_df['day'] == i]['revenue']))
#
# val_days_sum = []
# for i in val_df['day'].unique():
#     val_days_sum.append(sum(val_df[val_df['day'] == i]['revenue']))
#
# test_days_sum = []
# for i in test_df['day'].unique():
#     test_days_sum.append(sum(test_df[test_df['day'] == i]['revenue']))
#
# # np.concatenate((train_days_sum, val_days_sum, test_days_sum), axis=0)
#
# model = AR(train_days_sum)
# model_fit = model.fit()
#
# val_predictions = model_fit. \
#     predict(start=len(train_days_sum),
#             end=len(train_days_sum) + len(val_days_sum) - 1, dynamic=False)

# doing the same for pid
predictions = []
for p_id in train_df['pid'].unique():
    pid_specified_data = train_df[train_df['pid'] == p_id]
    daily_sale = []
    for i in train_df['day'].unique():
        daily_sale.append(sum(pid_specified_data[pid_specified_data['day'] == i]['revenue']))
    model = AR(daily_sale)
    model_fit = model.fit()

    predictions.append(
        model_fit.predict(start=len(daily_sale),
                          end=len(daily_sale) + len(val_df['day'].unique()) - 1,
                          dynamic=False))

pid_daily_sale = pd.DataFrame({'pid': train_df['pid'].unique(), 'pid_sale': predictions})


def pid_daily(df):
    temp = np.ndarray(pid_daily_sale['pid_sale'])
    print(temp[(df['day'] - 31)])
    return df


val_df.groupby('pid', sort=False).apply(pid_daily)
print(pid_daily_sale)
