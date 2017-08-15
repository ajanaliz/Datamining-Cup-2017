import math
import pandas as pd
from pathlib import Path

from sklearn.utils import shuffle

DATA_MERGED_PICKLE = 'data_merged_pickle.pkl'
DATA_FINAL_PICKLE = 'data_final_pickle.pkl'
DATA_CLUSTERED_PICKLE = 'data_clustered.pkl'
DATA_CLUSTERED_EXCEPT_PHARMFORM_FINAL_PICKLE = 'data_clustered_except_pharmform.pkl'
DATA_FINAL_DUMMY_MANUFACTURER_PICKLE = 'data_final_dummy_manufacturer_pickle.pkl'


def load_data(path, target_name=None, drop_cols=None, mode='csv'):
    if mode == 'csv':
        data = pd.read_csv(path)
    else:
        data = pd.read_pickle(path)

    if drop_cols:
        data = data.drop(drop_cols, axis=1)

    if target_name is None:
        return data
    else:
        return data_target(data, target_name)


def check_if_file_exists(path):
    file = Path(path)
    if file.is_file():
        return file
    else:
        return None


def split_train_val_test(data):
    train_df = data[data['day'] <= 31]
    val_df = data[(data['day'] > 31) & (data['day'] <= 62)]
    test_df = data[(data['day'] > 62) & (data['day'] <= 92)]
    return train_df, val_df, test_df


def data_target(data, target_name):
    data = data.astype('float32')
    target = data[target_name]
    data = data.drop(target_name, axis=1)
    return data, target


# To check how many columns have missing values - this can be repeated to see the progress made
def show_missing(data):
    missing = data.columns[data.isnull().any()].tolist()
    return data[missing].isnull().sum()


# Get missing value counts of a dataframe
def get_missing_count(df):
    return df.isnull().values.ravel().sum()


def merge_data(train_df, items_df):
    train_merged = train_df.copy()
    train_merged = train_merged.merge(train_df.merge(items_df, how='left', on='pid', sort=False))
    return train_merged


def split_abundant_target(data_df, ratio):
    data_df = shuffle(data_df)
    part_size = math.ceil(data_df.shape[0] / ratio)
    for i in range(0, ratio):
        yield data_df[i * part_size:min((i + 1) * part_size, data_df.shape[0])]
