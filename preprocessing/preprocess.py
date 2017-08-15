from preprocessing.preprocess_items import ItemsPreprocessor
from preprocessing.preprocess_train import TrainProcessor
from preprocessing.data_utils import *
from os import remove as remove_file


def preprocess(type=DATA_FINAL_PICKLE):
    items_processor = ItemsPreprocessor()
    items_processor.prepare(type != DATA_CLUSTERED_PICKLE)

    if check_if_file_exists('../data/{filename}'.format(filename=type)):
        data_df = load_data('../data/{filename}'.format(filename=type), mode='pkl')
    else:
        train_processor = TrainProcessor(items_processor.items_df)
        train_processor.prepare(type)

        data_df = train_processor.data_df

    return data_df


def regenerate_datasets(types=[DATA_FINAL_PICKLE]):
    for type in types:
        if check_if_file_exists('../data/{filename}'.format(filename=type)):
            remove_file('../data/{filename}'.format(filename=type))
        data = preprocess(type)
        print(list(data.columns))
        print(type, ' generated')


if __name__ == '__main__':
    regenerate_datasets(types=[DATA_FINAL_PICKLE, DATA_CLUSTERED_PICKLE, DATA_CLUSTERED_EXCEPT_PHARMFORM_FINAL_PICKLE])
