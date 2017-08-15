from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from sklearn import preprocessing
from sklearn.metrics import classification_report, precision_score, recall_score, auc, roc_curve, mean_squared_error
import pandas as pd
import numpy as np
import xgboost as xgb

from preprocessing.data_utils import *

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


class NeuralNets:
    def __init__(self):
        self.load_and_split()

    def load_and_split(self):
        data_df = load_data('../data/{filename}'.format(filename=DATA_FINAL_PICKLE),
                            drop_cols=['count', 'click', 'basket', 'revenue'],
                            mode='pkl')
        self.train_df, self.val_df, self.test_df = split_train_val_test(data_df)
        self.train_df.drop('day', axis=1)
        self.val_df.drop('day', axis=1)
        self.test_df.drop('day', axis=1)

    def preprocess_train(self):
        zero_df = self.train_df[self.train_df['order'] == 0]
        one_df = self.train_df[self.train_df['order'] == 1]

        ratio = round(zero_df.shape[0] / one_df.shape[0])

        for zero_df in split_abundant_target(zero_df, ratio):
            yield data_target(pd.concat([one_df, zero_df]), 'order')

    @staticmethod
    def preprocess_data(df):
        df, target = data_target(df, 'order')
        data = preprocessing.normalize(df, axis=1)
        return data, target

    @staticmethod
    def simple_nn(train_data, train_target, val_data, val_target, class_weight=None):
        model = Sequential()
        model.add(Dense(train_data.shape[1], activation='relu', input_dim=train_data.shape[1]))
        model.add(Dropout(0.5))
        model.add(Dense(train_data.shape[1], activation='relu', input_dim=100))
        model.add(Dropout(0.3))
        model.add(Dense(train_data.shape[1], activation='relu', input_dim=100))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='rmsprop',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        model.fit(train_data, train_target,
                  validation_data=(val_data, val_target),
                  epochs=20,
                  class_weight=class_weight,
                  batch_size=2048)
        return model

    @staticmethod
    def complex_nn():
        pass

    # def complex_nn(self):
    #     print('complex nn')
    #     model = Sequential([
    #         Dense(256, activation='relu', input_dim=self.train_df.shape[1]),
    #         Dropout(0.8),
    #         Dense(128, activation='relu'),
    #         Dropout(0.4),
    #         Dense(32, activation='relu'),
    #         Dense(1, activation='sigmoid'),
    #     ])
    #     model.compile(optimizer='rmsprop',
    #                   loss='binary_crossentropy',
    #                   metrics=['accuracy'])
    #
    #     class_weights = class_weight.compute_class_weight('auto', np.unique(self.train_target), self.train_target)
    #     print('class weights')
    #     print(class_weights)
    #     model.fit(self.train_df, self.train_target,
    #               validation_data=(self.val_df, self.val_target),
    #               epochs=1)
    #     model.save('complex_nn_v1.h5')
    #     return model

    @staticmethod
    def describe(target_true, target_pred):
        report = classification_report(target_true, target_pred)
        print()
        print('classification report')
        print(report)
        precision = precision_score(target_true, target_pred)
        print('precision score')
        print(precision)
        recall = recall_score(target_true, target_pred)
        print('recall score')
        print(recall)
        fpr, tpr, thresholds = roc_curve(target_true, target_pred, pos_label=2)
        auc_metric = auc(fpr, tpr)
        print('auc')
        print(auc_metric)


if __name__ == '__main__':
    nn = NeuralNets()

    val_data, val_target = nn.preprocess_data(nn.val_df)
    test_data, test_target = nn.preprocess_data(nn.test_df)

    # models_weights = []
    # for index, train_df, train_target in enumerate(nn.preprocess_train()):
    #     train_data = preprocessing.normalize(train_df, axis=1)
    #     model = nn.simple_nn(train_data, train_target, val_data, val_target)
    #     model.save_weights('simple_nn_v{model_number}.h5'.format(model_number=index))
    #     models_weights.append(model.get_weights())
    #     print('model')
    #     print(model.summary())
    #     print()
    #     print('target preds')
    #     target_pred = model.predict_classes(test_data)
    #     nn.describe(test_target, target_pred)

    train_data, train_target = nn.preprocess_data(nn.train_df)

    # model = nn.simple_nn(train_data, train_target, val_data, val_target)
    # print()
    # print('target preds')
    # target_pred = model.predict_classes(val_data)
    # nn.describe(val_target, target_pred)
    #
    model = nn.simple_nn(train_data, train_target, val_data, val_target, class_weight={0: 1., 1: 2.5})
    model.save('best_best.h5')
    print()
    print('target preds')
    val_target_pred = model.predict_classes(val_data)
    val_target_pred_prob = model.predict_proba(val_data)
    nn.describe(val_target, val_target_pred)

    # Regression part
    data = load_data(path='../data/{filename}'.format(filename=DATA_CLUSTERED_EXCEPT_PHARMFORM_FINAL_PICKLE),
                     drop_cols=['basket', 'click'
                                # 'competitorPrice',
                                # 'manufacturer',
                                # 'salesIndex',
                                # 'category',
                                # 'rrp',
                                # 'content_1', 'content_2', 'content_3',
                                # 'CM', 'G', 'ML', 'P',
                                # 'omitted_group',
                                ],
                     mode='pkl')
    train_df, val_df, test_df = split_train_val_test(data)
    train_df.drop('revenue', axis=1)
    train_df.drop('day', axis=1)
    val_df.drop('day', axis=1)
    test_df.drop('day', axis=1)

    train_data, train_target = data_target(train_df, 'count')
    val_data, val_target = data_target(val_df, 'count')
    val_data['order'] = val_target_pred_prob
    # print(val_target_pred)
    # print('------------------space----------------------')
    # print(val_data['order'])
    # val_df['order'] = val_order
    # train_order[train_order < 0] = 0
    # val_order[val_order < 0] = 0

    # x = train_df.drop(['revenue', 'count', 'day'], axis=1)
    # y = train_df['count']
    T_train_xgb = xgb.DMatrix(train_data, train_target)

    params = {"objective": "reg:linear", "booster": "gblinear", "eval_metric": "rmse",
              # "eta": 0.6,
              # "gamma": 100,
              # "max_depth": 18,
              }
    gbm = xgb.train(dtrain=T_train_xgb, params=params)
    val_pred = gbm.predict(xgb.DMatrix(val_data))
    val_pred = pd.Series(val_pred)
    val_pred[val_pred < 0] = 0
    print(val_pred.describe())
    val_pred = val_pred.values * val_df['price'].values * val_df['order'].values
    print(mean_squared_error(val_df['revenue'], val_pred))

    print('test')
    test_pred = gbm.predict(xgb.DMatrix(val_data))
    test_pred = pd.Series(test_pred)
    test_pred[test_pred < 0] = 0
    print(test_pred.describe())
    test_pred = test_pred.values * val_df['price'].values * val_df['order'].values
    print(mean_squared_error(val_df['revenue'], test_pred))
