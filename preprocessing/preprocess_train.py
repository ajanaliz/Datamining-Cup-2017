from preprocessing.data_utils import *
import xgboost as xgb
from sklearn.naive_bayes import GaussianNB
from preprocessing.feature_classifier import *


class TrainProcessor:
    def __init__(self, items_df):
        self.data_df = None
        self._train_df = load_data('../data/train.csv')
        self._items_df = items_df

    def prepare(self, type=DATA_FINAL_PICKLE):
        # if check_if_file_exists('../data/{filename}'.format(filename=DATA_MERGED_PICKLE)):
        #     self.data_df = load_data('../data/{filename}'.format(filename=DATA_MERGED_PICKLE), mode='pkl')
        # else:
        #     self.data_df = merge_data(self._train_df, self._items_df)
        # pd.to_pickle(self.data_df, '../data/{filename}'.format(filename=DATA_MERGED_PICKLE))

        self.data_df = merge_data(self._train_df, self._items_df)

        self._add_weekday_feature()
        self._impute_competitor_price()
        self._impute_campaign_index()
        self._prepare_availability()
        self._add_discount_rate_feature()
        self._add_count_feature()
        self.data_df = self.data_df.drop('lineID', axis=1)
        # self._prepare_manufacturer()
        if type == DATA_CLUSTERED_PICKLE:
            self.data_df['pharmForm'] = cluster_feature(self.data_df, 'pharmForm')
        if type != DATA_FINAL_PICKLE:
            self.data_df['manufacturer'] = cluster_feature(self.data_df, 'manufacturer')
            self.data_df['group'] = cluster_feature(self.data_df, 'group')
            self.data_df['category'] = cluster_feature(self.data_df, 'category')
        else:
            self.data_df = self.data_df.drop('group', axis=1)

        pd.to_pickle(self.data_df, '../data/{filename}'.format(filename=type))

    def _add_weekday_feature(self):
        self.data_df['weekDay'] = self.data_df['day'] % 7

    def _impute_competitor_price(self):
        df = self.data_df[['lineID', 'day', 'weekDay', 'rrp', 'price', 'competitorPrice']]
        train = df[pd.notnull(df['competitorPrice'])]
        train = train[train['competitorPrice'] != 0]

        x = train[['day', 'weekDay', 'rrp', 'price']]
        y = train['competitorPrice']
        T_train_xgb = xgb.DMatrix(x, y)

        params = {"objective": "reg:linear", "booster": "gblinear"}
        gbm = xgb.train(dtrain=T_train_xgb, params=params)

        competitor_missing_ids = set(df['lineID']) - set(train['lineID'])
        na_rows = df[['day', 'weekDay', 'rrp', 'price']][df['lineID'].isin(competitor_missing_ids)]
        y_pred = gbm.predict(xgb.DMatrix(na_rows))
        self.data_df.ix[self.data_df['lineID'].isin(competitor_missing_ids), 'competitorPrice'] = y_pred

    def _impute_campaign_index(self):
        # fills missing values for campaignIndex
        campaign_missing = self.data_df[pd.isnull(self.data_df['campaignIndex'])]['lineID']
        adFlag_missing = self.data_df[self.data_df['adFlag'] == 0]['lineID']
        intersections = pd.Series(list(set(campaign_missing).intersection(set(adFlag_missing))))

        # These are lineIDs with missing campaignIndex and adFlag=0
        ind = self.data_df.lineID.isin(intersections.tolist())

        # To be filled with D
        self.data_df['campaignIndex'].fillna(self.data_df[ind]['campaignIndex'].fillna('D'), inplace=True)

        # Filling the rest using naive bayes
        train_data = self.data_df[pd.notnull(self.data_df['campaignIndex'])]
        test_data = self.data_df[pd.isnull(self.data_df['campaignIndex'])]

        naive_bayes_clf = GaussianNB()
        naive_bayes_clf.fit(train_data[['pid', 'manufacturer', 'rrp']],
                            train_data['campaignIndex'])
        predictions = naive_bayes_clf.predict(
            test_data[['pid', 'manufacturer', 'rrp']])

        self.data_df.ix[self.data_df['lineID'].isin(test_data['lineID']),
                        'campaignIndex'] = predictions
        # campaignIndex filled completely

        self.data_df = pd.concat([self.data_df, pd.get_dummies(self.data_df['campaignIndex'])], axis=1)
        self.data_df = self.data_df.drop('campaignIndex', 1)

    def _prepare_availability(self):
        mapping = {1: 'available_1', 2: 'available_2', 3: 'available_3', 4: 'available_4'}
        self.data_df = self.data_df.replace({'availability': mapping})
        self.data_df = pd.concat([self.data_df, pd.get_dummies(self.data_df['availability'])], axis=1)
        self.data_df = self.data_df.drop('availability', 1)

    def _add_discount_rate_feature(self):
        self.data_df['discountRate'] = 100 * (1 - self.data_df['price'] / self.data_df['rrp'])
        self.data_df['discountRate'] = self.data_df['discountRate'].round()

    def _add_count_feature(self):
        self.data_df['count'] = self.data_df.revenue / self.data_df.price

    def _prepare_manufacturer(self):
        self.data_df = pd.concat([self.data_df, pd.get_dummies(self.data_df['manufacturer'], prefix='man')], axis=1)
        self.data_df = self.data_df.drop('manufacturer', axis=1)
