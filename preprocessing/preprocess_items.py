from preprocessing.data_utils import *


class ItemsPreprocessor:
    def __init__(self):
        self.items_df = load_data('../data/items.csv')

    def _prepare_pharm_form(self, get_dummy=True):
        # uppercase all pharmForm values
        self.items_df['pharmForm'] = self.items_df['pharmForm'].str.upper()

        items_with_dummy_pharmForms = pd.concat(
            [self.items_df, pd.get_dummies(self.items_df['pharmForm'])], axis=1)
        counted_pharmForm_per_group = items_with_dummy_pharmForms.groupby('group').agg('sum')
        counted_pharmForm_per_group = counted_pharmForm_per_group.drop(
            ['pid', 'manufacturer', 'genericProduct', 'salesIndex', 'category', 'rrp'], 1)
        pharm_of_group = counted_pharmForm_per_group.idxmax(axis=1)
        filled = self.items_df[['group', 'pharmForm']].apply(
            lambda x: pharm_of_group.get(x[0]) if pd.isnull(x[1]) else x[1], axis=1)
        self.items_df['pharmForm'] = filled
        if (get_dummy):
            # extract pharmForm values as binary feature and adding them to dataset
            self.items_df = pd.concat([self.items_df, pd.get_dummies(self.items_df['pharmForm'])], axis=1)
            self.items_df = self.items_df.drop('pharmForm', axis=1)

    def _prepare_content(self):
        def extract_numbers_from_content(input):
            x_index = input.find('X')
            if input == 'L   125':
                return 1, 1, 125
            if x_index == -1:
                if input == 'PAK':
                    return 1, 1, 1
                return 1, 1, input
            second_part = input[x_index + 1: len(input)]
            x_second_index = second_part.find('X')
            if x_second_index == -1:
                return 1, input[0: x_index], second_part
            return input[0: x_index], second_part[0: x_second_index], second_part[x_second_index + 1: len(second_part)]

        # split count of packs and amount of each to separate columns
        extracted_numbers = self.items_df['content'].apply(extract_numbers_from_content)
        extracted_numbers = pd.DataFrame(extracted_numbers.tolist(), columns=['content_1', 'content_2', 'content_3'],
                                         index=extracted_numbers.index)
        extracted_numbers['content_1'] = pd.to_numeric(extracted_numbers['content_1'])
        extracted_numbers['content_2'] = pd.to_numeric(extracted_numbers['content_2'])
        extracted_numbers['content_3'] = pd.to_numeric(extracted_numbers['content_3'])
        self.items_df = pd.concat([self.items_df, extracted_numbers], axis=1)
        self.items_df = self.items_df.drop('content', 1)
        return self.items_df

    def _prepare_unit(self):
        unit_map = {
            'KG': 1000,
            'ST': 6350,
            'M': 100,
            'L': 1000,
            'G': 1,
            'CM': 1,
            'ML': 1,
            'P': 1,
        }

        def unit_converter(row):
            return row['content_3'] * unit_map[row['unit']]

        self.items_df['content_3'] = self.items_df.apply(unit_converter, axis=1)
        mapping = {'KG': 'G', 'ST': 'G', 'L': 'ML', 'M': 'CM'}
        self.items_df = self.items_df.replace({'unit': mapping})
        self.items_df = pd.concat([self.items_df, pd.get_dummies(self.items_df['unit'])], axis=1)
        self.items_df = self.items_df.drop('unit', 1)
        return self.items_df

    def _impute_category(self):
        x_train = self.items_df[pd.notnull(self.items_df['category'])]
        y_train = x_train['category']
        pids = set(self.items_df['pid']) - set(x_train['pid'])
        x_train = x_train[
            ["manufacturer", "content_1", "content_2", "content_3", "G", "ML", "CM", "P", "genericProduct",
             "salesIndex",
             "rrp"]]
        from sklearn.neighbors import KNeighborsClassifier
        classifier = KNeighborsClassifier(n_neighbors=8, weights='distance', n_jobs=3)
        classifier.fit(x_train, y_train)
        x_test = self.items_df[self.items_df['pid'].isin(pids)]
        x_test = x_test[
            ["manufacturer", "content_1", "content_2", "content_3", "G", "ML", "CM", "P", "genericProduct",
             "salesIndex",
             "rrp"]]
        y_pred = classifier.predict(x_test)
        self.items_df.ix[self.items_df['pid'].isin(pids), 'category'] = y_pred
        return self.items_df

    def _prepare_group(self):
        def _get_first_char(string):
            return int(string[0])

        self.items_df['omitted_group'] = self.items_df['group'].apply(_get_first_char)

    def prepare(self, get_dummy=True):
        self._prepare_pharm_form(get_dummy=get_dummy)
        self._prepare_content()
        self._prepare_unit()
        self._impute_category()
        self._prepare_group()
