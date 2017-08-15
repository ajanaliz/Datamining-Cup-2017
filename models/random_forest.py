import pandas as pd
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.externals import joblib

data = pd.read_pickle('data/unit_fixed.pkl')
parameter_grid = {
    'n_estimators': [100, 500, 1000]
}

forest = RandomForestRegressor(n_jobs=-1,
                               max_features=None)
train_X = data.drop(['count', 'revenue'],
                    axis=1).fillna(data.mean())
train_Y = data['revenue']
cross_validation = StratifiedKFold(train_Y, n_folds=5)
grid_search = GridSearchCV(forest,
                           param_grid=parameter_grid,
                           cv=cross_validation)
grid_search.fit(train_X, train_Y)
print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))

joblib.dump(grid_search, 'forest_model.pkl')
