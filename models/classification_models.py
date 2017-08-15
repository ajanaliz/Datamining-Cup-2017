import numpy as np  
seed = 100
np.random.seed(seed)
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import time

from sklearn.metrics import silhouette_score
from pandas import *
from sklearn import cluster

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation

# uncomment if you want to run on GPU(nothing needs to be done if you are using Tensorflow backend)
# import theano
# theano.config.device = 'gpu'
# theano.config.floatX = 'float32'

data = pd.read_pickle('../data/train_v2.pkl')
# data.encode('utf-8').strip()
# data.to_csv('train_v2.csv', encoding='utf-8')

def cluster_feature(train_merged2, feature_title):
    train_merged = train_merged2[~train_merged2[feature_title].isnull()]
    category_order_list = train_merged.groupby(feature_title)['order'].apply(list)
    category_basket_list = train_merged.groupby(feature_title)['basket'].apply(list)
    category_click_list = train_merged.groupby(feature_title)['click'].apply(list)
    category_price_list = train_merged.groupby(feature_title)['price'].apply(list)
    #
    indices = category_order_list.keys()
    category_list = {}
    i = 1
    for item in category_order_list:
        category_list[i] = []
        category_list[i].append(float(sum(item)) / len(item))
        i += 1
    #
    i = 1
    for item in category_basket_list:
        category_list[i].append(float(sum(item)) / len(item))
        i += 1
    #
    i = 1
    for item in category_click_list:
        category_list[i].append(float(sum(item)) / len(item))
        i += 1
    #
    i = 1
    for item in category_price_list:
        category_list[i].append(float(sum(item)) / len(item))
        i += 1
    #
    unique_values_len = len(indices)
    #
    my_data = []
    for i in range(1, unique_values_len + 1):
        my_data.append(category_list[i])
    #
    numpy_data = np.array(my_data)
    #
    # k_list = range(unique_values_len / 40, unique_values_len / 10)
    k_list = range(10, 40)
    #
    best_k = 0
    best_score = 0
    #
    k_score = []
    for k in k_list:
        print(k)
        k_means = cluster.KMeans(n_clusters=k)
        k_means.fit(numpy_data)
        labels = k_means.labels_
        #
        silhouette_avg = silhouette_score(numpy_data, labels)
        #
        if silhouette_avg > best_score:
            best_score = silhouette_avg
            best_k = k
        k_score.append([k, silhouette_avg])
    #
    k_means = cluster.KMeans(n_clusters=best_k)
    k_means.fit(numpy_data)
    labels = k_means.labels_
    #
    label_map = {}
    for i in range(unique_values_len):
        label_map[indices[i]] = labels[i] + 1
    #
    for key in label_map:
        train_merged2.loc[train_merged2[feature_title] == key, feature_title] = label_map[key]
    #
    return train_merged2

# ----------- calculate mean coeff of price to produce Revenue
def calc_mult(train):
    coef = (train['revenue']).groupby(train['pid']).mean()
    coef[:] = 0
    num = coef.copy()
    #
    for index, row in train.iterrows():
        if row['order'] == 1:
            x = row['revenue'] / float(row['price'])
            coef[row['pid']] += x
            num[row['pid']] += 1
        if index % 1000 == 0:
            print(index)
    #
    coef[coef == 0] = 1.0
    num[num == 0] = 1.0
    #
    price_multiplier = coef / num
    #
    return price_multiplier, coef, num, train['pid'].unique().tolist()

def add_mult(data, price_multiplier, pids):
    # setting it to 0.0 decreases MSE
    data['mult'] = 1.0
    #
    mult_list = data['mult'].tolist()
    #
    i = 0
    t = 0
    for index, row in data.iterrows():
        if i == 0 and index > 0:
            t = index
        i = i + 1
        if row['order'] == 1 and row['pid'] in pids:
            mult_list[index - t] = price_multiplier[row['pid']]
        if (index - t) % 1000 == 0:
            print(index - t)
    #
    se = pd.Series(mult_list)
    # df_temp = pd.DataFrame({'mult', mult_list})
    data['mult'] = se.values
    #
    return data


def show_results(y_te, y_pred, y_prob, comb, mults):
    mat = confusion_matrix(y_te, y_pred)
    print('\nConfusion Matrix')
    print(mat)
    xx = y_pred - y_te
    accuracy = xx[xx==0].shape[0]/float(xx.shape[0])
    print('Accuracy = ' + str(accuracy))
    #
    print('\nOrder')
    mse_continues = ((y_prob - y_te)**2).sum() / float(y_te.shape[0])
    print('MSE(Continuous Probability) = ' + str(mse_continues))
    mse = ((y_pred - y_te)**2).sum() / float(y_te.shape[0])
    print('MSE(Discrete 0,1) = ' + str(mse))
    #
    print('\nRevenue')
    # y_r = comb[100000:150000]['revenue'].copy()
    y_r = comb[(comb['day'] >= 32) & (comb['day'] < 63)]['revenue'].copy()
    mse_continues = ((y_prob*te['price'] - y_r)**2).sum() / float(y_te.shape[0])
    print('MSE(Continuous Probability) = ' + str(mse_continues))
    mse = ((y_pred*te['price'] - y_r)**2).sum() / float(y_te.shape[0])
    print('MSE(Discrete 0,1) = ' + str(mse))
    #
    print('\nRevenue with multiplication')
    # y_r = comb[100000:150000]['revenue'].copy()
    y_r = comb[(comb['day'] >= 32) & (comb['day'] < 63)]['revenue'].copy()
    mse_continues = ((y_prob*te['price']*mults - y_r)**2).sum() / float(y_te.shape[0])
    print('MSE(Continuous Probability) = ' + str(mse_continues))
    mse = ((y_pred*te['price']*mults - y_r)**2).sum() / float(y_te.shape[0])
    print('MSE(Discrete 0,1) = ' + str(mse))


# trainfile = '../data/train.csv'
# itemsfile = '../data/items.csv'
# testfile = '../data/test.csv'

# train = pd.read_csv(trainfile)
# test = pd.read_csv(testfile)
# items = pd.read_csv(itemsfile)

data.drop(['count'], axis=1,inplace=True)

# ----------------- decrease number of values in features
cluster_features = False
if cluster_features:
    data = cluster_feature(data, 'pharmForm')
    data = cluster_feature(data, 'manufacturer')
    data = cluster_feature(data, 'group')
    data = cluster_feature(data, 'category')
    np.save('pharmForm.npy', data['pharmForm'])
    np.save('manufacturer.npy', data['manufacturer'])
    np.save('group.npy', data['group'])
    np.save('category.npy', data['category'])
else:
    data['pharmForm'] = np.load('pharmForm.npy')
    data['manufacturer'] = np.load('manufacturer.npy')
    data['group'] = np.load('group.npy')
    data['category'] = np.load('category.npy')


# data = cluster_feature(data, 'pid')


#------------------------------ adding some features
data['discount'] = (data['rrp'] - data['price']) / data['rrp']
data['compete'] = (data['price'] - data['competitorPrice']) / data['price']

data['f1'] = (data['rrp'] - data['price'])
data['f2'] = (data['rrp'] / data['price'])
data['f3'] = (data['rrp'] - data['competitorPrice'])
data['f4'] = (data['price'] - data['competitorPrice'])
data['f5'] = (data['price'] / data['competitorPrice'])

#---- mean and sum per day
data['revenue_mean'] = 0.0
data['revenue_sum'] = 0.0
data['order_sum'] = 0.0
data['order_mean'] = 0.0
for i in range(1, 93):
    data.loc[data['day'] == i,'revenue_mean'] = data[data['day'] == i]['revenue'].mean()
    data.loc[data['day'] == i,'revenue_sum'] = data[data['day'] == i]['revenue'].sum()
    data.loc[data['day'] == i,'order_mean'] = data[data['day'] == i]['order'].mean()
    data.loc[data['day'] == i,'order_sum'] = data[data['day'] == i]['order'].sum()
    print(i)


# data.drop(['compete'], axis=1,inplace=True)
# data.drop(['discount'], axis=1,inplace=True)

# data.drop(['f1'], axis=1,inplace=True)
# data.drop(['f2'], axis=1,inplace=True)
# data.drop(['f3'], axis=1,inplace=True)
# data.drop(['f4'], axis=1,inplace=True)
# data.drop(['f5'], axis=1,inplace=True)

# tr = data[:100000].copy()
# te = data[100000:150000].copy()
tr = data[data['day'] < 32].copy()
te = data[(data['day'] >= 32) & (data['day'] < 63)].copy()
comb = pd.concat([tr,te])
# comb.to_csv('train_merged.csv', index=False)

price_multiplier, coef, num, pids = calc_mult(tr)
# tr = add_mult(tr, price_multiplier, pids)
te = add_mult(te, price_multiplier, pids)

mults = te['mult'].copy()
# tr.drop(['mult'], axis=1,inplace=True)
te.drop(['mult'], axis=1,inplace=True)



#----------------- you can run from here if you need to test another model
tr = data[data['day'] < 32].copy()
# tr = pd.concat([tr.copy(), tr[tr['order'] == 1].copy()])
te = data[(data['day'] >= 32) & (data['day'] < 63)].copy()
comb = pd.concat([tr,te])


availability_dummies  = pd.get_dummies(comb['availability'])
comb = pd.concat([comb, availability_dummies], axis=1)
comb.drop(['availability'], axis=1,inplace=True)


pharmForm_dummies  = pd.get_dummies(comb['pharmForm'])
# pharmForm_dummies.drop(['S'], axis=1, inplace=True)
comb = pd.concat([comb, pharmForm_dummies], axis=1)
comb.drop(['pharmForm'], axis=1,inplace=True)


manufacturer_dummies  = pd.get_dummies(comb['manufacturer'])
comb = pd.concat([comb, manufacturer_dummies], axis=1)
comb.drop(['manufacturer'], axis=1,inplace=True)

group_dummies  = pd.get_dummies(comb['group'])
comb = pd.concat([comb, group_dummies], axis=1)
comb.drop(['group'], axis=1,inplace=True)

# content_dummies  = pd.get_dummies(comb['content'])
# comb = pd.concat([comb, content_dummies], axis=1)
# comb.drop(['content'], axis=1,inplace=True)

# unit_dummies  = pd.get_dummies(comb['unit'])
# comb = pd.concat([comb, unit_dummies], axis=1)
# comb.drop(['unit'], axis=1,inplace=True)

salesIndex_dummies  = pd.get_dummies(comb['salesIndex'])
comb = pd.concat([comb, salesIndex_dummies], axis=1)
comb.drop(['salesIndex'], axis=1,inplace=True)

category_dummies  = pd.get_dummies(comb['category'])
comb = pd.concat([comb, category_dummies], axis=1)
comb.drop(['category'], axis=1,inplace=True)

campaignIndex_dummies  = pd.get_dummies(comb['campaignIndex'])
comb = pd.concat([comb, campaignIndex_dummies], axis=1)
comb.drop(['campaignIndex'], axis=1,inplace=True)

# pid_dummies  = pd.get_dummies(comb['pid'])
# comb = pd.concat([comb, pid_dummies], axis=1)
comb.drop(['pid'], axis=1,inplace=True)

weekDay_dummies  = pd.get_dummies(comb['weekDay'])
comb = pd.concat([comb, weekDay_dummies], axis=1)
# comb.drop(['weekDay'], axis=1,inplace=True)
# comb.drop(['weekDay'], axis=1,inplace=True)

# tr = comb[0:100000].copy()
# te = comb[100000:150000].copy()
tr = comb[comb['day'] < 32].copy()
te = comb[(comb['day'] >= 32) & (comb['day'] < 63)].copy()

# tr.loc[tr['click'] == 1, 'order'] = 0
# tr.loc[tr['order'] == 1, 'order'] = 2
# tr.loc[tr['basket'] == 1, 'order'] = 1
# te.loc[te['click'] == 1, 'order'] = 0
# te.loc[te['order'] == 1, 'order'] = 2
# te.loc[te['basket'] == 1, 'order'] = 1
# tr.drop(['pid'], axis=1,inplace=True)
# te.drop(['pid'], axis=1,inplace=True)
tr.drop(['lineID'], axis=1,inplace=True)
te.drop(['lineID'], axis=1,inplace=True)
tr.drop(['click'], axis=1,inplace=True)
te.drop(['click'], axis=1,inplace=True)
tr.drop(['basket'], axis=1,inplace=True)
te.drop(['basket'], axis=1,inplace=True)
tr.drop(['revenue'], axis=1,inplace=True)
te.drop(['revenue'], axis=1,inplace=True)

# tr['competitorPrice'] = tr['price'] - tr['competitorPrice']
# te['competitorPrice'] = te['price'] - te['competitorPrice']

# tr = tr.drop('unit', 1)
# te = te.drop('unit', 1)

# svd = TruncatedSVD(n_components=100)
# pca = PCA(n_components=100)
# lsa = make_pipeline(svd, Normalizer(copy=False))
lsa = make_pipeline(Normalizer(copy=False))
weight_1 = tr[tr['order'] == 0].shape[0]/float(tr[tr['order'] == 1].shape[0])

# model = KNeighborsClassifier(10,verbose=1)
# model = DecisionTreeClassifier(max_depth=5)
# model = MLPClassifier(alpha=0.1,verbose=1)
# model = RandomForestClassifier(max_depth=10, n_estimators=1000, max_features=10, verbose=1)
model = RandomForestClassifier(n_estimators=100, verbose=1, class_weight={0:1., 1:weight_1})
# model = GaussianNB(,verbose=1)
# model = linear_model.LogisticRegression(verbose=1)
# model = svm.SVC(kernel='sigmoid', gamma=5,C=1,verbose=1)

DNN = False
if DNN:
    model = Sequential()
    model.add(Dense(2000, input_dim=tr.shape[1]-1, kernel_initializer='normal', activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(1000, input_dim=1000, kernel_initializer='normal', activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


X = tr.ix[:, tr.columns != 'order']
X = lsa.fit_transform(X)
y = tr['order']
t1 = time.time()
if DNN:
    model.fit(X, y, batch_size=32, epochs=10, class_weight = {0:1., 1:weight_1}) 
    # model.save('NN'
else:
    model.fit(X, y)


X_te = te.ix[:, te.columns != 'order']
X_te = lsa.transform(X_te)
y_te = te['order']
t3 = time.time()

if DNN:
    y_pred = model.predict_classes(X_te)
    y_pred = y_pred[:,-1]
else:
    y_pred = model.predict(X_te)

y_prob = model.predict_proba(X_te)
y_prob = y_prob[:,-1]


y_pred[data[(data['day'] >= 32) & (data['day'] < 63)]['availability'] == 4] = 0
y_prob[data[(data['day'] >= 32) & (data['day'] < 63)]['availability'] == 4] = 0

show_results(y_te, y_pred, y_prob, comb, mults)
# mults3 = mults2.copy()


#------------------- checking different cutoffs
mse_cont_list = []
mse_list = []
acc_list = []
for i in range(0,20):
    y_pred2 = y_prob.copy()
    y_pred2[y_pred2 > i * 0.05] = 1
    y_pred2[y_pred2 <= i * 0.05] = 0
    print(y_pred2.mean())
    y_r = comb[(comb['day'] >= 32) & (comb['day'] < 63)]['revenue'].copy()
    mse_continues = ((y_prob*te['price']*mults - y_r)**2).sum() / float(y_te.shape[0])
    mse_cont_list.append(mse_continues)
    # print('MSE(Continuous Probability) = ' + str(mse_continues))
    mse = ((y_pred2*te['price']*mults - y_r)**2).sum() / float(y_te.shape[0])
    mse_list.append(mse)
    xx = y_pred2 - y_te
    accuracy = xx[xx==0].shape[0]/float(xx.shape[0])
    acc_list.append(accuracy*100)
    # print('MSE(Discrete 0,1) = ' + str(mse))

plt.plot(mse_cont_list, '-b', mse_list, '-r', acc_list, '-k')
plt.show()
#--------- uncomment to save model ( keras )
# xx = y_pred - y_te
# accuracy = xx[xx==0].shape[0]/float(xx.shape[0])
# model.save('model_' + str(accuracy) + 'features_added')


#----- loading model
# from keras.models import load_model
# model = load_model('model_95')




#--
# np.save('model_' + str(accuracy) + 'features_added', model)
# model = np.load('filename')
