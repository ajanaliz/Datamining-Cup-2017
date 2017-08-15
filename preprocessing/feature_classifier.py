from sklearn import cluster
from sklearn.metrics import silhouette_score
import numpy as np

seed = 100
np.random.seed(seed)


def cluster_feature(train_merged2, feature_title):
    train_merged = train_merged2[~train_merged2[feature_title].isnull()]
    category_order_list = train_merged.groupby(feature_title)['order'].apply(list)
    category_basket_list = train_merged.groupby(feature_title)['basket'].apply(list)
    category_click_list = train_merged.groupby(feature_title)['click'].apply(list)
    category_price_list = train_merged.groupby(feature_title)['price'].apply(list)

    indices = category_order_list.keys()
    category_list = {}
    for index, item in enumerate(category_order_list):
        category_list[index] = []
        category_list[index].append(float(sum(item)) / len(item))

    for index, item in enumerate(category_basket_list):
        category_list[index].append(float(sum(item)) / len(item))

    for index, item in enumerate(category_click_list):
        category_list[index].append(float(sum(item)) / len(item))

    for index, item in enumerate(category_price_list):
        category_list[index].append(float(sum(item)) / len(item))

    unique_values_len = len(indices)
    my_data = []
    for i in range(1, unique_values_len + 1):
        my_data.append(category_list[i])

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
    return train_merged2[feature_title]
