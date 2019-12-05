import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def change_time(x):
    try:
        hour, minute = str(x)[:5].split(":")
        new_time = int(hour) * 60 + int(minute)
    except ValueError:
        new_time = -1
    return new_time


def get_leaf(train_x, train_y, val_x):
    from sklearn.tree import DecisionTreeClassifier
    train_x, train_y, val_x = map(np.array, [train_x, train_y, val_x])
    train_x = train_x.reshape(-1, 1)
    train_y = train_y.reshape(-1, 1)
    val_x = val_x.reshape(-1, 1)
    m = DecisionTreeClassifier(min_samples_leaf=0.001, max_leaf_nodes=25)
    m.fit(train_x, train_y)
    return m.apply(val_x)


def change_leaf(model, data_x, num_leaf=32):
    y_pred = model.predict(data_x,  pred_leaf=True)
    transformed = np.zeros([len(y_pred), len(y_pred[0]) * num_leaf], dtype=np.int64)
    for i in range(0, len(y_pred)):
        temp = np.arange(len(y_pred[0])) * num_leaf + np.array(y_pred[i])
        transformed[i][temp] += 1
    return transformed


def print_metric(m, train_x, train_y, test_x, test_y):
    try:
        train_prob, test_prob = map(m.predict_proba, [train_x, test_x])
        if train_prob.shape[1] > 2:
            auc_train = roc_auc_score(train_y, np.mean(train_prob[:, 1::2], axis=1))
            auc_test = roc_auc_score(test_y, np.mean(test_prob[:, 1::2], axis=1))
        else:
            auc_train = roc_auc_score(train_y, train_prob[:, 1])
            auc_test = roc_auc_score(test_y, test_prob[:, 1])
    except ImportError:
        train_prob, test_prob = map(m.predict, [train_x, test_x])
        auc_train = roc_auc_score(train_y, train_prob)
        auc_test = roc_auc_score(test_y, test_prob)
    print("train auc:%.4f\t test auc:%.4f" % (auc_train, auc_test))
    return None



