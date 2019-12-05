from fun import *
from feature_selsector import FeatureSelector

# 特征工程思路: 
# 1.决策树分组
# 2.信用评分卡提取woe iv 等指标
target_name = "Label"
id_name = "ID"
train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")
print("训练集数量%s\t 测试集数量%s" % (len(train), len(test)))
train_label = pd.read_csv("./data/train_label.csv")
train = train_label.merge(train, left_on=id_name, right_on=id_name, how="left")
data = pd.concat([train, test], sort=False, ignore_index=True)
a = data.isnull().sum()
data["mising_num"] = data.iloc[:, 23:].isnull().sum(axis=1) == 0
data["mising_num"] = data["mising_num"].astype(np.int)
cat_features = ['企业类型', '登记机关', '企业状态', '邮政编码', '行业代码',
                '行业门类', '企业类别', '管辖机关', "经营范围"]
time_feature = ['经营期限自', '经营期限至', '成立日期', '核准日期', '注销时间']
# Part1 清理纯字符串字段
data[cat_features] = data[cat_features].astype(np.str)
data["经营范围"] = data["经营范围"].apply(lambda x: x.count(",") + 1)
data['邮政编码'] = data['邮政编码'].apply(lambda x: str(x).strip(".0"))
data["注册资本/投资总额"] = data['注册资本'] / data['投资总额']
data["是否全资"] = data['注册资本'] >= data['投资总额']
data["注册资本/投资总额"] = data['注册资本'] / data['投资总额']
data["企业缴税"] = np.sum(data[['增值税', '企业所得税', '印花税', '城建税', "教育费"]], axis=1)
data["增值税/企业缴税"] = data["增值税"] / data["企业缴税"]
data["企业所得税/企业缴税"] = data["企业所得税"] / data["企业缴税"]
data["印花税/企业缴税"] = data["印花税"] / data["企业缴税"]
data["教育费/企业缴税"] = data["教育费"] / data["企业缴税"]
data["城建税/企业缴税"] = data["城建税"] / data["企业缴税"]
data["教育费/城建税"] = data["教育费"] / data["城建税"]
data["教育费/增值税"] = data["教育费"] / data["增值税"]
data["企业所得税/投资总额"] = data["企业所得税"] / data["投资总额"]
data["增值税/经营范围"] = data["增值税"] / data["经营范围"]
data["注册资本/增值税"] = data["注册资本"] / data["增值税"]
for feature in cat_features + time_feature:
    if feature in time_feature:
        data[feature] = data[feature].apply(lambda x: change_time(x))
    data[feature] = LabelEncoder().fit_transform(data[feature].astype(np.str) + "_" + data["mising_num"].astype(np.str))
    data[feature] = get_leaf(data[feature][:-len(test)], data[target_name][:-len(test)], data[feature])
    count_table = pd.pivot_table(data,
                                 index=feature,
                                 columns=target_name,
                                 values=id_name,
                                 aggfunc="count",
                                 fill_value=0)
    count_table[[1, 0]] = count_table[[1, 0]] + 1
    count_table["rate_pos"] = count_table[1] / np.sum(count_table[1]) * 100
    count_table["rate_neg"] = count_table[0] / np.sum(count_table[0]) * 100
    count_table["efficiency"] = count_table["rate_pos"] - count_table["rate_neg"]
    count_table["rate"] = count_table[1] / (count_table[1] + count_table[0])
    count_table["woe"] = np.log(count_table["rate_pos"] / count_table["rate_neg"])
    count_table["iv"] = count_table["woe"] * count_table["efficiency"]
    count_table.drop([0, 1, "rate_pos", "rate_neg"], axis=1, inplace=True)
    count_table.columns = [feature + i for i in count_table.columns]
    data = data.merge(count_table.reset_index(), left_on=feature, right_on=feature, how="left").drop(feature, axis=1)
# 处理年初年末数
before_feature = list(set([var for var in data.columns if "年初数" in var]))
before_feature.sort()
after_feature = list(set([var for var in data.columns if "年末数" in var]))
after_feature.sort()
delta_feature = data[after_feature].values - data[before_feature].values
rate_feature = delta_feature / data[before_feature].values
delta_feature = pd.DataFrame(delta_feature, columns=[var.strip("年初数") + "delta" for var in before_feature])
rate_feature = pd.DataFrame(rate_feature, columns=[var.strip("年初数") + "rate" for var in before_feature])
data = data.join(delta_feature).join(rate_feature)
data = data.replace(np.inf, np.nan)
data = data.replace(-np.inf, np.nan)
data.fillna(-99999, inplace=True)
X = data.drop(["Label", "ID"], axis=1)[:-9578]
y = data.Label[:-9578]
X_val = data.drop(["Label", "ID"], axis=1)[-9578:]

# 删除不重要的变量
fs = FeatureSelector(data=X, labels=y)
fs.identify_zero_importance(task='classification', eval_metric='auc',
                            n_iterations=100, early_stopping=True)
fs.identify_low_importance(cumulative_importance=0.99)
X = X.drop(fs.ops['low_importance'], axis=1).values
X_val = X_val.drop(fs.ops['low_importance'], axis=1).values

params_initial = {
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 32,
    'max_bin': 50,
    'max_depth': 5,
    'min_child_samples': 50,
    'min_child_weight': 2,
    'n_jobs': -1,
    'colsample_bytree': 0.8,
}
prob_base = []
for i in range(100):
    print("\niter %02d" % i)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=i + 100)
    clf = LGBMClassifier(n_estimators=1000, **params_initial)
    clf.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose=0, early_stopping_rounds=50)
    print_metric(clf, x_train, y_train, x_test, y_test)
    x_train_tran = change_leaf(clf, x_train)
    x_test_tran = change_leaf(clf, x_test)
    X_val_tran = change_leaf(clf, X_val)
    lm = LogisticRegression(solver="lbfgs", penalty='l2', C=0.1, max_iter=1000, tol=0.001)
    lm.fit(x_train_tran, y_train)
    print_metric(lm, x_train_tran, y_train, x_test_tran, y_test)
    prob = MinMaxScaler().fit_transform(lm.predict_proba(X_val_tran)[:, 1::2])
    prob_base.append(prob)
Label_base = np.mean(np.hstack(prob_base), axis=1)
submission = pd.read_csv("./data/submission.csv")
submission.Label = MinMaxScaler(feature_range=(0.001, 0.999)).fit_transform(Label_base.reshape(-1, 1))
submission.to_csv("./result/lgb_base_3.csv", index=False)
best = pd.read_csv("./result/15541b11-4344-46af-a669-65e9814e9638.csv")
print(np.corrcoef(best.Label, submission.Label))
