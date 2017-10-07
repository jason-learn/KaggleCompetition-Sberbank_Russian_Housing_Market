# *coding=utf-8*

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn import model_selection, preprocessing

macro_cols = ["balance_trade", "balance_trade_growth", "eurrub", "average_provision_of_build_contract",
              "micex_rgbi_tr", "micex_cbi_tr", "deposits_rate", "mortgage_value", "mortgage_rate",
              "income_per_cap", "rent_price_4+room_bus", "museum_visitis_per_100_cap", "apartment_build"]

df_train = pd.read_csv("../input/train.csv", parse_dates=['timestamp'])
df_test = pd.read_csv("../input/test.csv", parse_dates=['timestamp'])
df_macro = pd.read_csv("../input/macro.csv", parse_dates=['timestamp'], usecols=['timestamp'] + macro_cols)
df_train.head()


ylog_train_all = np.log1p(df_train['price_doc'].values)
id_test = df_test['id']

#去除id和要预测的价格
df_train.drop(['id', 'price_doc'], axis=1, inplace=True)
df_test.drop(['id'], axis=1, inplace=True)

# Build df_all = (df_train+df_test).join(df_macro)
num_train = len(df_train)
df_all = pd.concat([df_train, df_test])
df_all = pd.merge(df_all, df_macro, on='timestamp', how='left')
print(df_all.shape)

# Add month-year
month_year = (df_all.timestamp.dt.month + df_all.timestamp.dt.year * 100)
month_year_cnt_map = month_year.value_counts().to_dict()
df_all['month_year_cnt'] = month_year.map(month_year_cnt_map)

# Add week-year count
week_year = (df_all.timestamp.dt.weekofyear + df_all.timestamp.dt.year * 100)
week_year_cnt_map = week_year.value_counts().to_dict()
df_all['week_year_cnt'] = week_year.map(week_year_cnt_map)

# Add month and day-of-week
df_all['month'] = df_all.timestamp.dt.month
df_all['dow'] = df_all.timestamp.dt.dayofweek

# Other feature engineering
df_all['rel_floor'] = df_all['floor'] / df_all['max_floor'].astype(float)
df_all['rel_kitch_sq'] = df_all['kitch_sq'] / df_all['full_sq'].astype(float)

# Remove timestamp column (may overfit the model in train)
df_all.drop(['timestamp'], axis=1, inplace=True)

#标签预处理  标签编码  非数值类型的标签到数值类型标签的转化 比如['a','a','b']transform之后[0,0,1]
for c in df_all.columns:
    if df_all[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(df_all[c].values))
        df_all[c] = lbl.transform(list(df_all[c].values))
        # x_train.drop(c,axis=1,inplace=True)



# Deal with categorical values
df_numeric = df_all.select_dtypes(exclude=['object'])
df_obj = df_all.select_dtypes(include=['object']).copy()

for c in df_obj:
    df_obj[c] = pd.factorize(df_obj[c])[0]

df_values = pd.concat([df_numeric, df_obj], axis=1)

# Convert to numpy values
X_all = df_values.values
print(X_all.shape)



# Create a validation set, with last 20% of data
num_val = int(num_train * 0.2)

X_train_all = X_all[:num_train] #所有训练数据
X_train = X_all[:num_train - num_val] #训练数据
X_val = X_all[num_train - num_val:num_train] #验证数据
ylog_train = ylog_train_all[:-num_val] #训练数据标签
ylog_val = ylog_train_all[-num_val:]   #验证数据标签

X_test = X_all[num_train:]  #所有测试数据

df_columns = df_values.columns

print('X_train_all shape is', X_train_all.shape)
print('X_train shape is', X_train.shape)
print('y_train shape is', ylog_train.shape)
print('X_val shape is', X_val.shape)
print('y_val shape is', ylog_val.shape)
print('X_test shape is', X_test.shape)

#数据接口加载数据到DMatrix对象
dtrain_all = xgb.DMatrix(X_train_all, ylog_train_all, feature_names=df_columns) #所有数据
dtrain = xgb.DMatrix(X_train, ylog_train, feature_names=df_columns)  #所有数据0.8训练
dval = xgb.DMatrix(X_val, ylog_val, feature_names=df_columns)    #所有数据0.2验证
dtest = xgb.DMatrix(X_test, feature_names=df_columns)      #所有测试数据

xgb_params = {
    'eta': 0.05,  #学习率
    'max_depth': 5,#构建树的深度，越大越容易拟合
    'subsample': 1.0, #随机采样训练样本
    'colsample_bytree': 0.7,#生成树的列采样
    'objective': 'reg:linear',#线性回归
    'eval_metric': 'rmse',
    'silent': 1  #设置成1则没有运行信息输出，最好是设置为0
}


# Uncomment to tune XGB `num_boost_rounds`
partial_model = xgb.train(xgb_params, dtrain, num_boost_round=1000, evals=[(dval, 'val')],
                          early_stopping_rounds=20, verbose_eval=20)

num_boost_round = partial_model.best_iteration

fig, ax = plt.subplots(1, 1, figsize=(8, 16))
# xgb.plot_importance(partial_model, max_num_features=50, height=0.5, ax=ax)

num_boost_round = partial_model.best_iteration

model = xgb.train(dict(xgb_params, silent=0), dtrain_all, num_boost_round=num_boost_round)

fig, ax = plt.subplots(1, 1, figsize=(8, 16))
# xgb.plot_importance(model, max_num_features=50, height=0.5, ax=ax)

ylog_pred = model.predict(dtest)
y_pred = np.exp(ylog_pred) - 1

df_sub = pd.DataFrame({'id': id_test, 'price_doc': y_pred})

#df_sub.to_csv('sub.csv', index=False)


ddtrain = xgb.DMatrix(X_train_all, ylog_train_all)
ddtest = xgb.DMatrix(X_test)

cv_output = xgb.cv(xgb_params, ddtrain, num_boost_round=1000, early_stopping_rounds=20,
                   verbose_eval=50, show_stdv=False,nfold=10)
cv_output[['train-rmse-mean', 'test-rmse-mean']].plot()

num_boost_rounds = len(cv_output)
model = xgb.train(dict(xgb_params, silent=0), ddtrain, num_boost_round=num_boost_rounds)

fig, ax = plt.subplots(1, 1, figsize=(8, 13))
#xgb.plot_importance(model, max_num_features=50, height=0.5, ax=ax)

y_predict = model.predict(ddtest)

y_pred = np.exp(y_predict) - 1

df_sub = pd.DataFrame({'id': id_test, 'price_doc': y_pred})

df_sub.to_csv('../../subtest.csv', index=False)
