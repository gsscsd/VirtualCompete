#!/usr/bin/env python  
# encoding: utf-8  

""" 
@version: v1.0 
@author: Gsscsd   
@email: Gsscsd@qq.com 
@site: http://gsscsd.loan  
@file: main.py 
@time: 2017/9/6 
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Imputer
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier  
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


def cvPro(train,label):
    x_train,x_test,y_train,y_test = train_test_split(train,label,test_size = 0.3,random_state = 0)
    
    ##xgb
    xgb_clf = XGBClassifier()
    xgb_clf.fit(x_train,y_train)
    xgb_test_y = xgb_clf.predict_proba(x_test)
    xgb_y_lr_1 = [i[1] for i in xgb_test_y]

    loss = log_loss(y_test,xgb_y_lr_1)
    print("log_loss is :",loss)


def main(train_file_name,test_file_name,path):
    '''
    说明：lgb or xgb 默认参数:0.99 44 降维 成绩为：0.67447
    :param train_file_name:
    :param test_file_name:
    :param path:
    :return:
    '''

    print('Starting:...')

    last_col = 100
    median_col = 44
    train_data = pd.read_csv(train_file_name)
    test_data = pd.read_csv(test_file_name)
    one_hot_Fea = ['group1','group2','code_id']
    weight = train_data['weight']
    df_one_hot = train_data[one_hot_Fea]
    train_label = train_data['label']
    train_matrix_xgb = train_data.iloc[:,range(1,median_col)]
    test_matrix_xgb = test_data.iloc[:,range(1,median_col)]
    train_matrix_lgb = train_data.iloc[:,range(median_col,last_col)]
    test_matrix_lgb = test_data.iloc[:,range(median_col,last_col)]
    test_id = test_data['id'].astype("int64")
    test_id = [np.int64(i) for i in test_id]

    #数据清洗加归一化操作
    train_matrix_xgb.fillna(0)
    test_matrix_xgb.fillna(0)
    train_matrix_lgb.fillna(0)
    test_matrix_xgb.fillna(0)
    train_xgb_stand = Imputer().fit_transform(train_matrix_xgb)
    test_xgb_stand = Imputer().fit_transform(test_matrix_xgb)
    train_lgb_stand = Imputer().fit_transform(train_matrix_lgb)
    test_lgb_stand = Imputer().fit_transform(test_matrix_lgb)


    ##lgb模型
    lgb_clf = LGBMClassifier()
    lgb_clf.fit(train_lgb_stand,train_label)
    lgb_test_y = lgb_clf.predict_proba(test_lgb_stand)
    lgb_y_lr_1 = [i[1] for i in lgb_test_y]


    ###xgb模型
    xgb_clf = XGBClassifier()
    xgb_clf.fit(train_xgb_stand,train_label)
    xgb_test_y = xgb_clf.predict_proba(test_xgb_stand)
    xgb_y_lr_1 = [i[1] for i in xgb_test_y]

    pre_data = [(xgb_y_lr_1[i] * 0.5 + lgb_y_lr_1[i] * 0.5) for i in range(len(xgb_y_lr_1))]

    ##模型评估
#    cvPro(train_xgb_stand,train_label)

    #生成提交的数据
    subname = "xgb_sub_05.csv"
    sub_data_lr = pd.DataFrame(test_id)
    sub_data_lr.columns = ['id']
    sub_data_lr['proba'] = pd.Series(pre_data)
    # dats = sub_data_lr.loc[1:1000]
    # print(dats.shape)
    sub_data_lr.to_csv(path + subname,encoding='utf8',index = False)
    print('It is Ok.')

if __name__ == "__main__":
    test_file_name = "../data/stock_test_data_20171125.csv"
    train_file_name = "../data/stock_train_data_20171125.csv"
    path = "../data/"
    main(train_file_name,test_file_name,path)
