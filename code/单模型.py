#!/usr/bin/env python  
# encoding: utf-8  

""" 
@version: v1.0 
@author: Gsscsd   
@email: Gsscsd@qq.com 
@site: http://gsscsd.loan  
@file: test.py 
@time: 2017/9/6 
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Imputer
import numpy as np
from sklearn.decomposition import PCA
from xgboost import XGBClassifier

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

    last_col = 44
    train_data = pd.read_csv(train_file_name)
    test_data = pd.read_csv(test_file_name)
    train_label = train_data['label']
    train_matrix = train_data.iloc[:,range(1,last_col)]
    test_matrix = test_data.iloc[:,range(1,last_col)]
    test_id = test_data['id'].values
    test_id = [np.int64(i) for i in test_id]

    #数据清洗加归一化操作
    train_matrix.fillna(0)
    test_matrix.fillna(0)
    train_data_stand = Imputer().fit_transform(train_matrix)
    test_data_stand = Imputer().fit_transform(test_matrix)

    ## 降维处理
#    pca = PCA(n_components=0.99)
#    new_train = pca.fit_transform(train_data_stand)
#    new_test = pca.transform(test_data_stand)

#    ##lgb模型
#    lgb_clf = LGBMClassifier()
#    lgb_clf.fit(new_train,train_label)
#    lgb_test_y = lgb_clf.predict_proba(new_test)
#    lgb_y_lr_1 = [i[1] for i in lgb_test_y]

    ##模型评估
#    cvPro(train_xgb_stand,train_label)

    ##xgb模型
    xgb_clf = XGBClassifier()
    xgb_clf.fit(train_data_stand,train_label)
    xgb_test_y = xgb_clf.predict_proba(test_data_stand)
    xgb_y_lr_1 = [i[1] for i in xgb_test_y]

    #生成提交的数据
    subname = "xgb_submit.csv"
    sub_data_lr = pd.DataFrame(test_id)
    sub_data_lr.columns = ['id']
    sub_data_lr['proba'] = pd.Series(xgb_y_lr_1)
    sub_data_lr.to_csv(path + subname,encoding='utf8',index = False)
    print('It is Ok.')

if __name__ == "__main__":
    test_file_name = "../data/stock_test_data_20171125.csv"
    train_file_name = "../data/stock_train_data_20171125.csv"
    path = "../data/"
    main(train_file_name,test_file_name,path)
