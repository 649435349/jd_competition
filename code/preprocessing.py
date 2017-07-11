# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd
import os
import random
import csv
import copy
import datetime

from utils import *

from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
import xgboost as xgb

def alter_user():
    # 修改用户注册时间 改成到2016-04-15的距离
    os.chdir('..')
    with open('user.csv') as f1, open('user2.csv', 'w+') as f2:
        reader = csv.reader(f1)
        writer = csv.writer(f2)
        writer.writerow(['user_id', 'age', 'sex', 'user_lv_cd', 'user_reg_tm'])
        reader.next()
        for i in reader:
            i[-1] = (datetime.datetime(2016, 04, 25) - datetime.datetime.strptime(i[-1], '%Y-%m-%d')).days
            writer.writerow(i)

def get_brand_comment():
    #获得品牌的评论
    os.chdir('../')
    action=pd.read_csv('action_cate8.csv')
    skuid_brand=action[['sku_id','brand']].drop_duplicates()
    comment=pd.read_csv('comment.csv')
    brand_comment=pd.merge(comment,skuid_brand,on='sku_id')
    brand_comment['comment_num']=brand_comment['comment_num'].replace([0,1],2)
    brand_comment['comment_num']=brand_comment.apply(lambda x:x['comment_num']-2,axis=1)
    del brand_comment['sku_id']
    group=brand_comment.groupby(['brand','dt'])
    _brand_comment_comment_num=group['comment_num'].apply(np.mean,axis=0).reset_index()
    _brand_comment_bad_comment_rate=group['bad_comment_rate'].apply(np.mean,axis=0).reset_index()
    _brand_comment_has_bad_comment = group['has_bad_comment'].apply(lambda x:1 if np.max(x)>0 else 0).reset_index()
    brand_comment=pd.merge(_brand_comment_comment_num,_brand_comment_bad_comment_rate,on=['brand','dt'],how='outer')
    brand_comment = pd.merge(brand_comment, _brand_comment_has_bad_comment, on=['brand', 'dt'],
                             how='outer')
    brand_comment.to_csv('brand_comment.csv',index=False)

def get_average_time():
    os.chdir('../')

if __name__ == '__main__':
    get_brand_comment()