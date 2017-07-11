# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd
import os
import random
import csv
import copy
import datetime
import multiprocessing

from utils import *

from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
import xgboost as xgb


def analyse_user_xiguan(_type):
    #这个写法实在是太垃圾了
    global action, action_type4
    cnt=0
    # 分析用户习惯
    action_type4['farest_type{}'.format(_type)] = np.NAN
    action_type4['nearest_type{}'.format(_type)] = np.NAN
    for i in action_type4.index:
        cnt+=1
        print _type,cnt
        tmp = action[(action['user_id'] == action_type4.loc[i,'user_id']) & (
                        action['sku_id'] == action_type4.loc[i,'sku_id']) & (
                         action['time'] <= action_type4.loc[i,'time']) & (
                         action['type'] == _type)]

        if tmp.shape[0]:
            action_type4.loc[i, 'farest_type{}'.format(_type)] = \
                (datetime.datetime.strptime(
                    datetime.datetime.strptime(action_type4.loc[i, 'time'], '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d'),'%Y-%m-%d')
                 -
                 datetime.datetime.strptime(
                     datetime.datetime.strptime(tmp.iloc[0]['time'], '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d'),'%Y-%m-%d')).days


            action_type4.loc[i,'nearest_type{}'.format(_type)] = \
                (datetime.datetime.strptime(
                    datetime.datetime.strptime(action_type4.loc[i, 'time'], '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d'),'%Y-%m-%d')
                 -
                 datetime.datetime.strptime(
                     datetime.datetime.strptime(tmp.iloc[-1]['time'], '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d'),'%Y-%m-%d')).days

    return action_type4[['farest_type{}'.format(_type), 'nearest_type{}'.format(_type)]]


if __name__ == '__main__':
    print 'cate8', datetime.datetime.now()
    os.chdir('../')
    action=pd.read_csv('action_cate8.csv')
    action_type = {}
    action_type['action_type4'] = action[action['type'] == 4].copy()
    action_type['action_type1'] = action[action['type'] == 1].copy()
    action_type['action_type2'] = action[action['type'] == 2].copy()
    action_type['action_type5'] = action[action['type'] == 5].copy()
    action_type['action_type6'] = action[action['type'] == 6].copy()
    res=action_type['action_type4'][['user_id','sku_id','time']].copy()

    for i in [1,2,5,6]:
        for j in [('farest','first'),('nearest','last')]:
            tmp=pd.merge(action_type['action_type4'],action_type['action_type{}'.format(i)],on=['user_id','sku_id','brand'],how='left')
            tmp1=tmp[tmp['time_x']>tmp['time_y']].drop_duplicates(subset=['user_id','sku_id','time_x'],keep=j[1]).copy()
            tmp=tmp1[['time_x','time_y']].copy().apply(convert_datetime_day_gap,axis=1)
            tmp=pd.concat([tmp1[['user_id','sku_id','time_x']],tmp],axis=1).copy()
            tmp.rename(columns={0: '{}_type{}'.format(j[0], i),'time_x':'time'},inplace=True)
            res=pd.merge(res,tmp,on=['user_id','sku_id','time'],how='left').fillna(-1)

    res.to_csv('action_cate8_type4_analyse.csv',index=None)

    print datetime.datetime.now()