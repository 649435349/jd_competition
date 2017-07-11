# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd
import os
import random
import csv
import copy
import datetime
import multiprocessing
import pymysql

from utils import *

from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
import xgboost as xgb

insert_sql = 'insert into action(user_id,sku_id,time,model_id,type,cate,type) values(%s,%s,%s,%s,%s,%s,%s)'
index_sql='alter table action add index index1 (user_id,sku_id,time,type)'

def insert(x):
    cur.execute(insert_sql,x)
    conn.commit()

if __name__=='__main__':
    os.chdir('../raw_data')
    action = [tuple(i) for i in list(pd.read_csv('action.csv').fillna('').values)]

    conn = pymysql.connect(host='127.0.0.1',user='root', passwd='fyf!!961004', database='Jdata')
    cur = conn.cursor()

    pool = multiprocessing.Pool(4)
    pool.map(insert,action)
    pool.close()
    pool.close()

    cur.execute(index_sql)

    cur.close()
    conn.close()
