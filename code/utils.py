# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd
import os
import random
import csv
import copy
import traceback
import datetime

from utils import *

from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
import xgboost as xgb

delta=datetime.timedelta(days=1)

def convert_timedelta(duration):
    '''
    把timedelta类型转化成小时
    :param duration: datetime.timedelta
    :return: int:hours
    '''
    days, seconds = duration.days, duration.seconds
    hours = days * 24 + seconds // 3600
    #minutes = (seconds % 3600) // 60
    #seconds = (seconds % 60)
    return hours+1#, minutes, seconds

def convert_timedelta_minutes(duration):
    '''
    把timedelta类型转化成小时
    :param duration: datetime.timedelta
    :return: int:minutes
    '''
    days, seconds = duration.days, duration.seconds
    hours = days * 24 + seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = (seconds % 60)
    return  minutes

def convert_datetime_hour_gap(x):
    '''
    计算日期差,for goodluck.py
    :param Series
    :return: int:hours gap
    '''
    try:
        datetime1=datetime.datetime.strptime(convert_to_nextday(x['feature_end_date']),'%Y-%m-%d')
        datetime2 =datetime.datetime.strptime(x['time'], '%Y-%m-%d %H:%M:%S')
        return convert_timedelta(datetime1-datetime2)
    except:
        return np.NAN

def convert_datetime_day_gap(x):
    '''
    计算日期差,for analyse.py
    :param Series
    :return: int:days gap
    '''
    try:
        date1=datetime.datetime.strptime(
                        datetime.datetime.strptime(x['time_x'], '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d'),'%Y-%m-%d')
        date2 = datetime.datetime.strptime(
            datetime.datetime.strptime(x['time_y'], '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d'), '%Y-%m-%d')
        return (date1-date2).days
    except:
        return np.NAN

def get_selected_feature(model,modell):
    '''
    获得模型的挑选的特征
    :param line:
    :return: list
    '''
    if model=='xgb':
        selected_feature = [int(i) for i in
               pd.DataFrame(modell.get_fscore().items(), columns=['feature', 'importance']).sort_values('importance',
                                                                                                     ascending=False)[
                   'feature'].str.slice(1, ).values]
        return selected_feature
    elif model=='rf':
        pass

def convert_YMD_to_YMDHMM(s):
    '''

    :param s: str '%Y-%m-%d'
    :return: datetime
    '''
    pass

def convert_to_nextday(s):
    '''
    :param s:str '%Y-%m-%d'
    :return:str '%Y-%m-%(d+1)'
    '''
    return (datetime.datetime.strptime(s, '%Y-%m-%d')+delta).strftime('%Y-%m-%d')

def convert(x):
         return convert_timedelta(datetime.datetime.strptime(x['time'].iloc[-1],'%Y-%m-%d %H:%M:%S')-
                                  datetime.datetime.strptime(x['time'].iloc[0],'%Y-%m-%d %H:%M:%S'))

