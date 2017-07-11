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
from sklearn.svm import LinearSVC,SVR
from sklearn.feature_selection import SelectFromModel,RFE


import xgboost as xgb


delta = datetime.timedelta(days=1)

def create():
    # create trainset
    os.chdir('../raw_data/')
    whole_action = pd.read_csv('action.csv')

    whole_action_type = {}
    for i in range(1,7):
        whole_action_type['action_type{}'.format(i)] = whole_action[whole_action['type'] == i][['user_id', 'sku_id', 'brand', 'time']]

    os.chdir('../')
    action = pd.read_csv('action_cate8.csv')
    action_type = {}
    for i in range(1, 7):
        action_type['action_type{}'.format(i)] = action[action['type'] == i][['user_id','sku_id','brand','time']]
    comment = pd.read_csv('comment.csv')
    brand_comment=pd.read_csv('brand_comment.csv')
    product = pd.read_csv('product.csv')
    user = pd.read_csv('user.csv')

    # 建立训练集

    for www in [1,2]:
        trainset = pd.DataFrame()
        for i in list(pd.date_range('2016-02-01', '2016-02-25')) + list(pd.date_range('2016-03-16', '2016-03-28')):
            # 整体跳过0315，这一天有异常数据
            # 包含了线上线下，最后记得分裂
            feature_begin_datetime = datetime.datetime.strptime(str(i), '%Y-%m-%d %H:%M:%S')  # 注意是当天0点
            feature_end_datetime = feature_begin_datetime + 13 * delta
            label_begin_datetime = feature_begin_datetime + 14 * delta
            label_end_datetime = feature_begin_datetime + 18 * delta
            print '已经做到'+str(i)
            # 取前5天加入购物车，前3天点击，前3天浏览
            if www==0:
                user_item = action[(action['time'] >= ((feature_end_datetime-4*delta).strftime('%Y-%m-%d %H:%M:%S'))) & \
                                   (action['time'] <= ((label_begin_datetime).strftime('%Y-%m-%d %H:%M:%S')))&(action['type']==2)] \
                    [['user_id', 'sku_id', 'brand']].drop_duplicates()
                tmpp = action[(action['time'] >= ((feature_end_datetime-4*delta).strftime('%Y-%m-%d %H:%M:%S'))) & \
                              (action['time'] <= ((label_begin_datetime).strftime('%Y-%m-%d %H:%M:%S'))) & (
                                  action['type'] == 4)] \
                    [['user_id', 'sku_id', 'brand']].drop_duplicates()
            elif www==2:
                user_item = action[(action['time'] >= ((feature_end_datetime-2*delta).strftime('%Y-%m-%d %H:%M:%S'))) & \
                                   (action['time'] <= ((label_begin_datetime).strftime('%Y-%m-%d %H:%M:%S')))&
                                   (action['type']==1)] \
                    [['user_id', 'sku_id', 'brand']].drop_duplicates()
                tmpp = action[(action['time'] >= ((feature_end_datetime-2*delta).strftime('%Y-%m-%d %H:%M:%S'))) & \
                              (action['time'] <= ((label_begin_datetime).strftime('%Y-%m-%d %H:%M:%S'))) & (
                                  action['type'] == 4)] \
                    [['user_id', 'sku_id', 'brand']].drop_duplicates()
            else:
                user_item = action[(action['time'] >= ((feature_end_datetime - 2*delta).strftime('%Y-%m-%d %H:%M:%S'))) & \
                                   (action['time'] <= ((label_begin_datetime).strftime('%Y-%m-%d %H:%M:%S'))) &
                                   (action['type'] == 6)] \
                    [['user_id', 'sku_id', 'brand']].drop_duplicates()
                tmpp = action[(action['time'] >= ((feature_end_datetime - 2*delta).strftime('%Y-%m-%d %H:%M:%S'))) & \
                              (action['time'] <= ((label_begin_datetime ).strftime('%Y-%m-%d %H:%M:%S'))) & (
                                  action['type'] == 4)] \
                    [['user_id', 'sku_id', 'brand']].drop_duplicates()
            try:
                user_item = pd.DataFrame(np.array(list(set([tuple(i) for i in user_item.values]) -
                                                       (set([tuple(i) for i in user_item.values]) & set(
                                                           [tuple(i) for i in tmpp.values])))),
                                         columns=['user_id', 'sku_id', 'brand'])
            except:
                continue

            _user_item = [list(i) for i in user_item.values]

            # 加上样本日
            tmpset = copy.deepcopy(user_item)
            tmpset['feature_end_date']=feature_end_datetime.strftime('%Y-%m-%d')

            # 加label
            buy_set = [list(i) for i in
                       action[(action['time'] >= label_begin_datetime.strftime('%Y-%m-%d %H:%M:%S')) &
                              (action['time'] <= (label_end_datetime + delta).strftime('%Y-%m-%d %H:%M:%S')) &
                              (action['type'] == 4)][['user_id', 'sku_id', 'brand']].drop_duplicates().values]

            for i, j in enumerate(_user_item):
                if j in buy_set:
                    _user_item[i].append(1)
                else:
                    _user_item[i].append(0)

            tmpset = pd.merge(tmpset, pd.DataFrame(_user_item, columns=['user_id', 'sku_id', 'brand','label']),
                              on=['user_id', 'sku_id','brand'],how='left')

            # 随机采样正:负=1:8
            len1 = tmpset[tmpset['label'] == 1].shape[0]
            try:
                tmpset = pd.concat([tmpset[tmpset['label'] == 0].sample(4*(www+1) * len1), tmpset[tmpset['label'] == 1]])
            except:
                tmpset = pd.concat(
                    [tmpset[tmpset['label'] == 0].sample(2 * (www + 1) * len1), tmpset[tmpset['label'] == 1]])

            # 用户个人信息
            tmpset = pd.merge(tmpset, user, on='user_id', how='left')

            tmp_whole_action = whole_action[
                (whole_action['time'] >= (feature_begin_datetime.strftime('%Y-%m-%d %H:%M:%S'))) & \
                (whole_action['time'] <= ((feature_end_datetime + delta).strftime('%Y-%m-%d %H:%M:%S')))]
            tmp_action = action[(action['time'] >= (feature_begin_datetime.strftime('%Y-%m-%d %H:%M:%S'))) & \
                                (action['time'] <= ((feature_end_datetime + delta).strftime('%Y-%m-%d %H:%M:%S')))]

            # 用户的对商品，品牌，类别，整体的特征
            # 时间跨度为6h,12h,1d,3d,7d,14d和之前所有的
            # 浏览数,加购数,减购数,下单数，关注数，点击数
            # 该商品，该品牌
            # 时间跨度为6h,12h,1d,3d,7d,14d和之前所有的
            # 浏览数,加购数,减购数,下单数，关注数，点击数
            # 不需要类别的特征，因为大家都是cate8
            for _, i in enumerate([3,6, 12, 24, 24 * 3, 24 * 7, 24 * 14,
                                   (convert_timedelta((feature_end_datetime + delta) - datetime.datetime(2016, 2, 1)))]):
                for j in [1, 2, 3, 4, 5, 6]:
                    tmp_tmp_action = tmp_action[
                        (tmp_action['time'] <= ((feature_end_datetime + delta).strftime('%Y-%m-%d %H:%M:%S'))) & \
                        (tmp_action['time'] >= (
                            (feature_end_datetime + delta - datetime.timedelta(hours=i)).strftime(
                                '%Y-%m-%d %H:%M:%S'))) & \
                        (tmp_action['type'] == j)]
                    tmp_tmp_whole_action = tmp_whole_action[
                        (tmp_whole_action['time'] <= ((feature_end_datetime + delta).strftime('%Y-%m-%d %H:%M:%S'))) & \
                        (tmp_whole_action['time'] >= ((feature_end_datetime + delta - datetime.timedelta(hours=i)).strftime(
                                '%Y-%m-%d %H:%M:%S'))) & \
                        (tmp_whole_action['type'] == j)]

                    # 为了命名而做判断
                    if _ != 7:
                        tmpset = pd.merge(tmpset, tmp_tmp_action.groupby(['user_id', 'sku_id'])[
                            'cate'].count().reset_index().rename(
                            columns={'cate': 'skuid_' + str(i) + 'h' + '_type' + str(j)}),
                                          on=['user_id', 'sku_id'], how='left')
                        tmpset = pd.merge(tmpset,
                                          tmp_tmp_action.groupby(['user_id', 'brand'])['cate'].count().reset_index().rename(
                                              columns={'cate': 'brand_' + str(i) + 'h' + '_type' + str(j)}),
                                          on=['user_id', 'brand'], how='left')
                        tmpset = pd.merge(tmpset, tmp_tmp_action.groupby(['user_id'])['cate'].count().reset_index().rename(
                            columns={'cate': 'cate_' + str(i) + 'h' + '_type' + str(j)}),
                                          on=['user_id'], how='left')
                        tmpset = pd.merge(tmpset,
                                          tmp_tmp_whole_action.groupby(['user_id'])['cate'].count().reset_index().rename(
                                              columns={'cate': 'whole_' + str(i) + 'h' + '_type' + str(j)}),
                                          on=['user_id'], how='left')

                        tmpset = pd.merge(tmpset,
                                          tmp_tmp_whole_action.groupby(['sku_id'])['cate'].count().reset_index().rename(
                                              columns={'cate': 'skuid_whole_' + str(i) + 'h' + '_type' + str(j)}),
                                          on=['sku_id'], how='left')
                        tmpset = pd.merge(tmpset,
                                          tmp_tmp_whole_action.groupby(['brand'])['cate'].count().reset_index().rename(
                                              columns={'cate': 'brand_whole_' + str(i) + 'h' + '_type' + str(j)}),
                                          on=['brand'], how='left')
                    else:
                        tmpset = pd.merge(tmpset, tmp_tmp_action.groupby(['user_id', 'sku_id'])[
                            'cate'].count().reset_index().rename(
                            columns={'cate': 'skuid_' + 'before' + '_type' + str(j)}),
                                          on=['user_id', 'sku_id'], how='left')
                        tmpset = pd.merge(tmpset,
                                          tmp_tmp_action.groupby(['user_id', 'brand'])['cate'].count().reset_index().rename(
                                              columns={'cate': 'brand_' + 'before' + '_type' + str(j)}),
                                          on=['user_id', 'brand'], how='left')
                        tmpset = pd.merge(tmpset, tmp_tmp_action.groupby(['user_id'])['cate'].count().reset_index().rename(
                            columns={'cate': 'cate_' + 'before' + '_type' + str(j)}),
                                          on=['user_id'], how='left')
                        tmpset = pd.merge(tmpset,
                                          tmp_tmp_whole_action.groupby(['user_id'])['cate'].count().reset_index().rename(
                                              columns={'cate': 'whole_' + 'before' + '_type' + str(j)}),
                                          on=['user_id'], how='left')

                        tmpset = pd.merge(tmpset,
                                          tmp_tmp_whole_action.groupby(['sku_id'])['cate'].count().reset_index().rename(
                                              columns={'cate': 'skuid_whole_' + 'before' + '_type' + str(j)}),
                                          on=['sku_id'], how='left')
                        tmpset = pd.merge(tmpset,
                                          tmp_tmp_whole_action.groupby(['brand'])['cate'].count().reset_index().rename(
                                              columns={'cate': 'brand_whole_' + 'before' + '_type' + str(j)}),
                                          on=['brand'], how='left')

            # 填充0，之后相除会出现inf，再替换
            tmpset.fillna(0)

            tmp_timegap_list = ['3h_type','6h_type', '12h_type', '24h_type', '72h_type', '168h_type', '336h_type', 'before_type']
            for _, i in enumerate(tmp_timegap_list):
                # 商品特征的重构
                tmpset = pd.concat([tmpset,
                                    # 该商品，该品牌其他行为对下单的转化率,体现的是这个商品和品牌的特性
                                    pd.DataFrame(
                                        tmpset['skuid_whole_' + i + '4'] / tmpset['skuid_whole_' + i + '1'].astype('float'),
                                        columns=['skuid_whole_' + i + '4/1']),
                                    pd.DataFrame(
                                        tmpset['brand_whole_' + i + '4'] / tmpset['brand_whole_' + i + '1'].astype('float'),
                                        columns=['brand_whole_' + i + '4/1']),
                                    pd.DataFrame(
                                        tmpset['skuid_whole_' + i + '4'] / tmpset['skuid_whole_' + i + '2'].astype('float'),
                                        columns=['skuid_whole_' + i + '4/2']),
                                    pd.DataFrame(
                                        tmpset['brand_whole_' + i + '4'] / tmpset['brand_whole_' + i + '2'].astype('float'),
                                        columns=['brand_whole_' + i + '4/2']),
                                    pd.DataFrame(
                                        tmpset['skuid_whole_' + i + '4'] / tmpset['skuid_whole_' + i + '3'].astype('float'),
                                        columns=['skuid_whole_' + i + '4/3']),
                                    pd.DataFrame(
                                        tmpset['brand_whole_' + i + '4'] / tmpset['brand_whole_' + i + '3'].astype('float'),
                                        columns=['brand_whole_' + i + '4/3']),
                                    pd.DataFrame(
                                        tmpset['skuid_whole_' + i + '4'] / tmpset['skuid_whole_' + i + '5'].astype('float'),
                                        columns=['skuid_whole_' + i + '4/5']),
                                    pd.DataFrame(
                                        tmpset['brand_whole_' + i + '4'] / tmpset['brand_whole_' + i + '5'].astype('float'),
                                        columns=['brand_whole_' + i + '4/5']),
                                    pd.DataFrame(
                                        tmpset['skuid_whole_' + i + '4'] / tmpset['skuid_whole_' + i + '6'].astype('float'),
                                        columns=['skuid_whole_' + i + '4/6']),
                                    pd.DataFrame(
                                        tmpset['brand_whole_' + i + '4'] / tmpset['brand_whole_' + i + '6'].astype('float'),
                                        columns=['brand_whole_' + i + '4/6']),
                                    # 该商品对该品牌的行为比例
                                    pd.DataFrame(
                                        tmpset['skuid_whole_' + i + '1'] / tmpset['brand_whole_' + i + '1'].astype('float'),
                                        columns=['skuid/brand_whole_' + i + '1']),
                                    pd.DataFrame(
                                        tmpset['skuid_whole_' + i + '2'] / tmpset['brand_whole_' + i + '2'].astype('float'),
                                        columns=['skuid/brand_whole_' + i + '2']),
                                    pd.DataFrame(
                                        tmpset['skuid_whole_' + i + '3'] / tmpset['brand_whole_' + i + '3'].astype('float'),
                                        columns=['skuid/brand_whole_' + i + '3']),
                                    pd.DataFrame(
                                        tmpset['skuid_whole_' + i + '4'] / tmpset['brand_whole_' + i + '4'].astype('float'),
                                        columns=['skuid/brand_whole_' + i + '4']),
                                    pd.DataFrame(
                                        tmpset['skuid_whole_' + i + '5'] / tmpset['brand_whole_' + i + '5'].astype('float'),
                                        columns=['skuid/brand_whole_' + i + '5']),
                                    pd.DataFrame(
                                        tmpset['skuid_whole_' + i + '6'] / tmpset['brand_whole_' + i + '6'].astype('float'),
                                        columns=['skuid/brand_whole_' + i + '6']),
                                    ], axis=1)

                # 用户特征的重构
                for j in ['skuid_', 'brand_', 'cate_', 'whole_']:
                    # 该用户其他行为对下单的转化率,体现的是这个用户近期买东西的习惯
                    tmpset = pd.concat([tmpset,
                                        pd.DataFrame(tmpset[j + i + '4'] / tmpset[j + i + '1'].astype('float'),
                                                     columns=[j + i + '4/1']),
                                        pd.DataFrame(tmpset[j + i + '4'] / tmpset[j + i + '2'].astype('float'),
                                                     columns=[j + i + '4/2']),
                                        pd.DataFrame(tmpset[j + i + '4'] / tmpset[j + i + '3'].astype('float'),
                                                     columns=[j + i + '4/3']),
                                        pd.DataFrame(tmpset[j + i + '4'] / tmpset[j + i + '5'].astype('float'),
                                                     columns=[j + i + '4/5']),
                                        pd.DataFrame(tmpset[j + i + '4'] / tmpset[j + i + '6'].astype('float'),
                                                     columns=[j + i + '4/6'])], axis=1)

                    # 加用户的差分特征，体现的是用户对于该商品，品牌，类别，总体的浏览,加购……的递进情况
                    if _ != 7:
                        tmpset = pd.concat([tmpset,
                                            pd.DataFrame(tmpset[j + tmp_timegap_list[_] + '1'] - 2 * tmpset[
                                                j + tmp_timegap_list[_ + 1] + '1'],
                                                         columns=[j + tmp_timegap_list[_] + '-' + tmp_timegap_list[
                                                             _ + 1] + '1']),
                                            pd.DataFrame(tmpset[j + tmp_timegap_list[_] + '2'] - 2 * tmpset[
                                                j + tmp_timegap_list[_ + 1] + '2'],
                                                         columns=[j + tmp_timegap_list[_] + '-' + tmp_timegap_list[
                                                             _ + 1] + '2']),
                                            pd.DataFrame(tmpset[j + tmp_timegap_list[_] + '3'] - 2 * tmpset[
                                                j + tmp_timegap_list[_ + 1] + '3'],
                                                         columns=[j + tmp_timegap_list[_] + '-' + tmp_timegap_list[
                                                             _ + 1] + '3']),
                                            pd.DataFrame(tmpset[j + tmp_timegap_list[_] + '4'] - 2 * tmpset[
                                                j + tmp_timegap_list[_ + 1] + '4'],
                                                         columns=[j + tmp_timegap_list[_] + '-' + tmp_timegap_list[
                                                             _ + 1] + '4']),
                                            pd.DataFrame(tmpset[j + tmp_timegap_list[_] + '5'] - 2 * tmpset[
                                                j + tmp_timegap_list[_ + 1] + '5'],
                                                         columns=[j + tmp_timegap_list[_] + '-' + tmp_timegap_list[
                                                             _ + 1] + '5']),
                                            pd.DataFrame(tmpset[j + tmp_timegap_list[_] + '6'] - 2 * tmpset[
                                                j + tmp_timegap_list[_ + 1] + '6'],
                                                         columns=[
                                                             j + tmp_timegap_list[_] + '-' + tmp_timegap_list[_ + 1] + '6'])
                                            ], axis=1)

                # 该用户对商品的每个行为除去该品牌的每个行为的比例
                # 该用户对该品牌的每个行为除去对该类,全部的每个行为的比例
                # 该用户对该类的每个行为除去全部的每个行为的比例
                for typeid in range(1, 7):
                    tmpset = pd.concat([tmpset,
                                        pd.DataFrame(
                                            tmpset['skuid_' + tmp_timegap_list[_] + str(typeid)] / tmpset['brand_' +
                                                                                                          tmp_timegap_list[
                                                                                                              _] + str(
                                                typeid)],
                                            columns=['skuid/brand' + tmp_timegap_list[_] + str(typeid)]),
                                        pd.DataFrame(
                                            tmpset['skuid_' + tmp_timegap_list[_] + str(typeid)] / tmpset['cate_' +
                                                                                                          tmp_timegap_list[
                                                                                                              _] + str(
                                                typeid)],
                                            columns=['skuid/cate' + tmp_timegap_list[_] + str(typeid)]),
                                        pd.DataFrame(
                                            tmpset['skuid_' + tmp_timegap_list[_] + str(typeid)] / tmpset['whole_' +
                                                                                                          tmp_timegap_list[
                                                                                                              _] + str(
                                                typeid)],
                                            columns=['skuid/whole' + tmp_timegap_list[_] + str(typeid)]),
                                        pd.DataFrame(
                                            tmpset['brand_' + tmp_timegap_list[_] + str(typeid)] / tmpset['cate_' +
                                                                                                          tmp_timegap_list[
                                                                                                              _] + str(
                                                typeid)],
                                            columns=['brand/cate' + tmp_timegap_list[_] + str(typeid)]),
                                        pd.DataFrame(
                                            tmpset['brand_' + tmp_timegap_list[_] + str(typeid)] / tmpset['whole_' +
                                                                                                          tmp_timegap_list[
                                                                                                              _] + str(
                                                typeid)],
                                            columns=['brand/whole' + tmp_timegap_list[_] + str(typeid)]),
                                        pd.DataFrame(
                                            tmpset['cate_' + tmp_timegap_list[_] + str(typeid)] / tmpset['whole_' +
                                                                                                         tmp_timegap_list[
                                                                                                             _] + str(
                                                typeid)],
                                            columns=['cate/whole' + tmp_timegap_list[_] + str(typeid)]),
                                        ], axis=1)

            # 加商品信息
            tmpset = pd.merge(tmpset, product.ix[:, :'a3'], on='sku_id', how='left')

            # 加商品评论信息
            weekday = feature_end_datetime.weekday()
            tmpset = pd.merge(tmpset,
                              comment[comment['dt'] == ((feature_end_datetime - weekday * delta).strftime('%Y-%m-%d'))][
                                  ['sku_id', 'comment_num', 'has_bad_comment', 'bad_comment_rate']]
                              .rename(columns={'comment_num': 'before1stMon_comment_num',
                                               'has_bad_comment': 'before1stMon_has_bad_comment',
                                               'bad_comment_rate': 'before1stMon_bad_comment_rate'}), on='sku_id',
                              how='left')
            tmpset = pd.merge(tmpset,
                              comment[comment['dt'] == (
                                  (feature_end_datetime - (7 + weekday) * delta).strftime('%Y-%m-%d'))][
                                  ['sku_id', 'comment_num', 'has_bad_comment', 'bad_comment_rate']]
                              .rename(columns={'comment_num': 'before2ndMon_comment_num',
                                               'has_bad_comment': 'before2ndMon_has_bad_comment',
                                               'bad_comment_rate': 'before2ndMon_bad_comment_rate'}), on='sku_id',
                              how='left')

            # 商品评论差分信息
            tmpset = pd.concat([tmpset,
                                pd.DataFrame(tmpset['before1stMon_comment_num'] - tmpset['before2ndMon_comment_num'],
                                             columns=['before1stMon-before2ndMon_comment_num']),
                                pd.DataFrame(
                                    tmpset['before1stMon_bad_comment_rate'] - tmpset['before2ndMon_bad_comment_rate'],
                                    columns=['before1stMon-before2ndMon_bad_comment_rate'])], axis=1)

            # 加品牌评论信息
            tmpset = pd.merge(tmpset,
                              brand_comment[
                                  brand_comment['dt'] == ((feature_end_datetime - weekday * delta).strftime('%Y-%m-%d'))][
                                  ['brand', 'comment_num', 'has_bad_comment', 'bad_comment_rate']]
                              .rename(columns={'comment_num': 'brand_before1stMon_comment_num',
                                               'has_bad_comment': 'brand_before1stMon_has_bad_comment',
                                               'bad_comment_rate': 'brand_before1stMon_bad_comment_rate'}), on='brand',
                              how='left')
            tmpset = pd.merge(tmpset,
                              brand_comment[brand_comment['dt'] == (
                                  (feature_end_datetime - (7 + weekday) * delta).strftime('%Y-%m-%d'))][
                                  ['brand', 'comment_num', 'has_bad_comment', 'bad_comment_rate']]
                              .rename(columns={'comment_num': 'brand_before2ndMon_comment_num',
                                               'has_bad_comment': 'brand_before2ndMon_has_bad_comment',
                                               'bad_comment_rate': 'brand_before2ndMon_bad_comment_rate'}), on='brand',
                              how='left')

            # 商品评论差分信息
            tmpset = pd.concat([tmpset,
                                pd.DataFrame(
                                    tmpset['brand_before1stMon_comment_num'] - tmpset['brand_before2ndMon_comment_num'],
                                    columns=['brand_before1stMon-before2ndMon_comment_num']),
                                pd.DataFrame(
                                    tmpset['brand_before1stMon_bad_comment_rate'] - tmpset[
                                        'brand_before2ndMon_bad_comment_rate'],
                                    columns=['brand_before1stMon-before2ndMon_bad_comment_rate'])], axis=1)


            for i in [1,2,5,6]:
                group=action_type['action_type{}'.format(i)][
                                        (action_type['action_type{}'.format(i)]['time'] < label_begin_datetime.strftime(
                                            '%Y-%m-%d %H:%M:%S')) &
                                        (action_type['action_type{}'.format(i)]['time'] > feature_begin_datetime.strftime(
                                            '%Y-%m-%d %H:%M:%S'))
                                        ].groupby(['user_id','sku_id','brand','time']).apply(convert)
                whole_group = whole_action_type['action_type{}'.format(i)][
                    (whole_action_type['action_type{}'.format(i)]['time'] < label_begin_datetime.strftime(
                        '%Y-%m-%d %H:%M:%S')) &
                    (whole_action_type['action_type{}'.format(i)]['time'] > feature_begin_datetime.strftime(
                        '%Y-%m-%d %H:%M:%S'))
                    ].groupby(['user_id', 'sku_id', 'brand', 'time']).apply(convert)
                tmpset = pd.merge(tmpset, group.mean(level=['user_id', 'brand']).reset_index().rename(
                    columns={0: 'userid_brand_average_hours_type{}'.format(i)}), how='left')
                tmpset = pd.merge(tmpset, group.mean(level=['user_id']).reset_index().rename(
                    columns={0: 'userid_cate_average_hours_type{}'.format(i)}), how='left')
                tmpset = pd.merge(tmpset, whole_group.mean(level=['user_id']).reset_index().rename(
                    columns={0: 'userid_whole_average_hours_type{}'.format(i)}), how='left')
                tmpset = pd.merge(tmpset, group.mean(level=['sku_id']).reset_index().rename(
                    columns={0: 'skuid_average_hours_type{}'.format(i)}), how='left')
                tmpset = pd.merge(tmpset, group.mean(level=['brand']).reset_index().rename(
                    columns={0: 'brand_average_hours_type{}'.format(i)}), how='left')
                tmpset['skuid-brand_average_hours_type{}'.format(i)]=tmpset['skuid_average_hours_type{}'.format(i)]\
                                                                     -tmpset['brand_average_hours_type{}'.format(i)]

                for j in [('farest', 'first'), ('nearest', 'last')]:
                    # 加入用户商品时间信息
                    tmp1 = pd.merge(tmpset[['user_id', 'sku_id', 'feature_end_date']],
                                    action_type['action_type{}'.format(i)][
                                        (action_type['action_type{}'.format(i)]['time'] < label_begin_datetime.strftime(
                                            '%Y-%m-%d %H:%M:%S')) &
                                        (action_type['action_type{}'.format(i)]['time'] > feature_begin_datetime.strftime(
                                            '%Y-%m-%d %H:%M:%S'))
                                        ],
                                    on=['user_id', 'sku_id'], how='left') \
                        [['user_id', 'sku_id', 'feature_end_date', 'time']]. \
                        drop_duplicates(subset=['user_id', 'sku_id', 'feature_end_date'], keep=j[1])
                    tmp2 = tmp1[['feature_end_date', 'time']].apply(convert_datetime_hour_gap, axis=1)
                    tmp = pd.concat([tmp1, tmp2], axis=1)
                    tmp.rename(columns={0: 'userid_skuid_{}_type{}'.format(j[0], i)}, inplace=True)
                    del tmp['time']
                    # 加入用户商品品牌信息
                    tmpset = pd.merge(tmpset, tmp, on=['user_id', 'sku_id', 'feature_end_date'], how='left')
                    tmp1 = pd.merge(tmpset[['user_id', 'brand', 'feature_end_date']]
                                    , action_type['action_type{}'.format(i)][
                                        (action_type['action_type{}'.format(i)]['time'] < label_begin_datetime.strftime(
                                            '%Y-%m-%d %H:%M:%S')) &
                                        (action_type['action_type{}'.format(i)]['time'] > feature_begin_datetime.strftime(
                                            '%Y-%m-%d %H:%M:%S'))
                                        ],
                                    on=['user_id', 'brand'], how='left') \
                        [['user_id', 'brand', 'feature_end_date', 'time']]. \
                        drop_duplicates(subset=['user_id', 'brand', 'feature_end_date'], keep=j[1])
                    tmp2 = tmp1[['feature_end_date', 'time']].apply(convert_datetime_hour_gap, axis=1)
                    tmp = pd.concat([tmp1, tmp2], axis=1)
                    tmp.rename(columns={0: 'userid_brand_{}_type{}'.format(j[0], i)}, inplace=True)
                    del tmp['time']
                    tmpset = pd.merge(tmpset, tmp, on=['user_id', 'brand', 'feature_end_date'], how='left')
                    # 加入用户cate8信息
                    tmp1 = pd.merge(tmpset[['user_id', 'feature_end_date']], action_type['action_type{}'.format(i)][
                        (action_type['action_type{}'.format(i)]['time'] < label_begin_datetime.strftime(
                            '%Y-%m-%d %H:%M:%S'))
                        &
                        (action_type['action_type{}'.format(i)]['time'] > feature_begin_datetime.strftime(
                            '%Y-%m-%d %H:%M:%S'))],
                                    on=['user_id'], how='left') \
                        [['user_id', 'feature_end_date', 'time']]. \
                        drop_duplicates(subset=['user_id', 'feature_end_date'], keep=j[1])
                    tmp2 = tmp1[['feature_end_date', 'time']].apply(convert_datetime_hour_gap, axis=1)
                    tmp = pd.concat([tmp1, tmp2], axis=1)
                    tmp.rename(columns={0: 'userid_cate_{}_type{}'.format(j[0], i)}, inplace=True)
                    del tmp['time']
                    tmpset = pd.merge(tmpset, tmp, on=['user_id', 'feature_end_date'], how='left')
                    # 加入用户全集信息
                    tmp1 = pd.merge(tmpset[['user_id', 'feature_end_date']], whole_action_type['action_type{}'.format(i)][
                        (whole_action_type['action_type{}'.format(i)]['time'] < label_begin_datetime.strftime(
                            '%Y-%m-%d %H:%M:%S')) &
                        (whole_action_type['action_type{}'.format(i)]['time'] > feature_begin_datetime.strftime(
                            '%Y-%m-%d %H:%M:%S'))],
                                    on=['user_id'], how='left') \
                        [['user_id', 'feature_end_date', 'time']]. \
                        drop_duplicates(subset=['user_id', 'feature_end_date'], keep=j[1])
                    tmp2 = tmp1[['feature_end_date', 'time']].apply(convert_datetime_hour_gap, axis=1)
                    tmp = pd.concat([tmp1, tmp2], axis=1)
                    tmp.rename(columns={0: 'userid_whole_{}_type{}'.format(j[0], i)}, inplace=True)
                    del tmp['time']
                    tmpset = pd.merge(tmpset, tmp, on=['user_id', 'feature_end_date'], how='left')
                tmp1 = pd.DataFrame(
                    (tmpset['userid_skuid_farest_type{}'.format(i)] - tmpset['userid_skuid_nearest_type{}'.format(i)]),
                    columns=['userid_skuid_farest-nearest_type{}'.format(i)])
                tmp2 = pd.DataFrame(
                    (tmpset['userid_brand_farest_type{}'.format(i)] - tmpset['userid_brand_nearest_type{}'.format(i)]),
                    columns=['userid_brand_farest-nearest_type{}'.format(i)])
                tmp3 = pd.DataFrame(
                    (tmpset['userid_cate_farest_type{}'.format(i)] - tmpset['userid_cate_nearest_type{}'.format(i)]),
                    columns=['userid_cate_farest-nearest_type{}'.format(i)])
                tmp4 = pd.DataFrame(
                    (tmpset['userid_whole_farest_type{}'.format(i)] - tmpset['userid_whole_nearest_type{}'.format(i)]),
                    columns=['userid_whole_farest-nearest_type{}'.format(i)])
                tmpset = pd.concat([tmpset, tmp1,tmp2,tmp3,tmp4], axis=1)

            # 用户对该商品行为时间间隔/用户对该品牌时间间隔
            # 用户对该品牌行为时间间隔/用户对该类行为时间间隔
            # 用户对该类行为时间间隔/用户对所有行为时间间隔
            # 用户行为的频率
            for i in [1,2,5,6]:
                tmpset = pd.concat([tmpset,
                                    pd.DataFrame(
                                        tmpset['userid_skuid_farest-nearest_type{}'.format(i)] /
                                        tmpset['userid_brand_farest-nearest_type{}'.format(i)],
                                        columns=['userid_skuid/brand_timegap_type{}'.format(i)]),
                                    pd.DataFrame(
                                        tmpset['userid_skuid_farest-nearest_type{}'.format(i)] /
                                        tmpset['userid_cate_farest-nearest_type{}'.format(i)],
                                        columns=['userid_skuid/cate_timegap_type{}'.format(i)]),

                                     pd.DataFrame(
                                       tmpset['userid_skuid_farest-nearest_type{}'.format(i)] /
                                       tmpset['userid_whole_farest-nearest_type{}'.format(i)],
                                       columns=['userid_skuid/whole_timegap_type{}'.format(i)]),

                                    pd.DataFrame(
                                        tmpset['userid_brand_farest-nearest_type{}'.format(i)] /
                                        tmpset['userid_cate_farest-nearest_type{}'.format(i)],
                                        columns=['userid_brand/cate_timegap_type{}'.format(i)]),

                                     pd.DataFrame(
                                       tmpset['userid_brand_farest-nearest_type{}'.format(i)] /
                                       tmpset['userid_whole_farest-nearest_type{}'.format(i)],
                                       columns=['userid_brand/whole_timegap_type{}'.format(i)]),
                                     pd.DataFrame(
                                       tmpset['userid_cate_farest-nearest_type{}'.format(i)] /
                                       tmpset['userid_whole_farest-nearest_type{}'.format(i)],
                                       columns=['userid_cate/whole_timegap_type{}'.format(i)]),

                                    # 用户行为的频率
                                    pd.DataFrame(
                                        tmpset['skuid_before_type{}'.format(i)] /
                                        tmpset['userid_skuid_farest-nearest_type{}'.format(i)],
                                        columns=['userid_skuid_frequency_type{}'.format(i)]),
                                    pd.DataFrame(
                                        tmpset['brand_before_type{}'.format(i)] /
                                        tmpset['userid_brand_farest-nearest_type{}'.format(i)],
                                        columns=['userid_brand_frequency_timegap_type{}'.format(i)]),
                                    pd.DataFrame(
                                        tmpset['cate_before_type{}'.format(i)] /
                                        tmpset['userid_cate_farest-nearest_type{}'.format(i)],
                                        columns=['userid_cate_frequency_timegap_type{}'.format(i)]),
                                     pd.DataFrame(
                                      tmpset['whole_before_type{}'.format(i)] /
                                      tmpset['userid_whole_farest-nearest_type{}'.format(i)],
                                      columns=['userid_whole_frequency_timegap_type{}'.format(i)]),

                                    # 用户对该商品的最近点击、收藏、加购物车、购买时间减去同类其他商品的最近点击、收藏、加购物车、购买时间
                                    pd.DataFrame(
                                        -tmpset['userid_skuid_nearest_type{}'.format(i)] +
                                        tmpset['userid_brand_nearest_type{}'.format(i)],
                                        columns=['userid_skuid-brand_nearest_type{}'.format(i)]),
                                    pd.DataFrame(
                                        -tmpset['userid_skuid_nearest_type{}'.format(i)] +
                                        tmpset['userid_cate_nearest_type{}'.format(i)],
                                        columns=['userid_skuid-cate_nearest_type{}'.format(i)]),
                                    pd.DataFrame(
                                        -tmpset['userid_skuid_nearest_type{}'.format(i)] +
                                        tmpset['userid_whole_nearest_type{}'.format(i)],
                                        columns=['userid_skuid-whole_nearest_type{}'.format(i)]),
                                    pd.DataFrame(
                                        -tmpset['userid_brand_nearest_type{}'.format(i)] +
                                        tmpset['userid_cate_nearest_type{}'.format(i)],
                                        columns=['userid_brand-cate_nearest_type{}'.format(i)]),
                                    pd.DataFrame(
                                        -tmpset['userid_brand_nearest_type{}'.format(i)] +
                                        tmpset['userid_whole_nearest_type{}'.format(i)],
                                        columns=['userid_brand-whole_nearest_type{}'.format(i)]),
                                    pd.DataFrame(
                                        -tmpset['userid_cate_nearest_type{}'.format(i)] +
                                        tmpset['userid_whole_nearest_type{}'.format(i)],
                                        columns=['userid_cate-whole_nearest_type{}'.format(i)]),
                                    # 用户对该商品的最远点击、收藏、加购物车、购买时间
                                    # 减去同类其他商品的最近点远、收藏、加购物车、购买时间
                                    pd.DataFrame(
                                        -tmpset['userid_skuid_farest_type{}'.format(i)] +
                                        tmpset['userid_brand_farest_type{}'.format(i)],
                                        columns=['userid_skuid-brand_farest_type{}'.format(i)]),
                                    pd.DataFrame(
                                        -tmpset['userid_skuid_farest_type{}'.format(i)] +
                                        tmpset['userid_cate_farest_type{}'.format(i)],
                                        columns=['userid_skuid-cate_farest_type{}'.format(i)]),
                                    pd.DataFrame(
                                        -tmpset['userid_skuid_farest_type{}'.format(i)] +
                                        tmpset['userid_whole_farest_type{}'.format(i)],
                                        columns=['userid_skuid-whole_farest_type{}'.format(i)]),
                                    pd.DataFrame(
                                        -tmpset['userid_brand_farest_type{}'.format(i)] +
                                        tmpset['userid_cate_farest_type{}'.format(i)],
                                        columns=['userid_brand-cate_farest_type{}'.format(i)]),
                                    pd.DataFrame(
                                        -tmpset['userid_brand_farest_type{}'.format(i)] +
                                        tmpset['userid_whole_farest_type{}'.format(i)],
                                        columns=['userid_brand-whole_farest_type{}'.format(i)]),
                                    pd.DataFrame(
                                        -tmpset['userid_cate_farest_type{}'.format(i)] +
                                        tmpset['userid_whole_farest_type{}'.format(i)],
                                        columns=['userid_cate-whole_farest_type{}'.format(i)]),

                                    # 用户的行为减去用户平均水平
                                    pd.DataFrame(
                                        tmpset['userid_skuid_farest-nearest_type{}'.format(i)] -
                                        tmpset['userid_brand_average_hours_type{}'.format(i)],
                                        columns=['userid_skuid-brand_average_hours_type{}'.format(i)]),
                                    pd.DataFrame(
                                        tmpset['userid_skuid_farest-nearest_type{}'.format(i)] -
                                        tmpset['userid_cate_average_hours_type{}'.format(i)],
                                        columns=['userid_skuid-cate_average_hours_type{}'.format(i)]),
                                    pd.DataFrame(
                                        tmpset['userid_skuid_farest-nearest_type{}'.format(i)] -
                                        tmpset['userid_whole_average_hours_type{}'.format(i)],
                                        columns=['userid_skuid-whole_average_hours_type{}'.format(i)]),
                                    pd.DataFrame(
                                        tmpset['userid_brand_farest-nearest_type{}'.format(i)] -
                                        tmpset['userid_cate_average_hours_type{}'.format(i)],
                                        columns=['userid_brand-cate_average_hours_type{}'.format(i)]),
                                    pd.DataFrame(
                                        tmpset['userid_brand_farest-nearest_type{}'.format(i)] -
                                        tmpset['userid_whole_average_hours_type{}'.format(i)],
                                        columns=['userid_brand-whole_average_hours_type{}'.format(i)]),
                                    pd.DataFrame(
                                        tmpset['userid_cate_farest-nearest_type{}'.format(i)] -
                                        tmpset['userid_whole_average_hours_type{}'.format(i)],
                                        columns=['userid_cate-whole_average_hours_type{}'.format(i)]),
                                    pd.DataFrame(
                                        tmpset['userid_cate_farest-nearest_type{}'.format(i)] -
                                        tmpset['userid_whole_average_hours_type{}'.format(i)],
                                        columns=['userid_cate-whole_average_hours_type{}'.format(i)]),

                                    pd.DataFrame(
                                        tmpset['userid_skuid_farest-nearest_type{}'.format(i)] -
                                        tmpset['skuid_average_hours_type{}'.format(i)],
                                        columns=['userid_skuid-skuid_average_hours_type{}'.format(i)]),
                                    pd.DataFrame(
                                        tmpset['userid_brand_farest-nearest_type{}'.format(i)] -
                                        tmpset['brand_average_hours_type{}'.format(i)],
                                        columns=['userid_skuid-skuid_average_hours_type{}'.format(i)]),
                                    ], axis=1)




            # 合并到总trainset
            trainset = pd.concat([trainset, tmpset])

        # 保存训练集
        os.chdir('/home/fengyufei/PycharmProjects/jd_competition/trainset/online/')
        trainset[trainset['feature_end_date'] <= '2016-04-10'].to_csv('online_{}.csv'.format(www), index=False)
        os.chdir('/home/fengyufei/PycharmProjects/jd_competition/trainset/offline/')
        trainset[trainset['feature_end_date'] <= '2016-04-05'].to_csv('offline_{}.csv'.format(www), index=False)

        print 'trainset is saved and the shape is',trainset.shape

def create2():
    #create predictset
    os.chdir('../raw_data/')
    whole_action = pd.read_csv('action.csv')

    whole_action_type = {}
    for i in range(1, 7):
        whole_action_type['action_type{}'.format(i)] = whole_action[whole_action['type'] == i][
            ['user_id', 'sku_id', 'brand', 'time']]

    os.chdir('../')
    action = pd.read_csv('action_cate8.csv')
    action_type = {}
    for i in range(1, 7):
        action_type['action_type{}'.format(i)] = action[action['type'] == i][['user_id', 'sku_id', 'brand', 'time']]
    comment = pd.read_csv('comment.csv')
    brand_comment = pd.read_csv('brand_comment.csv')
    product = pd.read_csv('product.csv')
    user = pd.read_csv('user.csv')

    for www in range(3):
        for k,i in enumerate(['2016-03-28 00:00:00','2016-04-02 00:00:00']):
            feature_begin_datetime = datetime.datetime.strptime(str(i), '%Y-%m-%d %H:%M:%S')  # 注意是当天0点
            feature_end_datetime = feature_begin_datetime + 13 * delta
            label_begin_datetime = feature_begin_datetime + 14 * delta
            label_end_datetime = feature_begin_datetime + 18 * delta

            # 取前5天加入购物车，前3天点击，前3天浏览
            if www == 0:
                user_item = \
                action[(action['time'] >= ((feature_end_datetime - 4 * delta).strftime('%Y-%m-%d %H:%M:%S'))) & \
                       (action['time'] <= ((label_begin_datetime).strftime('%Y-%m-%d %H:%M:%S'))) & (
                       action['type'] == 2)] \
                    [['user_id', 'sku_id', 'brand']].drop_duplicates()
                tmpp = action[(action['time'] >= ((feature_end_datetime - 4 * delta).strftime('%Y-%m-%d %H:%M:%S'))) & \
                              (action['time'] <= ((label_begin_datetime).strftime('%Y-%m-%d %H:%M:%S'))) & (
                                  action['type'] == 4)] \
                    [['user_id', 'sku_id', 'brand']].drop_duplicates()
            elif www == 2:
                user_item = \
                action[(action['time'] >= ((feature_end_datetime - 2 * delta).strftime('%Y-%m-%d %H:%M:%S'))) & \
                       (action['time'] <= ((label_begin_datetime).strftime('%Y-%m-%d %H:%M:%S'))) &
                       (action['type'] == 1)] \
                    [['user_id', 'sku_id', 'brand']].drop_duplicates()
                tmpp = action[(action['time'] >= ((feature_end_datetime - 2 * delta).strftime('%Y-%m-%d %H:%M:%S'))) & \
                              (action['time'] <= ((label_begin_datetime).strftime('%Y-%m-%d %H:%M:%S'))) & (
                                  action['type'] == 4)] \
                    [['user_id', 'sku_id', 'brand']].drop_duplicates()
            else:
                user_item = \
                action[(action['time'] >= ((feature_end_datetime - 2 * delta).strftime('%Y-%m-%d %H:%M:%S'))) & \
                       (action['time'] <= ((label_begin_datetime).strftime('%Y-%m-%d %H:%M:%S'))) &
                       (action['type'] == 6)] \
                    [['user_id', 'sku_id', 'brand']].drop_duplicates()
                tmpp = action[(action['time'] >= ((feature_end_datetime - 2 * delta).strftime('%Y-%m-%d %H:%M:%S'))) & \
                              (action['time'] <= ((label_begin_datetime).strftime('%Y-%m-%d %H:%M:%S'))) & (
                                  action['type'] == 4)] \
                    [['user_id', 'sku_id', 'brand']].drop_duplicates()

            user_item = pd.DataFrame(np.array(list(set([tuple(i) for i in user_item.values]) -
                                                   (set([tuple(i) for i in user_item.values]) & set(
                                                       [tuple(i) for i in tmpp.values])))),
                                     columns=['user_id', 'sku_id', 'brand'])

            # 加上样本日
            __tmpset = copy.deepcopy(user_item)
            __tmpset['feature_end_date'] = feature_end_datetime.strftime('%Y-%m-%d')
            shape = __tmpset.shape[0]
            _tmpset = pd.DataFrame()

            for qqq in range(0, int(shape * 9 / 10.0), int(shape / 10.0)):
                tmpset=__tmpset.iloc[qqq:(qqq+int(shape / 10.0))]
                print qqq
                # 用户个人信息
                tmpset = pd.merge(tmpset, user, on='user_id', how='left')

                tmp_whole_action = whole_action[
                    (whole_action['time'] >= (feature_begin_datetime.strftime('%Y-%m-%d %H:%M:%S'))) & \
                    (whole_action['time'] <= ((feature_end_datetime + delta).strftime('%Y-%m-%d %H:%M:%S')))]
                tmp_action = action[(action['time'] >= (feature_begin_datetime.strftime('%Y-%m-%d %H:%M:%S'))) & \
                                    (action['time'] <= ((feature_end_datetime + delta).strftime('%Y-%m-%d %H:%M:%S')))]

                # 用户的对商品，品牌，类别，整体的特征
                # 时间跨度为6h,12h,1d,3d,7d,14d和之前所有的
                # 浏览数,加购数,减购数,下单数，关注数，点击数
                # 该商品，该品牌
                # 时间跨度为6h,12h,1d,3d,7d,14d和之前所有的
                # 浏览数,加购数,减购数,下单数，关注数，点击数
                # 不需要类别的特征，因为大家都是cate8
                for _, i in enumerate([3, 6, 12, 24, 24 * 3, 24 * 7, 24 * 14,
                                       (
                                       convert_timedelta((feature_end_datetime + delta) - datetime.datetime(2016, 2, 1)))]):
                    for j in [1, 2, 3, 4, 5, 6]:
                        tmp_tmp_action = tmp_action[
                            (tmp_action['time'] <= ((feature_end_datetime + delta).strftime('%Y-%m-%d %H:%M:%S'))) & \
                            (tmp_action['time'] >= (
                                (feature_end_datetime + delta - datetime.timedelta(hours=i)).strftime(
                                    '%Y-%m-%d %H:%M:%S'))) & \
                            (tmp_action['type'] == j)]
                        tmp_tmp_whole_action = tmp_whole_action[
                            (tmp_whole_action['time'] <= ((feature_end_datetime + delta).strftime('%Y-%m-%d %H:%M:%S'))) & \
                            (tmp_whole_action['time'] >= (
                                (feature_end_datetime + delta - datetime.timedelta(hours=i)).strftime(
                                    '%Y-%m-%d %H:%M:%S'))) & \
                            (tmp_whole_action['type'] == j)]

                        # 为了命名而做判断
                        if _ != 7:
                            tmpset = pd.merge(tmpset, tmp_tmp_action.groupby(['user_id', 'sku_id'])[
                                'cate'].count().reset_index().rename(
                                columns={'cate': 'skuid_' + str(i) + 'h' + '_type' + str(j)}),
                                              on=['user_id', 'sku_id'], how='left')
                            tmpset = pd.merge(tmpset,
                                              tmp_tmp_action.groupby(['user_id', 'brand'])[
                                                  'cate'].count().reset_index().rename(
                                                  columns={'cate': 'brand_' + str(i) + 'h' + '_type' + str(j)}),
                                              on=['user_id', 'brand'], how='left')
                            tmpset = pd.merge(tmpset,
                                              tmp_tmp_action.groupby(['user_id'])['cate'].count().reset_index().rename(
                                                  columns={'cate': 'cate_' + str(i) + 'h' + '_type' + str(j)}),
                                              on=['user_id'], how='left')
                            tmpset = pd.merge(tmpset,
                                              tmp_tmp_whole_action.groupby(['user_id'])[
                                                  'cate'].count().reset_index().rename(
                                                  columns={'cate': 'whole_' + str(i) + 'h' + '_type' + str(j)}),
                                              on=['user_id'], how='left')

                            tmpset = pd.merge(tmpset,
                                              tmp_tmp_whole_action.groupby(['sku_id'])['cate'].count().reset_index().rename(
                                                  columns={'cate': 'skuid_whole_' + str(i) + 'h' + '_type' + str(j)}),
                                              on=['sku_id'], how='left')
                            tmpset = pd.merge(tmpset,
                                              tmp_tmp_whole_action.groupby(['brand'])['cate'].count().reset_index().rename(
                                                  columns={'cate': 'brand_whole_' + str(i) + 'h' + '_type' + str(j)}),
                                              on=['brand'], how='left')
                        else:
                            tmpset = pd.merge(tmpset, tmp_tmp_action.groupby(['user_id', 'sku_id'])[
                                'cate'].count().reset_index().rename(
                                columns={'cate': 'skuid_' + 'before' + '_type' + str(j)}),
                                              on=['user_id', 'sku_id'], how='left')
                            tmpset = pd.merge(tmpset,
                                              tmp_tmp_action.groupby(['user_id', 'brand'])[
                                                  'cate'].count().reset_index().rename(
                                                  columns={'cate': 'brand_' + 'before' + '_type' + str(j)}),
                                              on=['user_id', 'brand'], how='left')
                            tmpset = pd.merge(tmpset,
                                              tmp_tmp_action.groupby(['user_id'])['cate'].count().reset_index().rename(
                                                  columns={'cate': 'cate_' + 'before' + '_type' + str(j)}),
                                              on=['user_id'], how='left')
                            tmpset = pd.merge(tmpset,
                                              tmp_tmp_whole_action.groupby(['user_id'])[
                                                  'cate'].count().reset_index().rename(
                                                  columns={'cate': 'whole_' + 'before' + '_type' + str(j)}),
                                              on=['user_id'], how='left')

                            tmpset = pd.merge(tmpset,
                                              tmp_tmp_whole_action.groupby(['sku_id'])['cate'].count().reset_index().rename(
                                                  columns={'cate': 'skuid_whole_' + 'before' + '_type' + str(j)}),
                                              on=['sku_id'], how='left')
                            tmpset = pd.merge(tmpset,
                                              tmp_tmp_whole_action.groupby(['brand'])['cate'].count().reset_index().rename(
                                                  columns={'cate': 'brand_whole_' + 'before' + '_type' + str(j)}),
                                              on=['brand'], how='left')

                # 填充0，之后相除会出现inf，再替换
                tmpset.fillna(0)

                tmp_timegap_list = ['3h_type', '6h_type', '12h_type', '24h_type', '72h_type', '168h_type', '336h_type',
                                    'before_type']
                for _, i in enumerate(tmp_timegap_list):
                    # 商品特征的重构
                    tmpset = pd.concat([tmpset,
                                        # 该商品，该品牌其他行为对下单的转化率,体现的是这个商品和品牌的特性
                                        pd.DataFrame(
                                            tmpset['skuid_whole_' + i + '4'] / tmpset['skuid_whole_' + i + '1'].astype(
                                                'float'),
                                            columns=['skuid_whole_' + i + '4/1']),
                                        pd.DataFrame(
                                            tmpset['brand_whole_' + i + '4'] / tmpset['brand_whole_' + i + '1'].astype(
                                                'float'),
                                            columns=['brand_whole_' + i + '4/1']),
                                        pd.DataFrame(
                                            tmpset['skuid_whole_' + i + '4'] / tmpset['skuid_whole_' + i + '2'].astype(
                                                'float'),
                                            columns=['skuid_whole_' + i + '4/2']),
                                        pd.DataFrame(
                                            tmpset['brand_whole_' + i + '4'] / tmpset['brand_whole_' + i + '2'].astype(
                                                'float'),
                                            columns=['brand_whole_' + i + '4/2']),
                                        pd.DataFrame(
                                            tmpset['skuid_whole_' + i + '4'] / tmpset['skuid_whole_' + i + '3'].astype(
                                                'float'),
                                            columns=['skuid_whole_' + i + '4/3']),
                                        pd.DataFrame(
                                            tmpset['brand_whole_' + i + '4'] / tmpset['brand_whole_' + i + '3'].astype(
                                                'float'),
                                            columns=['brand_whole_' + i + '4/3']),
                                        pd.DataFrame(
                                            tmpset['skuid_whole_' + i + '4'] / tmpset['skuid_whole_' + i + '5'].astype(
                                                'float'),
                                            columns=['skuid_whole_' + i + '4/5']),
                                        pd.DataFrame(
                                            tmpset['brand_whole_' + i + '4'] / tmpset['brand_whole_' + i + '5'].astype(
                                                'float'),
                                            columns=['brand_whole_' + i + '4/5']),
                                        pd.DataFrame(
                                            tmpset['skuid_whole_' + i + '4'] / tmpset['skuid_whole_' + i + '6'].astype(
                                                'float'),
                                            columns=['skuid_whole_' + i + '4/6']),
                                        pd.DataFrame(
                                            tmpset['brand_whole_' + i + '4'] / tmpset['brand_whole_' + i + '6'].astype(
                                                'float'),
                                            columns=['brand_whole_' + i + '4/6']),
                                        # 该商品对该品牌的行为比例
                                        pd.DataFrame(
                                            tmpset['skuid_whole_' + i + '1'] / tmpset['brand_whole_' + i + '1'].astype(
                                                'float'),
                                            columns=['skuid/brand_whole_' + i + '1']),
                                        pd.DataFrame(
                                            tmpset['skuid_whole_' + i + '2'] / tmpset['brand_whole_' + i + '2'].astype(
                                                'float'),
                                            columns=['skuid/brand_whole_' + i + '2']),
                                        pd.DataFrame(
                                            tmpset['skuid_whole_' + i + '3'] / tmpset['brand_whole_' + i + '3'].astype(
                                                'float'),
                                            columns=['skuid/brand_whole_' + i + '3']),
                                        pd.DataFrame(
                                            tmpset['skuid_whole_' + i + '4'] / tmpset['brand_whole_' + i + '4'].astype(
                                                'float'),
                                            columns=['skuid/brand_whole_' + i + '4']),
                                        pd.DataFrame(
                                            tmpset['skuid_whole_' + i + '5'] / tmpset['brand_whole_' + i + '5'].astype(
                                                'float'),
                                            columns=['skuid/brand_whole_' + i + '5']),
                                        pd.DataFrame(
                                            tmpset['skuid_whole_' + i + '6'] / tmpset['brand_whole_' + i + '6'].astype(
                                                'float'),
                                            columns=['skuid/brand_whole_' + i + '6']),
                                        ], axis=1)

                    # 用户特征的重构
                    for j in ['skuid_', 'brand_', 'cate_', 'whole_']:
                        # 该用户其他行为对下单的转化率,体现的是这个用户近期买东西的习惯
                        tmpset = pd.concat([tmpset,
                                            pd.DataFrame(tmpset[j + i + '4'] / tmpset[j + i + '1'].astype('float'),
                                                         columns=[j + i + '4/1']),
                                            pd.DataFrame(tmpset[j + i + '4'] / tmpset[j + i + '2'].astype('float'),
                                                         columns=[j + i + '4/2']),
                                            pd.DataFrame(tmpset[j + i + '4'] / tmpset[j + i + '3'].astype('float'),
                                                         columns=[j + i + '4/3']),
                                            pd.DataFrame(tmpset[j + i + '4'] / tmpset[j + i + '5'].astype('float'),
                                                         columns=[j + i + '4/5']),
                                            pd.DataFrame(tmpset[j + i + '4'] / tmpset[j + i + '6'].astype('float'),
                                                         columns=[j + i + '4/6'])], axis=1)

                        # 加用户的差分特征，体现的是用户对于该商品，品牌，类别，总体的浏览,加购……的递进情况
                        if _ != 7:
                            tmpset = pd.concat([tmpset,
                                                pd.DataFrame(tmpset[j + tmp_timegap_list[_] + '1'] - 2 * tmpset[
                                                    j + tmp_timegap_list[_ + 1] + '1'],
                                                             columns=[j + tmp_timegap_list[_] + '-' + tmp_timegap_list[
                                                                 _ + 1] + '1']),
                                                pd.DataFrame(tmpset[j + tmp_timegap_list[_] + '2'] - 2 * tmpset[
                                                    j + tmp_timegap_list[_ + 1] + '2'],
                                                             columns=[j + tmp_timegap_list[_] + '-' + tmp_timegap_list[
                                                                 _ + 1] + '2']),
                                                pd.DataFrame(tmpset[j + tmp_timegap_list[_] + '3'] - 2 * tmpset[
                                                    j + tmp_timegap_list[_ + 1] + '3'],
                                                             columns=[j + tmp_timegap_list[_] + '-' + tmp_timegap_list[
                                                                 _ + 1] + '3']),
                                                pd.DataFrame(tmpset[j + tmp_timegap_list[_] + '4'] - 2 * tmpset[
                                                    j + tmp_timegap_list[_ + 1] + '4'],
                                                             columns=[j + tmp_timegap_list[_] + '-' + tmp_timegap_list[
                                                                 _ + 1] + '4']),
                                                pd.DataFrame(tmpset[j + tmp_timegap_list[_] + '5'] - 2 * tmpset[
                                                    j + tmp_timegap_list[_ + 1] + '5'],
                                                             columns=[j + tmp_timegap_list[_] + '-' + tmp_timegap_list[
                                                                 _ + 1] + '5']),
                                                pd.DataFrame(tmpset[j + tmp_timegap_list[_] + '6'] - 2 * tmpset[
                                                    j + tmp_timegap_list[_ + 1] + '6'],
                                                             columns=[
                                                                 j + tmp_timegap_list[_] + '-' + tmp_timegap_list[
                                                                     _ + 1] + '6'])
                                                ], axis=1)

                    # 该用户对商品的每个行为除去该品牌的每个行为的比例
                    # 该用户对该品牌的每个行为除去对该类,全部的每个行为的比例
                    # 该用户对该类的每个行为除去全部的每个行为的比例
                    for typeid in range(1, 7):
                        tmpset = pd.concat([tmpset,
                                            pd.DataFrame(
                                                tmpset['skuid_' + tmp_timegap_list[_] + str(typeid)] / tmpset['brand_' +
                                                                                                              tmp_timegap_list[
                                                                                                                  _] + str(
                                                    typeid)],
                                                columns=['skuid/brand' + tmp_timegap_list[_] + str(typeid)]),
                                            pd.DataFrame(
                                                tmpset['skuid_' + tmp_timegap_list[_] + str(typeid)] / tmpset['cate_' +
                                                                                                              tmp_timegap_list[
                                                                                                                  _] + str(
                                                    typeid)],
                                                columns=['skuid/cate' + tmp_timegap_list[_] + str(typeid)]),
                                            pd.DataFrame(
                                                tmpset['skuid_' + tmp_timegap_list[_] + str(typeid)] / tmpset['whole_' +
                                                                                                              tmp_timegap_list[
                                                                                                                  _] + str(
                                                    typeid)],
                                                columns=['skuid/whole' + tmp_timegap_list[_] + str(typeid)]),
                                            pd.DataFrame(
                                                tmpset['brand_' + tmp_timegap_list[_] + str(typeid)] / tmpset['cate_' +
                                                                                                              tmp_timegap_list[
                                                                                                                  _] + str(
                                                    typeid)],
                                                columns=['brand/cate' + tmp_timegap_list[_] + str(typeid)]),
                                            pd.DataFrame(
                                                tmpset['brand_' + tmp_timegap_list[_] + str(typeid)] / tmpset['whole_' +
                                                                                                              tmp_timegap_list[
                                                                                                                  _] + str(
                                                    typeid)],
                                                columns=['brand/whole' + tmp_timegap_list[_] + str(typeid)]),
                                            pd.DataFrame(
                                                tmpset['cate_' + tmp_timegap_list[_] + str(typeid)] / tmpset['whole_' +
                                                                                                             tmp_timegap_list[
                                                                                                                 _] + str(
                                                    typeid)],
                                                columns=['cate/whole' + tmp_timegap_list[_] + str(typeid)]),
                                            ], axis=1)

                # 加商品信息
                tmpset = pd.merge(tmpset, product.ix[:, :'a3'], on='sku_id', how='left')

                # 加商品评论信息
                weekday = feature_end_datetime.weekday()
                tmpset = pd.merge(tmpset,
                                  comment[comment['dt'] == ((feature_end_datetime - weekday * delta).strftime('%Y-%m-%d'))][
                                      ['sku_id', 'comment_num', 'has_bad_comment', 'bad_comment_rate']]
                                  .rename(columns={'comment_num': 'before1stMon_comment_num',
                                                   'has_bad_comment': 'before1stMon_has_bad_comment',
                                                   'bad_comment_rate': 'before1stMon_bad_comment_rate'}), on='sku_id',
                                  how='left')
                tmpset = pd.merge(tmpset,
                                  comment[comment['dt'] == (
                                      (feature_end_datetime - (7 + weekday) * delta).strftime('%Y-%m-%d'))][
                                      ['sku_id', 'comment_num', 'has_bad_comment', 'bad_comment_rate']]
                                  .rename(columns={'comment_num': 'before2ndMon_comment_num',
                                                   'has_bad_comment': 'before2ndMon_has_bad_comment',
                                                   'bad_comment_rate': 'before2ndMon_bad_comment_rate'}), on='sku_id',
                                  how='left')

                # 商品评论差分信息
                tmpset = pd.concat([tmpset,
                                    pd.DataFrame(tmpset['before1stMon_comment_num'] - tmpset['before2ndMon_comment_num'],
                                                 columns=['before1stMon-before2ndMon_comment_num']),
                                    pd.DataFrame(
                                        tmpset['before1stMon_bad_comment_rate'] - tmpset['before2ndMon_bad_comment_rate'],
                                        columns=['before1stMon-before2ndMon_bad_comment_rate'])], axis=1)

                # 加品牌评论信息
                tmpset = pd.merge(tmpset,
                                  brand_comment[
                                      brand_comment['dt'] == (
                                      (feature_end_datetime - weekday * delta).strftime('%Y-%m-%d'))][
                                      ['brand', 'comment_num', 'has_bad_comment', 'bad_comment_rate']]
                                  .rename(columns={'comment_num': 'brand_before1stMon_comment_num',
                                                   'has_bad_comment': 'brand_before1stMon_has_bad_comment',
                                                   'bad_comment_rate': 'brand_before1stMon_bad_comment_rate'}), on='brand',
                                  how='left')
                tmpset = pd.merge(tmpset,
                                  brand_comment[brand_comment['dt'] == (
                                      (feature_end_datetime - (7 + weekday) * delta).strftime('%Y-%m-%d'))][
                                      ['brand', 'comment_num', 'has_bad_comment', 'bad_comment_rate']]
                                  .rename(columns={'comment_num': 'brand_before2ndMon_comment_num',
                                                   'has_bad_comment': 'brand_before2ndMon_has_bad_comment',
                                                   'bad_comment_rate': 'brand_before2ndMon_bad_comment_rate'}), on='brand',
                                  how='left')

                # 商品评论差分信息
                tmpset = pd.concat([tmpset,
                                    pd.DataFrame(
                                        tmpset['brand_before1stMon_comment_num'] - tmpset['brand_before2ndMon_comment_num'],
                                        columns=['brand_before1stMon-before2ndMon_comment_num']),
                                    pd.DataFrame(
                                        tmpset['brand_before1stMon_bad_comment_rate'] - tmpset[
                                            'brand_before2ndMon_bad_comment_rate'],
                                        columns=['brand_before1stMon-before2ndMon_bad_comment_rate'])], axis=1)

                for i in [1, 2, 5, 6]:
                    group = action_type['action_type{}'.format(i)][
                        (action_type['action_type{}'.format(i)]['time'] < label_begin_datetime.strftime(
                            '%Y-%m-%d %H:%M:%S')) &
                        (action_type['action_type{}'.format(i)]['time'] > feature_begin_datetime.strftime(
                            '%Y-%m-%d %H:%M:%S'))
                        ].groupby(['user_id', 'sku_id', 'brand', 'time']).apply(convert)
                    whole_group = whole_action_type['action_type{}'.format(i)][
                        (whole_action_type['action_type{}'.format(i)]['time'] < label_begin_datetime.strftime(
                            '%Y-%m-%d %H:%M:%S')) &
                        (whole_action_type['action_type{}'.format(i)]['time'] > feature_begin_datetime.strftime(
                            '%Y-%m-%d %H:%M:%S'))
                        ].groupby(['user_id', 'sku_id', 'brand', 'time']).apply(convert)
                    tmpset = pd.merge(tmpset, group.mean(level=['user_id', 'brand']).reset_index().rename(
                        columns={0: 'userid_brand_average_hours_type{}'.format(i)}), how='left')
                    tmpset = pd.merge(tmpset, group.mean(level=['user_id']).reset_index().rename(
                        columns={0: 'userid_cate_average_hours_type{}'.format(i)}), how='left')
                    tmpset = pd.merge(tmpset, whole_group.mean(level=['user_id']).reset_index().rename(
                        columns={0: 'userid_whole_average_hours_type{}'.format(i)}), how='left')
                    tmpset = pd.merge(tmpset, group.mean(level=['sku_id']).reset_index().rename(
                        columns={0: 'skuid_average_hours_type{}'.format(i)}), how='left')
                    tmpset = pd.merge(tmpset, group.mean(level=['brand']).reset_index().rename(
                        columns={0: 'brand_average_hours_type{}'.format(i)}), how='left')
                    tmpset['skuid-brand_average_hours_type{}'.format(i)] = tmpset['skuid_average_hours_type{}'.format(i)] \
                                                                           - tmpset['brand_average_hours_type{}'.format(i)]

                    for j in [('farest', 'first'), ('nearest', 'last')]:
                        # 加入用户商品时间信息
                        tmp1 = pd.merge(tmpset[['user_id', 'sku_id', 'feature_end_date']],
                                        action_type['action_type{}'.format(i)][
                                            (action_type['action_type{}'.format(i)]['time'] < label_begin_datetime.strftime(
                                                '%Y-%m-%d %H:%M:%S')) &
                                            (action_type['action_type{}'.format(i)][
                                                 'time'] > feature_begin_datetime.strftime(
                                                '%Y-%m-%d %H:%M:%S'))
                                            ],
                                        on=['user_id', 'sku_id'], how='left') \
                            [['user_id', 'sku_id', 'feature_end_date', 'time']]. \
                            drop_duplicates(subset=['user_id', 'sku_id', 'feature_end_date'], keep=j[1])
                        tmp2 = tmp1[['feature_end_date', 'time']].apply(convert_datetime_hour_gap, axis=1)
                        tmp = pd.concat([tmp1, tmp2], axis=1)
                        tmp.rename(columns={0: 'userid_skuid_{}_type{}'.format(j[0], i)}, inplace=True)
                        del tmp['time']
                        # 加入用户商品品牌信息
                        tmpset = pd.merge(tmpset, tmp, on=['user_id', 'sku_id', 'feature_end_date'], how='left')
                        tmp1 = pd.merge(tmpset[['user_id', 'brand', 'feature_end_date']]
                                        , action_type['action_type{}'.format(i)][
                                            (action_type['action_type{}'.format(i)]['time'] < label_begin_datetime.strftime(
                                                '%Y-%m-%d %H:%M:%S')) &
                                            (action_type['action_type{}'.format(i)][
                                                 'time'] > feature_begin_datetime.strftime(
                                                '%Y-%m-%d %H:%M:%S'))
                                            ],
                                        on=['user_id', 'brand'], how='left') \
                            [['user_id', 'brand', 'feature_end_date', 'time']]. \
                            drop_duplicates(subset=['user_id', 'brand', 'feature_end_date'], keep=j[1])
                        tmp2 = tmp1[['feature_end_date', 'time']].apply(convert_datetime_hour_gap, axis=1)
                        tmp = pd.concat([tmp1, tmp2], axis=1)
                        tmp.rename(columns={0: 'userid_brand_{}_type{}'.format(j[0], i)}, inplace=True)
                        del tmp['time']
                        tmpset = pd.merge(tmpset, tmp, on=['user_id', 'brand', 'feature_end_date'], how='left')
                        # 加入用户cate8信息
                        tmp1 = pd.merge(tmpset[['user_id', 'feature_end_date']], action_type['action_type{}'.format(i)][
                            (action_type['action_type{}'.format(i)]['time'] < label_begin_datetime.strftime(
                                '%Y-%m-%d %H:%M:%S'))
                            &
                            (action_type['action_type{}'.format(i)]['time'] > feature_begin_datetime.strftime(
                                '%Y-%m-%d %H:%M:%S'))],
                                        on=['user_id'], how='left') \
                            [['user_id', 'feature_end_date', 'time']]. \
                            drop_duplicates(subset=['user_id', 'feature_end_date'], keep=j[1])
                        tmp2 = tmp1[['feature_end_date', 'time']].apply(convert_datetime_hour_gap, axis=1)
                        tmp = pd.concat([tmp1, tmp2], axis=1)
                        tmp.rename(columns={0: 'userid_cate_{}_type{}'.format(j[0], i)}, inplace=True)
                        del tmp['time']
                        tmpset = pd.merge(tmpset, tmp, on=['user_id', 'feature_end_date'], how='left')
                        # 加入用户全集信息
                        tmp1 = \
                        pd.merge(tmpset[['user_id', 'feature_end_date']], whole_action_type['action_type{}'.format(i)][
                            (whole_action_type['action_type{}'.format(i)]['time'] < label_begin_datetime.strftime(
                                '%Y-%m-%d %H:%M:%S')) &
                            (whole_action_type['action_type{}'.format(i)]['time'] > feature_begin_datetime.strftime(
                                '%Y-%m-%d %H:%M:%S'))],
                                 on=['user_id'], how='left') \
                            [['user_id', 'feature_end_date', 'time']]. \
                            drop_duplicates(subset=['user_id', 'feature_end_date'], keep=j[1])
                        tmp2 = tmp1[['feature_end_date', 'time']].apply(convert_datetime_hour_gap, axis=1)
                        tmp = pd.concat([tmp1, tmp2], axis=1)
                        tmp.rename(columns={0: 'userid_whole_{}_type{}'.format(j[0], i)}, inplace=True)
                        del tmp['time']
                        tmpset = pd.merge(tmpset, tmp, on=['user_id', 'feature_end_date'], how='left')
                    tmp1 = pd.DataFrame(
                        (tmpset['userid_skuid_farest_type{}'.format(i)] - tmpset['userid_skuid_nearest_type{}'.format(i)]),
                        columns=['userid_skuid_farest-nearest_type{}'.format(i)])
                    tmp2 = pd.DataFrame(
                        (tmpset['userid_brand_farest_type{}'.format(i)] - tmpset['userid_brand_nearest_type{}'.format(i)]),
                        columns=['userid_brand_farest-nearest_type{}'.format(i)])
                    tmp3 = pd.DataFrame(
                        (tmpset['userid_cate_farest_type{}'.format(i)] - tmpset['userid_cate_nearest_type{}'.format(i)]),
                        columns=['userid_cate_farest-nearest_type{}'.format(i)])
                    tmp4 = pd.DataFrame(
                        (tmpset['userid_whole_farest_type{}'.format(i)] - tmpset['userid_whole_nearest_type{}'.format(i)]),
                        columns=['userid_whole_farest-nearest_type{}'.format(i)])
                    tmpset = pd.concat([tmpset, tmp1, tmp2, tmp3, tmp4], axis=1)

                # 用户对该商品行为时间间隔/用户对该品牌时间间隔
                # 用户对该品牌行为时间间隔/用户对该类行为时间间隔
                # 用户对该类行为时间间隔/用户对所有行为时间间隔
                # 用户行为的频率
                for i in [1, 2, 5, 6]:
                    tmpset = pd.concat([tmpset,
                                        pd.DataFrame(
                                            tmpset['userid_skuid_farest-nearest_type{}'.format(i)] /
                                            tmpset['userid_brand_farest-nearest_type{}'.format(i)],
                                            columns=['userid_skuid/brand_timegap_type{}'.format(i)]),
                                        pd.DataFrame(
                                            tmpset['userid_skuid_farest-nearest_type{}'.format(i)] /
                                            tmpset['userid_cate_farest-nearest_type{}'.format(i)],
                                            columns=['userid_skuid/cate_timegap_type{}'.format(i)]),

                                        pd.DataFrame(
                                            tmpset['userid_skuid_farest-nearest_type{}'.format(i)] /
                                            tmpset['userid_whole_farest-nearest_type{}'.format(i)],
                                            columns=['userid_skuid/whole_timegap_type{}'.format(i)]),

                                        pd.DataFrame(
                                            tmpset['userid_brand_farest-nearest_type{}'.format(i)] /
                                            tmpset['userid_cate_farest-nearest_type{}'.format(i)],
                                            columns=['userid_brand/cate_timegap_type{}'.format(i)]),

                                        pd.DataFrame(
                                            tmpset['userid_brand_farest-nearest_type{}'.format(i)] /
                                            tmpset['userid_whole_farest-nearest_type{}'.format(i)],
                                            columns=['userid_brand/whole_timegap_type{}'.format(i)]),
                                        pd.DataFrame(
                                            tmpset['userid_cate_farest-nearest_type{}'.format(i)] /
                                            tmpset['userid_whole_farest-nearest_type{}'.format(i)],
                                            columns=['userid_cate/whole_timegap_type{}'.format(i)]),

                                        # 用户行为的频率
                                        pd.DataFrame(
                                            tmpset['skuid_before_type{}'.format(i)] /
                                            tmpset['userid_skuid_farest-nearest_type{}'.format(i)],
                                            columns=['userid_skuid_frequency_type{}'.format(i)]),
                                        pd.DataFrame(
                                            tmpset['brand_before_type{}'.format(i)] /
                                            tmpset['userid_brand_farest-nearest_type{}'.format(i)],
                                            columns=['userid_brand_frequency_timegap_type{}'.format(i)]),
                                        pd.DataFrame(
                                            tmpset['cate_before_type{}'.format(i)] /
                                            tmpset['userid_cate_farest-nearest_type{}'.format(i)],
                                            columns=['userid_cate_frequency_timegap_type{}'.format(i)]),
                                        pd.DataFrame(
                                            tmpset['whole_before_type{}'.format(i)] /
                                            tmpset['userid_whole_farest-nearest_type{}'.format(i)],
                                            columns=['userid_whole_frequency_timegap_type{}'.format(i)]),

                                        # 用户对该商品的最近点击、收藏、加购物车、购买时间减去同类其他商品的最近点击、收藏、加购物车、购买时间
                                        pd.DataFrame(
                                            -tmpset['userid_skuid_nearest_type{}'.format(i)] +
                                            tmpset['userid_brand_nearest_type{}'.format(i)],
                                            columns=['userid_skuid-brand_nearest_type{}'.format(i)]),
                                        pd.DataFrame(
                                            -tmpset['userid_skuid_nearest_type{}'.format(i)] +
                                            tmpset['userid_cate_nearest_type{}'.format(i)],
                                            columns=['userid_skuid-cate_nearest_type{}'.format(i)]),
                                        pd.DataFrame(
                                            -tmpset['userid_skuid_nearest_type{}'.format(i)] +
                                            tmpset['userid_whole_nearest_type{}'.format(i)],
                                            columns=['userid_skuid-whole_nearest_type{}'.format(i)]),
                                        pd.DataFrame(
                                            -tmpset['userid_brand_nearest_type{}'.format(i)] +
                                            tmpset['userid_cate_nearest_type{}'.format(i)],
                                            columns=['userid_brand-cate_nearest_type{}'.format(i)]),
                                        pd.DataFrame(
                                            -tmpset['userid_brand_nearest_type{}'.format(i)] +
                                            tmpset['userid_whole_nearest_type{}'.format(i)],
                                            columns=['userid_brand-whole_nearest_type{}'.format(i)]),
                                        pd.DataFrame(
                                            -tmpset['userid_cate_nearest_type{}'.format(i)] +
                                            tmpset['userid_whole_nearest_type{}'.format(i)],
                                            columns=['userid_cate-whole_nearest_type{}'.format(i)]),
                                        # 用户对该商品的最远点击、收藏、加购物车、购买时间
                                        # 减去同类其他商品的最近点远、收藏、加购物车、购买时间
                                        pd.DataFrame(
                                            -tmpset['userid_skuid_farest_type{}'.format(i)] +
                                            tmpset['userid_brand_farest_type{}'.format(i)],
                                            columns=['userid_skuid-brand_farest_type{}'.format(i)]),
                                        pd.DataFrame(
                                            -tmpset['userid_skuid_farest_type{}'.format(i)] +
                                            tmpset['userid_cate_farest_type{}'.format(i)],
                                            columns=['userid_skuid-cate_farest_type{}'.format(i)]),
                                        pd.DataFrame(
                                            -tmpset['userid_skuid_farest_type{}'.format(i)] +
                                            tmpset['userid_whole_farest_type{}'.format(i)],
                                            columns=['userid_skuid-whole_farest_type{}'.format(i)]),
                                        pd.DataFrame(
                                            -tmpset['userid_brand_farest_type{}'.format(i)] +
                                            tmpset['userid_cate_farest_type{}'.format(i)],
                                            columns=['userid_brand-cate_farest_type{}'.format(i)]),
                                        pd.DataFrame(
                                            -tmpset['userid_brand_farest_type{}'.format(i)] +
                                            tmpset['userid_whole_farest_type{}'.format(i)],
                                            columns=['userid_brand-whole_farest_type{}'.format(i)]),
                                        pd.DataFrame(
                                            -tmpset['userid_cate_farest_type{}'.format(i)] +
                                            tmpset['userid_whole_farest_type{}'.format(i)],
                                            columns=['userid_cate-whole_farest_type{}'.format(i)]),

                                        # 用户的行为减去用户平均水平
                                        pd.DataFrame(
                                            tmpset['userid_skuid_farest-nearest_type{}'.format(i)] -
                                            tmpset['userid_brand_average_hours_type{}'.format(i)],
                                            columns=['userid_skuid-brand_average_hours_type{}'.format(i)]),
                                        pd.DataFrame(
                                            tmpset['userid_skuid_farest-nearest_type{}'.format(i)] -
                                            tmpset['userid_cate_average_hours_type{}'.format(i)],
                                            columns=['userid_skuid-cate_average_hours_type{}'.format(i)]),
                                        pd.DataFrame(
                                            tmpset['userid_skuid_farest-nearest_type{}'.format(i)] -
                                            tmpset['userid_whole_average_hours_type{}'.format(i)],
                                            columns=['userid_skuid-whole_average_hours_type{}'.format(i)]),
                                        pd.DataFrame(
                                            tmpset['userid_brand_farest-nearest_type{}'.format(i)] -
                                            tmpset['userid_cate_average_hours_type{}'.format(i)],
                                            columns=['userid_brand-cate_average_hours_type{}'.format(i)]),
                                        pd.DataFrame(
                                            tmpset['userid_brand_farest-nearest_type{}'.format(i)] -
                                            tmpset['userid_whole_average_hours_type{}'.format(i)],
                                            columns=['userid_brand-whole_average_hours_type{}'.format(i)]),
                                        pd.DataFrame(
                                            tmpset['userid_cate_farest-nearest_type{}'.format(i)] -
                                            tmpset['userid_whole_average_hours_type{}'.format(i)],
                                            columns=['userid_cate-whole_average_hours_type{}'.format(i)]),
                                        pd.DataFrame(
                                            tmpset['userid_cate_farest-nearest_type{}'.format(i)] -
                                            tmpset['userid_whole_average_hours_type{}'.format(i)],
                                            columns=['userid_cate-whole_average_hours_type{}'.format(i)]),

                                        pd.DataFrame(
                                            tmpset['userid_skuid_farest-nearest_type{}'.format(i)] -
                                            tmpset['skuid_average_hours_type{}'.format(i)],
                                            columns=['userid_skuid-skuid_average_hours_type{}'.format(i)]),
                                        pd.DataFrame(
                                            tmpset['userid_brand_farest-nearest_type{}'.format(i)] -
                                            tmpset['brand_average_hours_type{}'.format(i)],
                                            columns=['userid_skuid-skuid_average_hours_type{}'.format(i)]),
                                        ], axis=1)

                _tmpset=pd.concat([_tmpset,tmpset])
            # save predictset
            if k==0:
                os.chdir('/home/fengyufei/PycharmProjects/jd_competition/predictset/offline/')
                _tmpset.to_csv('offline_{}.csv'.format(www), index=False)
            else:
                os.chdir('/home/fengyufei/PycharmProjects/jd_competition/predictset/online/')
                _tmpset.to_csv('online_{}.csv'.format(www), index=False)

def feature_selection(line, model):
    os.chdir('../predictset/' + line)
    # 读取模型，获得得分特征的顺序
    modell = xgb.Booster()
    modell.load_model(line + '_' + model)
    best_res={'Score':0,'selected_items_number':0}

    #训练
    os.chdir('../../trainset/' + line)
    trainset = pd.read_csv(line + '.csv').fillna(-1).replace(np.inf,-1)

    train_x, train_y = trainset.ix[:, 5:].values, trainset.ix[:, 4].values
    if model == 'rf':
        modell = RandomForestRegressor(
            n_estimators=500,
            max_depth=20,
            max_features="auto",
            n_jobs=-1,
            oob_score=True,
        )
        modell.fit(train_x, train_y)
    elif model == 'xgb':
        dtrain = xgb.DMatrix(train_x, label=train_y)
        params = {'max_depth': 8, 'silent': 1, 'eta': 0.1, 'colsample_bytree': 0.8, 'eval_metric': 'logloss',
                  'seed': 2017, 'objective': 'reg:logistic'}
        num_round = 400
        modell = xgb.train(params, dtrain, num_round)

    # 读取预测集
    os.chdir('../../predictset/' + line)
    predictset = pd.read_csv(line + '.csv').fillna(-1).replace(np.inf,-1)
    predict_x = predictset.ix[:, 4:].values
    dpredict = xgb.DMatrix(predict_x)

    os.chdir('../../outcome/' + line)
    if model == 'rf':
        tmp1 = pd.concat([predictset, pd.DataFrame(modell.predict(predict_x), columns=['label'])],
                         axis=1).sort_values(
            by='label', ascending=False)
    elif model == 'xgb':
        tmp1 = pd.concat([predictset, pd.DataFrame(modell.predict(dpredict), columns=['label'])],
                         axis=1).sort_values(
            by='label', ascending=False)

    # 挑选输出多少结果
    for selected_items_number in range(200, 2201, 200):
        tmp = [tuple(i) for i in tmp1.iloc[:selected_items_number][['user_id', 'sku_id']].values]
        user_existed = []
        res = []
        for i in tmp:
            if i[0] not in user_existed:
                res.append(i)
                user_existed.append(i[0])

        # 线下评分
        if line == 'offline':
            '''
            Score=0.4*F11 + 0.6*F12
            此处的F1值定义为：
            F11=6*Recall*Precise/(5*Recall+Precise)
            F12=5*Recall*Precise/(2*Recall+3*Precise)
            其中，Precise为准确率，Recall为召回率.
            F11是label=1或0的F1值，F12是pred=1或0的F1值.
            '''
            test = [tuple(i) for i in pd.read_csv('test.csv').values]
            right1 = set([i[0] for i in res]) & set([i[0] for i in test])
            F11 = 6.0 * len(right1) / len(test) * len(right1) / len(res) / (
                5.0 * len(right1) / len(test) + 1.0 * len(right1) / len(res))

            right2 = set(res) & set(test)
            F22 = 5.0 * len(right2) / len(test) * len(right2) / len(res) / (
                2.0 * len(right2) / len(test) + 3.0 * len(right2) / len(res))
            Score = 0.4 * F11 + 0.6 * F22
            if Score>best_res['Score']:
                best_res['selected_items_number']=selected_items_number
                best_res['F11_score']=F11
                best_res['F22_score']=F22
                best_res['Score']=Score

                #保存结果
                pd.DataFrame(res).to_csv(line + '.csv', index=False, header=['user_id', 'sku_id'])

    print '最好的提交结果是前{}个'.format(best_res['selected_items_number']),
    print '最好的F11_score是{}'.format(best_res['F11_score']),
    print '最好的F22_score是{}'.format(best_res['F22_score']),
    print '最好的Score是{}'.format(best_res['Score'])
    return best_res

def feature_selection_L1(model):
    #训练
    os.chdir('../trainset/offline/')
    trainset = pd.read_csv('offline.csv').fillna(-1).replace(np.inf,-1)
    train_x, train_y = trainset.ix[:, 5:].values, trainset.ix[:, 4].values

    # L1筛选
    lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(train_x, train_y)
    transform_model = SelectFromModel(lsvc, prefit=True)


    # 线下评测
    train_x= transform_model.transform(train_x)
    print '选择了{}个特征'.format(train_x.shape[1])
    if model == 'rf':
        modell = RandomForestRegressor(
            n_estimators=500,
            max_depth=20,
            max_features="auto",
            n_jobs=-1,
            oob_score=True,
        )
        modell.fit(train_x, train_y)
    elif model == 'xgb':
        dtrain = xgb.DMatrix(train_x, label=train_y)
        params = {'max_depth': 8, 'silent': 1, 'eta': 0.1, 'colsample_bytree': 0.8, 'eval_metric': 'logloss',
                  'seed': 2017, 'objective': 'reg:logistic'}
        num_round = 400
        modell = xgb.train(params, dtrain, num_round)

    # 读取预测集
    os.chdir('../../predictset/offline' )
    predictset = pd.read_csv('offline.csv').fillna(-1).replace(np.inf,-1)
    predict_x = transform_model.transform(predictset.ix[:, 4:].values)
    dpredict = xgb.DMatrix(predict_x)

    os.chdir('../../outcome/offline')
    if model == 'rf':
        tmp1 = pd.concat([predictset, pd.DataFrame(modell.predict(predict_x), columns=['label'])],
                         axis=1).sort_values(
            by='label', ascending=False)
    elif model == 'xgb':
        tmp1 = pd.concat([predictset, pd.DataFrame(modell.predict(dpredict), columns=['label'])],
                         axis=1).sort_values(
            by='label', ascending=False)

    tmp = [tuple(i) for i in tmp1.iloc[:1200][['user_id', 'sku_id']].values]
    user_existed = []
    res = []
    for i in tmp:
        if i[0] not in user_existed:
            res.append(i)
            user_existed.append(i[0])

    # 线下评分
    '''
    Score=0.4*F11 + 0.6*F12
    此处的F1值定义为：
    F11=6*Recall*Precise/(5*Recall+Precise)
    F12=5*Recall*Precise/(2*Recall+3*Precise)
    其中，Precise为准确率，Recall为召回率.
    F11是label=1或0的F1值，F12是pred=1或0的F1值.
    '''
    test = [tuple(i) for i in pd.read_csv('test.csv').values]
    right1 = set([i[0] for i in res]) & set([i[0] for i in test])
    F11 = 6.0 * len(right1) / len(test) * len(right1) / len(res) / (
        5.0 * len(right1) / len(test) + 1.0 * len(right1) / len(res))
    print 'F11:', F11
    right2 = set(res) & set(test)
    F22 = 5.0 * len(right2) / len(test) * len(right2) / len(res) / (
        2.0 * len(right2) / len(test) + 3.0 * len(right2) / len(res))
    print 'F12:', F22
    Score = 0.4 * F11 + 0.6 * F22
    print 'Score:', Score

    #线上答案生成
    os.chdir('../../trainset/online')
    trainset = pd.read_csv('online.csv').fillna(-1).replace(np.inf, -1)
    train_x, train_y = trainset.ix[:, 5:].values, trainset.ix[:, 4].values
    train_x = transform_model.transform(train_x)
    if model == 'rf':
        modell = RandomForestRegressor(
            n_estimators=500,
            max_depth=20,
            max_features="auto",
            n_jobs=-1,
            oob_score=True,
        )
        modell.fit(train_x, train_y)
    elif model == 'xgb':
        dtrain = xgb.DMatrix(train_x, label=train_y)
        params = {'max_depth': 8, 'silent': 1, 'eta': 0.1, 'colsample_bytree': 0.8, 'eval_metric': 'logloss',
                  'seed': 2017, 'objective': 'reg:logistic'}
        num_round = 400
        modell = xgb.train(params, dtrain, num_round)

    # 读取预测集
    os.chdir('../../predictset/online/' )
    predictset = pd.read_csv('online.csv').fillna(-1).replace(np.inf, -1)
    predict_x = transform_model.transform(predictset.ix[:, 4:].values)
    dpredict = xgb.DMatrix(predict_x)

    os.chdir('../../outcome/online/')
    if model == 'rf':
        tmp1 = pd.concat([predictset, pd.DataFrame(modell.predict(predict_x), columns=['label'])],
                         axis=1).sort_values(
            by='label', ascending=False)
    elif model == 'xgb':
        tmp1 = pd.concat([predictset, pd.DataFrame(modell.predict(dpredict), columns=['label'])],
                         axis=1).sort_values(
            by='label', ascending=False)

    # 获得结果并保存
    tmp = [tuple(i) for i in tmp1.iloc[:1200][['user_id', 'sku_id']].values]
    user_existed = []
    res = []
    for i in tmp:
        if i[0] not in user_existed:
            res.append(i)
            user_existed.append(i[0])

def feature_selection_rfe():
    # 读取线下训练集
    os.chdir('../trainset/offline/')
    trainset = pd.read_csv('offline.csv').fillna(-1).replace(np.inf, -1)
    train_x, train_y = trainset.ix[:, 5:].values, trainset.ix[:, 4].values

    svr = SVR()
    rfe = RFE(estimator=svr, n_features_to_select=500, step=30)
    fit=rfe.fit(train_x, train_y)
    t=fit.support_
    print t

    # 线下评测
    os.chdir('../../outcome/offline')
    predictset = pd.read_csv('offline.csv').fillna(-1).replace(np.inf, -1)
    predict_x = predictset.ix[:, 4:][t].values
    tmp1 = pd.concat([predictset, pd.DataFrame(rfe.predict(predict_x), columns=['label'])],
                         axis=1).sort_values(
            by='label', ascending=False)

    # 获得结果
    tmp = [tuple(i) for i in tmp1.iloc[:1200][['user_id', 'sku_id']].values]
    user_existed = []
    res = []
    for i in tmp:
        if i[0] not in user_existed:
            res.append(i)
            user_existed.append(i[0])
            # 线下评分

    '''
    Score=0.4*F11 + 0.6*F12
    此处的F1值定义为：
    F11=6*Recall*Precise/(5*Recall+Precise)
    F12=5*Recall*Precise/(2*Recall+3*Precise)
    其中，Precise为准确率，Recall为召回率.
    F11是label=1或0的F1值，F12是pred=1或0的F1值.
    '''
    test = [tuple(i) for i in pd.read_csv('test.csv').values]
    right1 = set([i[0] for i in res]) & set([i[0] for i in test])
    F11 = 6.0 * len(right1) / len(test) * len(right1) / len(res) / (
        5.0 * len(right1) / len(test) + 1.0 * len(right1) / len(res))
    print 'F11:', F11
    right2 = set(res) & set(test)
    F22 = 5.0 * len(right2) / len(test) * len(right2) / len(res) / (
        2.0 * len(right2) / len(test) + 3.0 * len(right2) / len(res))
    print 'F12:', F22
    Score = 0.4 * F11 + 0.6 * F22
    print 'Score:', Score

def feature_selection_shoudong():
    os.chdir('../trainset/offline')
    trainset = pd.read_csv('_offline.csv').fillna(-1).replace(np.inf, -1)
    trainset=pd.concat([trainset.ix[:,:5],
                        trainset.ix[:,'age':'user_reg_tm'],
                        trainset.ix[:, 'a1':'brand_before1stMon-before2ndMon_bad_comment_rate'],
                        trainset.ix[:,'userid_brand_average_hours':'userid_whole_farest-nearest_type6'],
                        trainset.ix[:,'skuid_6h_type1':'brand_whole_before_type6'],
                        trainset.ix[:,'userid_skuid-brand_nearest_type1':'userid_cate-whole_nearest_type1'],
                        trainset.ix[:, 'userid_skuid-brand_nearest_type2': 'userid_cate-whole_nearest_type2'],
                        trainset.ix[:, 'userid_skuid-brand_nearest_type5': 'userid_cate-whole_nearest_type5'],
                        trainset.ix[:, 'userid_skuid-brand_nearest_type6': 'userid_cate-whole_nearest_type6'],
                        trainset.ix[:, 'skuid_whole_6h_type4/1':'brand_whole_6h_type4/6'],
                        trainset.ix[:, 'skuid_whole_12h_type4/1':'brand_whole_12h_type4/6'],
                        trainset.ix[:, 'skuid_whole_24h_type4/1':'brand_whole_24h_type4/6'],
                        trainset.ix[:, 'skuid_whole_72h_type4/1':'brand_whole_72h_type4/6'],
                        trainset.ix[:, 'skuid_whole_168h_type4/1':'brand_whole_168h_type4/6'],
                        trainset.ix[:, 'skuid_whole_336h_type4/1':'brand_whole_336h_type4/6'],
                        trainset.ix[:, 'skuid_whole_before_type4/1':'brand_whole_before_type4/6'],
                        trainset.ix[:, 'skuid_6h_type-12h_type1':'skuid_6h_type-12h_type6'],
                        trainset.ix[:, 'brand_6h_type-12h_type1':'brand_6h_type-12h_type6'],
                        trainset.ix[:, 'cate_6h_type-12h_type1':'cate_6h_type-12h_type6'],
                        trainset.ix[:, 'whole_6h_type-12h_type1':'whole_6h_type-12h_type6'],
                        trainset.ix[:, 'skuid_12h_type-24h_type1':'skuid_12h_type-24h_type6'],
                        trainset.ix[:, 'brand_12h_type-24h_type1':'brand_12h_type-24h_type6'],
                        trainset.ix[:, 'cate_12h_type-24h_type1':'cate_12h_type-24h_type6'],
                        trainset.ix[:, 'whole_12h_type-24h_type1':'whole_12h_type-24h_type6'],
                        trainset.ix[:, 'skuid_24h_type-72h_type1':'skuid_24h_type-72h_type6'],
                        trainset.ix[:, 'brand_24h_type-72h_type1':'brand_24h_type-72h_type6'],
                        trainset.ix[:, 'cate_24h_type-72h_type1':'cate_24h_type-72h_type6'],
                        trainset.ix[:, 'whole_24h_type-72h_type1':'whole_24h_type-72h_type6'],
                        trainset.ix[:, 'skuid_72h_type-168h_type1':'skuid_72h_type-168h_type6'],
                        trainset.ix[:, 'brand_72h_type-168h_type1':'brand_72h_type-168h_type6'],
                        trainset.ix[:, 'cate_72h_type-168h_type1':'cate_72h_type-168h_type6'],
                        trainset.ix[:, 'whole_72h_type-168h_type1':'whole_72h_type-168h_type6'],
                        trainset.ix[:, 'skuid_168h_type-336h_type1':'skuid_168h_type-336h_type6'],
                        trainset.ix[:, 'brand_168h_type-336h_type1':'brand_168h_type-336h_type6'],
                        trainset.ix[:, 'cate_168h_type-336h_type1':'cate_168h_type-336h_type6'],
                        trainset.ix[:, 'whole_168h_type-336h_type1':'whole_168h_type-336h_type6'],
                        trainset.ix[:, 'skuid_336h_type-before_type1':'skuid_336h_type-before_type6'],
                        trainset.ix[:, 'brand_336h_type-before_type1':'brand_336h_type-before_type6'],
                        trainset.ix[:, 'cate_336h_type-before_type1':'cate_336h_type-before_type6'],
                        trainset.ix[:, 'whole_336h_type-before_type1':'whole_336h_type-before_type6'],
                        ],axis=1)
    train_x, train_y = trainset.ix[:, 5:].values, trainset.ix[:, 4].values

    dtrain = xgb.DMatrix(train_x, label=train_y)
    params = {'max_depth': 8, 'silent': 1, 'eta': 0.1, 'colsample_bytree': 0.8, 'eval_metric': 'logloss',
              'seed': 2017, 'objective': 'reg:logistic'}
    num_round = 400
    modell = xgb.train(params, dtrain, num_round)

    # 读取预测集
    os.chdir('../../predictset/offline/' )
    predictset = pd.read_csv('_offline.csv').fillna(-1).replace(np.inf, -1)
    predictset = pd.concat([predictset.ix[:, :4],
                            predictset.ix[:, 'age':'user_reg_tm'],
                            predictset.ix[:, 'a1':'brand_before1stMon-before2ndMon_bad_comment_rate'],
                            predictset.ix[:, 'userid_brand_average_hours':'userid_whole_farest-nearest_type6'],
                            predictset.ix[:, 'skuid_6h_type1':'brand_whole_before_type6'],
                            predictset.ix[:, 'userid_skuid-brand_nearest_type1':'userid_cate-whole_nearest_type1'],
                            predictset.ix[:, 'userid_skuid-brand_nearest_type2': 'userid_cate-whole_nearest_type2'],
                            predictset.ix[:, 'userid_skuid-brand_nearest_type5': 'userid_cate-whole_nearest_type5'],
                            predictset.ix[:, 'userid_skuid-brand_nearest_type6': 'userid_cate-whole_nearest_type6'],
                            predictset.ix[:, 'skuid_whole_6h_type4/1':'brand_whole_6h_type4/6'],
                            predictset.ix[:, 'skuid_whole_12h_type4/1':'brand_whole_12h_type4/6'],
                            predictset.ix[:, 'skuid_whole_24h_type4/1':'brand_whole_24h_type4/6'],
                            predictset.ix[:, 'skuid_whole_72h_type4/1':'brand_whole_72h_type4/6'],
                            predictset.ix[:, 'skuid_whole_168h_type4/1':'brand_whole_168h_type4/6'],
                            predictset.ix[:, 'skuid_whole_336h_type4/1':'brand_whole_336h_type4/6'],
                            predictset.ix[:, 'skuid_whole_before_type4/1':'brand_whole_before_type4/6'],
                            predictset.ix[:, 'skuid_6h_type-12h_type1':'skuid_6h_type-12h_type6'],
                            predictset.ix[:, 'brand_6h_type-12h_type1':'brand_6h_type-12h_type6'],
                            predictset.ix[:, 'cate_6h_type-12h_type1':'cate_6h_type-12h_type6'],
                            predictset.ix[:, 'whole_6h_type-12h_type1':'whole_6h_type-12h_type6'],
                            predictset.ix[:, 'skuid_12h_type-24h_type1':'skuid_12h_type-24h_type6'],
                            predictset.ix[:, 'brand_12h_type-24h_type1':'brand_12h_type-24h_type6'],
                            predictset.ix[:, 'cate_12h_type-24h_type1':'cate_12h_type-24h_type6'],
                            predictset.ix[:, 'whole_12h_type-24h_type1':'whole_12h_type-24h_type6'],
                            predictset.ix[:, 'skuid_24h_type-72h_type1':'skuid_24h_type-72h_type6'],
                            predictset.ix[:, 'brand_24h_type-72h_type1':'brand_24h_type-72h_type6'],
                            predictset.ix[:, 'cate_24h_type-72h_type1':'cate_24h_type-72h_type6'],
                            predictset.ix[:, 'whole_24h_type-72h_type1':'whole_24h_type-72h_type6'],
                            predictset.ix[:, 'skuid_72h_type-168h_type1':'skuid_72h_type-168h_type6'],
                            predictset.ix[:, 'brand_72h_type-168h_type1':'brand_72h_type-168h_type6'],
                            predictset.ix[:, 'cate_72h_type-168h_type1':'cate_72h_type-168h_type6'],
                            predictset.ix[:, 'whole_72h_type-168h_type1':'whole_72h_type-168h_type6'],
                            predictset.ix[:, 'skuid_168h_type-336h_type1':'skuid_168h_type-336h_type6'],
                            predictset.ix[:, 'brand_168h_type-336h_type1':'brand_168h_type-336h_type6'],
                            predictset.ix[:, 'cate_168h_type-336h_type1':'cate_168h_type-336h_type6'],
                            predictset.ix[:, 'whole_168h_type-336h_type1':'whole_168h_type-336h_type6'],
                            predictset.ix[:, 'skuid_336h_type-before_type1':'skuid_336h_type-before_type6'],
                            predictset.ix[:, 'brand_336h_type-before_type1':'brand_336h_type-before_type6'],
                            predictset.ix[:, 'cate_336h_type-before_type1':'cate_336h_type-before_type6'],
                            predictset.ix[:, 'whole_336h_type-before_type1':'whole_336h_type-before_type6'],
                          ], axis=1)
    predict_x = predictset.ix[:, 4:].values
    dpredict = xgb.DMatrix(predict_x)

    os.chdir('../../outcome/offline/')
    tmp1 = pd.concat([predictset, pd.DataFrame(modell.predict(dpredict), columns=['label'])],
                     axis=1).sort_values(
        by='label', ascending=False)

    # 获得结果
    best_res={'F11':0,'F12':0,'Score':0,'selected_items_number':0}
    for selected_items_number in range(100,3000,200):
        tmp = [tuple(i) for i in tmp1.iloc[:selected_items_number][['user_id', 'sku_id']].values]
        user_existed = []
        res = []
        for i in tmp:
            if i[0] not in user_existed:
                res.append(i)
                user_existed.append(i[0])
        '''
        Score=0.4*F11 + 0.6*F12
        此处的F1值定义为：
        F11=6*Recall*Precise/(5*Recall+Precise)
        F12=5*Recall*Precise/(2*Recall+3*Precise)
        其中，Precise为准确率，Recall为召回率.
        F11是label=1或0的F1值，F12是pred=1或0的F1值.
        '''
        test = [tuple(i) for i in pd.read_csv('test.csv').values]
        right1 = set([i[0] for i in res]) & set([i[0] for i in test])
        F11 = 6.0 * len(right1) / len(test) * len(right1) / len(res) / (
            5.0 * len(right1) / len(test) + 1.0 * len(right1) / len(res))
        right2 = set(res) & set(test)
        F22 = 5.0 * len(right2) / len(test) * len(right2) / len(res) / (
            2.0 * len(right2) / len(test) + 3.0 * len(right2) / len(res))
        Score = 0.4 * F11 + 0.6 * F22
        if Score>best_res['Score']:
            best_res['Score']=Score
            best_res['F11'] = F11
            best_res['F12'] = Score
            best_res['selected_items_number'] = selected_items_number
    print 'selected_items_number:{}'.format(best_res['selected_items_number'])
    print 'F11:{}'.format(best_res['F11'])
    print 'F12:{}'.format(best_res['F12'])
    print 'Score:{}'.format(best_res['Score'])

def train_and_predict(line=None, model=None,best_res=None):
    os.chdir('../trainset/' + line)
    trainset = pd.read_csv(line + '.csv').fillna(-1).replace(np.inf,-1)
    train_x, train_y = trainset.ix[:, 5:].values, trainset.ix[:, 4].values

    if best_res:
        selected_items_number=best_res['selected_items_number']
    else:
        selected_items_number=1200

    if model == 'rf':
        modell = RandomForestRegressor(
            n_estimators=500,
            max_depth=20,
            max_features='auto',
            n_jobs=-1,
            oob_score=True,
        )
        modell.fit(train_x, train_y)
    elif model == 'xgb':
        dtrain = xgb.DMatrix(train_x, label=train_y)
        params = {'max_depth': 10, 'silent': 1, 'eta': 0.1, 'colsample_bytree': 0.3, 'eval_metric': 'logloss',
                  'seed': 2017, 'objective': 'reg:logistic'}
        num_round = 400
        modell = xgb.train(params, dtrain, num_round)

    # 读取预测集
    os.chdir('../../predictset/' + line)
    predictset = pd.read_csv(line + '.csv').fillna(-1).replace(np.inf, -1)
    predict_x = predictset.ix[:, 4:].values
    dpredict = xgb.DMatrix(predict_x)

    #如果是为了得到基础模型，就保存模型
    if not best_res:
        if model == 'xgb':
            modell.save_model(line+'_'+model)
        elif model=='rf':
            joblib.dump(modell, line+'_'+model)

    os.chdir('../../outcome/' + line)
    if model == 'rf':
        tmp1 = pd.concat([predictset, pd.DataFrame(modell.predict(predict_x), columns=['label'])],
                         axis=1).sort_values(
            by='label', ascending=False)
    elif model == 'xgb':
        tmp1 = pd.concat([predictset, pd.DataFrame(modell.predict(dpredict), columns=['label'])],
                         axis=1).sort_values(
            by='label', ascending=False)

    #获得结果并保存
    tmp = [tuple(i) for i in tmp1.iloc[:selected_items_number][['user_id','sku_id']].values]
    user_existed = []
    res = []
    for i in tmp:
        if i[0] not in user_existed:
            res.append(i)
            user_existed.append(i[0])
    pd.DataFrame(res).to_csv(line + '.csv', index=False,header=['user_id','sku_id'])

    if line == 'offline':
        '''
        Score=0.4*F11 + 0.6*F12
        此处的F1值定义为：
        F11=6*Recall*Precise/(5*Recall+Precise)
        F12=5*Recall*Precise/(2*Recall+3*Precise)
        其中，Precise为准确率，Recall为召回率.
        F11是label=1或0的F1值，F12是pred=1或0的F1值.
        '''
        test = [tuple(i) for i in pd.read_csv('test.csv').values]
        right1 = set([i[0] for i in res]) & set([i[0] for i in test])
        F11 = 6.0 * len(right1) / len(test) * len(right1) / len(res) / (
            5.0 * len(right1) / len(test) + 1.0 * len(right1) / len(res))
        print 'F11:', F11
        right2 = set(res) & set(test)
        F22 = 5.0 * len(right2) / len(test) * len(right2) / len(res) / (
            2.0 * len(right2) / len(test) + 3.0 * len(right2) / len(res))
        print 'F12:', F22
        Score = 0.4 * F11 + 0.6 * F22
        print 'Score:', Score

if __name__ == '__main__':
    # 参考PPT：https://wenku.baidu.com/view/e12e33ba59eef8c75fbfb3f6.html?re=view
    # 用前2个星期的数据做特征
    # 样本用这2个星期最后1天加入购物车的样本        #最后一天未加入购物车的样本，倒数第二天产生交互的样本，分别构造三个数据集
    # 数据的标注用之后五天内，是否购买来进行标注
    print 'begin at ',datetime.datetime.now()
    #建立数据集
    #create()
    #os.chdir('/home/fengyufei/PycharmProjects/jd_competition/code')
    create2()
    os.chdir('/home/fengyufei/PycharmProjects/jd_competition/code')
    print 'all sets are created at',datetime.datetime.now()

    #先进行一次整体的训练，为了得到一个基础模型
    #train_and_predict(line='offline', model='xgb')
    #os.chdir('/home/fengyufei/PycharmProjects/jd_competition/code')
    #print 'offline model comes out at', datetime.datetime.now()

    # 特征选择
    #best_res=feature_selection(line='offline',model='xgb')
    #feature_selection_rfe()#一条龙服务
    #feature_selection_shoudong()
    #os.chdir('/home/fengyufei/PycharmProjects/jd_competition/code')
    #print 'feature selection ends at', datetime.datetime.now()

    #线上答案生成
    #train_and_predict(line='online', model='xgb')
    #print 'online result comes out at',datetime.datetime.now()

    print 'end at ', datetime.datetime.now()