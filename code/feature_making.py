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

def create1():
    # 生成前面因为内存无法跑的全集用户特征，最后merge
    pass