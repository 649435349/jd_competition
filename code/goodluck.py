# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd
import os
import random
import datetime

def create(line=None):
    os.chdir('../raw_data')
    action=pd.read_csv('action.csv')

    for i in pd.date_range('2016-02-01',)






if __name__=='__main__':
    create()