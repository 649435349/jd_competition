# jd_competition
README
===========================
****
### Author:冯宇飞
### E-mail:649435349@qq.com
### Wechat:fyf649435349
****

## What is it?
```
仅作校招参考使用……
京东大赛代码，题目链接为http://www.datafountain.cn/#/competitions/247/ranking/a/0
简单来说是预测在未来五天内可能发生的购买行为
```
   
## What is used and what is the performance?
```
全部用Python写成，Python也是我最喜欢的语言。
用了大量的库,如Numpy,Pandas,Sklearn,Seaborn,Xgboost……
更多的数据工作在ipython中实现了。
最后ranked TOP4%(实习期间实在是没精力打比赛，太累了……)
第二次参加比赛，熟练使用了pandas之后感觉爽爆了
```

## The Structure of the project?
```
`./analyse.py`初步分析数据，生成图表
`./oodluck.py`建立数据集（没有用mysql，数据集保存下来，添加特征时导入再加），整合其他特征
`./goodluck.py`主要文件，作用：整合数据，构造特征，特征选择，建立数据集，训练，输出结果，交叉验证,线下测试……
`./utils.py`一些有用的函数
`./mysql.py`多进程插入mysql
`./preprossing.py`预处理数据
```
