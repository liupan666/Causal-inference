# _*_ coding:utf-8 _*_
import time
import requests
import json
import sys
import tensorflow as tf
import logging
import argparse
import os
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from scipy.stats import ttest_ind
from pyspark.sql.types import Row
from pyspark.sql import *
from pytoolkit import TDWSQLProvider, TDWUtil, TDWProvider, TableDesc, TableInfo
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import col
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import pyspark.sql.functions as F
from pyspark.sql.types import FloatType
from pytoolkit import TDWSQLProvider, TDWUtil, TDWProvider, TableDesc, TableInfo
from pyspark.ml.tuning import CrossValidatorModel
from pyspark.ml.feature import VectorAssembler
import pyspark.sql.functions as F
from pyspark.sql.types import FloatType, StringType
from pyspark.sql.functions import col,lit,split
import argparse
import tensorflow as tf
from pyspark.sql import SparkSession
from pytoolkit import TDWSQLProvider
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#XGB模型
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from xgboost import plot_importance
from matplotlib import pyplot
from commutil import spark_util
from commutil import tdw_util

#os.environ['GROUP_ID'] = 'g_wxg_wxplat_wxg_mmbiz_dw'
#os.environ['GAIA_ID'] = '3651'
#session = SparkSession.builder.config('spark.driver.memory', '16g').config('spark.executor.cores', 8).config('spark.executor.memory', '16g').getOrCreate()



ds = sys.argv[1]
#ds='20231125'
print(ds)
pri_parts = [ds]


# 实验组/对照组各特征差异显著性 
#target_feature为目标特征y，treatment为是否实验组
def compare(df,target_feature,treatment):   
    df_control = df[df[treatment]==0]
    df_treatment = df[df[treatment]==1]
    # 查看两组的均值
    print(df_control[target_feature].mean(), df_treatment[target_feature].mean()) 
 
    # t检验
    _, p = ttest_ind(df_control[target_feature], df_treatment[target_feature])
    print(f'p={p:.3f}')
 
    # 输出是否差异是否显著
    alpha = 0.05  # 显著水平设为 0.05
    if p > alpha:
        print('两者分布无差异（没有足够的证据拒绝原假设）')
    else:
        print('两者分布差异显著（拒绝原假设）')
        

# 计算SMD的方法
def SMD(d1, d2):
    # 计算样本数量
    n1, n2 = len(d1), len(d2)
    # 计算样本方差
    s1, s2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
    # 计算两组合并的标准差
    s = math.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    # 计算均值差
    u1, u2 = np.mean(d1), np.mean(d2)
    # 计算SMD标准平均差
    return (u1 - u2) / s

#输入数据集各特征的SMD
#treatment字段为是否实验组，cols为变量名组成的list
def feature_SMD(df,treatment,cols):
    df_control = df[df[treatment]==0]
    df_treatment = df[df[treatment]==1]
    effect_sizes = []
    for cl in cols:
        _, p_before = ttest_ind(df_control[cl], df_treatment[cl])
        #_, p_after = ttest_ind(df_matched_control[cl], df_matched_treatment[cl])
        SMD_before = SMD(df_treatment[cl], df_control[cl])       #SMD的绝对值在0.1以下则可认为差距很小
        #SMD_after = SMD(df_matched_treatment[cl], df_matched_control[cl])
        effect_sizes.append([cl,'before', SMD_before, p_before])
        #effect_sizes.append([cl,'after', SMD_after, p_after])
    #结果改为df
    df_effect_sizes = pd.DataFrame([x for x in effect_sizes], columns=['feature', 'matching', 'SMD', 'p-value'])
    return df_effect_sizes



#用逻辑回归模型计算ps（可选其他分类model）
#可增加model的分类效果来评判分类模型的性能（可选其他分类model）！！！！！！！！！
def compute_ps(train_X,train_y,pred_X,pred_y,df):
    ##1.建立逻辑回归模型（可选其他分类model）
    lr = LogisticRegression(max_iter=1000)
    lr.fit(train_X, train_y)
    # 使用逻辑回归模型对X进行概率预测
    y_pred_proba = lr.predict_proba(pred_X)
    
    # 在训练集上进行预测
    train_y_pred = lr.predict(train_X)
    # 在预测集上进行预测
    pred_y_pred = lr.predict(pred_X)
    # 计算训练集和预测集上的准确性
    train_accuracy = accuracy_score(train_y, train_y_pred)
    pred_accuracy = accuracy_score(pred_y, pred_y_pred)
    # 显示结果
    print("Training set accuracy:", train_accuracy)
    print("Prediction set accuracy:", pred_accuracy) 
    
    ##2.添加psm得分
    df['ps']=y_pred_proba[:,1]
    return df


##4.用knn为实验组元素匹配样本
#a. 首先为每个元素都进行knn，匹配得分范围在25%标准差内的10个最近的元素
def match(df,n_neighbors,ratio):
    #n_neighbors为匹配的元素个数，ratio为半径系数，caliper=np.std(df[ps])*ratio，ps为预测得分的列名
    #i：以ps得分标准差的25%作为半径，匹配相邻的10个元素，半径越大或者k越大，可匹配到的元素越多
    caliper = np.std(df['ps']) * ratio
    print(f'caliper (radius) is: {caliper:.4f}')
 
    #ii：拟合knn
    knn = NearestNeighbors(n_neighbors=n_neighbors, radius=caliper)
    ps = df[['ps']]  
    knn.fit(ps)
 
    #iii：返回每个点相邻点的索引和距离
    #distances：一个二维数组，其中每一行对应于 ps 中的一个点，每一列对应于该点的一个最近邻居。数组中的值是点与其最近邻居之间的距离。
    #neighbor_indexes：一个二维数组，与 distances 的形状相同。数组中的值是 ps 中点的最近邻居的索引。
    distances, neighbor_indexes = knn.kneighbors(ps)  
    
    #b.为实验组中的每个元素，在对照组中找到一个和它匹配的
    matched_control = []  # 保存对照组中匹配到的观测对象
 
    for current_index in df.index:  # 遍历df中的每行
        if df.loc[current_index,'treatment'] == 0:   # 如果当前行是对照组
            df.loc[current_index, 'matched'] = np.nan  # 将匹配到的对象设置为nan
        else:    # 如果当前行是实验组
            for idx in neighbor_indexes[current_index,:]: # 遍历实验组元素的10个最相邻元素，插入一个最近的符合条件的对照组样本
                # 排除10个相邻元素中的自己，且确保相邻元素都是对照组
                if (current_index != idx) and (df.loc[idx,'treatment'] == 0):
                    if idx not in matched_control:  # 且当前对象没被匹配过
                        df.loc[current_index, 'matched'] = idx  # 记录当前的匹配对象
                        matched_control.append(idx)  # 并将其插入到待保存的数组中
                        break
    return df

##可匹配重复的对照组个体
def match_0(df,n_neighbors,ratio):
    #n_neighbors为匹配的元素个数，ratio为半径系数，caliper=np.std(df[ps])*ratio，ps为预测得分的列名
    #i：以ps得分标准差的25%作为半径，匹配相邻的10个元素，半径越大或者k越大，可匹配到的元素越多
    caliper = np.std(df['ps']) * ratio
    print(f'caliper (radius) is: {caliper:.4f}')
 
    #ii：拟合knn
    knn = NearestNeighbors(n_neighbors=n_neighbors, radius=caliper)
    ps = df[['ps']]  
    knn.fit(ps)
 
    #iii：返回每个点相邻点的索引和距离
    #distances：一个二维数组，其中每一行对应于 ps 中的一个点，每一列对应于该点的一个最近邻居。数组中的值是点与其最近邻居之间的距离。
    #neighbor_indexes：一个二维数组，与 distances 的形状相同。数组中的值是 ps 中点的最近邻居的索引。
    distances, neighbor_indexes = knn.kneighbors(ps)  
    
    #b.为实验组中的每个元素，在对照组中找到一个和它匹配的
    matched_control = []  # 保存对照组中匹配到的观测对象
 
    for current_index in df.index:  # 遍历df中的每行
        if df.loc[current_index,'treatment'] == 0:   # 如果当前行是对照组
            df.loc[current_index, 'matched'] = np.nan  # 将匹配到的对象设置为nan
        else:    # 如果当前行是实验组
            for idx in neighbor_indexes[current_index,:]: # 遍历实验组元素的10个最相邻元素，插入一个最近的符合条件的对照组样本
                # 排除10个相邻元素中的自己，且确保相邻元素都是对照组
                if (current_index != idx) and (df.loc[idx,'treatment'] == 0):
                    df.loc[current_index, 'matched'] = idx  # 记录当前的匹配对象
                    matched_control.append(idx)  # 并将其插入到待保存的数组中
                    break
    return df

##分层匹配
def match_1(df,n_neighbors,ratio):
    #n_neighbors为匹配的元素个数，ratio为半径系数，caliper=np.std(df[ps])*ratio，ps为预测得分的列名
    #i：以ps得分标准差的25%作为半径，匹配相邻的10个元素，半径越大或者k越大，可匹配到的元素越多
    caliper = np.std(df['ps']) * ratio
    print(f'caliper (radius) is: {caliper:.4f}')
 
    #ii：拟合knn
    knn = NearestNeighbors(n_neighbors=n_neighbors, radius=caliper)
    ps = df[['ps']]  
    knn.fit(ps)
 
    #iii：返回每个点相邻点的索引和距离
    #distances：一个二维数组，其中每一行对应于 ps 中的一个点，每一列对应于该点的一个最近邻居。数组中的值是点与其最近邻居之间的距离。
    #neighbor_indexes：一个二维数组，与 distances 的形状相同。数组中的值是 ps 中点的最近邻居的索引。
    distances, neighbor_indexes = knn.kneighbors(ps)  
    original_indexes = df.iloc[neighbor_indexes.flatten()].index.values.reshape(neighbor_indexes.shape)
    #b.为实验组中的每个元素，在对照组中找到一个和它匹配的
    matched_control = []  # 保存对照组中匹配到的观测对象
 
    for current_index in df.index:  # 遍历df中的每行
        if df.loc[current_index,'treatment'] == 0:   # 如果当前行是对照组
            df.loc[current_index,'matched'] = np.nan  # 将匹配到的对象设置为nan
        else:    # 如果当前行是实验组
            for idx in original_indexes[df.index.get_loc(current_index),:]: # 遍历实验组元素的10个最相邻元素，插入一个最近的符合条件的对照组样本
                # 排除10个相邻元素中的自己，且确保相邻元素都是对照组
                if (current_index != idx) and (df.loc[idx,'treatment'] == 0):
                    if idx not in matched_control:  # 且当前对象没被匹配过
                        df.loc[current_index, 'matched'] = idx  # 记录当前的匹配对象
                        matched_control.append(idx)  # 并将其插入到待保存的数组中
                        break
    return df

##5.对比匹配前后各个特征在实验组、对照组上的SMD的绝对值、p值
#tm为是否为实验组的列名，cols为数据集特征构成的list
def feature_SMD_before_after(df,treatment,cols):
    df[treatment] = df[treatment].astype(int)
    df_control = df[df[treatment]==0]
    df_treatment = df[df[treatment]==1]
    df_matched_treatment=df[df['matched'].notna()]
    df_matched_control=df.loc[df['matched'].dropna().astype(int)]
    group_names = {'匹配前对照组数目': df_control,'匹配前实验组数目': df_treatment,'匹配后对照组数目': df_matched_control,'匹配后实验组数目': df_matched_treatment}
    for name, df in group_names.items():
        print(f'{name}: {len(df)}')
    effect_sizes = []

    for cl in cols:
        _, p_before = ttest_ind(df_control[cl], df_treatment[cl])
        _, p_after = ttest_ind(df_matched_control[cl], df_matched_treatment[cl])
        SMD_before = abs(SMD(df_treatment[cl], df_control[cl]) )      #SMD的绝对值在0.1以下则可认为差距很小
        SMD_after = abs(SMD(df_matched_treatment[cl], df_matched_control[cl]))
        effect_sizes.append([cl,'before', SMD_before, p_before])
        effect_sizes.append([cl,'after', SMD_after, p_after])
    #结果改为df
    df_effect_sizes = pd.DataFrame([x for x in effect_sizes], columns=['feature', 'matching', 'SMD', 'p-value'])
    return df_effect_sizes




def main():
    spark = spark_util.session(app_name='biz_title')
    #spark=session
    df_raw_0 = tdw_util.read_df(spark, db_name="wxg_mmbiz_dw",
                          table_name="dwmid_daily_wxapp_short_moive_uin_stat_dyn_feature_at",
                          pri_parts=pri_parts).toPandas()
    df_level=tdw_util.read_df(spark, db_name="wxg_mmbiz_dw",
                          table_name="dwmid_daily_wxapp_short_moive_uin_feature_level_at",
                          pri_parts=pri_parts).toPandas()
    df_level_copy=df_level.copy()
    
    
    ###数据预处理
    ##缺失值的处理!!!!!!
    # 计算 DataFrame 中每列的众数
    mode_values = df_raw_0.mode().iloc[0]
    # 用众数填充缺失数据
    df_raw_0 = df_raw_0.fillna(mode_values)
    df_raw_0 = df_raw_0.reset_index(drop=True)
    df_raw = df_raw_0.drop(['ds'],axis=1)
    # 删除包含null值的行
    df_raw = df_raw.dropna()
    #将treatment改为int型
    df_raw['treatment'] = df_raw['treatment'].astype(int)
    #重新设置df的索引
    df = df_raw.reset_index(drop=True)
    df_raw = df_raw.reset_index(drop=True)
    #将gender列化为分类变量
    df['gender'] = df['gender'].replace({'1': '男', '2': '女'})

    #对分类变量进行独热编码
    df = pd.get_dummies(df, columns=['city_level', 'gender'])


    #对数值型变量进行标准化
    # 初始化StandardScaler或MinMaxScaler
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    #scaler = StandardScaler()
    scaler = MinMaxScaler()

    # 对数值型变量列进行归一化（或标准化）
    numerical_columns = ['staytime_long','pay_amount_1d','pay_amount_30d','pv','history_pv','drama_num','history_drama_num',
                         'wxapp_num','history_wxapp_num','first_tonow','last_tonow','age']
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    df_1=df
    
    
    #查看匹配前两组的差异及显著性
    compare(df_raw,'pv','treatment')
    #查看为匹配前两组各协变量之间的差异情况
    cols = ['staytime_long','pay_amount_1d','pay_amount_30d','pv','history_pv','drama_num','history_drama_num',
        'wxapp_num','history_wxapp_num','first_tonow','last_tonow','age']
    feature_SMD(df,'treatment',cols)
    
    
    #把数据集分为训练集和测试集，训练集用来训练模型，测试集用来测试模型的效果
    seed = 20231212
    # 计算 'treatment' 列等于 1 的数据的数量
    count_treatment_1 = (df['treatment'] == 1).sum()

    # 从 'treatment' 列为 0 的数据中随机选择与 'treatment' 列等于 1 的数据的数量相同的数据
    df1 = df[df['treatment'] == 0].sample(n=count_treatment_1)

    # 将 'treatment' 列为 1 的数据与 df1 合并以创建新的 DataFrame df2
    df2 = pd.concat([df1,df[df['treatment'] == 1]], ignore_index=True)

    #不加入pv列
    X = df2.drop(['treatment','uin'], axis=1)
    Y = df2['treatment']

    X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                        test_size=0.33, 
                                                        random_state=seed)

    # 直接使用xgboost开源项目中封装好的分类器和回归器，可以直接使用XGBClassifier建立模型
    # 可视化测试集的loss
    # 改为True就能可视化loss
    xgboost_model = XGBClassifier()
    eval_set = [(X_test, y_test)]
    xgboost_model.fit(X_train, 
                      y_train, 
                      early_stopping_rounds=10, 
                      eval_metric="logloss",  # 损失函数的类型，分类一般都是用对数作为损失函数
                      eval_set=eval_set,
                      verbose=False)

    # xgboost的结果是每一个样本属于第一类的概率，要使用round将其转换为0 1值
    y_pred = xgboost_model.predict( X_test )
    predictions = [round(i) for i in y_pred] 

    # 计算准确率，也就是把准确率作为衡量模型好坏的指标
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
      
    
    # 设定要调节的各参数组合的dict
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [2,3,5,8],
        'learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3] ,
        'subsample': [0.3,0.5,0.8,1.0],}
    #k折交叉验证
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed) #test：n=3

    # 创建 XGBClassifier 模型
    model = XGBClassifier()
    # 创建 RandomizedSearchCV 对象
    random_search = RandomizedSearchCV(model, param_grid, scoring='accuracy', n_jobs=-1, verbose=1,cv=kfold,n_iter=40) #test：n=2

    # 使用训练数据进行网格搜索
    random_result=random_search.fit(X_train, y_train)

    # 输出最佳学习率和其对应的分数
    print("Best: %f using %s" % (random_result.best_score_, random_result.best_params_))

    # 使用最佳参数组合拟合模型
    best_model = random_search.best_estimator_

    # 预测测试集
    y_pred = best_model.predict(X_test)

    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    
    best_model.fit(X, Y)
    plot_importance( best_model )
    pyplot.show()
    
    # 预测测试集的概率
    X_all=df.drop(['treatment','uin'], axis=1)
    Y_all = df['treatment']
    y_pred_proba = best_model.predict_proba(X_all)
    df['ps']=y_pred_proba[:,1]
    
    ##查看预测ps后实验组和对照组ps的分布
    sns.histplot(data=df, x='ps', hue='treatment')
    # 在直方图上添加概率分布曲线
    sns.kdeplot(data=df[df['treatment'] == 0]['ps'], label='treatment=0')
    sns.kdeplot(data=df[df['treatment'] == 1]['ps'], label='treatment=1')
    # 添加图例
    plt.legend()
    # 显示图形
    plt.show()
    
    #根据ps进行匹配，并输出匹配后的df
    #参数分别为数据框，匹配个数n，ratio控制匹配半径
    df=match(df,10,0.5)
    
    ##将匹配结果带入原始数据框中，可得到对应uin
    df_raw_0['matched']=df['matched']
    matched = df_raw_0['matched'].dropna()
    df_raw_0_copy=df_raw_0.copy()
    df_raw_0_copy=df_raw_0_copy[df_raw_0_copy['matched'].notnull()]
    #完成匹配的实验组数据
    df_raw_0_copy['matched_uin'] = df_raw_0.loc[matched, 'uin'].values
    df_raw_0_copy
    
    #删除matched列
    df_raw_0_copy = df_raw_0_copy.drop('matched', axis=1)
    #变更列的类型为所创建的tdw表的列类型
    df_raw_0_copy = df_raw_0_copy.astype({'first_tonow': 'int64', 'last_tonow': 'int64','age':'int64'})
    
    ##将匹配后的用户特征以及被匹配用户存储到一个tdw表！！！
    #将pandas df转为spark df
    df_raw_0_copy_sp=spark.createDataFrame(df_raw_0_copy) 
    tdw_util.write_df(spark, df_raw_0_copy_sp, db_name='wxg_mmbiz_dw',
                      table_name='dwmid_daily_wxapp_short_moive_uin_feature_match_at', pri_part=ds,overwrite=True)
    
    #查看匹配前后数据集各特征的SMD
    df_effect_sizes_1=feature_SMD_before_after(df_raw_0,'treatment',cols)
    #用柱状图更直观的输出各特征SMD绝对值值的前后差异
    fig, ax = plt.subplots(figsize=(15, 5))
    sns.barplot(data=df_effect_sizes_1, x='SMD', y='feature', hue='matching', orient='h')
    
    
   


if __name__ == '__main__':
    main()