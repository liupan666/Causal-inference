from pyspark.sql import SparkSession
import os
import pandas as pd
import numpy as np
import pickle
from commutil import tdw_util
#from IPython.display import Image
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression
#import matplotlib.pyplot as plt
#from scipy.stats import randint
from sklearn.ensemble import GradientBoostingRegressor
#import multiprocessing as mp
#from collections import defaultdict
#from commutil import spark_util


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import sys

import causalml
from causalml.metrics import plot_gain, plot_qini, qini_score
from causalml.dataset import synthetic_data
from causalml.inference.tree import plot_dist_tree_leaves_values, get_tree_leaves_mask
from causalml.inference.meta import BaseSRegressor, BaseXRegressor, BaseTRegressor, BaseDRRegressor
from causalml.inference.tree import CausalRandomForestRegressor
from causalml.inference.tree import CausalTreeRegressor
from causalml.inference.tree.plot import plot_causal_tree
from causalml.inference.meta import XGBTRegressor
from causalml.inference.meta import BaseSLearner, BaseTLearner, BaseRLearner, BaseXLearner
from causalml.inference.tree import UpliftRandomForestClassifier
from causalml.dataset import *
from causalml.metrics import *

#import seaborn as sns
from pytoolkit import TDWSQLProvider,TDWUtil

#%config InlineBackend.figure_format = 'retina'
np.random.seed(40)


ds = sys.argv[1]
#ds='20231125'
print(ds)
pri_parts = [ds]


class Config():
    def __init__(self):
        self.GROUP_ID = 'g_wxg_wxa_wxa_offline_datamining'
        self.GAIA_ID = '3527'
        self.SHOW_LOG = 1
        self.db_name = 'wxg_mmbiz_dw'
        #self.origin_table_name = 'dwmid_daily_wxaapp_store_positive_comment_mid_at'
        #self.res_table_name = 'res_daily_wxaapp_store_positive_comment_mid_at'


##评估函数 uplift柱状图
def uplift_by_percentile(y_true, uplift, treatment):
    y_true_temp, uplift_temp, treatment_temp = np.array(y_true), np.array(uplift), np.array(treatment)

    order = np.argsort(uplift_temp, kind='mergesort')[::-1]
    trmnt_flag = 1
    bins = 10
    
    # 对样本按照uplift值划分分位数
    y_true_bin = np.array_split(y_true_temp[order], bins)
    trmnt_bin = np.array_split(treatment_temp[order], bins)
    
    n_trmnt = np.array([len(y[trmnt == trmnt_flag]) for y, trmnt in zip(y_true_bin, trmnt_bin)]) #treatment组样本数
    n_ctrl = np.array([len(y[trmnt != trmnt_flag]) for y, trmnt in zip(y_true_bin, trmnt_bin)])  #control组样本数
    
    trmnt_avg = np.array([np.mean(y[trmnt == trmnt_flag]) for y, trmnt in zip(y_true_bin, trmnt_bin)]) #treatment组y_true均值
    ctrl_avg = np.array([np.mean(y[trmnt != trmnt_flag]) for y, trmnt in zip(y_true_bin, trmnt_bin)]) #control组y_true均值
    
    uplift_score = trmnt_avg - ctrl_avg   #相减得到uplift score
    
    percentiles = [round(p * 100 / bins) for p in range(1, bins + 1)]
    df = pd.DataFrame({'percentile': percentiles,
                       'n_treatment': n_trmnt,
                       'n_control': n_ctrl,
                       'treatment_avg': trmnt_avg,
                       'control_avg': ctrl_avg,
                       'uplift': uplift_score})

    df = df.set_index('percentile', drop=True, inplace=False) \
           .astype({'n_treatment': 'int32', 'n_control': 'int32'})
    
    return df


def my_plot_uplift_by_percentile(y_true, uplift, treatment):
    df = uplift_by_percentile(y_true, uplift, treatment)
    bins = 10
    percentiles = [round(p * 100 / bins) for p in range(1, bins + 1)]
    delta = percentiles[0]
    treatment_avg = df['treatment_avg']
    control_avg = df['control_avg']
    uplift = df['uplift']
    
    fig, axes = plt.subplots(ncols=1, nrows=2, figsize=(8, 6), sharex=True, sharey=True, facecolor='white')
    #fig.text(0.04, 0.5, 'Uplift = treatment response rate - control response rate', va='center', ha='center', rotation='vertical')

    axes[1].bar(np.array(percentiles) - delta/6, treatment_avg, delta/3, color='forestgreen', label='treatment')
    axes[1].bar(np.array(percentiles) + delta/6, control_avg, delta/3,color='orange', label='control')
    axes[0].bar(np.array(percentiles), uplift, delta/1.5, color='red', label='uplift')

    axes[0].legend(loc='upper right')
    axes[0].tick_params(axis='x', bottom=False)
    axes[0].axhline(y=0, color='black', linewidth=1)
    axes[0].set_title(f'Uplift by percentile')

    axes[1].set_xticks(percentiles)

    axes[1].legend(loc='upper right',prop = {'size':9})
    axes[1].axhline(y=0, color='black', linewidth=1)
    axes[1].set_xlabel('Percentile')
    axes[1].set_title('Response rate by percentile')

    plt.show()
    

#增益曲线
#AUQC，AUUC

def get_auqc_auuc(label,pred_uplift,treatment):
    # Area Under Qini Curve
    qini_coef = qini_auc_score(label,pred_uplift,treatment)
    # Area Under Uplift Curve
    uplift_auc = uplift_auc_score(label,pred_uplift,treatment)
    print("AUQC: ",qini_coef)
    print("AUUC: ",uplift_auc)
    return qini_coef,uplift_auc

# Model treatment effect
def tree_model_fit(tree_models,ctrees,df_train,df_test,full_features,t,label_tag):
    '''
        Fit the models in dict and return the result
        Args:
        ----
        tree_models : a dict to save the models
        ctrees: a dict of models name and model params
        df_train: train set
        df_test: test set
        df_raw_0: all of the dataset
        full_features: a list of all the covariates
        t: treatment
        label_tag: target variable
    '''
    for ctree_name, ctree_info in ctrees.items():
        print(f"Fitting: {ctree_name}")
        ctree = CausalTreeRegressor(**ctree_info['params'])
        ctree.fit(X=df_train[full_features].values,
                  treatment=df_train[t].values,
                  y=df_train[label_tag].values)
        tree_models[ctree_name] = ctree

        ctrees[ctree_name].update({'model': ctree})
        df_train[ctree_name] = ctree.predict(df_train[full_features].values)
        df_test[ctree_name] = ctree.predict(df_test[full_features].values)

    #return [df_test,df_raw_0]
    
    
def forest_model_fit(forest_models,cforests,df_train,df_test,full_features,t,label_tag):
    '''
        Fit the models in dict and add the uplift result to df
        Args:
        ----
        forest_models : a dict to save the models
        cforests: a dict of models name and model params
        df_train: train set
        df_test: test set
        df_raw_0: all of the dataset
        full_features: a list of all the covariates
        t: treatment
        label_tag: target variable
    '''   
    for cforest_name, cforest_info in cforests.items():
        print(f"Fitting: {cforest_name}")
        cforest = CausalRandomForestRegressor(**cforest_info['params'])
        cforest.fit(X=df_train[full_features].values,
                  treatment=df_train[t].values,
                  y=df_train[label_tag].values)
        # save the model in a dict
        forest_models[cforest_name] = cforest

        cforests[cforest_name].update({'model': cforest})
        df_train[cforest_name] = cforest.predict(df_train[full_features].values)
        df_test[cforest_name] = cforest.predict(df_test[full_features].values)
        





##主函数：运行uplift model
def run_uplift_model(pri_parts):
    spark = SparkSession.builder.appName("short_movie_message_uplift").getOrCreate()
    group_name = 'tl'
    db_name = 'wxg_mmbiz_dw'
    origin_table_name_1 = 'dwmid_daily_wxapp_drama_message_uin_feature_at'
    origin_table_name_2 = 'dwd_daily_wxapp_short_moive_uin_msg_optim_feature_predict_at'
    res_table_name = 'dwmid_daily_wxapp_short_moive_message_uin_uplift_at'
    tdw = TDWSQLProvider(spark, db=db_name)
    real_parts = ["p_" + ds for ds in pri_parts] if pri_parts else None
    df_raw = tdw.table(tblName=origin_table_name_1, priParts=real_parts)
    df_raw_0=df_raw.toPandas()
    
    df_raw_1 = tdw.table(tblName=origin_table_name_2, priParts=real_parts)
    df_raw_1=df_raw_1.toPandas()
    df_raw_1['treatment'] = df_raw_1['treatment'].fillna(1)
    

    #删除部分和treatment直接相关的协变量
    df_raw_0= df_raw_0.drop(columns=['exp_pv', 'clk_pv','drama_msg_send_num_7d','drama_msg_clk_num_7d','drama_msg_exp_num_7d'])
    df_raw_1= df_raw_1.drop(columns=['exp_pv', 'clk_pv','drama_msg_send_num_7d','drama_msg_clk_num_7d','drama_msg_exp_num_7d','last_tonow'])


    ##数据预处理

    # 填充包含空值的行
    # 对于object类型的列，用众数填充；对于非object类型的列，用均值填充
    df_fill_0 = df_raw_0.apply(lambda x: x.fillna(x.mode()[0]) if x.dtype == 'object' else x.fillna(x.mean()))
    df_raw_1.loc[df_raw_1['pay_amount_level'] == '0', 'pay_amount_level'] = '未知'
    df_raw_1.loc[df_raw_1['active_level'] == '0', 'active_level'] = '0-无效'
    df_raw_1.loc[df_raw_1['city_level'] == '0', 'city_level'] = '未知'
    df_fill_1 = df_raw_1.apply(lambda x: x.fillna(x.mode()[0]) if x.dtype == 'object' else x.fillna(x.mean()))


    # 独热编码将离散型特征转换为数值类型
    df_fill_encoded_0= pd.get_dummies(df_fill_0, columns=['gender','pay_amount_level','active_level','city_level'])
    df_fill_encoded_1= pd.get_dummies(df_fill_1, columns=['gender','pay_amount_level','active_level','city_level'])


    # 将DataFrame分为训练集和测试集
    df_fill_encoded_0_filtered = df_fill_encoded_0[df_fill_encoded_0['treatment'] == 0]
    df_test= df_fill_encoded_1
    df_train=df_fill_encoded_0
    column_names = df_fill_encoded_0.columns.tolist()
    #干预变量
    t='treatment'
    # 结果变量
    label_tag='pv'
    # 混淆变量
    full_features=[col for col in column_names if col not in [t, label_tag,'uin','ds']]
    full_features_sl=[col for col in column_names if col not in [label_tag,'uin','ds']]


    ##因果树
    ctrees = {
        'ctree_mse': {
            'params':
            dict(criterion='standard_mse',
                 control_name=0,
                 min_impurity_decrease=0,
                 min_samples_leaf=400,
                 groups_penalty=0.,
                 groups_cnt=True),
        },
        'ctree_cmse': {
            'params':
            dict(
                criterion='causal_mse',
                control_name=0,
                min_samples_leaf=400,
                groups_penalty=0.,
                groups_cnt=True,
                max_depth=6
            ),
        },
        'ctree_cmse_p=0.1': {
            'params':
            dict(
                criterion='causal_mse',
                control_name=0,
                min_samples_leaf=400,
                groups_penalty=0.1,
                groups_cnt=True,
                max_depth=6
            ),
        },
        'ctree_cmse_p=0.25': {
            'params':
            dict(
                criterion='causal_mse',
                control_name=0,
                min_samples_leaf=400,
                groups_penalty=0.25,
                groups_cnt=True,
                max_depth=6
            ),
        },
        'ctree_cmse_p=0.5': {
            'params':
            dict(
                criterion='causal_mse',
                control_name=0,
                min_samples_leaf=400,
                groups_penalty=0.5,
                groups_cnt=True,
                max_depth=6
            ),
        },
        'ctree_ttest': {
            'params':
            dict(criterion='t_test',
                 control_name=0,
                 min_samples_leaf=400,
                 groups_penalty=0.,
                 groups_cnt=True,
                 max_depth=6
                ),
        },
    }


    tree_models=dict()
    tree_model_fit(tree_models,ctrees,df_train,df_test,full_features,t,label_tag)

    ##因果森林
    cforests = {
        'cforest_mse': {
            'params':
            dict(criterion='standard_mse',
                 control_name=0,
                 min_impurity_decrease=0,
                 min_samples_leaf=400,
                 groups_penalty=0.,
                 groups_cnt=True),
        },
        'cforest_cmse': {
            'params':
            dict(
                criterion='causal_mse',
                control_name=0,
                min_samples_leaf=400,
                groups_penalty=0.,
                groups_cnt=True
            ),
        },
        'cforest_cmse_p=0.2': {
            'params':
            dict(
                criterion='causal_mse',
                control_name=0,
                min_samples_leaf=400,
                groups_penalty=0.2,
                groups_cnt=True
            ),
        },
        'cforest_cmse_p=0.2_md=5': {
            'params':
            dict(
                criterion='causal_mse',
                control_name=0,
                max_depth=5,
                min_samples_leaf=400,
                groups_penalty=0.2,
                groups_cnt=True
            ),
        },

        'cforest_cmse_p=0.5': {
            'params':
            dict(
                criterion='causal_mse',
                control_name=0,
                min_samples_leaf=400,
                groups_penalty=0.5,
                groups_cnt=True,
            ),
        },
        'cforest_cmse_p=0.5_md=5': {
            'params':
            dict(
                criterion='causal_mse',
                control_name=0,
                max_depth=5,
                min_samples_leaf=400,
                groups_penalty=0.5,
                groups_cnt=True,
            ),
        },
        'cforest_ttest': {
            'params':
            dict(criterion='t_test',
                 control_name=0,
                 min_samples_leaf=400,
                 groups_penalty=0.,
                 groups_cnt=True),
        },
    }

    ### Model treatment effect
    ##a dict to save the model
    forest_models=dict()
    forest_model_fit(forest_models,cforests,df_train,df_test,full_features,t,label_tag)


    ##训练集上的qini
    # 获取两个字典的键名
    result = list(ctrees.keys()) + list(cforests.keys())
    # 添加 t和 label_tag到列表中
    result.extend([t, label_tag])

    df_qini = qini_score(df_train[result],
               outcome_col=label_tag,
               treatment_col=t)

    df_qini.sort_values(ascending=False)


    #测试数据集的qini
    df_qini_all = qini_score(df_test[result],
               outcome_col=label_tag,
               treatment_col=t)

    df_qini_all.sort_values(ascending=False)


    #计算各模型的qini系数，并根据qini系数选出排序前6的model
    #result.extend(['learner_x_ite','learner_s_ite','learner_t_ite','learner_dr_ite'])
    df_qini = qini_score(df_train[result],
               outcome_col=label_tag,
               treatment_col=t)
    df_qini.sort_values(ascending=False)


    # 取排序后的前六个值
    top_six = df_qini.sort_values(ascending=False).head(6)
    result_top6 = top_six.index.tolist()
    result_top6.extend([t,label_tag])


    ##特征重要性

    # 创建一个新的DataFrame，包含特征名和对应的重要性
    df_importances = pd.DataFrame({
        'feature': full_features, 
        'importance': forest_models[result_top6[0]].feature_importances_
    })

    # 按照重要性降序排列
    df_importances = df_importances.sort_values('importance', ascending=False)

    # 绘制柱状图
    '''
    fig, ax = plt.subplots(figsize=(12, 4))  # 设置图形大小
    df_importances.plot.bar(x='feature', y='importance', ax=ax)
    ax.set_title("Causal Forest feature importances")
    ax.set_ylabel("Mean decrease in impurity")
    ax.set_xticklabels(df_importances['feature'], rotation='vertical')
    plt.show()
    '''

    # 按照重要性降序排列并选出前15个
    top15 = df_importances.nlargest(15, 'importance')

    # 将这15个特征的名字转换为list
    top15_features = top15['feature'].tolist()

    # 特征选择后再拟合因果树
    forest_models_new=dict()
    forest_model_fit(forest_models_new,cforests,df_train,df_test,top15_features,t,label_tag)

    # 特征选择后再拟合因果树
    tree_models_new=dict()
    tree_model_fit(tree_models_new,ctrees,df_train,df_test,top15_features,t,label_tag)

    ##整个数据集的qini
    df_qini_new = qini_score(df_test[result],
               outcome_col=label_tag,
               treatment_col=t)

    df_qini_new.sort_values(ascending=False)

    #选择效果最好的前6个模型
    top_six = df_qini_new.sort_values(ascending=False).head(6)
    result_top6 = top_six.index.tolist()
    result_top6.extend([t,label_tag])
    result_top6

    #测试集上的gain曲线
    '''
    plot_gain(df_test[result_top6], 
              outcome_col=label_tag, 
              treatment_col=t,
              n = df_test.shape[0])

    #全部数据上的gain曲线
    plot_gain(df_train[result_top6],
              outcome_col=label_tag,
              treatment_col=t,
              n = df_raw_0.shape[0])

    #若uplift by percentile图的趋势为单调递减的柱状图则表示模型是合理的
    my_plot_uplift_by_percentile(df_test[label_tag], df_test[result_top6[0]], df_test['treatment'])
    '''

    #阈值选择，保存结果到tdw
    sorted_df = df_train.sort_values(by=result_top6[0], ascending=False).reset_index()

    # 初始化gain_cum列
    sorted_df['gain_cum'] = 0

    # 计算每一行的gain_cum值
    for index, row in sorted_df.iterrows():
        treatment_rows = sorted_df.loc[:index, 'treatment'] == 1
        control_rows = sorted_df.loc[:index, 'treatment'] == 0

        treatment_indices = treatment_rows.to_numpy().nonzero()[0]
        control_indices = control_rows.to_numpy().nonzero()[0]

        treatment_pv_sum = sorted_df.loc[treatment_indices, 'pv'].sum()
        control_pv_sum = sorted_df.loc[control_indices, 'pv'].sum()

        treatment_count = treatment_rows.sum()
        control_count = control_rows.sum()

        #计算累积增益gain
        if treatment_count > 0 and control_count > 0:
            gain_cum = ((treatment_pv_sum / treatment_count) - (control_pv_sum / control_count)) * (index + 1)
            sorted_df.at[index, 'gain_cum'] = gain_cum

    max_gain_cum_index = sorted_df['gain_cum'].idxmax()
    max_gain_cum_value = sorted_df['gain_cum'].max()
    uplift_thereshold=sorted_df.loc[max_gain_cum_index, result_top6[0]] #阈值
    
    #取前50%的uplift预测值对应的数据
    df_test['is_high_uplift'] = 0
    df_test_rank = df_test.sort_values(by=result_top6[0], ascending=False).reset_index()
    half_index = int(len(df_test_rank) * 0.5)
    df_test_rank.loc[df_test_rank.index[:half_index], 'is_high_uplift'] = 1
    
    
    
    #df_test.loc[df_test[result_top6[0]] >= uplift_thereshold, 'is_high_uplift'] = 1

    #df_test.groupby('is_high_uplift')[result_top6[0]].mean()

    ##将结果保存到tdw表中
    #将pandas df转为spark df
    #sorted_df_result=sorted_df[['ds','uin','is_high_uplift']]
    sorted_df_result=df_test_rank[df_test_rank['treatment'] == 1][['ds', 'uin', 'is_high_uplift']]
    sorted_df_result_sp=spark.createDataFrame(sorted_df_result)
    tdw_util.write_df(spark,sorted_df_result_sp,'wxg_mmbiz_dw','dwmid_daily_wxapp_short_moive_message_uin_uplift_at',sorted_df['ds'].iloc[0])
    
    
    
    
if __name__ == '__main__':
    config = Config()
    os.environ['GROUP_ID'] = config.GROUP_ID
    os.environ['GAIA_ID'] = config.GAIA_ID
    ds = sys.argv[1]
    pri_parts = [ds]
    run_uplift_model(pri_parts)