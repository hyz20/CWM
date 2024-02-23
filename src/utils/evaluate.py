import torch
import copy
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from utils.metrics import NDCG,Precision, MRR, MRR_nobi
from sklearn.metrics import log_loss, mean_squared_error, mean_absolute_error, ndcg_score
from model.trans_model import Use_inverse_model, TransModel_inverse


def _get_pred(data_ld, model):
    pred = []
    with torch.no_grad():
        model.eval()
        for _id, batch in enumerate(data_ld):
            pred_bacth = model(batch[0]).view(batch[0].size(0))
            pred_bacth = pred_bacth.cpu().tolist()
            pred.extend(pred_bacth)
            # print(len(pred))
    return pred

def _cal_quantile(x, ls):
    # idx = np.clip(len(ls) * x,0,len(ls)-1)
    idx = (len(ls)-1)* x
    lower = int(np.floor(idx))
    upper = int(np.ceil(idx))
    return 0.5 * (ls[lower] + ls[upper])

def _cal_wm_watch_time(row, eps):
    rel = 1/(1 + np.exp(-row['pred']))
    return min((eps/np.log(1/rel)) - 1, row['duration_ms'])
    # return (eps/np.log(1/rel)) - 1

def _my_sigmoid(x):
    return 1/(1 + np.exp(-x))


def cal_reg_metric3(df, model, data_ld, k, eps):
    pred_rel = []
    pred_usr = []
    with torch.no_grad():
        model.eval()
        for _id, batch in enumerate(data_ld):
            logsigmoid = torch.nn.LogSigmoid()
            pred_rel_bacth = -logsigmoid(model(batch[0]).view(batch[0].size(0)))
            pred_rel_bacth = pred_rel_bacth.cpu().tolist()
            pred_rel.extend(pred_rel_bacth)

    df['pred_rel'] = pred_rel
    # df['pred_rel'] = df['pred_rel'].apply(lambda x: _my_sigmoid(x))
    df['pred_usr'] = [k] * len(df)

    df['pred_wt'] = df.apply(lambda row: np.clip((row['pred_usr']/(row['pred_rel'] + 1e-6)) - 1, 0 , row['duration_ms']), axis=1)

    rmse = np.sqrt(mean_squared_error(df['play_time_truncate'].values, df['pred_wt'].values))
    mae = mean_absolute_error(df['play_time_truncate'].values, df['pred_wt'].values)

    temp_group = df.groupby('user_id')
    group_df = temp_group['play_time_truncate'].apply(list).reset_index()
    group_df['pred_wt'] = temp_group['pred_wt'].apply(list).reset_index()['pred_wt']
    xgauc_ls = group_df.apply(lambda row: _cal_one_usr_xauc(row) ,axis=1)
    # gpnr_ls = group_df.apply(lambda row: _cal_one_usr_pnr(row) ,axis=1)
    weight_ls = group_df.apply(lambda row: _cal_one_usr_weight2(row) ,axis=1)
    xgauc = sum(xgauc_ls)/sum(weight_ls)
    xauc = xauc_score(df['play_time_truncate'].values, df['pred_wt'].values)

    return rmse, mae, xgauc, xauc


def cal_reg_metric(df, model, data_ld, df_all, label_name, eps):
    df = copy.deepcopy(df)
    pred = _get_pred(data_ld, model)
    df['pred'] = pred

    if label_name == 'play_time_truncate':
        # df['pred_wt'] = pred
        df['pred_wt'] = df.apply(lambda row: np.clip(row['pred'], 0, row['duration_ms']), axis=1)
        rmse = np.sqrt(mean_squared_error(df['play_time_truncate'].values, df['pred'].values))
        mae = mean_absolute_error(df['play_time_truncate'].values, df['pred'].values)
        
    
    if label_name == 'PCR':
        # df['pred_wt'] = df.apply(lambda row: _my_sigmoid(row['pred'])*row['duration_ms'], axis=1)
        df['pred_wt'] = df.apply(lambda row: np.clip(_my_sigmoid(row['pred'])*row['duration_ms'], 0, row['duration_ms']), axis=1)
        rmse = np.sqrt(mean_squared_error(df['play_time_truncate'].values, df['pred_wt'].values))
        mae = mean_absolute_error(df['play_time_truncate'].values, df['pred_wt'].values)
        
    
    if label_name == 'percentile':
        temple = df_all.groupby('quantile_bin')['play_time_truncate']
        bin_ls = temple.apply(lambda x: x.to_list())
        bin_ls = bin_ls.apply(lambda x: np.sort(x))
        # df['pred_wt'] = df.apply(lambda row: _cal_quantile(row['pred'],bin_ls[row['quantile_bin']]), 
        #                                        axis=1)
        df['pred_wt'] = df.apply(lambda row: np.clip(_cal_quantile(_my_sigmoid(row['pred']),bin_ls[row['quantile_bin']]), 0, row['duration_ms']), 
                                               axis=1)
        rmse = np.sqrt(mean_squared_error(df['play_time_truncate'].values, df['pred_wt'].values))
        mae = mean_absolute_error(df['play_time_truncate'].values, df['pred_wt'].values)
        
    
    if label_name == 'gain':
        # df['pred_wt'] = df.apply(lambda row: row['pred']*row['std_play']+row['mean_play'], axis=1)
        df['pred_wt'] = df.apply(lambda row: np.clip(row['pred']*row['std_play']+row['mean_play'], 0, row['duration_ms']), axis=1)
        rmse = np.sqrt(mean_squared_error(df['play_time_truncate'].values, df['pred_wt'].values))
        mae = mean_absolute_error(df['play_time_truncate'].values, df['pred_wt'].values)

    if label_name == 'weighted_wt':
        c_model = eps
        pred_c = _get_pred(data_ld, c_model)
        df['pred_c'] = pred_c
        # df['pred_wt'] = df.apply(lambda row: (_my_sigmoid(row['pred'])/(1 - _my_sigmoid(row['pred']) + 1e-6)) * (1 - _my_sigmoid(row['pred_c'])), axis=1)
        df['pred_wt'] = df.apply(lambda row: np.clip((_my_sigmoid(row['pred'])/(1 - _my_sigmoid(row['pred']) + 1e-6)) * (1 - _my_sigmoid(row['pred_c'])), 0, row['duration_ms']), axis=1)
        rmse = np.sqrt(mean_squared_error(df['play_time_truncate'].values, df['pred_wt'].values))
        mae = mean_absolute_error(df['play_time_truncate'].values, df['pred_wt'].values)

    if label_name == 'D2Co':
        df['pred_wt'] = df.apply(lambda row: np.clip(row['pred']*(row['posi_mean'] - row['nega_mean']) + row['nega_mean'], 0, row['duration_ms']), axis=1)
        rmse = np.sqrt(mean_squared_error(df['play_time_truncate'].values, df['pred_wt'].values))
        mae = mean_absolute_error(df['play_time_truncate'].values, df['pred_wt'].values)

    temp_group = df.groupby('user_id')
    group_df = temp_group['play_time_truncate'].apply(list).reset_index()
    group_df['pred_wt'] = temp_group['pred_wt'].apply(list).reset_index()['pred_wt']
    xgauc_ls = group_df.apply(lambda row: _cal_one_usr_xauc(row) ,axis=1)
    # gpnr_ls = group_df.apply(lambda row: _cal_one_usr_pnr(row) ,axis=1)
    weight_ls = group_df.apply(lambda row: _cal_one_usr_weight2(row) ,axis=1)
    xgauc = sum(xgauc_ls)/sum(weight_ls)
    # gpnr =  sum(gpnr_ls)/sum(weight_ls)

    xauc = xauc_score(df['play_time_truncate'].values, df['pred_wt'].values)
    # return rmse, mae, xgauc, gpnr
    return rmse, mae, xgauc, xauc
    

def cal_group_metric(df,model,posi,data_ld, label_name=None):
    df = copy.deepcopy(df)
    if label_name == 'NDT':
        pred_m = []
        pred_c = []
        with torch.no_grad():
            model.eval()
            for _id, batch in enumerate(data_ld):
                pred_bacth_m, pred_bacth_c = model(batch[0])
                pred_bacth_m = pred_bacth_m.view(batch[0].size(0))
                pred_bacth_c = pred_bacth_c.view(batch[0].size(0))
                pred_bacth_m = pred_bacth_m.cpu().tolist()
                pred_bacth_c = pred_bacth_c.cpu().tolist()
                pred_m.extend(pred_bacth_m)
                pred_c.extend(pred_bacth_c)
        df['pred'] = np.array(pred_m) + np.array(pred_c)
    else:
        pred = _get_pred(data_ld, model)
        df['pred'] = pred

    df['rank'] = df.groupby('user_id')['pred'].rank(method='first', ascending=False)
    temp = df.groupby('user_id').apply(lambda x: x.sort_values('rank', ascending=True))
    temp.reset_index(drop=True, inplace=True)
    temp_group = temp.groupby('user_id')
    group_df = temp_group['long_view2'].apply(list).reset_index()
    group_df['pred_list'] = temp_group['pred'].apply(list).reset_index()['pred']
    # group_df['vwt_list'] = temp_group['VWT'].apply(list).reset_index()['VWT'] 
    group_df['wt_list'] = temp_group['play_time_truncate'].apply(list).reset_index()['play_time_truncate'] 
    group_df['pcr_list'] = temp_group['PCR'].apply(list).reset_index()['PCR']
    # print(group_df.head(10))
    result_ls = []
    # vwt_result_ls = []
    wt_result_ls = []
    pcr_result_ls = []

    for p in posi:
        ndcg_eval = NDCG(p)
        # mrr_eval.evaluate(targets)
        ndcg_ls = group_df['long_view2'].apply(lambda x:ndcg_eval.evaluate(x))
        result_ls.append(ndcg_ls.mean())

        wt_eval = Precision(p)
        wt_ls = group_df.apply(lambda row:wt_eval.evaluate([1]*len(row['wt_list']),row['wt_list']),axis=1)
        wt_result_ls.append(wt_ls.mean())

        pcr_eval = Precision(p)
        pcr_ls = group_df.apply(lambda row:pcr_eval.evaluate([1]*len(row['pcr_list']),row['pcr_list']),axis=1)
        pcr_result_ls.append(pcr_ls.mean())

    MRR_eval = MRR_nobi(999)
    mrr_val = group_df['long_view2'].apply(lambda x:MRR_eval.evaluate(x)).mean()

    roc_auc_score_val = roc_auc_score(df['long_view2'].values, df['pred'].values)
    # roc_auc_score_val = 0
    
    return result_ls,pcr_result_ls,wt_result_ls,roc_auc_score_val,mrr_val



def cal_gauc(df,model,data_ld):
    df = copy.deepcopy(df)
    pred = _get_pred(data_ld, model)
    df['pred'] = pred
    df['rank'] = df.groupby('user_id')['pred'].rank(method='first', ascending=False)
    temp = df.groupby('user_id').apply(lambda x: x.sort_values('rank', ascending=True))
    temp.reset_index(drop=True, inplace=True)
    temp_group = temp.groupby('user_id')
    group_df = temp_group['long_view2'].apply(list).reset_index()
    group_df['pred_list'] = temp_group['pred'].apply(list).reset_index()['pred']
    gauc_ls = group_df.apply(lambda row: _cal_one_usr_auc(row) ,axis=1)
    weight_ls = group_df.apply(lambda row: _cal_one_usr_weight(row) ,axis=1)
    gauc = sum(gauc_ls)/sum(weight_ls)
    return gauc


def cal_ndcg1(df,model,data_ld,label_name):
    df = copy.deepcopy(df)
    pred = _get_pred(data_ld, model)
    df['pred'] = pred
    df['rank'] = df.groupby('user_id')['pred'].rank(method='first', ascending=False)
    temp = df.groupby('user_id').apply(lambda x: x.sort_values('rank', ascending=True))
    temp.reset_index(drop=True, inplace=True)
    temp_group = temp.groupby('user_id')
    group_df = temp_group['long_view2'].apply(list).reset_index()
    # group_df = temp_group[label_name].apply(list).reset_index()
    group_df['pred_list'] = temp_group['pred'].apply(list).reset_index()['pred']
    ndcg_eval = NDCG(1)
    # mrr_eval.evaluate(targets)
    ndcg_ls = group_df['long_view2'].apply(lambda x:ndcg_eval.evaluate(x))
    # ndcg_ls = group_df[label_name].apply(lambda x:ndcg_eval.evaluate(x))
    return ndcg_ls.mean()


def cal_ndcg2(df,model,c_model,data_ld):
    df = copy.deepcopy(df)
    pred_m = []
    pred_c = []
    with torch.no_grad():
        model.eval()
        for _id, batch in enumerate(data_ld):
            pred_bacth_m, pred_bacth_c = model(batch[0])
            pred_bacth_m = pred_bacth_m.view(batch[0].size(0))
            pred_bacth_c = pred_bacth_c.view(batch[0].size(0))
            pred_bacth_m = pred_bacth_m.cpu().tolist()
            pred_bacth_c = pred_bacth_c.cpu().tolist()
            pred_m.extend(pred_bacth_m)
            pred_c.extend(pred_bacth_c)
    df['pred'] = np.array(pred_m) + np.array(pred_c)
    # df['pred'] = np.array(pred_c) 

    # df = copy.deepcopy(df)
    # pred = []
    # pred_m = _get_pred(data_ld, model)
    # df['pred_m'] = pred_m
    # pred_c = _get_pred(data_ld, c_model)
    # df['pred_c'] = pred_c
    # df['pred'] = df['pred_m'] + df['pred_c']

    df['rank'] = df.groupby('user_id')['pred'].rank(method='first', ascending=False)
    temp = df.groupby('user_id').apply(lambda x: x.sort_values('rank', ascending=True))
    temp.reset_index(drop=True, inplace=True)
    temp_group = temp.groupby('user_id')
    group_df = temp_group['long_view2'].apply(list).reset_index()
    # group_df = temp_group[label_name].apply(list).reset_index()
    group_df['pred_list'] = temp_group['pred'].apply(list).reset_index()['pred']
    ndcg_eval = NDCG(1)
    # mrr_eval.evaluate(targets)
    ndcg_ls = group_df['long_view2'].apply(lambda x:ndcg_eval.evaluate(x))
    # ndcg_ls = group_df[label_name].apply(lambda x:ndcg_eval.evaluate(x))
    return ndcg_ls.mean()

def cal_vwt1(df,model,data_ld,label_name):
    df = copy.deepcopy(df)
    pred = _get_pred(data_ld, model)
    df['pred'] = pred
    df['rank'] = df.groupby('user_id')['pred'].rank(method='first', ascending=False)
    temp = df.groupby('user_id').apply(lambda x: x.sort_values('rank', ascending=True))
    temp.reset_index(drop=True, inplace=True)
    temp_group = temp.groupby('user_id')
    # group_df = temp_group['long_view2'].apply(list).reset_index()
    group_df = temp_group['VWT'].apply(list).reset_index()
    group_df['pred_list'] = temp_group['pred'].apply(list).reset_index()['pred']
    vwt_eval = Precision(1)
    # vwt_ls = group_df['long_view2'].apply(lambda x:vwt_eval.evaluate(x))
    vwt_ls = group_df['VWT'].apply(lambda x:vwt_eval.evaluate([1]*len(x),x))
    return vwt_ls.mean()

def _cal_one_usr_auc(row):
    if 0<sum(row['long_view2'])<len(row['long_view2']):
        return sum(row['long_view2']) * roc_auc_score(row['long_view2'],row['pred_list'])
    else:
        return 0

def _cal_one_usr_weight(row):
    if 0<sum(row['long_view2'])<len(row['long_view2']):
        return sum(row['long_view2'])
    else:
        return 0


def cal_xauc(y_true,y_pred):
    # must be np.array
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    
    y_pred = y_pred[np.argsort(y_true)[::-1]]
    y_true = np.sort(y_true)[::-1]

    sum_indicator_value = 0 
    for i in range(len(y_true)):
        for j in range(i+1,len(y_true)):
            if y_pred[i] > y_pred[j]:
                sum_indicator_value += 1
            elif y_pred[i] == y_pred[j]:
                sum_indicator_value += 0.5
    if len(y_true)>1:
        auc_val = (2 * sum_indicator_value)/(len(y_true) * (len(y_true) - 1))
    else:
        auc_val = sum_indicator_value
    return auc_val

def cal_pnr(y_true,y_pred):
    # must be np.array
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    
    y_pred = y_pred[np.argsort(y_true)[::-1]]
    y_true = np.sort(y_true)[::-1]

    sum_indicator_value = 0 
    for i in range(len(y_true)):
        for j in range(i+1,len(y_true)):
            if y_pred[i] > y_pred[j]:
                sum_indicator_value += 1
            elif y_pred[i] == y_pred[j]:
                sum_indicator_value += 0.5
    if len(y_true)>1:
        auc_val = (2 * sum_indicator_value)/(len(y_true) * (len(y_true) - 1) - 2 * sum_indicator_value)
    else:
        auc_val = sum_indicator_value
    return auc_val

def _cal_one_usr_xauc(row):
    if sum(row['play_time_truncate']) > 0:
        # return len(row['play_time_truncate']) * cal_xauc(row['play_time_truncate'],row['pred_wt'])
        return len(row['play_time_truncate']) * xauc_score(np.array(row['play_time_truncate']),np.array(row['pred_wt']))
    else:
        return 0
    
def _cal_one_usr_pnr(row):
    if sum(row['play_time_truncate']) > 0:
        return len(row['play_time_truncate']) * cal_pnr(row['play_time_truncate'],row['pred_wt'])
    else:
        return 0

def _cal_one_usr_weight2(row):
    if sum(row['play_time_truncate']) > 0:
        return len(row['play_time_truncate'])    
    else:
        return 0


import math
import numpy as np

class InversePairsCalc:
    def InversePairs(self, data):
        if not data :
            return False
        if len(data)==1 :
            return 0
        def merge(tuple_fir,tuple_sec):
            array_before = tuple_fir[0]
            cnt_before = tuple_fir[1]
            array_after = tuple_sec[0]
            cnt_after = tuple_sec[1]
            cnt = cnt_before+cnt_after
            flag = len(array_after)-1
            array_merge = []
            for i in range(len(array_before)-1,-1,-1):
                while array_before[i]<array_after[flag] and flag>=0 :
                    array_merge.append(array_after[flag])
                    flag -= 1
                if flag == -1 :
                    break
                else:
                    array_merge.append(array_before[i])
                    cnt += (flag+1)
            if flag == -1 :
                for j in range(i,-1,-1):
                    array_merge.append(array_before[j])
            else:
                for j in range(flag ,-1,-1):
                    array_merge.append(array_after[j])
            return array_merge[::-1],cnt

        def mergesort(array):
            if len(array)==1:
                return (array,0)
            cut = math.floor(len(array)/2)
            tuple_fir=mergesort(array[:cut])
            tuple_sec=mergesort(array[cut:])
            return merge(tuple_fir, tuple_sec)
        return mergesort(data)[1]

def xauc_score(labels, pres):
    label_preds = zip(labels.reshape(-1), pres.reshape(-1))
    sorted_label_preds = sorted(
        label_preds, key=lambda lc: lc[1], reverse=True)
    label_preds_len = len(sorted_label_preds)
    pairs_cnt = label_preds_len * (label_preds_len-1) / 2
    
    labels_sort = [ele[0] for ele in sorted_label_preds]
    S=InversePairsCalc()
    total_positive = S.InversePairs(labels_sort)
    if pairs_cnt == 0:
        xauc = 1
    else:
        xauc = total_positive / pairs_cnt
    return xauc