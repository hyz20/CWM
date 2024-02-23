import numpy as np
import pandas as pd
from scipy.stats import percentileofscore
from scipy.stats import norm
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
import sys
sys.path.append("..")
from utils.moving_avg import freq_moving_ave, moving_ave, weighted_moving_ave


def cal_pcr_label(df_dat):
    # df_dat['PCR'] = df_dat.apply(lambda row: row['play_time_truncate']/row['duration_ms'], axis=1)
    df_dat['PCR'] =  np.nan_to_num(df_dat['play_time_truncate'].values/df_dat['duration_ms'].values, nan=0.0, posinf=0.0, neginf=0.0)
    assert np.all(np.isfinite(df_dat['PCR'].values))
    print(np.all(np.isfinite(df_dat['PCR'].values)))
    return df_dat


def cal_gain_label(df_dat, dat_name):
    # if dat_name == 'KuaiRand':
    df_cnt_duration = df_dat[['duration_ms','play_time_truncate']].groupby('duration_ms').mean()
    df_cnt_duration.reset_index(inplace=True)
    df_cnt_duration.rename(columns={'play_time_truncate':'mean_play'},inplace = True)
    df_dat = pd.merge(df_dat, df_cnt_duration, on=['duration_ms'], how='left')

    df_cnt_duration = df_dat[['duration_ms','play_time_truncate']].groupby('duration_ms').std()
    df_cnt_duration.reset_index(inplace=True)
    df_cnt_duration.rename(columns={'play_time_truncate':'std_play'},inplace = True)
    df_dat = pd.merge(df_dat, df_cnt_duration, on=['duration_ms'], how='left')
    df_dat['std_play'] = np.nan_to_num(df_dat['std_play'], nan=0.0, posinf=0.0, neginf=0.0)

    # df_dat['gain'] = df_dat.apply(lambda row:(row['play_time_truncate'] - row['mean_play'])/row['std_play'] if row['std_play']!=0 else 0, axis=1)
    df_dat['gain'] = np.nan_to_num((df_dat['play_time_truncate'].values - df_dat['mean_play'].values)/df_dat['std_play'].values, nan=0.0, posinf=0.0, neginf=0.0)
    assert np.all(np.isfinite(df_dat['gain'].values))
    print(np.all(np.isfinite(df_dat['gain'].values)))

    # df_dat['gain_prob'] = df_dat['gain'].apply(lambda x:norm.cdf(x))
    df_dat['gain_prob'] = norm.cdf(df_dat['gain'].values)
    assert np.all(np.isfinite(df_dat['gain_prob'].values))
    print(np.all(np.isfinite(df_dat['gain_prob'].values)))
    return df_dat


def cal_quantile_label(df_dat, group_num):
    df_dat['quantile_bin'] = pd.qcut(df_dat['duration_ms'], group_num, labels=False, duplicates='drop')
    df_dat['wt_rank'] = df_dat.groupby('quantile_bin')['play_time_truncate'].rank(method='dense', ascending=True)
    group_sample_num = df_dat.groupby('quantile_bin')['wt_rank'].max().values
    df_dat['percentile'] = (df_dat['wt_rank'])/group_sample_num[df_dat['quantile_bin']]
    # print(df_dat.groupby('quantile_bin')['percentile'].max())
    # print(df_dat.groupby('quantile_bin')['percentile'].min())

    df_dat.drop(['wt_rank'], axis=1, inplace=True)
    # df_dat.drop(['mean_play'], axis=1, inplace=True)
    # df_dat.drop(['std_play'], axis=1, inplace=True)
    return df_dat


def cal_wlr_label(df_dat):
    qt_60 = df_dat['play_time_truncate'].quantile(0.6)
    print('qt_60:',qt_60)
    df_dat['over_q60'] = df_dat['play_time_truncate'].apply(lambda x: 1 if x > qt_60 else 0)
    df_dat['weighted_wt'] = df_dat['play_time_truncate'].apply(lambda x: x if x > qt_60 else 1)
    return df_dat


def cal_ndt_label(df_dat):
    df_dat['log_wt'] = np.log(df_dat['play_time_truncate'].values + 1)
    vr_bound = np.exp(df_dat['log_wt'].mean() - df_dat['log_wt'].std())
    print('vr_bound:', vr_bound)
    df_dat['valid_read'] = df_dat['play_time_truncate'].apply(lambda x: 1 if x > vr_bound else 0)
    # kuairand_dat['NDT'] = kuairand_dat['play_time_truncate'].apply(lambda x: (A/(1+np.exp(-(x-offset)/tau)))-B)
    df_dat['NDT'] = df_dat['play_time_truncate'].apply(lambda x: (2.319/(1+np.exp(-(x-vr_bound)/20)))-0.744)
    return df_dat


def cal_d2co_label(df_dat, windows_size, alpha):
    mm_ls = []
    min_duration = df_dat['duration_ms'].min()
    max_duration = df_dat['duration_ms'].max()
    print('min_duration:',min_duration)
    print('max_duration:',max_duration)
    for d in tqdm(np.arange(min_duration,max_duration+1,1)):
        X = df_dat[df_dat['duration_ms']==d]['play_time_truncate'].values
        X = X.reshape(-1,1)
        # all duration>5 
        if len(X) > 2:
            gm = GaussianMixture(n_components=2, init_params='kmeans',covariance_type='spherical', max_iter=500, random_state=61).fit(X)
            means = np.sort(gm.means_.T[0])
            stds = np.sqrt(gm.covariances_[np.argsort(gm.means_.T[0])])
            weights = gm.weights_[np.argsort(gm.means_.T[0])]
            mm_d = list(zip(means,weights))
            mm_ls.append([d, mm_d[0][0],mm_d[1][0], mm_d[1][1], stds[0], stds[1]])

    mm_ls = np.array(mm_ls)

    df_stat = df_dat[(df_dat['duration_ms']<=max_duration) & (df_dat['duration_ms']>=min_duration)]['duration_ms'].value_counts()
    freq_ls = df_stat.sort_index().values 

    nega_map_GMM_ma = dict(zip(mm_ls[:,0],freq_moving_ave(mm_ls[:,1], freq_ls, windows_size=windows_size)))
    posi_map_GMM_ma = dict(zip(mm_ls[:,0],freq_moving_ave(mm_ls[:,2], freq_ls, windows_size=windows_size)))
    

    df_dat['posi_mean'] = df_dat['duration_ms'].apply(lambda x: posi_map_GMM_ma[x])
    df_dat['nega_mean'] = df_dat['duration_ms'].apply(lambda x: nega_map_GMM_ma[x])
    df_dat['D2Co'] = (df_dat['play_time_truncate'] - df_dat['nega_mean']) / (df_dat['posi_mean'] - df_dat['nega_mean'] + 1e-6)

    return df_dat



def cal_baseline_label(df_dat, group_num, dat_name, windows_size):

    df_dat = cal_pcr_label(df_dat)

    df_dat = cal_gain_label(df_dat, dat_name)

    df_dat = cal_quantile_label(df_dat, group_num)

    df_dat = cal_wlr_label(df_dat)
    
    df_dat = cal_ndt_label(df_dat)
    
    df_dat = cal_d2co_label(df_dat, windows_size, alpha=0.1)

    return df_dat

if __name__=="__main__":
    pass