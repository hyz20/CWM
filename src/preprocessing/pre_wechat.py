import pandas as pd 
import numpy as np


def pre_wechat():
    action = pd.read_csv("../rec_datasets/WeChat_data/user_action.csv")
    feed = pd.read_csv("../rec_datasets/WeChat_data/feed_info.csv")
    feed = feed[['feedid','authorid','videoplayseconds','bgm_song_id','bgm_singer_id']]
    df_wechat = pd.merge(left=action,right=feed,on=['feedid'],how='left')
    # df_wechat = df_wechat[df_wechat.videoplayseconds<=60]

    df_wechat['play'] = np.round(df_wechat['play']/1e3)
    df_wechat['play_time_truncate'] = np.clip(df_wechat['play'],0, df_wechat['videoplayseconds'])
    df_wechat.rename(columns={'play':'play_time_ms',
                          'videoplayseconds':'duration_ms',
                          'date_':'date'},inplace=True)
    
    df_wechat['bgm_song_id'] =  df_wechat['bgm_song_id'].apply(lambda x: 99999 if pd.isna(x) else x)
    df_wechat['bgm_singer_id'] =  df_wechat['bgm_song_id'].apply(lambda x: 99999 if pd.isna(x) else x)

    
    df_sel_dat = df_wechat[(df_wechat.duration_ms >= 5) & (df_wechat.duration_ms <= 60)]
    df_sel_dat = df_sel_dat[['date','userid', 'feedid', 'device', 'authorid','bgm_song_id', 'bgm_singer_id',
                            'like', 'forward','follow', 'comment','read_comment',
                            'duration_ms','play_time_ms','play_time_truncate']]
    
    userid_map = dict(zip(np.sort(df_sel_dat['userid'].unique()),range(len(df_sel_dat['userid'].unique()))))
    feedid_map = dict(zip(np.sort(df_sel_dat['feedid'].unique()),range(len(df_sel_dat['feedid'].unique()))))
    authorid_map = dict(zip(np.sort(df_sel_dat['authorid'].unique()),range(len(df_sel_dat['authorid'].unique()))))
    bgm_song_id_map = dict(zip(np.sort(df_sel_dat['bgm_song_id'].unique()),range(len(df_sel_dat['bgm_song_id'].unique()))))
    bgm_singer_id_map = dict(zip(np.sort(df_sel_dat['bgm_singer_id'].unique()),range(len(df_sel_dat['bgm_singer_id'].unique()))))


    df_sel_dat['userid'] = df_sel_dat['userid'].apply(lambda x: userid_map[x])
    df_sel_dat['feedid'] = df_sel_dat['feedid'].apply(lambda x: feedid_map[x])
    df_sel_dat['authorid'] = df_sel_dat['authorid'].apply(lambda x: authorid_map[x])
    df_sel_dat['bgm_song_id'] = df_sel_dat['bgm_song_id'].apply(lambda x: bgm_song_id_map[x])
    df_sel_dat['bgm_singer_id'] = df_sel_dat['bgm_singer_id'].apply(lambda x: bgm_singer_id_map[x])

    df_sel_dat.rename(columns={'userid':'user_id',
                          'feedid':'video_id'},inplace=True)

    return df_sel_dat
    # 检测df_wechat是否有空值,并且定位到空值
    # df_wechat.isnull().any()
    # df_wechat[df_wechat.isnull().values==True]
    # df_wechat[df_wechat['bgm_song_id'].isnull().values==True]

