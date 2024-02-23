import numpy as np
import pandas as pd

def make_feature(df, data_name):
    if data_name == 'KuaiRand':
        fe_names = ['user_id', 'follow_user_num_range','register_days_range', 'fans_user_num_range', 'friend_user_num_range','user_active_degree',
                    'video_id', 'author_id', 'music_id','tag_pop', 'video_type', 'upload_type', 'tab']
    elif data_name == 'WeChat':
        fe_names = ['user_id', 'video_id', 'device', 'authorid','bgm_song_id', 'bgm_singer_id']
    return df[fe_names].values

#     others = ['date','duration_ms','play_time_truncate','play_time_truncate_denoise','denoise_wt','PCR','PCR_denoise',
#             'mean_play','std_play','gain','gain_prob', 'duration_bin',
#             'mean_play_denoise','std_play_denoise','gain_denoise','gain_prob_denoise',
#             'quantile_bin','percentile','percentile_denoise','GMM','long_view2','scale_wt','wt_bin',
#             "WTG",'D2Q','WTG_denoise','D2Q_denoise','D2Co','BWT','play_time_raw','GMM_clip','VWT','posi_watch','nega_watch',
#             'watch_num','watch_time_sum','watch_time_ave','watch_time_median','watch_video_avelen','watch_video_midlen',
#             'over_q60','weighted_wt','log_wt','valid_read','NDT']
#     fe_names = [i for i in df.columns if i not in others]
#     # print(fe_names)
#     return df[fe_names].values

    

def cal_field_dims(df, data_name):
    if data_name == 'KuaiRand':
        fe_names = ['user_id', 'follow_user_num_range','register_days_range', 'fans_user_num_range', 'friend_user_num_range','user_active_degree',
                    'video_id', 'author_id', 'music_id','tag_pop', 'video_type', 'upload_type', 'tab']
    elif data_name == 'WeChat':
        fe_names = ['user_id', 'video_id', 'device', 'authorid','bgm_song_id', 'bgm_singer_id']
    field_dims = [len(df[fe].unique()) for fe in fe_names]
    print(fe_names)
    print(field_dims)
    print([df[fe].max() for fe in fe_names])
    return field_dims

#     others = ['date','duration_ms','play_time_truncate','play_time_truncate_denoise','denoise_wt','PCR','PCR_denoise',
#             'mean_play','std_play','gain','gain_prob', 'duration_bin',
#             'mean_play_denoise','std_play_denoise','gain_denoise','gain_prob_denoise',
#             'quantile_bin','percentile','percentile_denoise','GMM','long_view2','scale_wt','wt_bin',
#             'WTG','D2Q','WTG_denoise','D2Q_denoise','D2Co','BWT','play_time_raw','GMM_clip','VWT','posi_watch','nega_watch',
#             'watch_num','watch_time_sum','watch_time_ave','watch_time_median','watch_video_avelen','watch_video_midlen',
#             'over_q60','weighted_wt','log_wt','valid_read','NDT']
#     fe_names = [i for i in df.columns if i not in others]
#     # print(df[fe_names].head(10))
#     field_dims = [len(df[fe].unique()) for fe in fe_names]
#     # rint(fe_names)
#     # print(field_dims)
#     return field_dims