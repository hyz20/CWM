import pandas as pd
import numpy as np
from scipy.stats import norm


def cal_ground_truth(df_dat, data_name):

    def decide_long(row, data_name, w_qt_h):
        if data_name=='KuaiRand':

            if (row['play_time_ms']>=row['duration_ms']) & (row['duration_ms']<=w_qt_h):
                return 1
            elif (row['play_time_truncate']>w_qt_h) & (row['duration_ms']>w_qt_h):
                return 1
            else:
                return 0

        elif data_name=='WeChat':

            if (row['play_time_ms']>=row['duration_ms']) & (row['duration_ms']<=w_qt_h):
                return 1
            elif (row['play_time_truncate']>w_qt_h) & (row['duration_ms']>w_qt_h):
                return 1
            else:
                return 0
    
    w_qt_h = df_dat['play_time_truncate'].quantile(0.70) #23
    print(w_qt_h)
    df_dat['long_view2'] = df_dat.apply(lambda row: decide_long(row,data_name,w_qt_h), axis=1)
    # df_dat = df_dat[df_dat['duration_ms']>w_qt_h]
    return df_dat



if __name__=="__main__":
    pass