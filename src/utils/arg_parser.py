import argparse
from argparse import ArgumentTypeError

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')

def config_param_parser():
    parser = argparse.ArgumentParser(description="Experiment Configures and Model Parameters")
    parser.add_argument('--dat_name', type=str, choices=['KuaiRand','WeChat','KuaiShou2018'], required=True)
    parser.add_argument('--model_name', type=str, choices=['FM','DFM','AFM','NFM','AFI','DCN','xDFM'], required=True)
    parser.add_argument('--label_name', type=str, choices=['long_view2','PCR','PCR_denoise','play_time_truncate','play_time_truncate_denoise','percentile','percentile_denoise',
                                                            'gain','gain_denoise','gain_prob','gain_prob_denoise','GMM','scale_wt','D2Q','WTG','D2Co','BWT','WTG_denoise','D2Q_denoise','GMM_clip',
                                                            'WLR','NDT','JUMP','CWM','CWM2'], required=True)

    parser.add_argument('-g', '--group_num', type=int, default=30, help="Groups of percentile_label")
    parser.add_argument('-t', '--windows_size', type=int, default=10, help='Windows size of moving average')
    parser.add_argument('-s', '--sigma', type=float, default=0.3)
    parser.add_argument('--eps_usr', type=float, default=0.3)
    parser.add_argument('--c_inv', type=float, default=10, help='user cost inverse')

    # Experiment Configures
    # parser.add_argument('--fin', required=True)
    parser.add_argument('--fout',required=True)
    parser.add_argument('--use_cuda', type=str2bool, nargs='?', default=True)
    parser.add_argument('--load_to_eval', type=str2bool, nargs='?', default=False)
    parser.add_argument('--randseed', type=int , default=61)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--epoch_num', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--patience', type=int, default=5, help="waiting patience for early stop")
    parser.add_argument('--drop_out', type=float, default=0.1)
    return parser


if __name__=="__main__":
    parser = config_param_parser()
    args = parser.parse_args()
    print(args)
    