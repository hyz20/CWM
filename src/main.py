from train_model2 import Learner2
from utils.set_seed import setup_seed
from utils.arg_parser import config_param_parser
import torch
import warnings
import os

warnings.filterwarnings("ignore")

def main():
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #print(device)
    #torch.cuda.set_device(1)
    #os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    parser = config_param_parser()
    args = parser.parse_args()
    setup_seed(args.randseed)
    if args.label_name == 'CWM':
        _learner = Learner2(args)
        _learner.train()


if __name__=="__main__":
    print('Start ...')
    main()
    print('End ...')