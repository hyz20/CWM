import time
import torch
import pandas as pd
import numpy as np
from scipy.stats import norm
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam,Adadelta,RMSprop,SGD
from torch.nn import BCELoss,BCEWithLogitsLoss,MSELoss
from torch.optim.lr_scheduler import ExponentialLR
from sklearn.preprocessing import StandardScaler
# from torchfm.model.afi import AutomaticFeatureInteractionModel
# from torchfm.model.nfm import NeuralFactorizationMachineModel 
# from torchfm.model.afm import AttentionalFactorizationMachineModel
# from torchfm.model.dfm import DeepFactorizationMachineModel
# from torchfm.model.fm import FactorizationMachineModel
from model.fm import My_FactorizationMachineModel,Usr_FactorizationMachineModel
from model.dfm import My_DeepFactorizationMachineModel,Usr_DeepFactorizationMachineModel
from model.afi import My_AutomaticFeatureInteractionModel,Usr_AutomaticFeatureInteractionModel
from model.dcn import My_DeepCrossNetworkModel,Usr_DeepCrossNetworkModel
from model.xdfm import My_ExtremeDeepFactorizationMachineModel,Usr_ExtremeDeepFactorizationMachineModel
from model.loss_func import CWMLoss
from utils.set_seed import setup_seed
from utils.summary_dat import cal_field_dims, make_feature
from utils.data_wrapper import Wrap_Dataset, Wrap_Dataset2
from utils.early_stop import EarlyStopping, EarlyStopping2
from utils.evaluate import cal_gauc, cal_group_metric, cal_ndcg1, cal_vwt1, cal_reg_metric, cal_reg_metric3
from preprocessing.cal_ground_truth import cal_ground_truth

class Learner2(object):
    
    def __init__(self, args):
        self.dat_name = args.dat_name
        self.model_name = args.model_name
        self.label_name = args.label_name

        self.group_num = args.group_num
        self.windows_size = args.windows_size
        self.eps = args.sigma

        self.batch_size = args.batch_size
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.patience = args.patience
        self.use_cuda = args.use_cuda
        self.epoch_num = args.epoch_num
        self.seed = args.randseed
        self.fout = args.fout

        self.k = args.c_inv
        self.load_to_eval = args.load_to_eval


    def train(self):
        setup_seed(self.seed)
        self.all_dat, self.train_dat, self.vali_dat, self.test_dat, self.usr_dat= self._load_and_spilt_dat()
        self.train_loader, self.vali_loader, self.test_loader = self._wrap_dat()
        self.model, self.usr_model, self.optim, self.usr_optim, self.early_stopping, self.scheduler, self.usr_scheduler = self._init_train_env()
        if not self.load_to_eval:
            self._train_iteration()
        self._test_and_save()

    def _load_and_spilt_dat(self):
        if self.dat_name == 'KuaiRand':
            all_dat = pd.read_csv('../rec_datasets/WM_KuaiRand/KuaiRand_subset.csv') 
            all_dat = cal_ground_truth(all_dat, self.dat_name)

            train_dat = all_dat[(all_dat['date']<=20220421) & (all_dat['date']>=20220408)]
            vali_dat = all_dat[(all_dat['date']<=20220428) & (all_dat['date']>=20220422)]
            test_dat = all_dat[(all_dat['date']<=20220508) & (all_dat['date']>=20220429)]

        elif self.dat_name == 'WeChat':
            all_dat = pd.read_csv('../rec_datasets/WM_WeChat/WeChat_subset.csv')
            all_dat = cal_ground_truth(all_dat, self.dat_name)
            
            all_dat2 = all_dat[all_dat['duration_ms']<60]
            train_dat = all_dat2[(all_dat2['date']<=10) & (all_dat2['date']>=1)]
            vali_dat = all_dat2[(all_dat2['date']<=12) & (all_dat2['date']>=11)]
            test_dat = all_dat2[(all_dat2['date']<=14) & (all_dat2['date']>=13)]

        usr_dat = 0
        return all_dat, train_dat, vali_dat, test_dat, usr_dat


    def _wrap_dat(self):

        input_train = Wrap_Dataset2(make_feature(self.train_dat, self.dat_name),
                                self.train_dat['play_time_truncate'].tolist(),
                                self.train_dat['duration_ms'].tolist(),
                                self.train_dat['duration_ms'].tolist())
        train_loader = DataLoader(input_train, 
                                        batch_size=self.batch_size, 
                                        shuffle=True)
        print('Train dat is loaded...')

        input_vali = Wrap_Dataset2(make_feature(self.vali_dat, self.dat_name),
                                self.vali_dat['play_time_truncate'].tolist(),
                                self.vali_dat['duration_ms'].tolist(),
                                self.vali_dat['duration_ms'].tolist())
        vali_loader = DataLoader(input_vali, 
                                        batch_size=2048, 
                                        shuffle=False)
        print('Vali dat is loaded...')

        input_test = Wrap_Dataset2(make_feature(self.test_dat, self.dat_name),
                                self.test_dat['play_time_truncate'].tolist(),
                                self.test_dat['duration_ms'].tolist(),
                                self.test_dat['duration_ms'].tolist())
        test_loader = DataLoader(input_test, 
                                        batch_size=2048, 
                                        shuffle=False)
        print('Test dat is loaded...')
        return train_loader, vali_loader, test_loader

    
    def _init_train_env(self):
        if self.model_name == 'AFI':
            model = My_AutomaticFeatureInteractionModel(field_dims=cal_field_dims(self.all_dat, self.dat_name), 
                                                        embed_dim=10, 
                                                        num_heads=8, 
                                                        num_layers=1,
                                                        atten_embed_dim=64,
                                                        mlp_dims=[64], 
                                                        dropouts=[0.2,0.2])
        elif self.model_name == 'DFM':
            model = My_DeepFactorizationMachineModel(field_dims=cal_field_dims(self.all_dat, self.dat_name), embed_dim=10, mlp_dims=[64,64,64], dropout=0.2)
        elif self.model_name == 'FM':
            model = My_FactorizationMachineModel(field_dims=cal_field_dims(self.all_dat, self.dat_name), embed_dim=10)
        elif self.model_name == 'DCN':
            model = My_DeepCrossNetworkModel(field_dims=cal_field_dims(self.all_dat, self.dat_name), embed_dim=10, num_layers=3, mlp_dims=[64,64,64], dropout=0.2)
        elif self.model_name == 'xDFM':
            model = My_ExtremeDeepFactorizationMachineModel(field_dims=cal_field_dims(self.all_dat, self.dat_name), embed_dim=10, mlp_dims=[64,64,64], dropout=0.2, cross_layer_sizes=[64,64,64], split_half=True)

        
        # usr_model = Usr_FactorizationMachineModel(field_dims=[len(self.usr_dat[fe].unique()) for fe in self.usr_dat.columns], embed_dim=5)
        import copy
        usr_model = copy.deepcopy(model)

        if self.use_cuda:
            #model = nn.DataParallel(model)
            model = model.cuda()
            usr_model = usr_model.cuda()

        # lr = 1e-3
        if self.dat_name == 'KuaiRand':
            lr = 5e-4 # KuaiRand
            print('lr:',lr)
        elif self.dat_name == 'WeChat':
            lr = 5e-4 # WeChat
            print('lr:',lr)
        optim = Adam(model.parameters(), lr=lr, weight_decay=self.weight_decay)
        usr_optim = Adam(usr_model.parameters(), lr=lr, weight_decay=self.weight_decay)

        scheduler = ExponentialLR(optim, gamma=0.5,last_epoch=-1)
        usr_scheduler = ExponentialLR(usr_optim, gamma=0.5,last_epoch=-1)
        early_stopping = EarlyStopping2(self.fout + '_temp', patience=self.patience, verbose=True)

        print(model)
        print(usr_model)
        print('Training env has been initialized...')
        return model, usr_model, optim, usr_optim, early_stopping, scheduler, usr_scheduler


    def _train_iteration(self):
        dur=[]
        print('Training begin...')
        for epoch in range(self.epoch_num):
            if epoch >= 0:
                t0 = time.time()
            loss_log = []
            loss_log_usr = []
            self.model.train()
            self.usr_model.train()

            for _id, batch in enumerate(self.train_loader):
                self.model.train()
                self.optim.zero_grad()
                Lossfunc = CWMLoss()
                rel_score =  self.model(batch[0]).view(batch[0].size(0))
                wt = batch[1]
                duration = batch[3]

                k_tensor = torch.ones_like(rel_score) * self.k
                train_loss = Lossfunc(rel_score, k_tensor, wt, duration, self.eps)
                train_loss.backward()

                self.optim.step()
                loss_log.append(train_loss.item())
                loss_log_usr.append(0)


            if self.early_stopping.early_stop:
                print("Early stopping")
                break 


            if self.dat_name =='KuaiRand' or self.dat_name =='WeChat':
                rmse, mae, xgauc, xauc = cal_reg_metric3(self.vali_dat, self.model, self.vali_loader, self.k, self.eps)

                if self.dat_name == 'KuaiRand':
                    self.early_stopping(xgauc*(-1), self.model, self.usr_model)
                    # self.early_stopping(mae, self.model, self.usr_model)
                elif self.dat_name == 'WeChat':
                    self.early_stopping(mae, self.model, self.usr_model)
                # self.early_stopping(mae, self.model)

                if self.early_stopping.early_stop:
                    print("Early stopping")
                    break 

                if epoch >= 0:
                    dur.append(time.time() - t0)

                print("Epoch {:05d} | Time(s) {:.4f} | Train_Loss {:.4f} | Train_Usr_Loss {:.4f} | "
                        "Test_NDCG@1 {:.4f}| Test_RMSE {:.4f}| Test_MAE {:.4f}| Test_GXAUC {:.4f}| Test_XAUC {:.4f}|". format(epoch, np.mean(dur), np.mean(loss_log),np.mean(loss_log_usr),
                                                        0, rmse, mae, xgauc, xauc))
            # break

    def _test_and_save(self):
        if self.model_name == 'AFI':
            model = My_AutomaticFeatureInteractionModel(field_dims=cal_field_dims(self.all_dat,self.dat_name), 
                                                     embed_dim=10, 
                                                     num_heads=8, 
                                                     num_layers=1,
                                                     atten_embed_dim=64,
                                                     mlp_dims=[64], 
                                                     dropouts=[0.2,0.2])
        elif self.model_name == 'DFM':
            model = My_DeepFactorizationMachineModel(field_dims=cal_field_dims(self.all_dat, self.dat_name), embed_dim=10, mlp_dims=[64,64,64], dropout=0.2)
        elif self.model_name == 'FM':
            model = My_FactorizationMachineModel(field_dims=cal_field_dims(self.all_dat, self.dat_name), embed_dim=10)
        elif self.model_name == 'DCN':
            model = My_DeepCrossNetworkModel(field_dims=cal_field_dims(self.all_dat, self.dat_name), embed_dim=10, num_layers=3, mlp_dims=[64,64,64], dropout=0.2)
        elif self.model_name == 'xDFM':
            model = My_ExtremeDeepFactorizationMachineModel(field_dims=cal_field_dims(self.all_dat, self.dat_name), embed_dim=10, mlp_dims=[64,64,64], dropout=0.2, cross_layer_sizes=[64,64,64], split_half=True)


        # usr_model = Usr_FactorizationMachineModel(field_dims=[len(self.usr_dat[fe].unique()) for fe in self.usr_dat.columns], embed_dim=5)
        import copy
        usr_model = copy.deepcopy(model)

        model = model.cuda()
        usr_model = usr_model.cuda()

        model.load_state_dict(torch.load(self.fout + '_temp_checkpoint.pt'))
        usr_model.load_state_dict(torch.load(self.fout + '_temp_usr_checkpoint.pt'))
  
        
        ndcg_ls, pcr_ls, wt_ls, gauc_val, mrr_val= cal_group_metric(self.test_dat ,model,[1,3,5], self.test_loader)
        if self.label_name =='long_view2':
            rmse, mae, xgauc, xauc = 0,0,0
        else:
            rmse, mae, xgauc, xauc = cal_reg_metric3(self.test_dat, model, self.test_loader, self.k, self.eps)

        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("{}_{} | Log_loss {:.4f} | AUC {:.4f} | GAUC {:.4f} | MRR {:.4f} | "
                    "nDCG@1 {:.4f}| nDCG@3 {:.4f}| nDCG@5 {:.4f}| "
                    "PCR@1 {:.4f}| PCR@3 {:.4f}| PCR@5 {:.4f}| WT@1 {:.4f}| WT@3 {:.4f}| WT@5 {:.4f}| RMSE {:.4f} | MAE {:.4f}| XGAUC {:.4f}| XAUC {:.4f} |". format(self.model_name, self.label_name, 0,0, gauc_val, mrr_val,
                                                    ndcg_ls[0],ndcg_ls[1],ndcg_ls[2],pcr_ls[0],pcr_ls[1],pcr_ls[2],wt_ls[0],wt_ls[1],wt_ls[2], rmse, mae, xgauc, xauc))
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        
        df_result = pd.DataFrame([],columns=['AUC','MRR','nDCG@1','nDCG@3','nDCG@5','PCR@1','PCR@3','PCR@5','WT@1','WT@3','WT@5','RMSE', 'MAE','XGAUC', 'XAUC'])
        df_result.loc[1] =  [gauc_val, mrr_val] + ndcg_ls + pcr_ls + wt_ls + [rmse, mae, xgauc, xauc]

        df_result.to_csv('{}_result.csv'.format(self.fout))
        torch.save(model.state_dict(), '{}_model.pt'.format(self.fout))
        torch.save(usr_model.state_dict(), '{}_usr_model.pt'.format(self.fout))


        
if __name__=="__main__":
    pass

        