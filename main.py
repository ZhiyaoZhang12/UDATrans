# -*- coding: utf-8 -*-
import argparse
import os
import torch
import pandas as pd
import numpy as np
from exp.exp_informer import Exp_Informer
from results_analyse import results_analysis

parser = argparse.ArgumentParser(description='[Informer] Long Sequences Forecasting')
parser.add_argument('--model', type=str, required=False, default='informer',help='model of experiment, options: [informer, informerstack, informerlight(TBD)，dcnn，biLSTM]')
parser.add_argument('--root_path', type=str, default="Case/UDATrans/", help='root path of the data file') #"./" or "MTL/"
parser.add_argument('--data_path', type=str, default="Data/")
parser.add_argument('--features', type=str, default='MS', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--checkpoints', type=str, default='autodl-tmp/checkpoints/', help='location of model checkpoints')
parser.add_argument('--seq_len', type=int, default=36, help='input sequence length of Informer encoder')
parser.add_argument('--c_out', type=int, default=1, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=6, help='num of heads')#8
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')#default=2
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')#default=1
parser.add_argument('--s_layers', type=str, default='3,2,1', help='num of stack encoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn') #2048
parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor') #5  
parser.add_argument('--padding', type=int, default=0, help='padding type')
parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder, using this argument means not using distilling', default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--sensor_features', type=str, default= ['sensor 2','sensor 3','sensor 4','sensor 7','sensor 8','sensor 9','sensor 11','sensor 12','sensor 13','sensor 14','sensor 15','sensor 17','sensor 20','sensor 21'], help='sensor feature used')
parser.add_argument('--is_descrsing', type=str, default=False,help='whether to make the sensor data in a uniform trend')
parser.add_argument('--normal_style', type=str, default='StandardScaler',help='MinMaxScale or StandardScaler')
parser.add_argument('--attn', type=str, default='prob', help='attention used in encoder, options:[prob, full]')
parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')                                                         
parser.add_argument('--activation', type=str, default='gelu',help='activation')
parser.add_argument('--output_attention',type=bool, default=False, help='whether to output attention in ecoder')  #action='store_true'
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
parser.add_argument('--mix', action='store_false', help='use mix attention in generative decoder', default=True)
parser.add_argument('--cols', type=str, nargs='+', help='certain cols from the data files as the input features')
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
parser.add_argument('--batch_size', type=int, default=64, help='batch size of train input data') #default=32
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate') #0.0001
parser.add_argument('--train_dataset_name', type=str, default= ['FD003'], help='dataset for training')
parser.add_argument('--test_dataset_name', type=str, default= ['FD001'], help='dataset for test')  #['FD001','FD002','FD003','FD004']
parser.add_argument('--OP_features', type=bool, default=False, help='wheather to use the OP features')
parser.add_argument('--test_data', type=str, default='test_data', help='use the train or test data to be the data in test') 
parser.add_argument('--loss_rul', type=str, default='mse',help='loss function for rul')  #default mse & score
parser.add_argument('--loss_weight', type=list, default=[0.01,2,0.006,0.2], help='batch size of train input data') #classifier,discriminator,diff

parser.add_argument('--lradj', type=str, default='type1',help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu') #default=True, False use CPU
parser.add_argument('--gpu', type=int, default=0, help='gpu') #default=0
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False) #default=False
parser.add_argument('--devices', type=str, default='0,1,2,3,4,5,6,7,8,9,10',help='device ids of multile gpus')
parser.add_argument('--des', type=str, default='test',help='exp description')
parser.add_argument('--num_engines', type=dict, default={'FD001':100,'FD002':259,'FD003':100,'FD004':248,'PHM08':65},help='number of engines in each dataset')

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

os.environ['CUDA_VISIBLE_DEVICES']='0'
'''
if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ','')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]  
'''
args.s_layers = [int(s_l) for s_l in args.s_layers.replace(' ','').split(',')]


print('Args in experiment:')
print(args)

Exp = Exp_Informer     

for ii in range(args.itr): 
    name_prior = '4.12'  
    setting  = '{0}_s{1}t{2}_{3}_epo{8}_wei{7}_bs{4}_lr{5}_hd{9}_{6}'.format( 
                            name_prior,args.train_dataset_name,args.test_dataset_name,
                            args.model,args.batch_size,args.learning_rate,
                            args.des,args.loss_weight,args.train_epochs,args.n_heads)
    settingsss = setting + '{}'.format(ii)

    print('Settings in this run:',settingsss)
    
    exp = Exp(args) # set experiments
  

    print('\n')
    print('>>>>>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(settingsss))
    #exp.train(settingsss) 
    try:
        exp.train(settingsss)            
    except RuntimeError as exception:
        if "out of memory" in str(exception):
            print("WARNING: out of memory")
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
        else:
            raise exception    
    
    for test_dataset in args.test_dataset_name:
        print('\n')
        print('>>>>>>>>>>>testing {}: {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(test_dataset,settingsss))
        exp.test_window(settingsss)
    torch.cuda.empty_cache()
 

total_results = pd.DataFrame([])

test_window_rul = results_analysis(args,setting,args.test_dataset_name,flag='test_window')
test_window_rul.index = [i+1 for i in range(test_window_rul.shape[0]-1)]+['mean']  
results_analyse_path = args.root_path + 'results_analyse/{}_{}/'.format(args.train_dataset_name,args.test_dataset_name)
if not os.path.exists(results_analyse_path):
        os.makedirs(results_analyse_path)
test_window_rul.to_csv(path_or_buf=results_analyse_path+'{}.csv'.format(setting),index=True, header=True)

