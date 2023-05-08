# -*- coding: utf-8 -*-
from data_loader import DataReaderTrajactory,DataPrepare
from exp.exp_basic import Exp_Basic
from models.model import Informer  
from sklearn.preprocessing import LabelBinarizer
from utils.tools import EarlyStopping, adjust_learning_rate
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import os
import time
import warnings
import torch.nn.functional as F
from models.rul_process import RUL_results_save, Draw_RUL_decreasing_fig,Draw_RUL_unit_fig
from models.score_fun import score_loss, score_focal_loss
from utils.metrics import metric
from sklearn.metrics import classification_report,precision_recall_fscore_support
from exp.diss import Diss
from models.score_fun import score_loss, score_focal_loss
from utils.metrics import metric, error_function, score_function, prec_function, accuracy_function
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings('ignore')


class Exp_Informer(Exp_Basic):
    def __init__(self, args):
        super(Exp_Informer, self).__init__(args)   
        args = self.args
        self.__data_prepare__()
              
        
    def __data_prepare__(self):
        data_prepare = DataPrepare(root_path=self.args.root_path,
                               data_path=self.args.data_path,
                               train_dataset_name=self.args.train_dataset_name,
                               test_dataset_name=self.args.test_dataset_name,       
                               sensor_features=self.args.sensor_features,
                               OP_features=self.args.OP_features,
                               normal_style=self.args.normal_style,
                               seq_len=self.args.seq_len)
        self.train_data, self.validation_data, self.test_whole_data, self.test_window_data, \
                    self.train_target_data, self.validation_target_data = data_prepare.process()
        

    def _build_model(self):

        model_dict = {
            'informer':Informer,
        }
        
        if self.args.model=='informer' or self.args.model=='informerstack':
            e_layers = self.args.e_layers if self.args.model=='informer' else self.args.s_layers
            model = model_dict[self.args.model](
                self.args,
                self.args.OP_features,
                self.args.c_out, 
                self.args.seq_len,
                self.args.factor,
                self.args.d_model, 
                self.args.n_heads, 
                e_layers, # self.args.e_layers,
                self.args.d_layers, 
                self.args.d_ff,
                self.args.dropout, 
                self.args.attn,
                self.args.embed,
                self.args.activation,
                self.args.output_attention,
                self.args.distil,
                self.args.mix,
                self.device
            ).float()
       
        elif self.args.model == 'biLSTM':
            model = model_dict[self.args.model](input_size=self.args.enc_in,hidden_size=512,
                                                num_layers=5,output_size=self.args.c_out,seq_len=self.args.seq_len,
                                                out_len=self.args.pred_len).float() #hidden_size=512,num_layers=5
            
        elif self.args.model == 'dcnn':
            model = model_dict[self.args.model](pred_len=self.args.pred_len).float()
            
            
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model
    

    def _get_data(self, flag):

        
        if flag in ['test_window','test_whole']:  
            shuffle_flag = False; drop_last = False; batch_size = args.batch_size 
        elif flag=='pred':   
            shuffle_flag = False; drop_last = False; batch_size = 20
        else:  #train,val
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size

        data_set = DataReaderTrajactory(
            data,
            seq_len=args.seq_len,
        )
        
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _select_criterion_rul(self):
        if self.args.loss_rul == 'mse':
            criterion_rul =  nn.MSELoss()
        elif self.args.loss_rul == 'mae':
            criterion_rul =  nn.L1Loss()
        elif self.args.loss_rul == 'score':            
            criterion_rul = score_loss              
        return criterion_rul

    
    # sortmax 结果转 onehot
    def _props_to_onehot(self,props):
        if isinstance(props, list):
            props = np.array(props)
        a = np.argmax(props, axis=1)
        b = np.zeros((len(a), props.shape[1]))
        b[np.arange(len(a)), a] = 1
        return torch.tensor(b)     
        
        
    def vali(self, validation_dataset_source, validation_dataset_target, validation_loader_source, validation_loader_target, criterion_rul):
        
        alpha = []
        
        self.model.eval()
        with torch.no_grad():  
          
            total_loss_rul = torch.tensor([]).to(self.device)
            total_loss_domain_classifier_source = torch.tensor([]).to(self.device)
            total_loss_domain_discriminator_source = torch.tensor([]).to(self.device)
            total_loss_diss_source = torch.tensor([]).to(self.device)
            

            total_loss_domain_classifier_target= torch.tensor([]).to(self.device)
            total_loss_domain_discriminator_target = torch.tensor([]).to(self.device)
            total_loss_diss_target = torch.tensor([]).to(self.device)
            
            
            criterion_domain_classifier = torch.nn.NLLLoss()
            criterion_domain_discriminator = torch.nn.NLLLoss()
            
            criterion_domain_classifier = criterion_domain_classifier.to(self.device)
            criterion_domain_discriminator = criterion_domain_discriminator.to(self.device)
            
            
            validation_source_iter = iter(validation_loader_source)
        
            for i in range(len(validation_loader_source)):
                
                batch_data_source = validation_source_iter.next()
                domain_label_source = torch.zeros(self.args.batch_size,1).long().to(self.device)

                indexs, Y_RUL, Y_RUL_out, class_output, domain_output, share_feature, private_feature = \
                                            self._process_one_batch(batch_data_source,'vali_source',alpha)

                loss_rul = criterion_rul(Y_RUL_out, Y_RUL)    
                loss_domain_classifier_source = criterion_domain_classifier(class_output,domain_label_source)
                loss_domain_discriminator_source = criterion_domain_discriminator(domain_output,domain_label_source)
                loss_diss_source = Diss(share_feature, private_feature)

                total_loss_rul=torch.cat([total_loss_rul,loss_rul.reshape([-1])])
                total_loss_domain_classifier_source=torch.cat([total_loss_domain_classifier_source,loss_domain_classifier_source.reshape([-1])])
                total_loss_domain_discriminator_source=torch.cat([total_loss_domain_discriminator_source,loss_domain_discriminator_source.reshape([-1])])
                total_loss_diss_source=torch.cat([total_loss_diss_source,loss_diss_source.reshape([-1])])
            
            total_loss_rul = torch.mean(total_loss_rul)  
            total_loss_domain_classifier_source = torch.mean(total_loss_domain_classifier_source)  
            total_loss_domain_discriminator_source = torch.mean(total_loss_domain_discriminator_source)  
            total_loss_diss_source = torch.mean(total_loss_diss_source)  

            
            validation_target_iter = iter(validation_loader_target)
            for i in range(len(validation_loader_target)):
                
                batch_data_target = validation_target_iter.next()
                domain_label_target = torch.ones(self.args.batch_size,1).long().to(self.device)

                indexs, Y_RUL, Y_RUL_out, class_output, domain_output, share_feature, private_feature = \
                                                self._process_one_batch(batch_data_target,'vali_target',alpha)

                loss_rul = criterion_rul(Y_RUL_out, Y_RUL)    
                loss_domain_classifier_target = criterion_domain_classifier(class_output,domain_label_target)
                loss_domain_discriminator_target = criterion_domain_discriminator(domain_output,domain_label_target)
                loss_diss_target = Diss(share_feature, private_feature)

                #print('xxxxxxxxxx',total_loss_rul,loss_rul.reshape([-1]))
                #total_loss_rul=torch.cat([total_loss_rul,loss_rul.reshape([-1])])
                total_loss_domain_classifier_target=torch.cat([total_loss_domain_classifier_target,loss_domain_classifier_target.reshape([-1])])
                total_loss_domain_discriminator_target=torch.cat([total_loss_domain_discriminator_target,loss_domain_discriminator_target.reshape([-1])])
                total_loss_diss_target=torch.cat([total_loss_diss_target,loss_diss_target.reshape([-1])])
            
            #total_loss_rul = torch.mean(total_loss_rul)  
            total_loss_domain_classifier_target = torch.mean(total_loss_domain_classifier_target)  
            total_loss_domain_discriminator_target = torch.mean(total_loss_domain_discriminator_target)  
            total_loss_diss_target = torch.mean(total_loss_diss_target)
            
            loss_weight = self.args.loss_weight 
            total_loss  = loss_weight[0]*loss_rul + \
                            loss_weight[1]*(loss_domain_classifier_source+loss_domain_classifier_target) + \
                            loss_weight[2]*(loss_domain_discriminator_source+loss_domain_discriminator_target) + \
                            loss_weight[3]*(loss_diss_source+loss_diss_target)
            
            print('\n')
            print('validation')
            print('rul',loss_rul)
            print('classifier,s,t,sum',loss_domain_classifier_source,loss_domain_classifier_target,\
                  loss_domain_classifier_source+loss_domain_classifier_target) 
            print('adv',loss_domain_discriminator_source,loss_domain_discriminator_target,loss_domain_discriminator_source+loss_domain_discriminator_target)
            print('diss',loss_diss_source,loss_diss_target,loss_diss_source+loss_diss_target)
            print('\n')
            print('wrul',loss_weight[0]*loss_rul)
            print('wclassifier',loss_weight[1]*(loss_domain_classifier_source+loss_domain_classifier_target)) 
            print('wadv',loss_weight[2]*(loss_domain_discriminator_source+loss_domain_discriminator_target))
            print('wdiss',loss_weight[3]*(loss_diss_source+loss_diss_target))
            
            self.model.train()
            return total_loss, loss_rul

        
    def train(self, setting):
        
        #train_source
        train_dataset_source = DataReaderTrajactory(data=self.train_data,
                                                       seq_len=self.args.seq_len)
        train_loader_source  = DataLoader(
                                            dataset=train_dataset_source,
                                            batch_size=self.args.batch_size,
                                            shuffle=True,
                                            num_workers=self.args.num_workers,
                                            drop_last=True)
        #train_target
        train_dataset_target = DataReaderTrajactory(data=self.train_target_data,
                                                       seq_len=self.args.seq_len)
        train_loader_target  = DataLoader(
                                            dataset=train_dataset_target,
                                            batch_size=self.args.batch_size,
                                            shuffle=True,
                                            num_workers=self.args.num_workers,
                                            drop_last=True)
        
        
        #validation_source
        validation_dataset_source = DataReaderTrajactory(data=self.validation_data,
                                                           seq_len=self.args.seq_len)
        validation_loader_source  = DataLoader(
                                                dataset=validation_dataset_source,
                                                batch_size=self.args.batch_size,
                                                shuffle=True,
                                                num_workers=self.args.num_workers,
                                                drop_last=True)
        #validation_target
        validation_dataset_target = DataReaderTrajactory(data=self.validation_target_data,
                                                               seq_len=self.args.seq_len)
        validation_loader_target  = DataLoader(
                                                dataset=validation_dataset_target,
                                                batch_size=self.args.batch_size,
                                                shuffle=True,
                                                num_workers=self.args.num_workers,
                                                drop_last=True)
        
        #train_data, train_loader = self._get_data(flag='train')
        #vali_data, vali_loader = self._get_data(flag='val')

              
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion_rul =  self._select_criterion_rul()
        criterion_domain_classifier = torch.nn.NLLLoss()
        criterion_domain_discriminator = torch.nn.NLLLoss()
        
        criterion_domain_classifier = criterion_domain_classifier.to(self.device)
        criterion_domain_discriminator = criterion_domain_discriminator.to(self.device)


        if self.args.use_amp==True:
            scaler = torch.cuda.amp.GradScaler()

            
        for p in self.model.parameters():
            p.requires_grad = True
            
        
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            #train_loss = []
            train_total_loss = torch.tensor([]).to(self.device)

            self.model.train()
            epoch_time = time.time()
            
            len_dataloader = min(len(train_loader_source), len(train_loader_target))
            data_source_iter = iter(train_loader_source)
            data_target_iter = iter(train_loader_target)

            for i in range(len_dataloader):
            #for i, (batch_x) in enumerate(train_loader):
                p = float(i + epoch * len_dataloader) / self.args.train_epochs / len_dataloader
                alpha = 2. / (1. + np.exp(-10 * p)) - 1
                
                #iter_count += 1 
                model_optim.zero_grad() 
                
                # training model using source data
                batch_data_source = data_source_iter.next()
                domain_label_source = torch.zeros(self.args.batch_size,1).long().to(self.device)
                indexs, Y_RUL, Y_RUL_out, class_output, domain_output, share_feature, private_feature = \
                                                            self._process_one_batch(batch_data_source,'train_source',alpha)
         
                loss_rul = criterion_rul(Y_RUL_out, Y_RUL)
                loss_domain_classifier_source = criterion_domain_classifier(class_output,domain_label_source)
                loss_domain_discriminator_source = criterion_domain_discriminator(domain_output,domain_label_source)
                loss_diss_source = Diss(share_feature, private_feature)
            
                
                # training model using target data
                batch_data_target = data_target_iter.next()
                domain_label_target = torch.ones(self.args.batch_size,1).long().to(self.device)
                indexs, Y_RUL, Y_RUL_out, class_output, domain_output, share_feature, private_feature = \
                                                            self._process_one_batch(batch_data_target,'train_target',alpha)
         
                #loss_rul = criterion_rul(Y_RUL_out, Y_RUL)
                loss_domain_classifier_target = criterion_domain_classifier(class_output,domain_label_target)
                loss_domain_discriminator_target = criterion_domain_discriminator(domain_output,domain_label_target)
                loss_diss_target = Diss(share_feature, private_feature)
        
            
                loss_weight = self.args.loss_weight 
                loss  = loss_weight[0]*loss_rul  + \
                        loss_weight[1]*(loss_domain_classifier_source+loss_domain_classifier_target) + \
                        loss_weight[2]*(loss_domain_discriminator_source+loss_domain_discriminator_target) + \
                        loss_weight[3]*(loss_diss_source+loss_diss_target)
                '''
                print('train')
                print('rul',loss_rul)
                print('classifier,s,t,sum',loss_domain_classifier_source,loss_domain_classifier_target,\
                      loss_domain_classifier_source+loss_domain_classifier_target) 
                print('adv',loss_domain_discriminator_source,loss_domain_discriminator_target,\
                      loss_domain_discriminator_source+loss_domain_discriminator_target)
                print('diss',loss_diss_source,loss_diss_target,loss_diss_source+loss_diss_target)
                '''
                
                #train_loss.append(loss.item())
                train_total_loss = torch.cat([train_total_loss,loss.reshape([-1])])
                

                if (i+1) % 50==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))

                    #speed = (time.time()-time_now)/iter_count
                    speed = (time.time()-time_now)/(epoch+1)
                    left_time = speed*((self.args.train_epochs - epoch)*len_dataloader - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))

            #train_loss = np.average(train_loss)
            train_total_loss = torch.mean(train_total_loss)

            
                      
            vali_loss, vali_loss_rul = self.vali(validation_dataset_source, validation_dataset_target, validation_loader_source, validation_loader_target, criterion_rul)
            

            print("Epoch: {0}, Steps: {1} | Train loss: {2:.7f} Vali loss: {3:.7f}".format(
                epoch + 1, len_dataloader, train_total_loss, vali_loss))

            early_stopping(vali_loss, self.model, path)
            
            #2.14.2023
            #early_stopping(vali_loss_rul, self.model, path)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch+1, self.args)

        best_model_path = path + '/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model
        


    def test_window(self, setting):       
        #test_data, test_loader = self._get_data(flag=flag)
        flag = 'test_window'
        
        #test_window
        test_window_dataset = DataReaderTrajactory(data=self.test_window_data,
                                            seq_len=self.args.seq_len)
        
        if self.args.test_dataset_name in [['FD002']]:
            batch_size=259
        if self.args.test_dataset_name in [['FD004']]:
            batch_size=248
        elif self.args.test_dataset_name in [['FD003'],['FD001']]:  
            batch_size=100
            
        test_window_loader  = DataLoader(
                                         dataset=test_window_dataset,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         num_workers=self.args.num_workers,
                                         drop_last=False)
    
        self.model.eval()
        with torch.no_grad():  
            Y_RUL_outs = torch.tensor([])
            Y_RULs = torch.tensor([])
            indexss = torch.tensor([])
            
            alpha = []
            for i, (batch_x) in enumerate(test_window_loader):  
                indexs, Y_RUL, Y_RUL_out, class_output, domain_output, share_feature, private_feature = \
                                                            self._process_one_batch(batch_x,'test_window',alpha)                
             
                if i in [246,247]:
                    print(batch_x.size())
                    print('out:',Y_RUL_out)
                    print('true',Y_RUL)
                
                            
                Y_RUL_outs = torch.cat([Y_RUL_outs,Y_RUL_out.detach().cpu()])
                Y_RULs = torch.cat([ Y_RULs,Y_RUL.detach().cpu()])      
                indexss = torch.cat([indexss,indexs.detach().cpu()])

            Y_RULs = np.array(Y_RULs)
            Y_RUL_outs = np.array(Y_RUL_outs)
            indexss = np.array(indexss)

            #shape   RUL num_batch*batch_size*1    SOH num_batch*batch_size*3   HI  num_batch*batch_size*16(pred_len)*1        
            #shape   RUL num_units*1    SOH num_units*3   HI  num_units*16(pred_len)
            Y_RUL_outs = Y_RUL_outs.reshape(-1, Y_RUL_outs.shape[-1]).round()
            Y_RULs = Y_RULs.reshape(-1, Y_RULs.shape[-1])
            indexss = indexss.reshape(-1, indexss.shape[-1])
            indexss = torch.tensor(indexss,dtype=torch.int64)
            
            #if indexss[-1].mean() == 2.:
            #    Y_RUL_outs[-1] = 10
            
            results = np.hstack([indexss,Y_RULs,Y_RUL_outs])
            
            results = pd.DataFrame(data=results,columns=['dataset','unit_id','True_RUL','Predicted_RUL'])                 
            results['Error'] = results.apply(lambda df: error_function(df, 'Predicted_RUL', 'True_RUL'), axis=1)
            results['Score'] = results.apply(lambda df: score_function(df, 'Error'), axis=1)
            results['Prec'] = results.apply(lambda df: prec_function(df, 'Predicted_RUL', 'True_RUL'), axis=1)
            results['Accuracy'] = results.apply(lambda df: accuracy_function(df, 'Error'), axis=1)

            r2 = r2_score(results['True_RUL'], results['Predicted_RUL'])
            prec = results['Prec'].sum()/len(results['Prec'])
            mae, mse, rmse, mape, mspe = metric(results['Predicted_RUL'], results['True_RUL'])
            score = score_loss(torch.tensor(results['Predicted_RUL']), torch.tensor(results['True_RUL'])) 
            #print(results.head(20))

            
            # result save
            folder_path = self.args.root_path + 'results/{}_{}/'.format(self.args.train_dataset_name,self.args.test_dataset_name) + setting +'/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path) 
            
            #rul desreasing  
            Draw_RUL_decreasing_fig(self.args,results,score,rmse,prec,self.args.test_dataset_name,flag,indexss)

            print('***************************{}_{}*******************************'.format(flag,self.args.test_dataset_name))
            print('train datasets:',self.args.train_dataset_name)
            print('test datasets:',self.args.test_dataset_name)
            print('weight:',self.args.loss_weight)
            print('op factors:',self.args.OP_features)
            print('score_rul:{0:.2f},     rmse_rul:{1:.4f}'.format(score, rmse))
            print('************************************************************')
                
            np.save(folder_path+'{}_{}_rul.npy'.format(self.args.test_dataset_name,flag),np.array([score,rmse,prec,mse,mae,r2]))
            results.to_csv(folder_path+'{}_{}_RUL.csv'.format(flag,self.args.test_dataset_name),index=True,header=True)
            

    def test_whole(self, setting):
        #test_data, test_loader = self._get_data(flag=flag)
        flag = 'test_whole'
        #test_whole
        test_whole_dataset = DataReaderTrajactory(data=self.test_whole_data,
                                                  seq_len=self.args.seq_len)
        test_whole_loader  = DataLoader(
                                        dataset=test_whole_dataset,
                                        batch_size=self.args.batch_size,
                                        shuffle=False,
                                        num_workers=self.args.num_workers,
                                        drop_last=True)
    
        self.model.eval()
        with torch.no_grad():  
            Y_RUL_outs = torch.tensor([])
            Y_RULs = torch.tensor([])
            indexss = torch.tensor([])
            
            alpha = []
            for i, (batch_x) in enumerate(test_whole_loader):  
                indexs, Y_RUL, Y_RUL_out, class_output, domain_output, share_feature, private_feature = \
                                                            self._process_one_batch(batch_x,'test_whole',alpha)                

                            
                Y_RUL_outs = torch.cat([Y_RUL_outs,Y_RUL_out.detach().cpu()])
                Y_RULs = torch.cat([ Y_RULs,Y_RUL.detach().cpu()])      
                indexss = torch.cat([indexss,indexs.detach().cpu()])

            Y_RULs = np.array(Y_RULs)
            Y_RUL_outs = np.array(Y_RUL_outs)
            indexss = np.array(indexss)

            #shape   RUL num_batch*batch_size*1    SOH num_batch*batch_size*3   HI  num_batch*batch_size*16(pred_len)*1        
            #shape   RUL num_units*1    SOH num_units*3   HI  num_units*16(pred_len)
            Y_RUL_outs = Y_RUL_outs.reshape(-1, Y_RUL_outs.shape[-1])
            Y_RULs = Y_RULs.reshape(-1, Y_RULs.shape[-1])
            indexss = indexss.reshape(-1, indexss.shape[-1])
            indexss = torch.tensor(indexss,dtype=torch.int64)

            # result save
            folder_path = self.args.root_path + 'results/{}_{}/'.format(self.args.train_dataset_name,self.args.test_dataset_name) + setting +'/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            
            df_RUL_results, mse_rul, mae_rul, r2_rul, score_rul, prec_rul, rmse_rul = \
                        RUL_results_save(self.args,Y_RUL_outs,Y_RULs,self.args.train_dataset_name,flag,setting,indexss)
               
            
            Draw_RUL_unit_fig(self.args,df_RUL_results,score_rul,rmse_rul,prec_rul,self.args.train_dataset_name,flag,indexss)
            
            print('***************************{}_{}*******************************'.format(flag,self.args.train_dataset_name))
            print('train datasets:',self.args.train_dataset_name)
            print('test datasets:',self.args.test_dataset_name)
            print('op factors:',self.args.OP_features)
            print('score_rul:{0:.2f},     rmse_rul:{1:.4f}'.format(score_rul, rmse_rul))
            print('************************************************************')
                
            np.save(folder_path+'{}_{}_rul.npy'.format(self.args.train_dataset_name,flag),np.array([score_rul,rmse_rul,prec_rul,mse_rul,mae_rul,r2_rul]))
            df_RUL_results.to_csv(folder_path+'{}_{}_RUL.csv'.format(flag,self.args.train_dataset_name),index=True,header=True)
            

    
    
    def predict(self, setting, load=False):
        pass           
    
    
    def _process_one_batch(self, batch_x,flag,alpha): #NO decoder
        if self.args.model=='informer' or self.args.model=='informerstack':
            batch_x = batch_x.float().to(self.device)
            Y_RUL = batch_x[:,-1,-1:].to(self.device)  
            indexs = batch_x[:,-1,:2] 
            
            batch_x = batch_x.float()[:,:,2:-1]   #delete  RUL column   the first two columns are dataset_id and unit_id
            Y_RUL_out, class_output, domain_output, share_feature, private_feature = self.model(batch_x,flag,alpha)

            return  indexs, Y_RUL, Y_RUL_out, class_output, domain_output, share_feature, private_feature
      
