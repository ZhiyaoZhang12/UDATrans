# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler,MinMaxScaler,LabelBinarizer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import pandas as pd
import os
from sklearn.utils import shuffle
from torch.utils.data import Dataset

class DataPrepare(object):   
    def __init__(self,
                  root_path,
                  data_path,
                  seq_len,
                  train_dataset_name, 
                  test_dataset_name,
                  sensor_features,
                  OP_features,
                  normal_style,
                  **kwargs):
        
        self.root_path = root_path
        self.data_path = data_path
        self.train_dataset_name = train_dataset_name
        self.test_dataset_name = test_dataset_name
        self.sensor_features = sensor_features
        self.OP_features = OP_features
        self.normal_style = normal_style
        self.seq_len = seq_len
        self.type_1 = ['FD001', 'FD003']
        self.type_2 = ['FD002', 'FD004','PHM08']
        
    
    # data load
    def loader_engine_train(self, dataset_name,**kwargs):
        train = pd.read_csv(self.data_path+'{}/train_{}.csv'.format(dataset_name,dataset_name),header=None, **kwargs)
        return train
    
    def loader_engine_test(self,dataset_name, **kwargs):
        test = pd.read_csv( self.data_path +'{}/test_{}.csv'.format(dataset_name,dataset_name) ,header=None, **kwargs)
        test_RUL = pd.read_csv(self.data_path +'{}/RUL_{}.csv'.format(dataset_name,dataset_name),header=None, **kwargs)
        return test, test_RUL
    

    def add_columns_name_train(self,train):
        sensor_columns = ["sensor {}".format(s) for s in range(1,22)]
        info_columns = ['unit_id','cycle']
        settings_columns = ['setting 1', 'setting 2', 'setting 3']
        train.columns = info_columns + settings_columns + sensor_columns
        return train

            
    def add_columns_name_test(self,test,test_RUL):
        sensor_columns = ["sensor {}".format(s) for s in range(1,22)]
        info_columns = ['unit_id','cycle']
        settings_columns = ['setting 1', 'setting 2', 'setting 3']
        test.columns = info_columns + settings_columns + sensor_columns
        test_RUL.columns = ['RUL']
        test_RUL['unit_id'] = [i for i in range(1,len(test_RUL)+1,1)]
        test_RUL.set_index('unit_id',inplace=True,drop=True)
        return test
        
        
    def labeling_train(self,data, piecewise_point):
        ##for train
        maxRUL_dict = data.groupby('unit_id')['cycle'].max().to_dict()
        data['maxRUL'] = data['unit_id'].map(maxRUL_dict)
        data['RUL'] = data['maxRUL'] - data['cycle'] 
        
        #RUL filter
        filter_RUL = (data['RUL'] >= piecewise_point)
        data.loc[filter_RUL,['RUL']] = piecewise_point
        return data
        

    def labeling_test(self,data,test_RUL,piecewise_point):
        ###for test
        RUL_dict = test_RUL.to_dict()
        data['RUL_test'] =data['unit_id'].map(RUL_dict['RUL'])

        maxT_dict_train = data.groupby('unit_id')['cycle'].max().to_dict()
        data['maxT'] = data['unit_id'].map(maxT_dict_train)

        data['RUL'] = data['RUL_test'] + data['maxT'] - data['cycle']
        max_RUL_test = data.groupby('unit_id')['RUL'].max().to_dict()
        data['maxRUL'] = data['unit_id'].map(max_RUL_test)
                
        #RUL filter
        filter_RUL = (data['RUL'] >= piecewise_point)
        data.loc[filter_RUL,['RUL']] = piecewise_point
        return data
      
    
    def normalization(self,data,dataset_name):  

        if len(data):
            data_normalize = data.copy()

        if dataset_name in self.type_1:
            if self.normal_style == 'StandardScaler':
                scaler = StandardScaler().fit(data[self.sensor_features])
            elif self.normal_style == 'MinMaxScaler':
                scaler = MinMaxScaler().fit(data[self.sensor_features])

            data_normalize[self.sensor_features] = scaler.transform(data[self.sensor_features])

        elif dataset_name in self.type_2:   
            #给他们聚类['OP']
            self.settings_columns = ['setting 1', 'setting 2', 'setting 3']
            kmeans = KMeans(n_clusters=6, random_state=0).fit(data[self.settings_columns])
            data['OP'] = kmeans.labels_
            
            if len(data):
                data_normalize = data.copy()
                
            gb = data.groupby('OP')[self.sensor_features]
            d = {}
            for x in gb.groups:
                if self.normal_style == 'StandardScaler':
                    d["scaler_{0}".format(x)] = StandardScaler().fit(gb.get_group(x))
                elif self.normal_style == 'MinMaxScaler':
                    d["scaler_{0}".format(x)] = MinMaxScaler().fit(gb.get_group(x))

                data_normalize.loc[data_normalize['OP'] == x, self.sensor_features] = d["scaler_{0}".format(x)].transform(
                    data.loc[data['OP'] == x, self.sensor_features])

        data = data_normalize.copy()
        del data_normalize
        return data
   

    def onehot_coding_OP(self,data,dataset_name):
        if dataset_name in self.type_2:
            onehot = LabelBinarizer()
            data_op = onehot.fit_transform(data['OP'])
          
            op_columns = ["OP {}".format(s) for s in range(1,7)]
            data[op_columns] = data_op
            data.drop(['OP'],inplace=True,axis=1)
        
        else: 
            #pass
            op_columns = ["OP {}".format(s) for s in range(1,7)]
            data_op = np.zeros((len(data),6))
            data[op_columns] = data_op
        return data
        
        
    def del_unuseful_columns(self,data,dataset_name): 
        op_columns = ["OP {}".format(s) for s in range(1,7)]
        #if self.dataset_name in self.type_2:
        if self.OP_features == True:
            useful_columns =  ['unit_id'] + self.sensor_features + op_columns + ['RUL']  
        #elif self.dataset_name in self.type_1:
        if self.OP_features == False:
            useful_columns =  ['unit_id'] + self.sensor_features + ['RUL']        
        data = data.loc[:,useful_columns]     
        
        #做双索引
        data['dataset_id'] = ['{}'.format(dataset_name)]*data.shape[0]
        data.set_index(['dataset_id','unit_id'],drop=True,inplace=True)
        return data
    
    def train_vali_split(self,data):
        data_turbines = np.arange(len(data.index.to_series().unique()))
        train_turbines, validation_turbines = train_test_split(data_turbines, test_size=0.3,random_state = 1334) 
        idx_train = data.index.to_series().unique()[train_turbines]
        idx_validation = data.index.to_series().unique()[validation_turbines]     
  
        validation = data.loc[idx_validation]
        train = data.loc[idx_train]
        return train, validation
    
    def test_window(self,test):
        ###test get the last window
        test_window = pd.DataFrame([])
        for unit_index in (test.index.to_series().unique()):
            trajectory_df = pd.DataFrame(test.loc[unit_index])
            if len(trajectory_df) >= self.seq_len:
                temp_last_new = trajectory_df.iloc[-self.seq_len:,:]  
                test_window = pd.concat([test_window,temp_last_new])             
            else:                
                padding_data = pd.DataFrame(data=np.full(shape=[-len(trajectory_df)+self.seq_len,\
                                            trajectory_df.shape[1]],fill_value=1),\
                                            columns=trajectory_df.columns)
                temp_last_new = pd.concat([padding_data,trajectory_df])
                temp_last_new['unit_id'] = [unit_index[1]]*len(temp_last_new)
                temp_last_new['dataset_id'] = [unit_index[0]]*len(temp_last_new)

                temp_last_new.set_index(['dataset_id','unit_id'],inplace=True,drop=True)
                test_window = pd.concat([test_window,temp_last_new])
        return test_window
            
        
    def process(self):
        path_save = self.root_path + 'data/{}_{}/'.format(self.train_dataset_name,self.test_dataset_name)   
        if not os.path.exists(path_save):
            os.makedirs(path_save)
            
        ###for train_dataset_name
        trains = pd.DataFrame([])
        for dataset_name in self.train_dataset_name:
            train = self.loader_engine_train(dataset_name)
            train = self.add_columns_name_train(train)            
            train = self.labeling_train(train,piecewise_point=125)
            train = train.astype({'RUL':np.int64})
            train = self.normalization(train,dataset_name)
            if self.OP_features == True:
                train = self.onehot_coding_OP(train,dataset_name)
            train = self.del_unuseful_columns(train,dataset_name)  
            trains = pd.concat([trains,train],axis=0)
        
        train, validation = self.train_vali_split(train)
        trains.to_csv(path_save + 'train.csv')
        validation.to_csv(path_save + 'validation.csv')
        
        ###for test_dataset_name
        tests = pd.DataFrame([])
        for dataset_name in self.test_dataset_name:
            test, test_RUL = self.loader_engine_test(dataset_name)
            test = self.add_columns_name_test(test, test_RUL)            
            test = self.labeling_test(test,test_RUL,piecewise_point=125)
            test = test.astype({'RUL':np.int64})
            test = self.normalization(test,dataset_name)
            if self.OP_features == True:
                test = self.onehot_coding_OP(test,dataset_name)
            test = self.del_unuseful_columns(test,dataset_name)  
            tests = pd.concat([tests,test],axis=0)
        
        test = tests    
        test_window = self.test_window(test)
        test.to_csv(path_save + 'test.csv')
        test_window.to_csv(path_save + 'test_window.csv')
        
        
        ###for target in training process 
        train_target, validation_target = self.train_vali_split(test)

        return train, validation, test, test_window, train_target, validation_target

    
    
class DataReaderTrajactory(Dataset):
    def __init__(self, 
                 data,
                 seq_len,
                 ):

        self.all_seq_x = []
        self.seq_len = seq_len
        self.data_x = data

        self.__read_data__()  # data preprocessing
              

    def __read_data__(self):         
        self.all_seq_x = self.transform_data(self.data_x)

        
    def transform_data(self,data):
        ### enc, dec for save the precessed data(time window) 
        enc = []
        #print('\n')
        print('There are {} trajectories in dataset'.format(len(data.index.to_series().unique())))
          
        data.reset_index(inplace=True,drop=False)
        
        data['dataset'] = data['dataset_id'] 
        filter1 = data['dataset_id'] == 'FD001'
        data.loc[filter1,'dataset'] = 1
        filter2 = data['dataset_id'] == 'FD002'
        data.loc[filter2,'dataset'] = 2
        filter3 =  data['dataset_id'] == 'FD003'
        data.loc[filter3,'dataset'] = 3
        filter4 = data['dataset_id'] == 'FD004'
        data.loc[filter4,'dataset'] = 4
        filter8 = data['dataset_id'] == 'PHM08'
        data.loc[filter8,'dataset'] = 8

        data['unit'] = data['unit_id']
        data.set_index(['dataset_id','unit_id'],inplace=True,drop=True)
        orders = [-2,-1] + [i for i in range(len(data.columns)-2)]
        data = data.iloc[:,orders]
        ##用unit和dataset 去保存 'dataset_id','unit_id'

        #Loop through each trajectory
        for unit_index in (data.index.to_series().unique()): 
            #get the whole trajectory (index)
            temp_df = pd.DataFrame(data.loc[unit_index])             
             
            # Loop through the data in the object (index) trajectory
            data_enc_npc, data_dec_npc, array_data_enc, array_data_dec = [],[],[],[]
            len_trajectory = len(temp_df) 
            enc_last_index = len_trajectory
            for i in range(enc_last_index - self.seq_len + 1):
                s_begin = i
                s_end = s_begin + self.seq_len
                data_enc_npc = temp_df.iloc[s_begin:s_end]    
                array_data_enc.append(data_enc_npc)
            enc = enc + array_data_enc
        return enc
    
    
    def __getitem__(self,index):          
        seq_x = self.all_seq_x[index].values    
        return seq_x.astype(np.float32)
    
    def __len__(self):
        return len(self.all_seq_x)
        
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
