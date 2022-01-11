import pickle
import datetime
import pgeocode
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

import torch

import warnings
warnings.filterwarnings('ignore')


def one_hot_feat(feat, feat_list, val):
    out_array = np.zeros(len(feat_list))
    val = feat+str(val)
    pos = np.where(feat_list == val)[0][0]
    out_array[pos] = 1

    return np.asarray([pos])#out_array

def get_big_seller(seller_id):
    big_sellers = [0, 3, 4, 5, 6, 8, 7, 1, 9, 2, 11, 10, 12, 13, 14, 15, 16, 17, 18, 20, 19, 21, 22, 24, 26, 25, 23, 31, 29, 34, 30, 33, 32, 38, 42, 52, 36, 45, 40, 49, 46, 37, 41, 51, 53, 48, 43, 56, 47, 55, 66, 54, 63, 58, 44, 57, 67, 73, 62, 71, 65, 77, 74, 89, 75, 88, 82, 84, 76, 90, 72, 69, 78, 81, 108, 92, 85, 68, 94, 86, 109, 105, 98, 107, 87, 99, 102, 96, 97, 103, 135, 100, 91, 95, 114, 104, 119, 110, 125, 118]
    if seller_id in big_sellers:
        return big_sellers.index(seller_id)
    else:
        return 100

def get_int(zipc):
    return int(zipc)

def get_deliv_cat(deliv_days):
    if deliv_days >= 0 and deliv_days < 8:
        return np.asarray([1,0])
    else:
        return np.asarray([0,1])


class EbayDataset(Dataset):
    def __init__(self, data_df, data_dir='/Users/pamelakatali/Downloads/Ebay_ML/data/'):
        data_df = data_df.loc[(data_df['delivery_days'] >= 0)]
        data_df['delivery_days'] = data_df['delivery_days'].apply(get_deliv_cat)
        #data_df = data_df.loc[data_df['delivery_days'] < 10]

        #data_df = data_df.drop(labels=drop_lst, axis=0)#data_df.drop(drop_lst)

        self.ship_cols = pickle.load( open( data_dir+"ship_id_cols.pkl", "rb" ) )
        self.cat_cols = pickle.load( open( data_dir+"cat_id_cols.pkl", "rb" ) )
        self.pack_cols = pickle.load( open( data_dir+"pack_size_cols.pkl", "rb" ) )
        self.b2c_cols = pickle.load( open( data_dir+"b2c_c2c_cols.pkl", "rb" ) )

        self.scaler = pickle.load(open( data_dir+"train_minmax_scaler_3.pkl", "rb" ))

            
        self.cols_num = ['declared_handling_days', 'carrier_min_estimate', 'carrier_max_estimate', 'carrier_diff_estimate', 'accept_days',  'shipping_dist', 'shipping_fee', 'flat_rate', 'item_price','quantity', 'weight', 'non_weight_rate','weight_units'] #13 
        #self.cols_num = ['shipping_dist', 'shipping_fee']
        self.cols_one_hot = ['ship_id','cat_id','pack_size','b2c_c2c'] #4
        self.cols_cat = ['payment_month','seller_id','item_group','buyer_group']#,'item_region','buyer_region','item_addr','buyer_addr'] #8
        self.cols = self.cols_num + self.cols_one_hot + self.cols_cat + ['delivery_days'] #

        #print(data_df.columns)
        data_df['seller_id'] = data_df['seller_id'].apply(get_big_seller)
        #data_df['buyer_group'] = data_df['buyer_group'].apply(get_int)
        self.data = data_df[self.cols]
        #self.y = data_df['delivery_days']
        
        self.target_scaler = pickle.load( open( data_dir+'train_target_stadrd_scaler.pkl', 'rb'))
        #self.y = self.target_scaler.transform(data_df['delivery_days'].values.reshape(-1, 1))
        #self.y = np.tanh(data_df['delivery_days'].values / 10)
        self.y = data_df['delivery_days'].values 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        
        ship = one_hot_feat('ship_id_', self.ship_cols, sample['ship_id'])
        cat = one_hot_feat('cat_id_', self.cat_cols, sample['cat_id'])
        pack = one_hot_feat('pack_size_', self.pack_cols, sample['pack_size'])
        b2c_c2c = one_hot_feat('b2c_c2c_', self.b2c_cols, sample['b2c_c2c'])
        

        x = self.scaler.transform(sample[self.cols_num].values.reshape(1,len(self.cols_num)))

        x = x.reshape(len(self.cols_num),)
        #x = x[5:6]


        one_hot = np.concatenate((ship, cat, pack, b2c_c2c), axis=0)
        x = np.concatenate((x, one_hot), axis=0)
        x = x.reshape(len(self.cols_num)+len(self.cols_one_hot),)
        
        x_cat = sample[self.cols_cat].values
        #print(x_cat)
        #samp_seller
        x = np.concatenate((x, [int(sample['payment_month'])-1,int(sample['seller_id']),int(sample['item_group'])+1,int(sample['buyer_group'])+1]), axis=0)
        len_x = len(self.cols_num)+len(self.cols_one_hot)+len(self.cols_cat)
        x = x.reshape(len_x,)
        #print(x)
        
        #print(sample['delivery_days'])
        return x, self.y[idx], sample['delivery_days']





class EbayDatasetXGB(Dataset):
    def __init__(self, data_df, data_dir='/Users/pamelakatali/Downloads/Ebay_ML/data/'):
        #data_df = data_df.loc[(data_df['delivery_days'] >= 1)]
        #data_df['delivery_days'] = data_df['delivery_days'].apply(get_deliv_cat)
        #data_df = data_df.loc[data_df['delivery_days'] < 10]

        #data_df = data_df.drop(labels=drop_lst, axis=0)#data_df.drop(drop_lst)

        self.ship_cols = pickle.load( open( data_dir+"ship_id_cols.pkl", "rb" ) )
        self.cat_cols = pickle.load( open( data_dir+"cat_id_cols.pkl", "rb" ) )
        self.pack_cols = pickle.load( open( data_dir+"pack_size_cols.pkl", "rb" ) )
        self.b2c_cols = pickle.load( open( data_dir+"b2c_c2c_cols.pkl", "rb" ) )

        self.scaler = pickle.load(open( data_dir+"train_minmax_scaler_3.pkl", "rb" ))

            
        self.cols_num = ['declared_handling_days', 'carrier_min_estimate', 'carrier_max_estimate', 'carrier_diff_estimate', 'accept_days',  'shipping_dist', 'shipping_fee', 'flat_rate', 'item_price','quantity', 'weight', 'non_weight_rate','weight_units'] #13 
        #self.cols_num = ['shipping_dist', 'shipping_fee']
        self.cols_one_hot = ['ship_id','cat_id','pack_size','b2c_c2c'] #4
        self.cols_cat = ['payment_month','seller_id','item_group','buyer_group']#,'item_region','buyer_region','item_addr','buyer_addr'] #8
        self.cols = self.cols_num + self.cols_one_hot + self.cols_cat + ['delivery_days'] #

        #print(data_df.columns)
        data_df['seller_id'] = data_df['seller_id'].apply(get_big_seller)
        #data_df['buyer_group'] = data_df['buyer_group'].apply(get_int)
        self.data = data_df[self.cols]
        #self.y = data_df['delivery_days']
        
        self.target_scaler = pickle.load( open( data_dir+'train_target_stadrd_scaler.pkl', 'rb'))
        #self.y = self.target_scaler.transform(data_df['delivery_days'].values.reshape(-1, 1))
        #self.y = np.tanh(data_df['delivery_days'].values / 10)
        self.y = data_df['delivery_days'].values 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        
        ship = one_hot_feat('ship_id_', self.ship_cols, sample['ship_id'])
        cat = one_hot_feat('cat_id_', self.cat_cols, sample['cat_id'])
        pack = one_hot_feat('pack_size_', self.pack_cols, sample['pack_size'])
        b2c_c2c = one_hot_feat('b2c_c2c_', self.b2c_cols, sample['b2c_c2c'])
        

        x = sample[self.cols_num].values.reshape(1,len(self.cols_num))#self.scaler.transform()

        x = x.reshape(len(self.cols_num),)
        #x = x[5:6]


        one_hot = np.concatenate((ship, cat, pack, b2c_c2c), axis=0)
        x = np.concatenate((x, one_hot), axis=0)
        x = x.reshape(len(self.cols_num)+len(self.cols_one_hot),)
        
        x_cat = sample[self.cols_cat].values
        #print(x_cat)
        #samp_seller
        x = np.concatenate((x, [int(sample['payment_month'])-1,int(sample['seller_id']),int(sample['item_group'])+1,int(sample['buyer_group'])+1]), axis=0)
        len_x = len(self.cols_num)+len(self.cols_one_hot)+len(self.cols_cat)
        x = x.reshape(len_x,)
        #print(x)
        
        #print(x.astype(np.float).dtype)
        #print(x.astype(np.float))
        return x.astype(np.float), self.y[idx], sample['delivery_days']

