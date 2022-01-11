import os
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import StepLR

import pgeocode
import datetime
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from MLP import MLP, MLP2
from EbayLoss import ebay_loss
from EbayDataset import EbayDataset, EbayDatasetXGB


import datetime
from preprocess import convert_timestamp

def get_delivery_date_quiz(row):
  
  #payment_date = datetime.datetime.fromisoformat(row['payment_datetime'][:10])
  delivery_date = row['acceptance_scan_timestamp'] + datetime.timedelta(days=row['mlp_pred'])
  #delivery_date = row['payment_datetime'] + datetime.timedelta(days=row['mlp_pred'])

  return delivery_date.date()


def get_delivery_days(inpt, mlp_pred, xgb_0, knn_1, data_dir):
    scaler = pickle.load(open( data_dir+"train_minmax_scaler_3.pkl", "rb" ))
    final_preds = []
    for i in range(len(inpt)):
        x = inpt[i:i+1]
        if mlp_pred[i] == 0:
            final_preds.append(xgb_0.predict(x)[0])
        else:
            x_num = x[:,:13]
            x_one_hot = x[:,13:]
            x_num = scaler.transform(x_num)
            x = np.concatenate((x_num,x_one_hot), axis=1)
            final_preds.append(knn_1.predict(x)[0])
    return torch.Tensor(np.round(final_preds))

def preprocess(data_dir, quiz_pred=0):
    
    
    BATCH_SIZE = 4096
    CHUNK_SIZE = 10000

    # Set fixed random number seed
    torch.manual_seed(42)
    
    batch_count = 0
    chunk_count = 0
    
    targets_all = torch.tensor([])
    outputs_all = torch.tensor([])
    inputs_all = torch.tensor([])
    targets_actual_all = torch.tensor([])
    outputs_all_actual = torch.tensor([])

    quiz_filename = data_dir+"clean_ebay_quiz.tsv.gz"

    sftmx = nn.Softmax(dim=1)

    with torch.no_grad():
        model = MLP()
        model.load_state_dict(torch.load(data_dir+'mlp_small/mlp_train.pt',map_location=torch.device('cpu')))
        model.eval()

        if quiz_pred == 0:
            for chunk in pd.read_csv(quiz_filename, sep='\t', chunksize=CHUNK_SIZE):
                chunk_count += 1
                #process chunk
                dataset = EbayDatasetXGB(chunk,data_dir=data_dir)
                # Prepare dataset
                quizloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
                print(chunk_count)
            
                # Iterate over the DataLoader for training data
                for i, data in enumerate(quizloader):
                    
                    # Get inputs
                    inputs, targets, targets_actual = data
        
                    #inputs = inputs
                    #targets = targets
        
                    # Perform forward pass
                    outputs = model(inputs.float())
                    inputs_all = torch.cat((inputs_all, inputs), 0)
                    outputs_all = torch.cat((outputs_all, outputs), 0)
                    #targets_all = torch.cat((targets_all, targets), 0)
                    #targets_actual_all = torch.cat((targets_actual_all, targets_actual.cpu().detach()), 0)
                    #break;
                #break;
        else:
            inputs_all = pickle.load(open(data_dir+'quiz_pred_inputs.pkl', 'rb' ))
            #inputs_all = inputs_all[:50000,:]
            #outputs = model(inputs.float())
            outputs_all = model(inputs_all.float())

            
    print("done chunking")
    y_pred_cat = sftmx(outputs_all.detach())
    y_pred_cat = np.argmax(y_pred_cat, axis=1)

    xg_0 = pickle.load(open(data_dir+'class_0_bin_xgb.pkl', 'rb' ))
    neigh_1 = pickle.load(open(data_dir+'class_1_bin_knn.pkl', 'rb' ))
    #xg_2 = pickle.load(open(data_dir+'class_2_xgboost.pkl', 'rb' ))

    y_pred = get_delivery_days(inputs_all.detach(),y_pred_cat, xgb_0=xg_0, knn_1=neigh_1, data_dir=data_dir)
    print(y_pred)
    
    quiz_df = pd.read_csv(data_dir+"ebay_quiz.tsv.gz", sep="\t")
    quiz_df['acceptance_scan_timestamp'] = quiz_df['acceptance_scan_timestamp'].apply(convert_timestamp)
    quiz_df['mlp_pred'] = y_pred

    quiz_df['mlp_pred'] = quiz_df[['acceptance_scan_timestamp', 'mlp_pred']].apply(get_delivery_date_quiz, axis=1)
    quiz_df[['record_number', 'mlp_pred']].to_csv(data_dir+'ebay_quiz_mlp_pred.tsv.gz',sep='\t',header=False, index=False, compression='infer')

    if quiz_pred == 0: 
        pickle.dump( inputs_all, open( data_dir+'quiz_pred_inputs.pkl', 'wb' ) )


if __name__ == '__main__':
    #data_dir = '/content/drive/MyDrive/Colab_Notebooks/Ebay/data/'
    data_dir = '/Users/pamelakatali/Downloads/Ebay_ML/data/'

    preprocess(data_dir, quiz_pred=1)
