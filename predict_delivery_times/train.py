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
from sklearn.metrics import precision_recall_fscore_support

from MLP import MLP
from EbayLoss import ebay_loss
from EbayDataset import EbayDataset


def shuffle_train_set(data_dir):
    train_chunk = pd.read_csv(data_dir+'clean_ebay_train.tsv.gz', sep='\t')
    train_chunk = train_chunk.sample(frac=1)
    shuffle_lst = train_chunk.index.tolist()
    train_chunk.to_csv(data_dir+'ebay_train_small_shuffle.tsv.gz', sep='\t')
    train_chunk = None
    del train_chunk


def train(data_dir):

    BATCH_SIZE = 4096
    CHUNK_SIZE = 4096
    TEST_CHUNK_SIZE = 25000
    EPOCHS = 100
    EVAL_BATCH = int(CHUNK_SIZE /  BATCH_SIZE) - 10
    LR = 1e-3
    CLASS_WEIGHTS = torch.Tensor([0.0001,30])#torch.Tensor([0.6361927962000341, 0.8996882893824599, 3.1580273176823166])
    LOSS_FN = nn.BCEWithLogitsLoss(weight=CLASS_WEIGHTS)#ebay_loss #nn.L1Loss()#nn.MSELoss() #EbayLoss.apply
    MODEL_DIR = 'mlp_small'
    EVAL_CHUNK = 2
    EVAL_BATCH = 0
    DEVICE = torch.device("cpu") #"cuda:0" if torch.cuda.is_available() else "cpu")
    WEIGHT_DECAY = 0.0
    print(DEVICE)

    # Set fixed random number seed
    torch.manual_seed(42)

    # Initialize the MLP
    mlp = MLP()
    mlp = mlp.to(DEVICE)


    # Define the loss function and optimizer
    loss_function = LOSS_FN
    optimizer = torch.optim.AdamW(mlp.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.95)

    sftmx = nn.Softmax(dim=1)
    loss_lst = []
    batch_count = 0

    best_loss = 1000.0
    
    dev_set_calc = 0
    target_scaler = pickle.load( open( data_dir+'train_target_scaler.pkl', 'rb'))

    # Run the training loop
    for epoch in range(0, EPOCHS): # 5 epochs at maximum
        # Print epoch
        print(f'Starting epoch {epoch+1}')

        # Shuffle data
        print("Shuffling data")
        #shuffle_train_set(data_dir)

        train_filename = data_dir+"ebay_train_small_shuffle.tsv.gz"
        test_filename = data_dir+"clean_ebay_dev.tsv.gz"

        current_loss = 0.0
        print("Start training")
        
        chunk_count = 0
        
        for chunk in pd.read_csv(train_filename, sep='\t', chunksize=CHUNK_SIZE):
            chunk_count += 1
            #process chunk
            dataset = EbayDataset(chunk,data_dir=data_dir)
            # Prepare dataset
            trainloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
            
            # Iterate over the DataLoader for training data
            for i, data in enumerate(trainloader):
                mlp.train()

                # Get inputs
                inputs, targets, targets_actual = data

                #print(targets)
                #print(np.unique(targets.detach().numpy()))
                
                inputs = inputs.to(DEVICE)
                targets = targets.to(DEVICE)


                # Zero the gradients
                optimizer.zero_grad()

                # Perform forward pass
                outputs = mlp(inputs.float())

                #print(torch.reshape(targets.float(),(len(outputs),1)))
                #print(outputs)

                # Compute loss
                #loss = loss_function(outputs, torch.reshape(targets.float(),(len(outputs),1)))
                #print(sftmx(outputs))
                loss = loss_function(outputs, targets.float())



                # Perform backward pass
                loss.backward()


                # Perform optimization
                optimizer.step()


                # Print statistics
                current_loss += loss.item()
                batch_count += 1

                
                if i == EVAL_BATCH and (chunk_count % EVAL_CHUNK == 0):
                #if i % EVAL_BATCH == (EVAL_BATCH - 1):
                #    print('Loss after mini-batch %5d: %.3f' %
                #            (i + 1, current_loss / EVAL_BATCH))
                
                    #print("Batch_count:",batch_count)
                    print("Chunk:",chunk_count)
                    print("Mini-batch:",(chunk_count*EVAL_CHUNK))
                    print('Train loss: %.3f' %
                            (current_loss / batch_count))#(current_loss / (chunk_count*(CHUNK_SIZE/BATCH_SIZE) )) )
                          
                    f = open(data_dir+"model_loss.txt", "a")
                    f.write(str(chunk_count)+",")
                    f.write(str(chunk_count*EVAL_CHUNK)+",")
                    f.write(str((current_loss / batch_count))+",")
                    f.close()
                    
                    with torch.no_grad():
                        test_chunk = pd.read_csv(test_filename, sep='\t',nrows=TEST_CHUNK_SIZE)

                        #process chunk
                        test_dataset = EbayDataset(test_chunk,data_dir=data_dir)
                        # Prepare dataset
                        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=TEST_CHUNK_SIZE, shuffle=False, num_workers=0)
                        
                        if dev_set_calc == 0:
                            targets_all = torch.tensor([])
                            outputs_all = torch.tensor([])
                            inputs_all = torch.tensor([])
                            targets_actual_all = torch.tensor([])

                            for i, data in enumerate(testloader):
                                inputs, targets, targets_actual = data
                                
                                inputs = inputs.to(DEVICE)
                                targets = targets.to(DEVICE)
                                

                                outputs = mlp(inputs.float())
                                inputs_all = torch.cat((inputs_all, inputs.cpu().detach()), 0)
                                outputs_all = torch.cat((outputs_all, outputs.cpu().detach()), 0)
                                targets_all = torch.cat((targets_all, targets.cpu().detach()), 0)
                                targets_actual_all = torch.cat((targets_actual_all, targets_actual.cpu().detach()), 0)
                                dev_set_calc = 1
                        else:
                            outputs_all = mlp(inputs_all.float())
                            #outputs_all = torch.cat((outputs_all, outputs.cpu().detach()), 0)

                        loss_test = loss_function(outputs_all, targets_all)
                        #loss_test = loss_function(outputs_all, torch.reshape(targets_all.float(),(len(outputs_all),1)))
                        #loss_test = ebay_loss(outputs_all, torch.reshape(targets_all.float(),(len(outputs_all),1)))
                        y_pred = sftmx(outputs_all.detach())
                        y_pred = np.argmax(outputs_all.detach(), axis=1)
                        targets_all_ = np.argmax(targets_all.detach(), axis=1)

                        print("Test loss:", loss_test.item())
                        prec, rec, f1, supp = precision_recall_fscore_support(targets_all_, y_pred, average=None)
                        print("Prec:", prec)
                        print("Reca:", rec)
                        print("F1  :", f1)
                        print("Supp:", supp)

                        
                        
                        if loss_test.item() <= best_loss:
                            best_lost = loss_test.item() 
                            torch.save(mlp.state_dict(), data_dir+MODEL_DIR+"/mlp_mid_train_"+str(epoch)+"_best_loss_"+"epoch.pt")
                        #outputs_all_actual = torch.Tensor(target_scaler.inverse_transform(outputs_all))
                        #outputs_all_actual = torch.Tensor(np.arctanh(torch.clamp(outputs_all, min=-0.90, max=0.90)))
                        '''
                        x = torch.clamp(outputs_all, min=0.05, max=9.95)
                        outputs_all_actual = (1/2)*(np.log((1+x)/(1-x)))*10
                        loss_test = loss_function(outputs_all_actual, torch.reshape(targets_actual_all.float(),(len(outputs_all_actual),1)))                     
                        print("Test loss:", loss_test.item())
                        '''
                        f = open(data_dir+"model_loss.txt", "a")
                        f.write(str(loss_test.item())+"\n")
                        f.close()                        
                        
                        loss_lst.append(current_loss)
                        loss_lst.append(loss_test.item())
                        


                    current_loss = 0.0
                    batch_count = 0
            
        torch.save(mlp.state_dict(), data_dir+MODEL_DIR+"/mlp_mid_train_"+str(epoch)+"_epoch.pt")
        pickle.dump(loss_lst, open( data_dir+MODEL_DIR+"/loss_lst.pkl", "wb" ))


if __name__ == '__main__':
    #data_dir = '/content/drive/MyDrive/Colab_Notebooks/Ebay/data/'
    data_dir = '/Users/pamelakatali/Downloads/Ebay_ML/data/'
    train(data_dir)
