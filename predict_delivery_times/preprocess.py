import pickle
import datetime
import pgeocode
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler


def get_zip(zipc):
    return zipc[:5]

def convert_timestamp(timestamp):
  #converts gmt with offset to utc
  
  curr_timestamp = timestamp[:23]
  offset_h = int(timestamp[24:26])
  offset_m = int(timestamp[27:])

  if timestamp[23] == '-':
    return datetime.datetime.fromisoformat(curr_timestamp) - datetime.timedelta(hours=offset_h, minutes=offset_m)
  else:
    return datetime.datetime.fromisoformat(curr_timestamp) + datetime.timedelta(hours=offset_h, minutes=offset_m)

def get_accept_days(row):
    return row.days

def get_month(date):
    return date.month

def convert_weight(row):
    if row['weight_units'] == 1: #convert everything to lbs
        return row['weight']
    if row['weight_units']  == 2:
        return 2.20462 * row['weight']

def get_flat_rate(rate):
    rates_lst = [0.0, 3.99, 5.0, 4.99, 4.5, 6.0, 7.0, 5.99, 4.25, 6.5, 4.75, 5.75, 5.25, 4.85]
    if rate in rates_lst:
        return rates_lst.index(rate)
    else:
        return -1

def get_non_weight_rate(row):
    if row['flat_rate'] != -1:
        if row['weight'] > 0:
            return row['shipping_fee'] / row['weight']
        else:
            return row['shipping_fee'] 
    else:
        return -1


def get_delivery_days(row):
    delivery_date = datetime.datetime.fromisoformat(row['delivery_date']) + datetime.timedelta(hours=12)
    days_delivery = delivery_date - row['acceptance_scan_timestamp']
    return days_delivery.days

def get_zip_region(zipcode, start, end):
    zipcode = zipcode[:5]
    try:
        return int(zipcode[start:end])
    except:
        return -1

def preprocess(data_dir, dataset, dev_set=0, quiz_set=0):
    train_df = pd.read_csv(data_dir+dataset, sep='\t')
    train_preprocess = pd.DataFrame([])

    drop_lst = np.where(train_df['buyer_zip'].isnull().values)[0]
    train_df = train_df.drop(drop_lst)
    drop_lst = np.where(train_df['item_zip'].isnull().values)[0]
    train_df = train_df.drop(drop_lst)

    
    train_preprocess['declared_handling_days'] = train_df['declared_handling_days'].fillna(1.6209234583705132)#mean

    train_preprocess['carrier_min_estimate'] = train_df['carrier_min_estimate'].replace(-1,0)
    train_preprocess['carrier_max_estimate'] = train_df['carrier_max_estimate']
    train_preprocess['carrier_diff_estimate'] = train_df['carrier_max_estimate'] - train_df['carrier_min_estimate']

    train_df['acceptance_scan_timestamp'] = train_df['acceptance_scan_timestamp'].apply(convert_timestamp)
    train_df['payment_datetime'] = train_df['payment_datetime'].apply(convert_timestamp)
    c =  train_df['acceptance_scan_timestamp'] - train_df['payment_datetime']
    train_preprocess['accept_days'] = c.apply(get_accept_days)
    del c

    
    
    dist = pgeocode.GeoDistance('US')
    '''
    train_df['item_zip'] = train_df['item_zip'].apply(get_zip)
    train_df['buyer_zip'] = train_df['buyer_zip'].apply(get_zip)
    '''

    print(train_df['item_zip'] )
    print(train_df['buyer_zip'] )
    train_preprocess['shipping_dist'] = dist.query_postal_code(train_df['item_zip'].apply(get_zip).values, train_df['buyer_zip'].apply(get_zip).values)
    train_preprocess['shipping_dist'] = train_preprocess['shipping_dist'].fillna(1752.4414692310288)

    train_preprocess['shipping_fee'] = train_df['shipping_fee']
    train_preprocess['flat_rate'] = train_df['shipping_fee'].apply(get_flat_rate)
    train_preprocess['item_price'] = train_df['item_price']
    train_preprocess['quantity'] = train_df['quantity']

    train_preprocess['weight'] = train_df[['weight', 'weight_units']].apply(convert_weight, axis=1)

    train_preprocess['non_weight_rate'] = train_preprocess[['shipping_fee', 'weight', 'flat_rate']].apply(get_non_weight_rate, axis=1)
    
    #a = np.where(train_df['weight'] != 0)[0]
    #weight_mean = 23.673040684516266 #train_df['weight'][a].mean()
    #train_preprocess['weight'] = train_preprocess['weight'].replace(0, weight_mean)
    train_preprocess['weight_units'] = train_df['weight_units'] - 1


    #scaler
    cols = train_preprocess.columns
    #print()


    train_preprocess['payment_month'] = train_df['payment_datetime'].apply(get_month)
    


    train_preprocess['seller_id'] = train_df['seller_id']

    train_preprocess['item_zip'] = train_df['item_zip']
    train_preprocess['buyer_zip'] = train_df['buyer_zip']
    
    train_preprocess['item_group'] = train_df['item_zip'].apply(get_zip_region, start=0, end=1)
    train_preprocess['buyer_group'] = train_df['buyer_zip'].apply(get_zip_region, start=0, end=1)

    train_preprocess['item_region'] = train_df['item_zip'].apply(get_zip_region, start=1, end=3)
    train_preprocess['buyer_region'] = train_df['buyer_zip'].apply(get_zip_region, start=1, end=5)

    train_preprocess['item_addr'] = train_df['item_zip'].apply(get_zip_region, start=3, end=5)
    train_preprocess['buyer_addr'] = train_df['buyer_zip'].apply(get_zip_region, start=3, end=5)
    
    #one-hot features
    train_preprocess['ship_id'] = train_df['shipment_method_id']
    train_preprocess['cat_id'] = train_df['category_id']
    train_preprocess['pack_size'] = train_df['package_size']
    train_preprocess['b2c_c2c'] = train_df['b2c_c2c']




    if quiz_set == 0:
        #delivery days
        train_preprocess['delivery_days'] = train_df[['acceptance_scan_timestamp','delivery_date']].apply(get_delivery_days, axis=1)
        drop_lst = np.where(train_preprocess['delivery_days'] < 0)[0]
        train_preprocess = train_preprocess.drop(drop_lst)

    else:
        train_preprocess['delivery_days'] = np.zeros(len(train_preprocess))

    #save data
    train_preprocess.to_csv(data_dir+'clean_'+dataset, sep="\t")



    if dev_set == 0 and quiz_set == 0:
        scaler = MinMaxScaler(feature_range=(-1,1))
        scaler.fit(train_preprocess[cols].values)

        pickle.dump(scaler, open( data_dir+"train_minmax_scaler_3.pkl", "wb"))




if __name__ == '__main__':
    #data_dir = '/content/drive/MyDrive/Colab_Notebooks/Ebay/data/'
    data_dir = '/Users/pamelakatali/Downloads/Ebay_ML/data/'
    
    #dataset = 'ebay_train.tsv.gz'
    #dataset = 'ebay_dev.tsv.gz'
    dataset = 'ebay_quiz.tsv.gz'
    preprocess(data_dir, dataset, dev_set=0,quiz_set=1)


