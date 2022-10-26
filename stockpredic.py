#!/usr/bin/env python                                                           
#                                                                               
# file: stockpredic.py                                                          
#                                                                               
#-----------------------------------------------------------------------------  

# import system modules                                                         
import sys
import pandas as pd
import time
import datetime
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pylab import rcParams
from tensorflow.python import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional
from keras import callbacks

def process_data(stockdata):

    # read and print unformatted stock data                                     
    mmm_stock_data = pd.read_csv(stockdata)
    print("\n")
    print(mmm_stock_data.head())

    # only keep the date and adjusted close prices                            
    mmm_stock_data = mmm_stock_data[['Date', 'Adj Close']]

    # convert Date to datetime and set Date as index                            
    mmm_stock_data['Date'] = pd.to_datetime(mmm_stock_data['Date'].apply(lambda x: x.split()[0]))
    mmm_stock_data.set_index('Date', drop = True, inplace = True)

    # print the first 10 items in mmm_stock_data                                
    print("\n")
    print(mmm_stock_data.head())

    # graph adjusted closing prices                                             
    rcParams['figure.figsize'] = 14, 8
    ax1 = mmm_stock_data.plot(y = 'Adj Close', style='b-', grid = True, label = "Closing Price")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price (USD)")
    ax1.set_title("MMM Closing Price 1970-2022")
    plt.legend()

    # data preprocessing                                                        
    MMS = MinMaxScaler()
    mmm_stock_data[mmm_stock_data.columns] = MMS.fit_transform(mmm_stock_data)
    print("\n")
    print(mmm_stock_data.shape)

    # train dataset is 80 percent of overall dataset                            
    training_size = round(len(mmm_stock_data) * 0.80)
    print("Training Size: %s" % (training_size))
    train_data = mmm_stock_data[:training_size]
    test_data = mmm_stock_data[training_size:]
    print("\n")
    print(train_data.shape, test_data.shape)

    # sequence for training and testing                                         
    train_seq, train_label = create_sequence(train_data)
    test_seq, test_label = create_sequence(test_data)
    print("\n")
    print(train_seq.shape, train_label.shape, test_seq.shape, test_label.shape)
    print("\n")

    # create LSTM model using Keras                                             
    model = Sequential()

    # 50 units assigned in LSTM parameter, dropout rate of 10%                  
    model.add(LSTM(units = 50, return_sequences = True, input_shape = (train_seq.shape[1], train_seq.shape[2])))
    model.add(Dropout(0.1))
    model.add(LSTM(units = 50))
    model.add(Dense(1, name = "layer1"))
    model.compile(loss = 'mean_squared_error', optimizer = 'adam', metrics = ['mean_absolute_error'])

    print("\n")
    print(model.summary())

    # stop epochs if model starts overfitting                                   
    earlystopping = callbacks.EarlyStopping(monitor = "val_loss", mode = "min", patience = 5, restore_best_weights = True)

    # fit the model, 25 epochs but could stop early                             
    model.fit(train_seq, train_label, epochs = 25, validation_data = (test_seq, test_label), verbose = 1, callbacks = [earlystopping])

    # predicted test values                                                     
    test_predicted = model.predict(test_seq)
    print(test_predicted[:5])

    # inverse needs to be taken on test_predicted to get back original values   
    test_inverse_predicted = MMS.inverse_transform(test_predicted)
    print(test_inverse_predicted[:5])

    # combine the stock data with the predicted data                            
    slic_data = pd.concat([mmm_stock_data.iloc[-2579:].copy(), pd.DataFrame(test_inverse_predicted, columns = ['Adj Close Predicted'], index = mmm_stock_data.iloc[-2579:].index)], axis=1)

    slic_data[['Adj Close']] = MMS.inverse_transform(slic_data[['Adj Close']])
    print(slic_data[['Adj Close', 'Adj Close Predicted']].head(10))
    print(slic_data[['Adj Close', 'Adj Close Predicted']].tail(10))

    # graph real vs. predicted closing prices                                   
    slic_data[['Adj Close', 'Adj Close Predicted']].plot(figsize = (10, 6))
    plt.xticks(rotation = 45)
    plt.xlabel("Date", size = 15)
    plt.ylabel("Stock Price", size = 15)
    plt.title("Real vs. Predicted Close Price", size=15)

    print(model.summary())
    plt.show()

    return True

def create_sequence(dataset):

    sequences = []
    labels = []

    start_idx = 0
    for stop_idx in range(50, len(dataset)):
        sequences.append(dataset.iloc[start_idx:stop_idx])
        labels.append(dataset.iloc[stop_idx])
        start_idx += 1

    return (np.array(sequences), np.array(labels))

# function: main                                                                
#                                                                               
def main(argv):

    x = process_data(argv[1])

# begin gracefully 
#           
if __name__ == '__main__':
    main(sys.argv[0:])
#       
# end of file 