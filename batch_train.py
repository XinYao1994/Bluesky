# -*- coding: utf-8 -*-
import csv
import collections
# Location Name item[0]
# Time item[1]
# Total (Wh) item[2]
# Ac (Wh) Light (Wh) Socket (Wh) item[3, 4, 5]
# Fridge (Wh) Water Heater (Wh) Cooking Appliance (Wh)  Other (Wh)  Mixed Usage (Wh)

#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function
#from __future__ import unicode_literals

#from caffe2.python import core, workspace, model_helper, utils, brew
#from caffe2.python.rnn_cell import LSTM
#from caffe2.proto import caffe2_pb2
#from caffe2.python.optimizer import build_sgd

import math
# using Keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# drawing
from pandas import *
import string
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  
import numpy as np

# utils
import argparse
import logging
from datetime import datetime

split = 0.8
# loop = 24 this means a day
# --gpu

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
  dataX, dataY = [], []
  for i in range(len(dataset)-look_back-1):
    a = dataset[i:(i+look_back), 0]
    dataX.append(a)
    dataY.append(dataset[i + look_back, 0])
  return np.array(dataX), np.array(dataY)


def main():
  parser = argparse.ArgumentParser(description="Blue Sky Training")
  parser.add_argument("--train_data", type=str, default="hku_jcsviii.csv", help="Path to training data in a text file format")
  parser.add_argument("--train_iter", type=int, default=1000, help="max training iteration")
  parser.add_argument("--seq_length", type=int, default=24, help="One training example sequence length")
  parser.add_argument("--layers", type=int, default=1, help="the number of layers of the NN")
  parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")
  #parser.add_argument("--iters_to_report", type=int, default=500, help="How often to report loss and generate text")
  #parser.add_argument("--hidden_size", type=int, default=100, help="Dimension of the hidden representation")
  #parser.add_argument("--gpu", action="store_true", help="If set, training is going to use GPU 0")
  
  args = parser.parse_args()
  
  # read data from csv
  csv_data = csv.reader(open(args.train_data, "r"))
  loop = args.seq_length
  layers = args.layers
  dict = collections.OrderedDict()
  # put data in dict
  for item in csv_data:
    if dict.has_key(item[0]):
      dict[item[0]][item[1]] = item[2:6]
    else:
      dict[item[0]] = collections.OrderedDict()
      dict[item[0]][item[1]] = item[2:6]
  
  print "Available Location Name:"
  dict.pop('Location Name')
  #print dict.keys()

  #while True:
  order = "NC6F 10"
        locate = order.split(' ')[0]
        period = order.split(' ')[1]
        # training with the data
        print "train the data with location " + locate
        data = dict[locate]
        dateAsKey = data.keys()
        # Train total:
        total_data = map(lambda x:float(x[0]), data.values())
        c = {"1": total_data}
        df = DataFrame(c)
        dataset = df.values
        dataset = dataset.astype('float32')
        # normalize the dataset
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        
        train_size = int(len(dataset) * split)
        test_size = len(dataset) - train_size
        train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
        # print train
        # print test

        look_back = loop
        trainX, trainY = create_dataset(train, look_back)
        testX, testY = create_dataset(test, look_back)
        # reshape
        print trainX
        trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

        # print testX, len(testX[0][0]), testY 
        # create and fit the LSTM network
        model = Sequential()
        if layers > 1:
          model.add(LSTM(4, return_sequences=True, input_shape=(1, look_back)))
          for i in xrange(layers-2):
            model.add(LSTM(4, return_sequences=True))
          model.add(LSTM(4))
        else:
          model.add(LSTM(4, input_shape=(1, look_back)))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(trainX, trainY, epochs=args.train_iter, batch_size=args.batch_size, verbose=2)
        # make predictions
        trainPredict = model.predict(trainX)
        testPredict = model.predict(testX)
        # invert predictions
        trainPredict = scaler.inverse_transform(trainPredict)
        trainY = scaler.inverse_transform([trainY])
        testPredict = scaler.inverse_transform(testPredict)
        testY = scaler.inverse_transform([testY])
        # calculate root mean squared error
        trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
        print('Train Score: %.2f RMSE' % (trainScore))
        testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
        print('Test Score: %.2f RMSE' % (testScore))
        # print dateAsKey
        # draw now:
        # model.TrainModel()

        # forecast
        
        trainPredictPlot = np.empty_like(dataset)
        trainPredictPlot[:, :] = np.nan
        trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

        testPredictPlot = np.empty_like(dataset)
        testPredictPlot[:, :] = np.nan
        testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

        plt.figure(figsize=(12,4))
        plt.plot(scaler.inverse_transform(dataset))
        plt.plot(trainPredictPlot)
        plt.plot(testPredictPlot)
        plt.savefig(locate+"_output_l"+str(args.layers)+"_RMSE_"+str(testScore)+".pdf")

        '''
        print "predict the tendency within " + period + " hours"
        dataAsPredict = range(int(period)) 
        Predictresult = []
        for i in dataAsPredict:
          begin = len(dataset)-look_back+i
          end = len(dataset)
          window = np.append(dataset[begin:end,:], np.array(Predictresult))
          print window
          PX, PY = create_dataset(window, look_back)
          print PX, PY
          PX = np.reshape(PX, (PX.shape[0], 1, PX.shape[1]))
          Predictresult.append(model.predict(PX));
        
        plt.figure(figsize=(12,5))#
        plt.plot(dataAsPredict, Predictresult, linestyle='-', color='k', label = dataAsPredict)
        plt.xlabel("TimeStamp (hours)")
        plt.ylabel("Total Enerage Usage")
        # plt.show()
        plt.savefig("output.pdf")
        '''

if __name__ == '__main__':
  main()




