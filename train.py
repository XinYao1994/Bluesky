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
import numpy as np
from datetime import datetime

split = 0.67
# --gpu

def main():
  #parser = argparse.ArgumentParser(description="Caffe2: Blue Sky Training")
  parser.add_argument("--train_data", type=str, default="hku_jcsviii.csv", help="Path to training data in a text file format")
  #parser.add_argument("--train_iter", type=int, default=10000, help="max training iteration")
  #parser.add_argument("--seq_length", type=int, default=25, help="One training example sequence length")
  #parser.add_argument("--batch_size", type=int, default=1, help="Training batch size")
  #parser.add_argument("--iters_to_report", type=int, default=500, help="How often to report loss and generate text")
  #parser.add_argument("--hidden_size", type=int, default=100, help="Dimension of the hidden representation")
  #parser.add_argument("--gpu", action="store_true", help="If set, training is going to use GPU 0")
  
  args = parser.parse_args()
  
  # read data from csv
  csv_data = csv.reader(open(args.train_data, "r"))
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
  print dict.keys()

  while True:
    order = raw_input("\nInput locations and forecast period: ")
    if "exit" in order: break;
    else:
      #try:
        locate = order.split(' ')[0]
        period = order.split(' ')[1]
        #device = core.DeviceOption(caffe2_pb2.CUDA if args.gpu else caffe2_pb2.CPU, 0)
        #with core.DeviceScope(device):
        #  model = CharRNN(args)
        #  model.CreateModel()
        # training with the data
        print "train the data with location " + locate
        data = dict[locate]
        dateAsKey = data.keys()
        # Train total:
        total_data = map(lambda x:float(x[0]), data.values())
        c = {"a": range(len(dateAsKey)), "b": total_data}
        df = DataFrame(c)
        dataset = df.values
        dataset = dataset.astype('float32')
        print dataset
        # normalize the dataset
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        
        # print dateAsKey
        # draw now:
        # model.TrainModel()
        # forecast
        print "predict the tendency within " + period + " hours"
        plt.figure(figsize=(12,5))#
        plt.plot(np.arange(len(dataset)), dataset, linestyle='-', color='k', label = dateAsKey)
        plt.xlabel("TimeStamp (hours)")
        plt.ylabel("Total Enerage Usage")
        plt.show()
        plt.savefig("x.pdf")
      #except Exception as E:
      #  print "illegal input"

if __name__ == '__main__':
  #workspace.GlobalInit(['caffe2', '--caffe2_log_level=2'])
  main()




