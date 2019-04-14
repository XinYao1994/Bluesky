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
from keras.layers import Dense, Activation
from keras.layers import LSTM

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import keras.backend as K

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

def tilted_loss(q,y,f):
  e = (y-f)
  return K.mean(K.maximum(q*e, (q-1)*e), axis=-1)

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
  times_ = map(lambda x: x%24, xrange(len(total_data)))
  
  model = Sequential()
  model.add(Dense(units=24, input_dim=1, activation='relu'))
  model.add(Dense(units=24, input_dim=1, activation='relu'))
  model.add(Dense(1))
  model.compile(loss='mae', optimizer='adadelta')
  model.fit(times_, total_data, epochs=args.train_iter, batch_size=args.batch_size, verbose=0)
  model.evaluate(times_, total_data)
  
  plt.scatter(times_, total_data)
  time_test = np.array(xrange(24))
  p_test = model.predict(time_test)
  plt.plot(time_test, p_test, 'r', label="Sequential")
  
  #Quantiles
  Q_range = [0.1, 0.3, 0.5, 0.7, 0.9]
  for q in Q_range:
    model = Sequential()
    model.add(Dense(units=24, input_dim=1, activation='relu'))
    model.add(Dense(units=24, input_dim=1, activation='relu'))
    model.add(Dense(1))
    model.compile(loss=lambda y,f: tilted_loss(q,y,f), optimizer='adadelta')
    model.fit(times_, total_data, epochs=args.train_iter, batch_size=args.batch_size, verbose=0)
    
    q_test = model.predict(time_test)
    plt.plot(time_test, q_test, label=q)
  
  
  plt.savefig(locate+"_output_l"+str(args.layers)+"_"+str(args.train_iter)+"_RMSE_"+str(testScore)+".pdf")
  
plt.legend()
if __name__ == '__main__':
  main()




