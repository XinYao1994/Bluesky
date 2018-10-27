# -*- coding: utf-8 -*-
import csv
import collections
# Location Name item[0]
# Time item[1]
# Total (Wh) item[2]
# Ac (Wh) Light (Wh) Socket (Wh) item[3, 4, 5]
# Fridge (Wh) Water Heater (Wh) Cooking Appliance (Wh)  Other (Wh)  Mixed Usage (Wh)
train_iter = 10000
path = "hku_jcsviii.csv"
# read data from csv
csv_data = csv.reader(open(path, "r"))
dict = collections.OrderedDict()
# put data in dict
for item in csv_data:
	if dict.has_key(item[0]):
		dict[item[0]][item[1]] = item[2:6]
	else:
		dict[item[0]] = {}
		dict[item[0]][item[1]] = item[2:6]

print "Available Location Name:"
dict.pop('Location Name')
print dict.keys()

while True:
	order = raw_input("Input locations and forecast period: ")
	if "exit" in order: break;
	else:
		try:
			locate = order.split(' ')[0]
			period = order.split(' ')[1]
			# training with the data
			print "train the data with location " + locate
		
			# forecast
			print "predict the tendency within " + period + " hours"
		except Exception as E:
			print "illegal input"






