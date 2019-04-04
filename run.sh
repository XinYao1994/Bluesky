#!/bin/bash
for i in 1 2 4 8 16 32; do
	for j in 1000 2000 4000 8000; do
		python batch_train.py --train_iter=$j --layers=$i
	done
done
#apython batch_train.py --train_iter=2000 --layers=2
