#!/bin/bash
for j in 1000 2000 4000 8000; do
	python batch_DQR.py --train_iter=$j
done
