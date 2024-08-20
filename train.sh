#!/bin/bash

train_dir='./officalData/TrainSet/'
train_4_image_list='./TrainSet_4_image.txt'
python train.py $train_dir $train_4_image_list
