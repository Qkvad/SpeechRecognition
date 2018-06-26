#!/bin/bash

python tf/src/label_wav.py \
--graph=tf/models/conv0875.pb \
--wav=$1 \
--labels=tf/logs/single_fc/train_progress/single_fc_labels.txt
