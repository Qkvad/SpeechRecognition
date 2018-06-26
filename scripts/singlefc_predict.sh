#!/bin/bash

python tf/src/label_wav.py \
--graph=tf/models/single_fc_032.pb \
--wav=$1 \
--labels=tf/logs/single_fc/train_progress/single_fc_labels.txt
