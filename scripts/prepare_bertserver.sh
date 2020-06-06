#!/bin/bash

if [ $# -lt 1 ]; then
  cat >&2 <<EOF 
Usage: $0 <bert-server-dir> [<num-worker>]
e.g.: $0 utils/bert 4
EOF
  exit 1;
fi

bertserver=$1
num_worker=3
if [ $# -eq 2 ]; then
    num_worker=$2
fi

[ ! -d $bertserver ] && mkdir $bertserver
cd $bertserver

if [ ! -d chinese_L-12_H-768_A-12 ]; then
    wget https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip
    unzip chinese_L-12_H-768_A-12.zip 
fi

bert-serving-start -model_dir chinese_L-12_H-768_A-12 -num_worker $num_worker
 
