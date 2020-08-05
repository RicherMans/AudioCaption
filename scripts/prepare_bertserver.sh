#!/bin/bash
#SBATCH --job-name bert-server
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --output=logs/bert-server-%j.log
#SBATCH --error=logs/bert-server-%j.err

if [ $# -lt 1 ]; then
  cat >&2 <<EOF 
Usage: $0 <bert-server-dir> <lang> [<num-worker>]
e.g.: $0 utils/bert zh 4
EOF
  exit 1;
fi

node=$(hostname -s)

module load cuda/10.0
source activate bertserver

bertserver=$1
num_worker=3
lang=$2

if [ $# -eq 3 ]; then
    num_worker=$3
fi

[ ! -d $bertserver ] && mkdir $bertserver
cd $bertserver

if [ $lang == "zh" ]; then
    model_dir="chinese_L-12_H-768_A-12"
    if [ ! -d chinese_L-12_H-768_A-12 ]; then
        [ ! -f chinese_L-12_H-768_A-12.zip ] && wget https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip
        unzip chinese_L-12_H-768_A-12.zip 
    fi
else
    model_dir="uncased_L-12_H-768_A-12"
    if [ ! -d uncased_L-12_H-768_A-12 ]; then
        mkdir uncased_L-12_H-768_A-12
        [ ! -f uncased_L-12_H-768_A-12.zip ] && wget https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-12_H-768_A-12.zip
        unzip uncased_L-12_H-768_A-12.zip -d uncased_L-12_H-768_A-12
    fi
fi

# print which machine the server is running on
echo -e "
Bert server is running on ${node}
"

bert-serving-start -model_dir ${model_dir} -num_worker ${num_worker}

