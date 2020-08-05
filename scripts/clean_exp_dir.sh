#!/bin/bash

if [ "$1" == "--help" ]; then
    printf -- 'Cleaning invalid training directory\n';
    printf -- 'Usage: %s <experiment-root>\n' $0;
    exit 0;
fi;

experiment_root=$1

if [ ! $experiment_root ]; then
    echo "need experiment root directory!"
else 
    for experiment_dir in $(ls $experiment_root)
    do
        num_files=$(ls $experiment_root/$experiment_dir | wc -l)
        if [ $num_files -eq 1 ]; then
            echo "Empty training directory: "$experiment_root/$experiment_dir
            rm -r $experiment_root/$experiment_dir
        fi
    done
fi

