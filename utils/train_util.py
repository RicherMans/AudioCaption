# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import os
import sys
import logging
import datetime
import yaml
import torch
import numpy as np
import tableprint as tp
import pandas as pd
import sklearn.preprocessing as pre
from pprint import pformat

sys.path.append(os.getcwd())

def genlogger(outputfile, level="INFO"):
    formatter = logging.Formatter(
        "[ %(levelname)s : %(asctime)s ] - %(message)s")
    logger = logging.getLogger(__name__ + "." + outputfile)
    logger.setLevel(getattr(logging, level))
    # Log results to std
    # stdhandler = logging.StreamHandler(sys.stdout)
    # stdhandler.setFormatter(formatter)
    # Dump log to file
    filehandler = logging.FileHandler(outputfile)
    filehandler.setFormatter(formatter)
    logger.addHandler(filehandler)
    # logger.addHandler(stdhandler)
    return logger


def pprint_dict(in_dict, outputfun=sys.stdout.write, formatter='yaml'):
    """pprint_dict

    :param outputfun: function to use, defaults to sys.stdout
    :param in_dict: dict to print
    """
    if formatter == 'yaml':
        format_fun = yaml.dump
    elif formatter == 'pretty':
        format_fun = pformat
    for line in format_fun(in_dict).split('\n'):
        outputfun(line)


def encode_labels(labels: pd.Series, encoder=None):
    """encode_labels

    Encodes labels

    :param labels: pd.Series representing the raw labels e.g., Speech, Water
    :param encoder (optional): Encoder already fitted 
    returns encoded labels (one hot) and the encoder
    """
    assert isinstance(labels, pd.Series), "Labels need to series"
    if not encoder:
        encoder = pre.LabelEncoder()
        encoder.fit(labels)
    labels_encoded = encoder.transform(labels)
    return labels_encoded.tolist(), encoder


def parse_config_or_kwargs(config_file, **kwargs):
    with open(config_file) as con_read:
        yaml_config = yaml.load(con_read, Loader=yaml.FullLoader)
    # passed kwargs will override yaml config
    return dict(yaml_config, **kwargs)


def parse_augments(augment_list):
    """parse_transforms
    parses the transformation string in configuration file to corresponding methods

    :param transform_list: list
    """
    from datasets import augment

    augments = []
    for transform in augment_list:
        if transform == "timemask":
            augments.append(augment.TimeMask(1, 50))
        elif transform == "freqmask":
            augments.append(augment.FreqMask(1, 10))
    return torch.nn.Sequential(*augments)


def criterion_improver(mode):
    assert mode in ("loss", "acc", "score")
    best_value = np.inf if mode == "loss" else 0

    def comparator(x, best_x):
        return x < best_x if mode == "loss" else x > best_x

    def inner(x):
        nonlocal best_value

        if comparator(x, best_value):
            best_value = x
            return True
        return False
    return inner


def on_training_started(engine, outputfun=sys.stdout.write, header=[]):
    outputfun("<== Training Started ==>")
    for line in tp.header(header, style="grid").split("\n"):
        outputfun(line)


def log_results(engine,
                cv_evaluator, 
                cv_dataloader, 
                outputfun=sys.stdout.write,
                train_metrics=["loss", "accuracy"], 
                cv_metrics=["loss", "accuracy"]):
    train_results = engine.state.metrics
    cv_evaluator.run(cv_dataloader)
    cv_results = cv_evaluator.state.metrics
    output_str_list = [
        "Validation Results - Epoch : {:<4}".format(engine.state.epoch)
    ]
    for metric in train_metrics:
        output = train_results[metric]
        if isinstance(output, torch.Tensor):
            output = output.item()
        output_str_list.append("{} {:<5.2g} ".format(
            metric, output))
    for metric in cv_metrics:
        output = cv_results[metric]
        if isinstance(output, torch.Tensor):
            output = output.item()
        output_str_list.append("{} {:5<.2g} ".format(
            metric, output))

    outputfun(" ".join(output_str_list))


def save_model_on_improved(engine,
                           criterion_improved, 
                           metric_key,
                           dump,
                           save_path):
    if criterion_improved(engine.state.metrics[metric_key]):
        torch.save(dump, save_path)


def on_training_ended(engine, n, outputfun=sys.stdout.write):
    outputfun(tp.bottom(n, style="grid"))


def update_reduce_on_plateau(engine, scheduler, metric):
    cv_loss = engine.state.metrics[metric]
    scheduler.step(cv_loss)
