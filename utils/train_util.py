# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import os
import sys
import logging
import datetime
import yaml
import torch
import numpy as np
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



def log_results(engine,
                cv_evaluator, 
                cv_dataloader, 
                outputfun=sys.stdout.write,
                cv_metrics=["loss", "accuracy"]):
    cv_evaluator.run(cv_dataloader)
    cv_results = cv_evaluator.state.metrics
    output_str_list = [
        "Validation Results - Epoch : {:<4}".format(engine.state.epoch)
    ]
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


def update_lr(engine, scheduler, metric):
    if scheduler.__class__.__name__ == "ReduceLROnPlateau":
        cv_result = engine.state.metrics[metric]
        scheduler.step(cv_result)
    else:
        scheduler.step()
