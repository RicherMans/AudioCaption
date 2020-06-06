import sys
import logging
import yaml
import numpy as np
import torch
from pprint import pformat

def genlogger(outputfile, level="INFO"):
    formatter = logging.Formatter(
        "[ %(levelname)s : %(asctime)s ] - %(message)s")
    logger = logging.getLogger(__name__ + "." + outputfile)
    logger.setLevel(getattr(logging, level))
    # Dump log to file
    filehandler = logging.FileHandler(outputfile)
    filehandler.setFormatter(formatter)
    logger.addHandler(filehandler)
    # Log results to std
    # stdhandler = logging.StreamHandler(sys.stdout)
    # stdhandler.setFormatter(formatter)
    return logger

def parse_config_or_kwargs(config_file, **kwargs):
    with open(config_file) as con_read:
        yaml_config = yaml.load(con_read, Loader=yaml.FullLoader)
    # passed kwargs will override yaml config
    kwargs.keys()
    return dict(yaml_config, **kwargs)

def criterion_improver(mode):
    """Returns a function to ascertain if criterion did improve

    :mode: can be 'loss', 'acc' or 'score'
    :returns: function that can be called, function returns true if criterion improved

    """
    assert mode in ('loss', 'acc', 'score')
    best_value = np.inf if mode == 'loss' else 0

    def comparator(x, best_x):
        return x < best_x if mode == 'loss' else x > best_x

    def inner(x):
        # rebind parent scope variable
        nonlocal best_value
        if comparator(x, best_value):
            best_value = x
            return True
        return False
    return inner

def pprint_dict(in_dict, outputfun=sys.stdout.write, formatter="yaml"):
    """pprint_dict
    :param outputfun: function to use, defaults to sys.stdout
    :param in_dict: dict to print
    """
    if formatter == "yaml":
        format_fun = yaml.dump
    elif formatter == "pretty":
        format_fun = pformat
    for line in format_fun(in_dict).split("\n"):
        outputfun(line)

def log_results(engine,
                cv_evaluator,
                cv_dataloader,
                outputfun=sys.stdout.write,
                train_metrics=["loss", "accuracy"],
                cv_metrics=["loss", "accuracy"]):
    train_results = engine.state.metrics
    cv_evaluator.run(cv_dataloader, max_epochs=1)
    cv_results = cv_evaluator.state.metrics
    output_str_list = [
        "Results - Epoch : {:<4}".format(engine.state.epoch)
    ]
    output_str_list.append("Train - ")
    for metric in train_metrics:
        output = train_results[metric]
        if isinstance(output, torch.Tensor):
            output = output.item()
        output_str_list.append("{} {:<5.2g}  ".format(
            metric, output))
    output_str_list.append("Validation - ")
    for metric in cv_metrics:
        output = cv_results[metric]
        if isinstance(output, torch.Tensor):
            output = output.item()
        output_str_list.append("{} {:5<.3g} ".format(
            metric, output))
    outputfun(" ".join(output_str_list))


def save_model_on_improved(engine,
                           criterion_improved,
                           metric_key,
                           dump,
                           save_path):
    if criterion_improved(engine.state.metrics[metric_key]):
        torch.save(dump, save_path)

def update_reduce_on_plateau(engine, scheduler, metric):
    val_loss = engine.state.metrics[metric]
    if  scheduler.__class__.__name__ == "ReduceLROnPlateau":
        scheduler.step(val_loss)
    else:
        scheduler.step()
