# coding=utf-8
#!/usr/bin/env python3
import datetime
import torch
from pprint import pformat
import models
import fire
import logging
from SJTUDataSet import create_dataloader_train_cv
import kaldi_io
import yaml
import os
from build_vocab import Vocabulary
import numpy as np
import tableprint as tp
import sklearn.preprocessing as pre
import torchnet as tnt
import random
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def parsecopyfeats(feat, cmvn=False, delta=False, splice=None):
    outstr = "copy-feats ark:{} ark:- |".format(feat)
    if cmvn:
        outstr += "apply-cmvn-sliding --center ark:- ark:- |"
    if delta:
        outstr += "add-deltas ark:- ark:- |"
    if splice and splice > 0:
        outstr += "splice-feats --left-context={} --right-context={} ark:- ark:- |".format(
            splice, splice)
    return outstr


def sample_cv(dataloader, encodermodel, decodermodel,
              criterion, sample_length: int = 20):
    """
    Samples from decoder for evaluation
    """
    encodermodel = encodermodel.eval()
    decodermodel = decodermodel.eval()
    avg_value_meter = tnt.meter.AverageValueMeter()
    acc_meter = tnt.meter.ClassErrorMeter(accuracy=True)
    with torch.no_grad():
        for i, (features, captions, lengths) in enumerate(dataloader):
            features = features.float().to(device)
            captions = captions.long().to(device)
            features, encoder_state = encodermodel(features)
            # print(features.shape, features.mean(1))
            for idx in range(len(features)):
                outputs, probs = decodermodel.sample(
                    features[idx].unsqueeze(0),
                    states=None, maxlength=lengths[idx],
                    return_probs=True)
                probs = probs.squeeze(0)
                target_trimmed = captions[idx][:lengths[idx]]
                loss = criterion(probs, target_trimmed)
                avg_value_meter.add(loss.item())
                acc_meter.add(probs.data, target_trimmed.data)
    return avg_value_meter.value(), acc_meter.value()[0]


def trainepoch(dataloader, encodermodel, decodermodel,
               criterion, optimizer, vocab, teacher_forcing=True):
    encodermodel = encodermodel.train()
    decodermodel = decodermodel.train()
    avg_value_meter = tnt.meter.AverageValueMeter()
    acc_meter = tnt.meter.ClassErrorMeter(accuracy=True)
    with torch.set_grad_enabled(True):
        for i, (features, captions, lengths) in enumerate(dataloader):
            features = features.float().to(device)
            captions = captions.long().to(device)
            features, encoder_state = encodermodel(features)
            loss = 0
            # print("Train:",features.shape, features.mean(1))
            if teacher_forcing:
                outputs = decodermodel(
                    features, captions, lengths, state=encoder_state)
                # Remove padding from targets
                targets = torch.nn.utils.rnn.pack_padded_sequence(
                    captions, lengths, batch_first=True)[0]
                loss = criterion(outputs, targets)
                acc_meter.add(outputs.data, targets.data)
                # print(outputs.max(1)[1], targets[0)
                avg_value_meter.add(loss.item())
            else:
                for idx in range(len(features)):
                    outputs, probs = decodermodel.sample(
                        features[idx].unsqueeze(0),
                        states=None, maxlength=lengths[idx],
                        return_probs=True)
                    probs = probs.squeeze(0)
                    target_trimmed = captions[idx][:lengths[idx]]
                    cur_loss = criterion(probs, target_trimmed)
                    loss += cur_loss
                    avg_value_meter.add(cur_loss.item())
                    acc_meter.add(probs.data, target_trimmed.data)
            encodermodel.zero_grad()
            decodermodel.zero_grad()
            loss.backward()
            optimizer.step()
    return avg_value_meter.value(), acc_meter.value()[0]


def genlogger(outdir, fname):
    formatter = logging.Formatter(
        "[ %(levelname)s : %(asctime)s ] - %(message)s")
    logging.basicConfig(
        level=logging.DEBUG,
        format="[ %(levelname)s : %(asctime)s ] - %(message)s")
    logger = logging.getLogger("Pyobj, f")
    # Dump log to file
    fh = logging.FileHandler(os.path.join(outdir, fname))
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def parse_config_or_kwargs(config_file, **kwargs):
    with open(config_file) as con_read:
        yaml_config = yaml.load(con_read)
    # passed kwargs will override yaml config
    kwargs.keys()
    return dict(yaml_config, **kwargs)


def criterion_improver(mode):
    """Returns a function to ascertain if criterion did improve

    :mode: can be ether 'loss' or 'acc'
    :returns: function that can be called, function returns true if criterion improved

    """
    assert mode in ('loss', 'acc')
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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(1)


def main(features: str, vocab_file: str,
         config='config/trainconfig.yaml', **kwargs):
    """Trains a model on the given features and vocab.

    :features: str: Input features. Needs to be kaldi formatted file
    :vocab_file:str: Vocabulary generated by using build_vocab.py
    :config: A training configuration. Note that all parameters in the config can also be manually adjusted with --ARG=VALUE
    :returns: None
    """

    config_parameters = parse_config_or_kwargs(config, **kwargs)
    outputdir = os.path.join(
        config_parameters['encodermodel'] + '_' +
        config_parameters['decodermodel'],
        datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    try:
        os.makedirs(outputdir)
    except IOError:
        pass
    logger = genlogger(outputdir, 'train.log')
    logger.info("Storing data at: {}".format(outputdir))
    logger.info("<== Passed Arguments ==>")
    # Print arguments into logs
    for line in pformat(config_parameters).split('\n'):
        logger.info(line)

    kaldi_string = parsecopyfeats(
        features, **config_parameters['feature_args'])

    scaler = getattr(
        pre, config_parameters['scaler'])(
        **config_parameters['scaler_args'])
    inputdim = -1
    logger.info(
        "<== Estimating Scaler ({}) ==>".format(
            scaler.__class__.__name__))
    for kid, feat in kaldi_io.read_mat_ark(kaldi_string):
        scaler.partial_fit(feat)
        inputdim = feat.shape[-1]
    assert inputdim > 0, "Reading inputstream failed"
    vocabulary = torch.load(vocab_file)
    vocab_size = len(vocabulary)
    logger.info(
        "Features: {} Input dimension: {} Vocab Size: {}".format(
            features, inputdim, vocab_size))
    if 'load_pretrained' in config_parameters and config_parameters['load_pretrained']:
        encodermodeldump = torch.load(
            config_parameters['load_pretrained'],
            map_location=lambda storage, loc: storage)
        pretrainedmodel = encodermodeldump['encodermodel']
        encodermodel = models.PreTrainedCNN(
            inputdim=inputdim, pretrained_model=pretrainedmodel, **
            config_parameters['encodermodel_args'])
    else:
        encodermodel = getattr(
            models, config_parameters['encodermodel'])(
            inputdim=inputdim, **config_parameters['encodermodel_args'])
    decodermodel = getattr(
        models, config_parameters['decodermodel'])(
        vocab_size=vocab_size, **config_parameters['decodermodel_args'])
    logger.info("<== EncoderModel ==>")
    for line in pformat(encodermodel).split('\n'):
        logger.info(line)
    logger.info("<== DecoderModel ==>")
    for line in pformat(decodermodel).split('\n'):
        logger.info(line)

    params = list(encodermodel.parameters()) + list(decodermodel.parameters())

    train_dataloader, cv_dataloader = create_dataloader_train_cv(
        kaldi_string,
        config_parameters['captions_file'],
        vocab_file,
        transform=scaler.transform,
        **config_parameters['dataloader_args'])
    optimizer = getattr(
        torch.optim, config_parameters['optimizer'])(
        params,
        **config_parameters['optimizer_args'])

    scheduler = getattr(
        torch.optim.lr_scheduler,
        config_parameters['scheduler'])(
        optimizer,
        **config_parameters['scheduler_args'])
    criterion = torch.nn.CrossEntropyLoss()
    trainedmodelpath = os.path.join(outputdir, 'model.th')

    encodermodel = encodermodel.to(device)
    decodermodel = decodermodel.to(device)

    criterion_improved = criterion_improver(
        config_parameters['improvecriterion'])
    for line in tp.header(
        ['Epoch', 'MeanLoss(T)', 'StdLoss(T)', 'Loss(CV)', 'StdLoss(CV)',
         "Acc(T)", "Acc(CV)", "Forcing?"],
            style='grid').split('\n'):
        logger.info(line)
    teacher_forcing_ratio = config_parameters['teacher_forcing_ratio']
    for epoch in range(1, config_parameters['epochs']+1):
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        train_loss_mean_std, train_acc = trainepoch(
            train_dataloader, encodermodel, decodermodel, criterion, optimizer,
            vocabulary, use_teacher_forcing)
        cv_loss_mean_std, cv_acc = sample_cv(
            cv_dataloader, encodermodel, decodermodel, criterion)
        logger.info(
            tp.row(
                (epoch,) + train_loss_mean_std + cv_loss_mean_std +
                (train_acc, cv_acc, use_teacher_forcing),
                style='grid'))
        epoch_meanloss = cv_loss_mean_std[0]
        if epoch % config_parameters['saveinterval'] == 0:
            torch.save({'encodermodel': encodermodel,
                        'decodermodel': decodermodel, 'scaler': scaler,
                        'config': config_parameters},
                       os.path.join(outputdir, 'model_{}.th'.format(epoch)))
        # ReduceOnPlateau needs a value to work
        schedarg = epoch_meanloss if scheduler.__class__.__name__ == 'ReduceLROnPlateau' else None
        scheduler.step(schedarg)
        if criterion_improved(epoch_meanloss):
            torch.save({'encodermodel': encodermodel,
                        'decodermodel': decodermodel, 'scaler': scaler,
                        'config': config_parameters},
                       trainedmodelpath)
        else:
            dump = torch.load(trainedmodelpath)
            encodermodel.load_state_dict(dump['encodermodel'].state_dict())
            decodermodel.load_state_dict(dump['decodermodel'].state_dict())
        if optimizer.param_groups[0]['lr'] < 1e-6:
            break
    logger.info(tp.bottom(8, style='grid'))
    # Sample results
    from sample import sample

    sample(
        data_path=features,
        encoder_path=trainedmodelpath,
        vocab_path=vocab_file,
        output=os.path.join(
            outputdir,
            'output_word.txt'))


if __name__ == '__main__':
    fire.Fire(main)
