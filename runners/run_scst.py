# coding=utf-8
#!/usr/bin/env python3
import os
import re
import sys
import logging
import datetime
import random
import uuid
from pprint import pformat

from tqdm import tqdm
import fire
import numpy as np
import pandas as pd
import sklearn.preprocessing as pre
import torch
from ignite.engine.engine import Engine, Events
from ignite.utils import convert_tensor
from ignite.handlers import EarlyStopping, ModelCheckpoint
from ignite.contrib.handlers import ProgressBar
from ignite.metrics import Accuracy, Loss, RunningAverage, Average

sys.path.append(os.getcwd())
# sys.path.append("/mnt/lustre/sjtu/home/xnx98/utils")
import models
import utils.train_util as train_util
from utils.build_vocab import Vocabulary
from datasets.SJTUDataSet import SJTUDataset, collate_fn
from runners.base_runner import BaseRunner

device = "cpu"
if torch.cuda.is_available() and "SLURM_JOB_PARTITION" in os.environ and \
    "gpu" in os.environ["SLURM_JOB_PARTITION"]:
    device = "cuda"
    torch.backends.cudnn.deterministic = True
device = torch.device(device)


class ScstRunner(BaseRunner):
    """Main class to run experiments"""

    @staticmethod
    def _get_model(config, vocabulary):
        embed_size = config["model_args"]["embed_size"]
        encodermodel = getattr(
            models.encoder, config["encodermodel"])(
            inputdim=config["inputdim"], 
            embed_size=embed_size,
            **config["encodermodel_args"])
        decodermodel = getattr(
            models.decoder, config["decodermodel"])(
            vocab_size=len(vocabulary),
            embed_size=embed_size,
            **config["decodermodel_args"])
        model = getattr(models.SeqTrainModel, config["model"])(
            encodermodel, decodermodel, vocabulary, **config["model_args"])

        if config["load_pretrained"]:
            dump = torch.load(
                config["pretrained"],
                map_location=lambda storage, loc: storage)
            model.load_state_dict(dump["model"].state_dict(), strict=False)

        return model

    @staticmethod
    def _forward(model, batch, mode, **kwargs):
        assert mode in ("train", "sample")

        if mode == "sample":
            # SJTUDataSetEval
            feats = batch[1]
            feat_lens = batch[-1]

            feats = convert_tensor(feats.float(),
                                   device=device,
                                   non_blocking=True)
            sampled = model(feats, feat_lens, mode="sample", **kwargs)
            return sampled

        # mode is "train"
        assert "train_mode" in kwargs, "need to provide training mode (XE or scst)"
        assert kwargs["train_mode"] in ("XE", "scst"), "unknown training mode"

        feats = batch[0]
        caps = batch[1]
        keys = batch[2]
        feat_lens = batch[-2]
        cap_lens = batch[-1]
        feats = convert_tensor(feats.float(),
                               device=device,
                               non_blocking=True)
        caps = convert_tensor(caps.long(),
                              device=device,
                              non_blocking=True)

        
        if kwargs["train_mode"] == "XE":
            # trained by cross entropy loss
            assert "tf" in kwargs, "need to know whether to use teacher forcing"
            ce = torch.nn.CrossEntropyLoss()
            # pack labels to remove padding from caption labels
            targets = torch.nn.utils.rnn.pack_padded_sequence(
                caps, cap_lens, batch_first=True).data
            if kwargs["tf"]:
                probs = model(feats, feat_lens, caps, cap_lens, mode="forward")
            else:
                sampled = model(feats, feat_lens, mode="sample", max_length=max(cap_lens))
                probs = torch.nn.utils.rnn.pack_padded_sequence(
                    sampled["probs"], cap_lens, batch_first=True).data
                probs = convert_tensor(probs, device=device, non_blocking=True)
            loss = ce(probs, targets)
            output = {"loss": loss}
        else:
            # trained by self critical reward (reinforcement learning)
            assert "key2refs" in kwargs, "missing references"
            scorer = kwargs.get("scorer", None)
            output = model(feats, feat_lens, keys, kwargs["key2refs"], 
                           mode="scst", max_length=max(cap_lens), scorer=scorer)
        
        return output

    def train(self, config, **kwargs):
        """Trains a model on the given configurations.
        :param config:str: A training configuration. Note that all parameters in the config can also be manually adjusted with --ARG=VALUE
        :param **kwargs: parameters to overwrite yaml config
        """

        from pycocoevalcap.cider.cider import Cider

        config_parameters = train_util.parse_config_or_kwargs(config, **kwargs)
        config_parameters["seed"] = self.seed
        zh = config_parameters["zh"]
        outputdir = os.path.join(
            config_parameters["outputpath"], config_parameters["model"],
            "{}_{}".format(
                datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%m'),
                uuid.uuid1().hex))

        # Early init because of creating dir
        checkpoint_handler = ModelCheckpoint(
            outputdir,
            "run",
            n_saved=1,
            require_empty=False,
            create_dir=True,
            score_function=lambda engine: -engine.state.metrics["loss"],
            score_name="loss")

        logger = train_util.genlogger(os.path.join(outputdir, "train.log"))
        # print passed config parameters
        logger.info("Storing files in: {}".format(outputdir))
        train_util.pprint_dict(config_parameters, logger.info)

        vocabulary = torch.load(config_parameters["vocab_file"])
        trainloader, cvloader, info = self._get_dataloaders(config_parameters, vocabulary)
        config_parameters["inputdim"] = info["inputdim"]
        logger.info("<== Estimating Scaler ({}) ==>".format(info["scaler"].__class__.__name__))
        logger.info(
                "Stream: {} Input dimension: {} Vocab Size: {}".format(
                config_parameters["feature_stream"], info["inputdim"], len(vocabulary)))
        train_key2refs = info["train_key2refs"]
        # train_scorer = BatchCider(train_key2refs)
        cv_key2refs = info["cv_key2refs"]
        # cv_scorer = BatchCider(cv_key2refs)

        model = self._get_model(config_parameters, vocabulary)
        model = model.to(device)
        train_util.pprint_dict(model, logger.info, formatter="pretty")
        optimizer = getattr(
            torch.optim, config_parameters["optimizer"]
        )(model.parameters(), **config_parameters["optimizer_args"])
        train_util.pprint_dict(optimizer, logger.info, formatter="pretty")

        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            # optimizer, **config_parameters["scheduler_args"])
        crtrn_imprvd = train_util.criterion_improver(config_parameters["improvecriterion"])

        def _train_batch(engine, batch):
            model.train()
            with torch.enable_grad():
                optimizer.zero_grad()
                train_scorer = Cider(zh=zh)
                output = self._forward(model, batch, "train", train_mode="scst", 
                                       key2refs=train_key2refs, scorer=train_scorer)
                output["loss"].backward()
                optimizer.step()
                return output

        trainer = Engine(_train_batch)
        RunningAverage(output_transform=lambda x: x["loss"]).attach(trainer, "running_loss")
        pbar = ProgressBar(persist=False, ascii=True)
        pbar.attach(trainer, ["running_loss"])

        key2pred = {}

        def _inference(engine, batch):
            model.eval()
            keys = batch[2]
            with torch.no_grad():
                cv_scorer = Cider(zh=zh)
                output = self._forward(model, batch, "train", train_mode="scst",
                                       key2refs=cv_key2refs, scorer=cv_scorer)
                seqs = output["sampled_seqs"].cpu().numpy()
                for idx, seq in enumerate(seqs):
                    if keys[idx] in key2pred:
                        continue
                    candidate = self._convert_idx2sentence(seq, vocabulary, zh=zh)
                    key2pred[keys[idx]] = [candidate,]
                return output

        evaluator = Engine(_inference)
        RunningAverage(output_transform=lambda x: x["loss"]).attach(trainer, "running_loss")

        metrics = {
            "loss": Average(output_transform=lambda x: x["loss"]),
            "reward": Average(output_transform=lambda x: x["reward"].reshape(-1, 1)),
        }

        for name, metric in metrics.items():
            metric.attach(trainer, name)
            metric.attach(evaluator, name)

        RunningAverage(output_transform=lambda x: x["loss"]).attach(evaluator, "running_loss")
        pbar.attach(evaluator, ["running_loss"])

        # @trainer.on(Events.STARTED)
        # def log_initial_result(engine):
            # evaluator.run(cvloader, max_epochs=1)
            # logger.info("Initial Results - loss: {:<5.2f}\tscore: {:<5.2f}".format(evaluator.state.metrics["loss"], evaluator.state.metrics["score"].item()))


        trainer.add_event_handler(
              Events.EPOCH_COMPLETED, train_util.log_results, evaluator, cvloader,
              logger.info, metrics.keys(), ["loss", "reward", "score"])

        def eval_cv(engine, key2pred, key2refs, scorer):
            # if len(cv_key2refs) == 0:
                # for key, _ in key2pred.items():
                    # cv_key2refs[key] = key2refs[key]
            score, scores = scorer.compute_score(key2refs, key2pred)
            engine.state.metrics["score"] = score
            key2pred.clear()

        evaluator.add_event_handler(
            Events.EPOCH_COMPLETED, eval_cv, key2pred, cv_key2refs, Cider(zh=zh))

        evaluator.add_event_handler(
            Events.EPOCH_COMPLETED, train_util.save_model_on_improved, crtrn_imprvd,
            "score", {
                "model": model,
                "config": config_parameters,
                "scaler": info["scaler"]
            }, os.path.join(outputdir, "saved.pth"))

        evaluator.add_event_handler(
            Events.EPOCH_COMPLETED, checkpoint_handler, {
                "model": model,
            }
        )

        trainer.run(trainloader, max_epochs=config_parameters["epochs"])
        return outputdir


if __name__ == "__main__":
    fire.Fire(ScstRunner)
