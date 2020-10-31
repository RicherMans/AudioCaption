# coding=utf-8
#!/usr/bin/env python3
import os
import sys
import datetime
import random
import uuid

import fire
import numpy as np
import pandas as pd
import sklearn.preprocessing as pre
import torch
from ignite.engine.engine import Engine, Events
from ignite.metrics import Accuracy, Loss, RunningAverage, Average
from ignite.handlers import EarlyStopping, ModelCheckpoint
from ignite.contrib.handlers import ProgressBar
from ignite.utils import convert_tensor

sys.path.append(os.getcwd())
import models
import utils.train_util as train_util
from utils.build_vocab import Vocabulary
from runners.base_runner import BaseRunner
from datasets.SJTUDataSet import SJTUDataset, SJTUDatasetEval, collate_fn

class Runner(BaseRunner):

    @staticmethod
    def _get_model(config, vocab_size):
        embed_size = config["model_args"]["embed_size"]
        if config["encodermodel"] == "E2EASREncoder":
            encodermodel = models.encoder.load_espnet_encoder(config["pretrained_encoder"])
        else:
            encodermodel = getattr(
                models.encoder, config["encodermodel"])(
                inputdim=config["inputdim"],
                embed_size=embed_size,
                **config["encodermodel_args"])
            if "pretrained_encoder" in config:
                encoder_state_dict = torch.load(
                    config["pretrained_encoder"],
                    map_location="cpu")
                encodermodel.load_state_dict(encoder_state_dict, strict=False)

        decodermodel = getattr(
            models.decoder, config["decodermodel"])(
            vocab_size=vocab_size,
            input_size=embed_size,
            **config["decodermodel_args"])
        model = getattr(
            models.WordModel, config["model"])(encodermodel, decodermodel, **config["model_args"])
        return model

    def _forward(self, model, batch, mode, **kwargs):
        assert mode in ("train", "validation", "eval")
        kwargs["mode"] = mode

        if mode == "eval":
            feats = batch[1]
            feat_lens = batch[-1]

            feats = convert_tensor(feats.float(),
                                   device=self.device,
                                   non_blocking=True)
            output = model(feats, feat_lens, **kwargs)
            return output

        # mode is "train"

        feats = batch[0]
        caps = batch[1]
        feat_lens = batch[-2]
        cap_lens = batch[-1]
        feats = convert_tensor(feats.float(),
                               device=self.device,
                               non_blocking=True)
        caps = convert_tensor(caps.long(),
                              device=self.device,
                              non_blocking=True)
        # pack labels to remove padding from caption labels
        targets = torch.nn.utils.rnn.pack_padded_sequence(
            caps, cap_lens, batch_first=True).data

        output = model(feats, feat_lens, caps, cap_lens, **kwargs)

        packed_logits = torch.nn.utils.rnn.pack_padded_sequence(
            output["logits"], cap_lens, batch_first=True).data
        packed_logits = convert_tensor(packed_logits, device=self.device, non_blocking=True)

        output["packed_logits"] = packed_logits
        output["targets"] = targets
        return output

    def train(self, config, **kwargs):
        """Trains a model on the given configurations.
        :param config: A training configuration. Note that all parameters in the config can also be manually adjusted with --ARG=VALUE
        :param **kwargs: parameters to overwrite yaml config
        """
        from pycocoevalcap.cider.cider import Cider

        config_parameters = train_util.parse_config_or_kwargs(config, **kwargs)
        config_parameters["seed"] = self.seed
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
            score_function=lambda engine: engine.state.metrics["score"],
            score_name="score")

        logger = train_util.genlogger(os.path.join(outputdir, "train.log"))
        # print passed config parameters
        logger.info("Storing files in: {}".format(outputdir))
        train_util.pprint_dict(config_parameters, logger.info)

        zh = config_parameters["zh"]
        vocabulary = torch.load(config_parameters["vocab_file"])
        train_loader, val_loader, info = self._get_dataloaders(config_parameters, vocabulary)
        config_parameters["inputdim"] = info["inputdim"]
        val_key2refs = info["val_key2refs"]
        logger.info("<== Estimating Scaler ({}) ==>".format(info["scaler"].__class__.__name__))
        logger.info(
            "Feature: {} Input dimension: {} Vocab Size: {}".format(
                config_parameters["feature_file"], info["inputdim"], len(vocabulary)))

        model = self._get_model(config_parameters, len(vocabulary))
        if "pretrained_word_embedding" in config_parameters:
            embeddings = np.load(config_parameters["pretrained_word_embedding"])
            model.load_word_embeddings(embeddings, tune=config_parameters["tune_word_embedding"], projection=True)
        model = model.to(self.device)
        train_util.pprint_dict(model, logger.info, formatter="pretty")
        optimizer = getattr(
            torch.optim, config_parameters["optimizer"]
        )(model.parameters(), **config_parameters["optimizer_args"])
        train_util.pprint_dict(optimizer, logger.info, formatter="pretty")


        criterion = torch.nn.CrossEntropyLoss().to(self.device)
        crtrn_imprvd = train_util.criterion_improver(config_parameters['improvecriterion'])

        def _train_batch(engine, batch):
            model.train()
            with torch.enable_grad():
                optimizer.zero_grad()
                output = self._forward(
                    model, batch, "train",
                    ss_ratio=config_parameters["scheduled_sampling_args"]["ss_ratio"]
                )
                loss = criterion(output["packed_logits"], output["targets"]).to(self.device)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                output["loss"] = loss.item()
                return output

        trainer = Engine(_train_batch)
        RunningAverage(output_transform=lambda x: x["loss"]).attach(trainer, "running_loss")
        pbar = ProgressBar(persist=False, ascii=True, ncols=100)
        pbar.attach(trainer, ["running_loss"])

        key2pred = {}

        def _inference(engine, batch):
            model.eval()
            keys = batch[2]
            with torch.no_grad():
                output = self._forward(model, batch, "validation") 
                seqs = output["seqs"].cpu().numpy()
                for (idx, seq) in enumerate(seqs):
                    if keys[idx] in key2pred:
                        continue
                    candidate = self._convert_idx2sentence(seq, vocabulary, zh)
                    key2pred[keys[idx]] = [candidate,]
                return output

        metrics = {
            "loss": Loss(criterion, output_transform=lambda x: (x["packed_logits"], x["targets"])),
            "accuracy": Accuracy(output_transform=lambda x: (x["packed_logits"], x["targets"])),
        }

        evaluator = Engine(_inference)

        def eval_cv(engine, key2pred, key2refs):
            scorer = Cider(zh=zh)
            score, scores = scorer.compute_score(key2refs, key2pred)
            engine.state.metrics["score"] = score
            key2pred.clear()

        evaluator.add_event_handler(
            Events.EPOCH_COMPLETED, eval_cv, key2pred, val_key2refs)

        for name, metric in metrics.items():
            metric.attach(trainer, name)
            metric.attach(evaluator, name)

        trainer.add_event_handler(
              Events.EPOCH_COMPLETED, train_util.log_results, evaluator, val_loader,
              logger.info, metrics.keys(), ["loss", "accuracy", "score"])

        if config_parameters["scheduled_sampling"]:
            trainer.add_event_handler(
                Events.GET_BATCH_COMPLETED, train_util.update_ss_ratio, config_parameters, len(train_loader))

        evaluator.add_event_handler(
            Events.EPOCH_COMPLETED, train_util.save_model_on_improved, crtrn_imprvd,
            "score", {
                "model": model.state_dict(),
                "config": config_parameters,
                "scaler": info["scaler"]
        }, os.path.join(outputdir, "saved.pth"))

        scheduler = getattr(torch.optim.lr_scheduler, config_parameters["scheduler"])(
            optimizer, **config_parameters["scheduler_args"])
        evaluator.add_event_handler(
            Events.EPOCH_COMPLETED, train_util.update_lr,
            scheduler, "score")

        evaluator.add_event_handler(
            Events.EPOCH_COMPLETED, checkpoint_handler, {
                "model": model,
            }
        )

        # early_stop_handler = EarlyStopping(
            # patience=config_parameters["early_stop"],
            # score_function=lambda engine: engine.state.metrics["score"],
            # trainer=trainer)
        # evaluator.add_event_handler(Events.COMPLETED, early_stop_handler)

        trainer.run(train_loader, max_epochs=config_parameters["epochs"])
        return outputdir


if __name__ == "__main__":
    fire.Fire(Runner)
