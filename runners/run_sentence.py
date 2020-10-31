# coding=utf-8
#!/usr/bin/env python3
import os
import sys
import datetime
import random
import uuid
import re
from pprint import pformat
from contextlib import contextmanager

from tqdm import tqdm
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
from datasets.SJTUDataSet import SJTUSentenceDataset, collate_fn
import utils.score_util as score_util
from runners.base_runner import BaseRunner

class Runner(BaseRunner):

    @staticmethod
    def _get_dataloaders(config, vocabulary):
        scaler = getattr(
            pre, config["scaler"])(
            **config["scaler_args"])
        inputdim = -1
        caption_df = pd.read_json(config["caption_file"], dtype={"key": str})

        sentence_embedding = np.load(config["sentence_embedding"], allow_pickle=True)

        for batch in tqdm(
            torch.utils.data.DataLoader(
                SJTUSentenceDataset(
                    kaldi_stream=config["feature_stream"],
                    caption_df=caption_df,
                    vocabulary=vocabulary,
                    sentence_embedding=sentence_embedding,
                ),
                collate_fn=collate_fn([0, 1]),
                **config["dataloader_args"]
            ), 
            ascii=True
        ):
            feat = batch[0]
            feat = feat.reshape(-1, feat.shape[-1])
            scaler.partial_fit(feat)
            inputdim = feat.shape[-1]
        assert inputdim > 0, "Reading inputstream failed"

        augments = train_util.parse_augments(config["augments"])
        cv_keys = np.random.choice(
            caption_df["key"].unique(), 
            int(len(caption_df["key"].unique()) * (1 - config["train_percent"] / 100.)), 
            replace=False
        )
        cv_df = caption_df[caption_df["key"].apply(lambda x: x in cv_keys)]
        train_df = caption_df[~caption_df.index.isin(cv_df.index)]
        # train_df = caption_df.sample(frac=config["train_percent"] / 100., 
                                     # random_state=0)
        trainloader = torch.utils.data.DataLoader(
            SJTUSentenceDataset(
                kaldi_stream=config["feature_stream"],
                caption_df=train_df,
                vocabulary=vocabulary,
                sentence_embedding=sentence_embedding,
                transform=[scaler.transform, augments]
            ),
            shuffle=True,
            collate_fn=collate_fn([0, 1]),
            **config["dataloader_args"]
        )

        # cv_df = caption_df[~caption_df.index.isin(train_df.index)]
        if config["zh"]:
            cv_key2refs = cv_df.groupby("key")["tokens"].apply(list).to_dict()
        else:
            cv_key2refs = cv_df.groupby("key")["caption"].apply(list).to_dict()
        cvloader = torch.utils.data.DataLoader(
            SJTUSentenceDataset(
                kaldi_stream=config["feature_stream"],
                caption_df=cv_df,
                vocabulary=vocabulary,
                sentence_embedding=sentence_embedding,
                transform=scaler.transform,
            ),
            shuffle=False,
            collate_fn=collate_fn([0, 1]),
            **config["dataloader_args"]
        )
        return trainloader, cvloader, {"scaler": scaler, "inputdim": inputdim, "cv_key2refs": cv_key2refs}

    @staticmethod
    def _get_model(config, vocab_size):
        embed_size = config["model_args"]["embed_size"]
        encodermodel = getattr(
            models.encoder, config["encodermodel"])(
            inputdim=config["inputdim"], 
            embed_size=embed_size,
            **config["encodermodel_args"])
        if "pretrained_encoder" in config:
            encoder_state_dict = torch.load(
                config["pretrained_encoder"],
                map_location="cpu")
            encodermodel.load_state_dict(encoder_state_dict)

        decodermodel = getattr(
            models.decoder, config["decodermodel"])(
            vocab_size=vocab_size,
            embed_size=embed_size,
            **config["decodermodel_args"])
        model = getattr(
            models.WordModel, config["model"])(encodermodel, decodermodel, **config["model_args"])
        return model

    def _forward(self, model, batch, mode="train", **kwargs):
        assert mode in ("train", "sample")

        if mode == "sample":
            feats = batch[1]
            feat_lens = batch[-1]

            feats = convert_tensor(feats.float(),
                                   device=self.device,
                                   non_blocking=True)
            sampled = model(feats, feat_lens, mode="sample", **kwargs)
            return sampled

        # mode is "train"
        assert "tf" in kwargs, "need to know whether to use teacher forcing"

        feats = batch[0]
        caps = batch[1]
        sent_embeds = batch[2]
        feat_lens = batch[-2]
        cap_lens = batch[-1]
        feats = convert_tensor(feats.float(),
                               device=self.device,
                               non_blocking=True)
        caps = convert_tensor(caps.long(),
                              device=self.device,
                              non_blocking=True)
        sent_embeds = convert_tensor(sent_embeds.float(),
                                     device=self.device,
                                     non_blocking=True)
        # pack labels to remove padding from caption labels
        targets = torch.nn.utils.rnn.pack_padded_sequence(
            caps, cap_lens, batch_first=True).data

        if kwargs["tf"]:
            output = model(feats, feat_lens, caps, cap_lens, mode="forward")
            # keys include: ["probs", "seq_outputs"]
        else:
            output = model(feats, feat_lens, mode="sample", max_length=max(cap_lens))
            probs = torch.nn.utils.rnn.pack_padded_sequence(
                output["probs"], cap_lens, batch_first=True).data
            probs = convert_tensor(probs, device=self.device, non_blocking=True)
            output["probs"] = probs

        output["sentence_targets"] = sent_embeds
        output["word_targets"] = targets

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
            score_function=lambda engine: -engine.state.metrics["loss"],
            score_name="loss")

        logger = train_util.genlogger(os.path.join(outputdir, "train.log"))
        # print passed config parameters
        logger.info("Storing files in: {}".format(outputdir))
        train_util.pprint_dict(config_parameters, logger.info)

        vocabulary = torch.load(config_parameters["vocab_file"])
        zh = config_parameters["zh"]
        trainloader, cvloader, info = self._get_dataloaders(config_parameters, vocabulary)
        cv_key2refs = info["cv_key2refs"]
        config_parameters["inputdim"] = info["inputdim"]
        logger.info("<== Estimating Scaler ({}) ==>".format(info["scaler"].__class__.__name__))
        logger.info(
            "Stream: {} Input dimension: {} Vocab Size: {}".format(
                config_parameters["feature_stream"], info["inputdim"], len(vocabulary)))

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


        XE_criterion = torch.nn.CrossEntropyLoss().to(self.device)
        seq_criterion = torch.nn.CosineEmbeddingLoss().to(self.device)
        crtrn_imprvd = train_util.criterion_improver(config_parameters['improvecriterion'])
        tf_ratio = config_parameters["teacher_forcing_ratio"]

        def _train_batch(engine, batch):
            model.train()
            tf = True if random.random() < tf_ratio else False
            with torch.enable_grad():
                optimizer.zero_grad()
                output = self._forward(model, batch, tf=tf)
                XE_loss = XE_criterion(output["probs"], output["word_targets"])
                seq_loss = seq_criterion(output["seq_outputs"], output["sentence_targets"], torch.ones(batch[0].shape[0]).to(self.device))
                loss = XE_loss + seq_loss * config_parameters["seq_loss_ratio"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                output["XE_loss"] = XE_loss.item()
                output["seq_loss"] = seq_loss.item()
                output["loss"] = loss.item()
                return output

        trainer = Engine(_train_batch)
        RunningAverage(output_transform=lambda x: x["loss"]).attach(trainer, "running_loss")
        pbar = ProgressBar(persist=False, ascii=True)
        pbar.attach(trainer, ["running_loss"])

        key2pred = {}

        def _inference(engine, batch):
            model.eval()
            keys = batch[3]
            with torch.no_grad():
                output = self._forward(model, batch, tf=config_parameters["teacher_forcing_on_validation"])
                output["XE_loss"] = XE_criterion(output["probs"], output["word_targets"])
                output["seq_loss"] = seq_criterion(output["seq_outputs"], output["sentence_targets"], torch.ones(len(keys)).to(self.device))
                output["loss"] = output["XE_loss"] + output["seq_loss"] * config_parameters["seq_loss_ratio"]
                seqs = output["seqs"].cpu().numpy()
                for (idx, seq) in enumerate(seqs):
                    if keys[idx] in key2pred:
                        continue
                    candidate = self._convert_idx2sentence(seq, vocabulary, zh)
                    key2pred[keys[idx]] = [candidate,]
                return output

        metrics = {
            "loss": Average(output_transform=lambda x: x["loss"]),
            "XE_loss": Average(output_transform=lambda x: x["XE_loss"]),
            "seq_loss": Average(output_transform=lambda x: x["seq_loss"]),
        }

        evaluator = Engine(_inference)

        for name, metric in metrics.items():
            metric.attach(trainer, name)
            metric.attach(evaluator, name)

        def eval_cv(engine, key2pred, key2refs):
            scorer = Cider(zh=zh)
            score, scores = scorer.compute_score(key2refs, key2pred)
            engine.state.metrics["score"] = score
            key2pred.clear()

        evaluator.add_event_handler(
            Events.EPOCH_COMPLETED, eval_cv, key2pred, cv_key2refs)
            
        trainer.add_event_handler(
              Events.EPOCH_COMPLETED, train_util.log_results, evaluator, cvloader,
              logger.info, metrics.keys(), ["XE_loss", "seq_loss", "score"])

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

        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            # optimizer, **config_parameters["scheduler_args"])

        # evaluator.add_event_handler(
            # Events.EPOCH_COMPLETED, train_util.update_reduce_on_plateau,
            # scheduler, "score")

        # early_stop_handler = EarlyStopping(
            # patience=config_parameters["early_stop"],
            # score_function=lambda engine: engine.state.metrics["score"],
            # trainer=trainer)
        # evaluator.add_event_handler(Events.COMPLETED, early_stop_handler)

        trainer.run(trainloader, max_epochs=config_parameters["epochs"])
        return outputdir


if __name__ == "__main__":
    fire.Fire(Runner)
