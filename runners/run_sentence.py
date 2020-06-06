# coding=utf-8
#!/usr/bin/env python3
import os
import sys
import datetime
import random
import uuid
import pickle
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
import utils.gpu_selection as gpu_selection
from utils.kaldi_util import parsecopyfeats
from utils.build_vocab import Vocabulary
from SJTUDataSet import SJTUDatasetSentence, collate_fn


deviceId, gpu_name, valid_gpus = gpu_selection.auto_select_gpu()
torch.cuda.set_device(deviceId)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Runner(object):
    """Main class to run experiments"""
    def __init__(self, seed=1):
        super().__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)

    @staticmethod
    def _get_dataloaders(config, vocabulary):
        scaler = getattr(
            pre, config["scaler"])(
            **config["scaler_args"])
        inputdim = -1
        # kaldi_stream = parsecopyfeats(config["features"], 
                                      # **config["feature_args"])
        caption_df = pd.read_json(config["caption_file"])

        with open(config["sentence_embedding"], "rb") as f:
            sent_embeds = pickle.load(f)

        for batch in tqdm(
            torch.utils.data.DataLoader(
                SJTUDatasetSentence(
                    kaldi_stream=config["feature_stream"],
                    caption_df=caption_df,
                    sent_embeds=sent_embeds,
                    vocabulary=vocabulary
                ),
                shuffle=False,
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

        train_df = caption_df.sample(frac=config["train_percent"] / 100., random_state=0)
        cv_df = caption_df[~caption_df.index.isin(train_df.index)]
        trainset = SJTUDatasetSentence(
            config["feature_stream"],
            train_df,
            sent_embeds,
            vocabulary,
            scaler.transform
        )
        trainloader = torch.utils.data.DataLoader(
            trainset,
            shuffle=True,
            collate_fn=collate_fn([0, 1]),
            **config["dataloader_args"]
        )
        cvset = SJTUDatasetSentence(
            config["feature_stream"],
            cv_df,
            sent_embeds,
            vocabulary,
            scaler.transform
        )
        cvloader = torch.utils.data.DataLoader(
            cvset,
            shuffle=False,
            collate_fn=collate_fn([0, 1]),
            **config["dataloader_args"]
        )
        return trainloader, cvloader, {"scaler": scaler, "inputdim": inputdim}

    @staticmethod
    def _get_model(config, vocab_size):
        embed_size = config["model_args"]["embed_size"]
        if "load_pretrained" in config and config["load_pretrained"]:
            encodermodeldump = torch.load(
                config["load_pretrained"],
                map_location=lambda storage, loc: storage)
            pretrainedmodel = encodermodeldump["encodermodel"]
            encodermodel = models.PreTrainedCNN(
                inputdim=config["inputdim"], 
                pretrained_model=pretrainedmodel,
                embed_size=embed_size,
                **config["encodermodel_args"])
        else:
            encodermodel = getattr(
                models.encoder, config["encodermodel"])(
                inputdim=config["inputdim"], 
                embed_size=embed_size,
                **config["encodermodel_args"])
        decodermodel = getattr(
            models.decoder, config["decodermodel"])(
            vocab_size=vocab_size,
            embed_size=embed_size,
            **config["decodermodel_args"])
        model = getattr(
            models.model, config["model"])(encodermodel, decodermodel, **config["model_args"])
        return model

    @staticmethod
    def _forward(model, batch, tf=None, mode="train", **kwargs):

        assert mode in ("train", "sample")

        if mode == "sample":
            max_length = kwargs["max_length"]
            method = kwargs["method"]
            feats = batch[1]
            feat_lens = batch[-1]

            feats = convert_tensor(feats.float(),
                                   device=device,
                                   non_blocking=True)
            sampled = model(feats, feat_lens, mode="sample", max_length=max_length, method=method)
            return sampled["seqs"]

        # mode is "train"
        assert tf is not None

        feats = batch[0]
        caps = batch[1]
        sent_embeds = batch[2]
        feat_lens = batch[-2]
        cap_lens = batch[-1]
        feats = convert_tensor(feats.float(),
                               device=device,
                               non_blocking=True)
        sent_embeds = convert_tensor(sent_embeds.float(),
                                     device=device,
                                     non_blocking=True)
        caps = convert_tensor(caps.long(),
                              device=device,
                              non_blocking=True)
        # pack labels to remove padding from caption labels
        targets = torch.nn.utils.rnn.pack_padded_sequence(
            caps, cap_lens, batch_first=True).data

        output = {}

        if tf:
            output = model(feats, feat_lens, caps, cap_lens, mode="forward")
        else:
            sampled = model(feats, feat_lens, mode="sample", max_length=max(cap_lens))
            probs = torch.nn.utils.rnn.pack_padded_sequence(
                sampled["probs"], cap_lens, batch_first=True).data
            probs = convert_tensor(probs, device=device, non_blocking=True)
            output["probs"] = probs
            output["seqs"] = sampled["seqs"]

        output["targets"] = targets
        output["sentence_targets"] = model.sent_projection(sent_embeds)

        return output

    def train(self, config, **kwargs):
        """Trains a model on the given configurations.
        :param config: A training configuration. Note that all parameters in the config can also be manually adjusted with --ARG=VALUE
        :param **kwargs: parameters to overwrite yaml config
        """

        config_parameters = train_util.parse_config_or_kwargs(config, **kwargs)
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

        model = self._get_model(config_parameters, len(vocabulary))
        if "pretrained_word_embedding" in config_parameters:
            embeddings = np.load(config_parameters["pretrained_word_embedding"])
            model.load_word_embeddings(embeddings, tune=False, projection=True)
        model = model.to(device)
        train_util.pprint_dict(model, logger.info, formatter="pretty")
        optimizer = getattr(
            torch.optim, config_parameters["optimizer"]
        )(model.parameters(), **config_parameters["optimizer_args"])
        train_util.pprint_dict(optimizer, logger.info, formatter="pretty")

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **config_parameters['scheduler_args'])

        criterion = torch.nn.CrossEntropyLoss().to(device)
        crtrn_imprvd = train_util.criterion_improver(config_parameters['improvecriterion'])
        tf_ratio = config_parameters["teacher_forcing_ratio"]

        def _train_batch(engine, batch):
            model.train()
            tf = True if random.random() < tf_ratio else False
            with torch.enable_grad():
                optimizer.zero_grad()
                output = self._forward(model, batch, tf=tf)
                loss = criterion(output["probs"], output["targets"])
                loss.backward()
                optimizer.step()
                output["loss"] = loss.item()
                return output

        trainer = Engine(_train_batch)
        RunningAverage(output_transform=lambda x: x["loss"]).attach(trainer, "running_loss")
        pbar = ProgressBar(persist=False, ascii=True)
        pbar.attach(trainer, ["running_loss"])

        def _inference(engine, batch):
            model.eval()
            with torch.no_grad():
                output = self._forward(model, batch, tf=config_parameters["teacher_forcing_on_validation"])
                return output

        metrics = {
            "loss": Loss(criterion, output_transform=lambda x: (x["probs"], x["targets"])),
            "accuracy": Accuracy(output_transform=lambda x: (x["probs"], x["targets"]))
        }

        evaluator = Engine(_inference)

        for name, metric in metrics.items():
            metric.attach(trainer, name)
            metric.attach(evaluator, name)
            
        trainer.add_event_handler(
              Events.EPOCH_COMPLETED, train_util.log_results, evaluator, cvloader,
              logger.info, metrics.keys())

        # evaluator.add_event_handler(
            # Events.EPOCH_COMPLETED, train_util.save_model_on_improved, crtrn_imprvd,
            # "loss", {
                # "model": model,
                # "config": config_parameters,
                # "scaler": info["scaler"]
        # }, os.path.join(outputdir, "saved.pth"))
        trainer.add_event_handler(
            Events.EPOCH_COMPLETED, train_util.save_model_on_improved, crtrn_imprvd,
            "loss", {
                "model": model,
                "config": config_parameters,
                "scaler": info["scaler"]
        }, os.path.join(outputdir, "saved.pth"))

        # evaluator.add_event_handler(
            # Events.EPOCH_COMPLETED, train_util.update_reduce_on_plateau,
            # scheduler, "loss")

        evaluator.add_event_handler(
            Events.EPOCH_COMPLETED, checkpoint_handler, {
                "model": model,
            }
        )

        # early_stop_handler = EarlyStopping(
            # patience=config_parameters["early_stop"],
            # score_function=lambda engine: -engine.state.metrics["loss"],
            # trainer=trainer)
        # evaluator.add_event_handler(Events.COMPLETED, early_stop_handler)

        trainer.run(trainloader, max_epochs=config_parameters["epochs"])
        return outputdir

    def sample(self,
               experiment_path: str,
               kaldi_stream,
               kaldi_scp,
               max_length=None,
               output: str="output_word.txt"):
        """Generate captions given experiment model"""
        import tableprint as tp
        from SJTUDataSet import SJTUDatasetEval, collate_fn

        dump = torch.load(os.path.join(experiment_path, "saved.pth"),
                          map_location=lambda storage, loc: storage)
        model = dump["model"]
        # Some scaler (sklearn standardscaler)
        scaler = dump["scaler"]
        # Also load previous training config
        config = dump["config"]
        vocabulary = torch.load(config["vocab_file"])
        model = model.to(device)
        dataset = SJTUDatasetEval(
            kaldi_stream=kaldi_stream,
            kaldi_scp=kaldi_scp,
            transform=scaler.transform)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=False,
            collate_fn=collate_fn((1,)),
            batch_size=16,
            num_workers=0)
        
        if max_length is None:
            max_length = model.max_length
        width_length = max_length * 4
        pbar = ProgressBar(persist=False, ascii=True)
        writer = open(os.path.join(experiment_path, output), "w")
        writer.write(
            tp.header(
                ["InputUtterance", "Output Sentence"], width=[len("InputUtterance"), width_length]))
        writer.write('\n')

        sentences = []
        def _sample(engine, batch):
            # batch: [ids, feats, feat_lens]
            with torch.no_grad():
                model.eval()
                ids = batch[0]
                # seqs = self._forward(model, batch, mode="sample", method="beam", beam_size=3, max_length=max_length)
                seqs = self._forward(model, batch, mode="sample", method="greedy", max_length=max_length)
                seqs = seqs.cpu().numpy()
                for idx, seq in enumerate(seqs):
                    caption = []
                    for word_id in seq:
                        word = vocabulary.idx2word[word_id]
                        caption.append(word)
                        if word == "<end>":
                            break
                    sentence = "".join(caption)
                    writer.write(tp.row([ids[idx], sentence], width=[len("InputUtterance"), width_length]) + "\n")
                    sentences.append(sentence)

        sample_engine = Engine(_sample)
        pbar.attach(sample_engine)
        sample_engine.run(dataloader)
        writer.write(tp.bottom(2, width=[len("InputUtterance"), width_length]) + "\n")
        writer.write("Unique sentence number: {}\n".format(len(set(sentences))))
        writer.close()

    def bleu_evaluate(self, 
                      experiment_path: str, 
                      caption_file: str, 
                      max_length=None,
                      smoothing="method1",
                      N=4,
                      output: str="bleu_score.txt"):
        """Calculating BLEU score of the given model outputs"""
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        import utils.kaldi_io as kaldi_io

        dump = torch.load(os.path.join(experiment_path, "saved.pth"),
                          map_location=lambda storage, loc: storage)
        model = dump["model"]
        # Some scaler (sklearn standardscaler)
        scaler = dump["scaler"]
        # Also load previous training config
        config = dump["config"]
        vocabulary = torch.load(config["vocab_file"])
        model = model.to(device)
        caption_df = pd.read_json(caption_file)
        # caption_df["filename"] = caption_df["filename"].apply(
            # lambda x: os.path.splitext(os.path.basename(x))[0])
        id2refs = caption_df.groupby(["key"])["tokens"].apply(list).to_dict()

        if max_length is None:
            max_length = model.max_length

        smoothing_function = getattr(SmoothingFunction(), smoothing)
        bleu_weights = [1./N]*N

        model.eval()

        human_scores = []
        model_scores = []

        f = open(os.path.join(experiment_path, output), "w")

        with torch.no_grad(), tqdm(total=len(caption_df["filename"].unique()), ascii=True) as pbar:
            for data_id, feat in kaldi_io.read_mat_ark(config["feature_stream"]):
                if data_id not in id2refs:
                    continue
                feat = scaler.transform(feat)
                feat = torch.as_tensor(feat).to(device).unsqueeze(0)
                sampled = model(feat, (feat.size(1),), mode="sample", max_length=max_length)
                seq = sampled["seqs"].squeeze(0).cpu().numpy()

                candidate = []
                for word_id in seq:
                    word = vocabulary.idx2word[word_id]
                    if word == "<end>":
                        break
                    elif word == "<start>":
                        continue
                    candidate.append(word)
                refs = id2refs[data_id]


                human_score = []
                model_score = []

                for turn, ref in enumerate(refs):
                    ref_without_turn = [x for i, x in enumerate(refs) if i != turn]
                    human_score.append(
                        sentence_bleu(
                            ref_without_turn, 
                            ref,
                            smoothing_function=smoothing_function,
                            weights=bleu_weights))
                    model_score.append(
                        sentence_bleu(
                            ref_without_turn,
                            candidate,
                            smoothing_function=smoothing_function,
                            weights=bleu_weights))

                human_scores.append(max(human_score))
                model_scores.append(max(model_score))
                pbar.update()

        f.write("Human BLEU_{}: {:10.3f}\n".format(N, np.mean(human_scores)))
        f.write("Model BLEU_{}: {:10.3f}\n".format(N, np.mean(model_scores)))
        f.close()

    def bert_evaluate(self, 
                      experiment_path: str, 
                      caption_file: str, 
                      reference_embeddings,
                      max_length=None,
                      output: str="bert_score.txt"):
        """Calculating BERT score of the given model outputs"""
        import utils.kaldi_io as kaldi_io
        from bert_serving.client import BertClient
        import tableprint as tp


        def cosine_similarity(vec1, vec2):
            s = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            return s

        dump = torch.load(os.path.join(experiment_path, "saved.pth"),
                          map_location=lambda storage, loc: storage)
        model = dump["model"]
        # Some scaler (sklearn standardscaler)
        scaler = dump["scaler"]
        # Also load previous training config
        config = dump["config"]
        vocabulary = torch.load(config["vocab_file"])
        model = model.to(device)
        caption_df = pd.read_json(caption_file)
        # caption_df["filename"] = caption_df["filename"].apply(
            # lambda x: os.path.splitext(os.path.basename(x))[0])
        # id2refs = caption_df.groupby(
            # ["filename"])["tokens"].apply(list).to_dict()

        if max_length is None:
            max_length = model.max_length

        model.eval()

        human_scores = []
        model_scores = []

        bert_client = BertClient()

        if isinstance(reference_embeddings, str):
            ref_embeddings = np.load(reference_embeddings, allow_pickle=True)
        else:
            ref_embeddings = {}
            for reference_embedding in reference_embeddings:
                ref_embeddings.update(np.load(reference_embedding, allow_pickle=True))

        writer = open(os.path.join(experiment_path, output), "w")
        width_length = max_length * 4
        writer.write(
            tp.header(
                ["InputUtterance", "Output Sentence"], width=[len("InputUtterance"), width_length]))
        writer.write('\n')

        with torch.no_grad(), tqdm(total=len(caption_df["key"].unique()), ascii=True) as pbar:
            for data_id, feat in kaldi_io.read_mat_ark(config["feature_stream"]):
                if data_id not in caption_df["key"].values:
                    continue
                feat = scaler.transform(feat)
                feat = torch.as_tensor(feat).to(device).unsqueeze(0)
                sampled = model(feat, (feat.size(1),), mode="sample", max_length=max_length)
                seq = sampled["seqs"].squeeze(0).cpu().numpy()

                candidate = []
                for word_id in seq:
                    word = vocabulary.idx2word[word_id]
                    if word == "<end>":
                        break
                    elif word == "<start>":
                        continue
                    candidate.append(word)
                candidate = "".join(candidate)
                candidate_embed = bert_client.encode([candidate])

                # tokened_refs = id2refs[data_id]
                # refs = []

                # for tokened_ref in tokened_refs:
                    # refs.append("".join(tokened_ref))

                # ref_embeddings = bert_client.encode(refs)
                ref_embedding = ref_embeddings[data_id]
                
                human_score = []
                model_score = []

                for turn in range(len(ref_embedding)):
                    ref_embed = ref_embedding[turn]
                    for i in range(len(ref_embedding)):
                        if i != turn:
                            human_score.append(cosine_similarity(
                                ref_embed.reshape(-1), ref_embedding[i].reshape(-1)))

                    model_score.append(cosine_similarity(
                        ref_embed.reshape(-1), candidate_embed.reshape(-1)))

                human_scores.append(max(human_score))
                model_scores.append(max(model_score))
                writer.write(tp.row([data_id, "{} ({:.3f})".format(candidate, max(model_score))], width=[len("InputUtterance"), width_length]) + "\n")
                pbar.update()

        writer.write(tp.bottom(2, width=[len("InputUtterance"), width_length]) + "\n")
        writer.write("Human BERT score: {:10.3f}\n".format(np.mean(human_scores)))
        writer.write("Model BERT score: {:10.3f}\n".format(np.mean(model_scores)))
        writer.close()

if __name__ == "__main__":
    fire.Fire(Runner)
