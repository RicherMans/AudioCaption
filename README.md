# AudioCaption : Listen and Tell

This repository provides source code for several models on audio captioning as well as several datasets.

Firstly please checkout this repository.

```bash
git clone --recurse-submodules https://www.github.com/Richermans/AudioCaption
```

# Dataset

The two datasets, hospital and car, can be downloaded via Zenodo: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4671263.svg)](https://doi.org/10.5281/zenodo.4671263). 

# Related Papers
Here are papers related to this repository:
* [Audio Caption: Listen And Tell](https://arxiv.org/abs/1902.09254)
* [Audio Caption in a Car Setting with a Sentence-Level Loss](http://arxiv.org/abs/1905.13448)

If you'd like to use the AudioCaption dataset, please cite:
```
@inproceedings{Wu2019,
  author    = {Mengyue Wu and
               Heinrich Dinkel and
               Kai Yu},
  title     = {Audio Caption: Listen and Tell},
  booktitle = {{IEEE} International Conference on Acoustics, Speech and Signal Processing,
               {ICASSP} 2019, Brighton, United Kingdom, May 12-17, 2019},
  pages     = {830--834},
  publisher = {{IEEE}},
  year      = {2019},
  url       = {https://doi.org/10.1109/ICASSP.2019.8682377},
  doi       = {10.1109/ICASSP.2019.8682377},
  timestamp = {Wed, 16 Oct 2019 14:14:52 +0200},
}
```

# Baseline

In order to sucessfully run the baseline, the following packages and frameworks are required:

1. Kaldi (mostly for data processing)
2. A bunch of Python3 packages ( most notably [torch](https://pytorch.org/), see `requirements.txt` )

## Prequisite Installation

The code is written exclusively in Python3. In order to install all required packages use the included `requirements.txt`. `pip install -r requirements.txt` does the job.

### Kaldi

For this code, only the feature pipeline of kaldi is utlilized, thus only the feature packages need to be installed in order to function

```bash
git clone https://github.com/kaldi-asr/kaldi.git kaldi --origin upstream
cd kaldi && git pull
cd tools; make
cd ../src; make -j4 featbin
```

Lastly, create a new environment variable for the `kaldi_io.py` script to function properly. Either locally export in your current session the variable `KALDI_ROOT` or put it into `~/.bashrc` or `~/.profile`.

```bash
export KALDI_ROOT=/PATH/TO/YOUR/KALDI
```

### (Optional) NLP Tokenizer

This repository already provided the tokenized dataset in the json format. However, if one wishes to tokenize differently (e.g., tokenize by some custom NLP tokenizer), we also provide a simple script to install and run the Stanford NLP Tokenizer.

This dataset is labelled in Chinese. Chinese has some specific differences to most Indo-European languages, including its script. In particular, Chinese does not use an indicator for word separation, as English does with a blank space. Rather it depends on the reader to split a sentence into semantically sound tokens.

However, the [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/) software provides support for tokenization of Chinese. The script `prepare_dataserver.sh` downloads all the necessary plugins for the CoreNLP tool in order to enable tokenization. The script `utils/build_vocab.py` does need a running server in the background in order to work.

Downloading and running the CoreNLP tokenization server only needs to execute:

```bash
bash scripts/prepare_dataserver.sh
```

It requires at least `java` being installed on your machine. It is recommended to run this script in the background.


### (Optional) BERT Pretrained Embeddings

In [this paper](Audio Caption in a Car Setting with a Sentence-Level Loss), [BERT](https://github.com/google-research/bert#pre-trained-models) embeddings are used to provide sequence-level supervision. The scripts in `utils/bert` need [bert-as-service](https://github.com/hanxiao/bert-as-service) running in the background.

To use bert-as-service, you need to first install the repository. It is recommended that you create a new environment with Tensorflow 1.3 to run BERT server since it is incompatible with Tensorflow 2.x.

After successful installation of [bert-as-service](https://github.com/hanxiao/bert-as-service), downloading and running the BERT server: 

```bash
bash scripts/prepare_bert_server.sh <path-to-server> <num-workers> zh
```

By default, server based on BERT base Chinese model is running in the background. You can change to other models by changing corresponding model name and path in `scripts/prepare_bert_server.sh`.

To extract sentence embeddings, you need to execute `utils/bert/create_sent_embedding.py`, where the usage is shown.

## Extract Features

The kaldi scp format requires a tab or space separated line with the information: `FEATURENAME WAVEPATH`

For example, to extract feature from hospital data, assume the raw data is placed in `DATA_DIR` (`data/hospital/wav` here) and you will store features in `FEATURE_DIR` (`data/hospital` here):

```bash
DATA_DIR=`pwd`/data/hospital/wav
FEATURE_DIR=`pwd`/data/hospital
PREFIX=hospital
find $DATA_DIR -type f | awk -F[./] '{print "'$PREFIX'""_"$(NF-1),$0}' > $FEATURE_DIR/wav.scp
```

* Filterbank:

```bash
compute-fbank-feats --config=config/kaldi/fbank.conf scp:$FEATURE_DIR/wav.scp ark:$FEATURE_DIR/fbank.ark
python utils/copyark2hdf5.py $FEATURE_DIR/fbank.ark $FEATURE_DIR/fbank.hdf5
rm $FEATURE_DIR/fbank.ark
```

* Logmelspectrogram:

```bash
python utils/extract_feat.py $FEATURE_DIR/wav.scp $FEATURE_DIR/logmel.hdf5 $FEATURE_DIR/logmel.scp mfcc -win_length 1764 -hop_length 882
```

The kaldi scp file can be further split into a development scp and an evaluation scp:
```bash
python utils/split_scp.py $FEATURE_DIR/fbank.scp $FEATURE_DIR/zh_eval.json
python utils/split_scp.py $FEATURE_DIR/logmel.scp $FEATURE_DIR/zh_eval.json
```

## Dump vocabulary

Vocabulary should be prepared and dumped to a file for later use. To use the tokenized hospital dataset, run:
```bash
python utils/build_vocab.py "['data/hospital/zh_dev.json', 'data/hospital/zh_eval.json']" data/hospital/vocab_zh.pth --pretokenized True
```
A vocabulary file `data/hospital/vocab_zh.pth` will be generated.

## Training Configurator

Training configuration is done in `config/*.yaml`. Here one can adjust some hyperparameters e.g., number of hidden layers or embedding size. You can also write your own models in `models/*.py` and adjust the config to use that model (e.g. `encoder: MYMODEL`). 

Note: All parameters within the `runners/*.py` script use exclusively parameters with the same name as their `.yaml` file counterpart. They can all be switched and changed on the fly by passing `--ARG VALUE`, e.g., if one wishes to switch the captions file to use English captions, pass `--caption_file data/hospital/en_dev.json`.


## Training models

In order to train a model (for example using standard cross entropy loss), simply run:

```bash
python runners/run.py train config/xe.yaml
```

This will store the training logs and model checkpoints in `OUTPUTPATH/MODEL/TIMESTAMP`.

## Predicting and Evaluating

Predicting and evaluating is done by running `evaluate` (e.g. using Logmelspectrogram feature):

```bash
export experiment_path=experiments/***
python runners/run.py predict_evaluate $experiment_path $FEATURE_DIR/logmel.hdf5 $FEATURE_DIR/logmel_eval.scp $FEATURE_DIR/zh_eval.json
```

Standard machine translation metrics (BLEU@1-4, ROUGE-L, CIDEr, METEOR and SPICE) are included, where METEOR and SPICE can only be used on English datasets.

If you just want to do inference, do not provide caption reference file:
```bash
python runners/run.py predict_evaluate $experiment_path $FEATURE_DIR/logmel.hdf5 $FEATURE_DIR/logmel_eval.scp
```



