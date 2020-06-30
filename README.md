# AudioCaption : Listen and Tell

This repository provides source code for several models on audio captioning as well as labels of several datasets.

Firstly please checkout this repository.

```bash
git clone https://www.github.com/Richermans/AudioCaption
```

# Dataset

For all datasets, labels are provided in the directory `data/*/labels`.

## AudioCaption

### hospital

The full AudioCaption hospital dataset (3710 video clips) can be downloaded via [google drive](https://drive.google.com/open?id=1_osRNYzRQf4siCHHKwudZQc6x0XPSAb9) .
There is also a Zenodo link: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3715277.svg)](https://doi.org/10.5281/zenodo.3715277)
The audio-only part of the dataset can be downloaded via [google drive](https://drive.google.com/file/d/1tixUQAuGobL-O94D0Gwmxs94jeyMmPlC/view?usp=sharing).

An easy way to download the dataset is by using the pip script `gdown`. `pip install gdown` will install that script. Then:

```
cd data
gdown https://drive.google.com/uc?id=1tixUQAuGobL-O94D0Gwmxs94jeyMmPlC
unzip hospital_audio.zip
```

If you need a proxy to download the dataset, we recommend using [Proxychains](https://github.com/rofl0r/proxychains-ng).

### car

The dataset on car scene can be downloaded via [google drive](https://drive.google.com/file/d/1D1h4_orPBVOlLX9rrnxYBtObD3tpp43B/view?usp=sharing).

The source code and dataset in the paper [What does a car-sette tape tell?](http://arxiv.org/abs/1905.13448) is also provided here.

# Related Papers
Here are papers related to this repository:
* [Audio Caption: Listen And Tell](https://arxiv.org/abs/1902.09254)
* [What Does A Car-sette Tape Tell?](http://arxiv.org/abs/1905.13448)

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

You can load pretrained word embeddings in Google [BERT](https://github.com/google-research/bert#pre-trained-models) instead of training word embeddings from scratch. The scripts in `utils/bert` need a BERT server in the background. We use BERT server from [bert-as-service](https://github.com/hanxiao/bert-as-service).

To use bert-as-service, you need to first install the repository. It is recommended that you create a new environment with Tensorflow 1.3 to run BERT server since there is something wrong with this repository when using Tensorflow 2.x.

After successful installation of [bert-as-service](https://github.com/hanxiao/bert-as-service), downloading and running the BERT server needs to execute:

```bash
bash scripts/prepare_bert_server.sh <path-to-server> <num-workers>
```

By default, server based on BERT base Chinese model is running in the background. You can change to other models by changing corresponding model name and path in `srcipts/prepare_bert_server.sh`.


## Extract Features


The kaldi scp format requires a tab or space separated line with the information: `FEATURENAME WAVEPATH`

For example, to extract feature from hospital data:

* Filterbank:

```bash
find `pwd`/data/hospital/hospital_3707/ -type f | awk -F[./] '{print $(NF-1),$0}' > data/hospital/wav.scp
compute-fbank-feats --config=fbank_config/fbank.conf scp,p:`pwd`/hospital/wav.scp ark:- | copy-feats ark:- ark,scp:`pwd`/hospital/fbank.ark,`pwd`/hospital/fbank.scp
```

* Logmelspectrogram:

```bash
python utils/featextract.py `find data/hospital/hospital_3707/ -type f` data/hospital/logmel.ark mfcc -win_length 1764 -hop_length 882
```

## Training Configurator

Training configuration is done in `config/*.yaml`. Here one can adjust some hyperparameters e.g., number of hidden layers or embedding size. You can also write your own models in `models/*.py` and adjust the config to use that model (e.g. `encoder: MYMODEL`). 

Note: All parameters within the `runners/*.py` script use exclusively parameters with the same name as their `.yaml` file counterpart. They can all be switched and changed on the fly by passing `--ARG VALUE`, e.g., if one wishes to switch the captions file to use english captions, pass `--caption_file data/hospital/en_dev.json`.


## Training models

In order to train a model (for example using standard cross entropy loss), simply run:

```bash
python runners/run.py train config/xe.yaml
```

This will store the training logs and model checkpoints in `OUTPUTPATH/MODEL/TIMESTAMP`.

## Predicting and Evaluating

Predicting and evaluating is done by running `coco_evaluate` function in `runners/*.py`:

```bash
export kaldi_stream="copy-feats scp:data/hospital/fbank_eval.scp ark:- |"
export experiment_path=experiments/***
python runners/run.py coco_evaluate $experiment_path "$kaldi_stream" data/hospital/zh_eval.json
```

Standard machine translation metrics (BLEU@1-4, ROUGE-L, CIDEr, METEOR and SPICE) are included, where METEOR and SPICE can only be used on English datasets.

BERT similarity score (proposed in [What does a car-sette tape tell?](http://arxiv.org/abs/1905.13448)) can also be calculated by `bert_evaluate` function in `runners/*.py`, with BERT server running in the background.



