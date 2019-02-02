# AudioCaption : Listen and Tell

This repository provides the baseline model source code as well as the labels to the ICASSP2019 Paper *Audiocaption: Listen and Tell*

Firstly please checkout this repository.

```bash
git clone https://www.github.com/Richermans/AudioCaption
```

# Dataset
The full AudioCaption dataset (3710 video clips) can be downloaded via [google drive](https://drive.google.com/open?id=1_osRNYzRQf4siCHHKwudZQc6x0XPSAb9) .
The audio of the dataset can be downloaded via [google drive](https://drive.google.com/file/d/1tixUQAuGobL-O94D0Gwmxs94jeyMmPlC/view?usp=sharing).

A easy way to download the dataset is by using the pip script `gdown`. `pip install gdown` will install that script. Then:

```
cd data
gdown https://drive.google.com/uc?id=1tixUQAuGobL-O94D0Gwmxs94jeyMmPlC
unzip hospital_audio.zip
```

If you need a proxy to download the dataset, we recommend using [Proxychains](https://github.com/rofl0r/proxychains-ng).

The labels can be found in this repository in the directory `data/labels`. The experiments can be run with the Chinese labels `hospital_cn.csv` or the English ones `hospital_en.csv`.

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

However, the [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/) software provides support for tokenization of Chinese. The script `prepare_dataserver.sh` downloads all the necessary plugins for the CoreNLP tool in order to enable tokenization. The scripts `build_vocab.py` and `create_labels.py` do need a running server in the background in order to work.

Downloading and running the CoreNLP tokenization server only needs to execute:

```bash
./prepare_dataserver.sh
```

It requires at least `java` being installed on your machine. It is recommended to run this script in the background.


## Extract Features

* Filterbank (extracted with 25ms windows and 10ms shift):

The kaldi scp format requires a tab or space separated line with the information: `FEATURENAME WAVEPATH`

```bash
cd data
find `pwd`/hospital_3707/ -type f | awk -F[./] '{print $(NF-1),$0}' > hospital.scp
compute-fbank-feats --config=fbank_config/fbank.conf scp:hospital.scp ark:hospital_fbank.ark
```

* Logmelspectrogram

```bash
cd data
python3 ../feature/featextract.py `find hospital_* -type f` hospital_logmel.ark mfcc -win_length 1764 -hop_length 882
```

## Training Configurator

Training configuration is done in `config/*.yaml`. Here one can adjust some hyperparameters e.g., number of hidden layers or embedding size. You can also write your own Models in `models.py` and Adjust the config to use that model (e.g. `encoder:MYMODEL`). 
Note: All parameters within the `train.py` script use exclusively parameters with the same name as their `.yaml` file counterpart. They can all be switched and changed on the fly by passing `--ARG VALUE`, e.g., if one wishes to switch the captions file to use english captions, pass `--captions_file data/labels/hospital_en.json`.


## Training models

In order to train a model (Chinese by default, with FBank features), simply run:

```bash
python train.py data/hospital_fbank.ark data/vocab.cn
```

This script creates a new directory `ENCODERMODEL_DECODERMODEL/TIMESTAMP`. On a CPU the training might take about half an hour for one epoch ( we run by default for 20 ). On a GPU the training is much faster, approximately couple of minutes on a GTX1080.

The training creates for each epoch one `model_EPOCH.th` file in the output directory. Finally after training finished, the script will use the best model (according to PPL) in order to generate sentences given some input features (by default the same that were used for training)

## Evaluating

Run the `score.py` script with the following parameters:

```bash
python score.py DATA_DIR MODEL_PATH VOCAB_PATH
```
e.g.,
```bash
python score.py data/hospital_fbank.ark GRUEncoder_GRUDecoder/SOMEPATH/model.th data/vocab_cn.th
```

## Sampling

Sampling is done by the `sample.py` script. It is ran by default after training finishes. 

