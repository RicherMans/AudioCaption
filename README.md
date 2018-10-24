# AudioCaption : Show and Tell

This repository provides the baseline to the Paper [Audiocaption: Show and Tell]().


In order to sucessfully run the baseline, the following packages and frameworks are required:

1. Kaldi (mostly for data processing)
2. A bunch of Python packages ( most notably [torch]() ), Python3

## Prequisite Installation

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

This dataset is labelled in Chinese. Chinese has some specific differences to most Indo-European languages, including its script. In particular, Chinese does not use an indicator for word separation, as English does with a blank space. Rather it depends on the reader to split a sentence into semantically sound tokens.

However, the [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/) software provides support for tokenization of Chinese. The script `prepare_dataserver.sh` downloads all the necessary plugins for the CoreNLP tool in order to enable tokenization. The scripts `build_vocab.py` and `create_labels.py` do need a running server in the background in order to work.

Downloading and running the CoreNLP tokenization server only needs to execute:

```bash
./prepare_dataserver.sh
```

It requires at least `java` being installed on your machine. It is recommended to run this script in the background.

## Training models

In order to train a model, simply run:

## Download the dataset

The Audio dataset can be downloaded via [google drive](https://drive.google.com/file/d/1tixUQAuGobL-O94D0Gwmxs94jeyMmPlC/view?usp=sharing).

A easy way to download the dataset is by using the pip script `gdown`. `pip install gdown` will install that script. Then:
```
cd data
gdown https://drive.google.com/uc?id=1tixUQAuGobL-O94D0Gwmxs94jeyMmPlC
unzip hospital_audio.zip
```

## Extract Features

* Filterbank (extracted with 25ms windows and 10ms shift):

The kaldi scp format requires a tab or space separated line with the information: `FEATURENAME WAVEPATH`

```bash
find `pwd`/hospital_3707/ -type f | awk -F[./] '{print $(NF-1),$0}' > hospital.scp
compute-fbank-feats --config=fbank_config/fbank.conf scp:hospital.scp ark:hospital_fbank.ark
```

* Logmelspectrogram

```bash
python3 ../feature/featextract.py `find hospital_* -type f` hospital_logmel.ark mfcc -win_length 1764 -hop_length 882
```

## Training Configurator

Training configuration is done in `config/*.yaml`. Here one can adjust some hyperparameters e.g., number of hidden layers or embedding size. You can also write your own Models in `models.py` and Adjust the config to use that model (e.g. `encoder:MYMODEL`)

