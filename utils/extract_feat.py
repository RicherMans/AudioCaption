# -*- coding: utf-8 -*-
# @Author: richman
# @Date:   2018-03-29
# @Last Modified by:   richman
# @Last Modified time: 2018-03-29
import sys
import os
import librosa
import numpy as np
import argparse
import h5py
from tqdm import tqdm as tqdm

sys.path.append(os.getcwd())
import utils.kaldi_io as kaldi_io

def extractmfcc(y, fs=44100, **mfcc_params):
    eps = np.spacing(1)
    # Calculate Static Coefficients
    power_spectrogram = np.abs(librosa.stft(y + eps,
                                            center=True,
                                            n_fft=mfcc_params['n_fft'],
                                            win_length=mfcc_params[
                                                'win_length'],
                                            hop_length=mfcc_params[
                                                'hop_length'],
                                            ))**2
    mel_basis = librosa.filters.mel(sr=fs,
                                    n_fft=mfcc_params['n_fft'],
                                    n_mels=mfcc_params['n_mels'],
                                    fmin=mfcc_params['fmin'],
                                    fmax=mfcc_params['fmax'],
                                    htk=mfcc_params['htk'])
    mel_spectrum = np.dot(mel_basis, power_spectrogram)
    if mfcc_params['no_mfcc']:
        return np.log(mel_spectrum+eps)
    else:
        S = librosa.power_to_db(mel_spectrum)
        return librosa.feature.mfcc(S=S,
                                    n_mfcc=mfcc_params['n_mfcc'])


def extractstft(y, fs=44100, ** params):
    """Extracts short time fourier transform with either energy or power

    Args:
        y (np.array float): input array, usually from librosa.load()
        fs (int, optional): Sampling Rate
        **params: Extra parameters

    Returns:
        numpy array: Log amplitude of either energy or power stft
    """
    eps = np.spacing(1)
    # Calculate Static Coefficients
    spectogram = np.abs(librosa.stft(y + eps, center=params['center'],
                                     n_fft=params['n_fft'], win_length=params[
                                         'win_length'],
                                     hop_length=params['hop_length'], ))
    if params['power']:
        spectogram = spectogram**2
    return librosa.logamplitude(spectogram)


def extractwavelet(y, fs, **params):
    import pywt
    frames = librosa.util.frame(
        y, frame_length=4096, hop_length=2048).transpose()
    features = []
    for frame in frames:
        res = []
        for _ in range(params['level']):
            frame, d = pywt.dwt(frame, params['type'])
            res.append(librosa.power_to_db(np.sum(d * d) / len(d)))
        res.append(librosa.power_to_db(np.sum(frame * frame) / len(frame)))
        features.append(res)
    return np.array(features)


def extractraw(y, fs, **params):
    return librosa.util.frame(y, frame_length=params['frame_length'], hop_length=params['hop_length']).transpose()



parser = argparse.ArgumentParser()
""" Arguments: wavfilelist, n_mfcc, n_fft, win_length, hop_length, htk, fmin, fmax """
parser.add_argument('-config', type=argparse.FileType('r'), default=None)
parser.add_argument('wavfilelist', type=str, default=sys.stdin)
parser.add_argument('featureout', type=str, default=sys.stdout)
parser.add_argument('keyout', type=str, default=sys.stdout)

parser.add_argument('-norm', default='mean')
# parser.add_argument('-concat', default=1, type=int,
# help="concatenates samples over time")
parser.add_argument('-nomono', default=False, action="store_true")
subparsers = parser.add_subparsers(help="subcommand help")
mfccparser = subparsers.add_parser('mfcc')
mfccparser.add_argument('-n_mfcc', type=int, default=20)
mfccparser.add_argument('-n_mels', type=int, default=128)
mfccparser.add_argument('-n_fft', type=int, default=2048)
mfccparser.add_argument('-win_length', type=int, default=2048)
mfccparser.add_argument('-hop_length', type=int, default=1024)
mfccparser.add_argument('-htk', default=True,
                        action="store_true", help="Uses htk formula for MFCC est.")
mfccparser.add_argument('-fmin', type=int, default=12)
mfccparser.add_argument('-fmax', type=int, default=8000)
mfccparser.add_argument('-no_mfcc', default=True, action='store_false', help="Does not extract mfcc, rather logmelspect")
mfccparser.set_defaults(extractfeat=extractmfcc)
stftparser = subparsers.add_parser('stft')
stftparser.add_argument('-n_fft', type=int, default=2048)
stftparser.add_argument('-win_length', type=int, default=2048)
stftparser.add_argument('-hop_length', type=int, default=1024)
stftparser.add_argument('-center', default=False, action="store_true")
stftparser.add_argument('-power', default=False, action="store_true")
stftparser.set_defaults(extractfeat=extractstft)
rawparser = subparsers.add_parser('raw')
rawparser.add_argument('-hop_length', type=int, default=1024)
rawparser.add_argument('-frame_length', type=int, default=2048)
rawparser.set_defaults(extractfeat=extractraw)
waveletparser = subparsers.add_parser('wave')
waveletparser.add_argument('-level', default=10, type=int)
waveletparser.add_argument('-type', default='db4', type=str)
waveletparser.set_defaults(extractfeat=extractwavelet)


args = parser.parse_args()

argsdict = vars(args)

# Just for TQDM, usually its not that large anyway
# for line in tqdm(args.wavfilelist, ascii=True):
with h5py.File(args.featureout, "w") as feature_store, open(args.keyout, "w") as key_store, open(args.wavfilelist, "r") as wav_reader:
    for line in tqdm(wav_reader.readlines(), ascii=True, ncols=100):
        key, filename = line.strip().split()
        assert os.path.exists(filename), filename + "not exists!"
        y, sr = librosa.load(filename, sr=None, mono=not args.nomono)
        # Stereo
        if y.ndim > 1:
            feat = np.array([args.extractfeat(i, sr, **argsdict) for i in y])
        else:
            feat = args.extractfeat(y, sr, **argsdict)
        # Transpose feat, nsamples to nsamples, feat
        feat = np.vstack(feat).transpose()
        feature_store[key] = feat
        key_store.write(key + "\n")
