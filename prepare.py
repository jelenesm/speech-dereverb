"""Dataset preparation and evaluation.

Downloads the MIT IR survey, LibriSpeech (dev-clean or train-clean-100), and ESC-50
noise, resamples everything to 16 kHz, then materializes degraded/clean wav pairs
under ./datasets/dereverb/{train,val,test}-{ver}/{X,Y}/ for training.
"""
import argparse
import csv
import glob
import os
import pickle
import random
import tarfile
import time
import zipfile

import librosa
import numpy as np
import scipy
import soundfile as sf
from matplotlib import pyplot as plt
from tqdm import tqdm

TARGET_SR = 16000


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def download(url, path):
    """Download url and unpack under path. Skips if path already exists."""
    if os.path.exists(path):
        return
    import wget  # lazy import — only needed for first-time preparation
    os.makedirs(path)
    print(f'downloading {url} to {path}')
    file = wget.download(url)
    print('\nextracting')
    ext = os.path.splitext(file)[1]
    if ext == '.zip':
        with zipfile.ZipFile(file, 'r') as zf:
            zf.extractall(path)
    elif ext == '.gz':
        with tarfile.open(file, 'r:gz') as tf_:
            tf_.extractall(path=path)
    os.remove(file)


def prepare_impulse_responses(ir_dir='./IR'):
    download('http://mcdermottlab.mit.edu/Reverb/IRMAudio/Audio.zip', ir_dir)
    fname = os.path.join(ir_dir, 'impulse_responses.pkl')
    if os.path.exists(fname):
        return fname
    responses = []
    for path in sorted(glob.glob(ir_dir + '/Audio/*.wav')):
        ir, sr = sf.read(path)
        if sr != TARGET_SR:
            ir = librosa.resample(ir, orig_sr=sr, target_sr=TARGET_SR)
        responses.append(ir)
    print(f'writing impulse responses to {fname}')
    with open(fname, 'wb') as f:
        pickle.dump(responses, f)
    return fname


def prepare_librispeech(ver, dataset_dir='.'):
    if ver == 'dev':
        path = os.path.join(dataset_dir, 'SpeechDev')
        download('https://www.openslr.org/resources/12/dev-clean.tar.gz', path)
        return os.path.join(path, 'LibriSpeech/dev-clean')
    path = os.path.join(dataset_dir, 'Speech')
    download('https://www.openslr.org/resources/12/train-clean-100.tar.gz', path)
    return os.path.join(path, 'LibriSpeech/train-clean-100')


def prepare_esc50(dataset_dir='.'):
    target_dir = os.path.join(dataset_dir, 'Noise')
    esc_master = os.path.join(dataset_dir, 'ESC-50-master')
    # The archive unpacks to ESC-50-master/, so we can't use download() directly
    # (it would skip because the parent dir always exists). Check for the extracted dir instead.
    if not os.path.exists(esc_master):
        import wget
        url = 'https://github.com/karoldvl/ESC-50/archive/master.zip'
        print(f'downloading {url}')
        file = wget.download(url)
        print('\nextracting')
        with zipfile.ZipFile(file, 'r') as zf:
            zf.extractall(dataset_dir)
        os.remove(file)
    csv_file = os.path.join(esc_master, 'meta/esc50.csv')
    source_dir = os.path.join(esc_master, 'audio')
    classes = [
        'washing_machine', 'vacuum_cleaner', 'clock_alarm', 'helicopter', 'chainsaw',
        'siren', 'car_horn', 'engine', 'train', 'church_bells', 'airplane', 'crying_baby',
        'door_wood_knock', 'door_wood_creaks', 'rooster', 'sea_waves', 'chirping_birds',
        'clock_tick', 'laughing', 'breathing',
    ]
    for cls in classes:
        dir_name = os.path.join(target_dir, cls)
        if os.path.exists(dir_name):
            continue
        os.makedirs(dir_name)
        print(f'processing class: {cls}')
        with open(csv_file, 'r') as f:
            rows = [r for r in csv.reader(f) if r[3] == cls]
        for row in rows:
            y, sr = sf.read(os.path.join(source_dir, row[0]))
            y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
            sf.write(os.path.join(dir_name, row[0]), y, samplerate=TARGET_SR)
    return target_dir


# ---------------------------------------------------------------------------
# Degradation primitives
# ---------------------------------------------------------------------------

def split_dataset(ds, val_split=20, seed=1337):
    random.Random(seed).shuffle(ds)
    n_val = len(ds) * val_split // 100
    return ds[:-n_val], ds[-n_val:]


def load(path, segment_len):
    x, sr = sf.read(path)
    if len(x) <= segment_len:
        x = np.concatenate([x, np.zeros(segment_len - len(x))])
    else:
        offset = np.random.randint(0, len(x) - segment_len)
        x = x[offset:offset + segment_len]
    return x, sr


def add_reverb(x, ir_list, prob):
    y = x.copy()
    if np.random.binomial(1, prob):
        h = ir_list[np.random.randint(0, len(ir_list))]
        y = scipy.signal.fftconvolve(y, h)[:len(y)]
        y *= (np.std(x)/np.std(y))
    return y


def add_distortion(x, clip_level, prob):
    if np.random.binomial(1, prob):
        th = np.random.uniform(clip_level[0], clip_level[1])
        x = np.clip(x, -th, th)
    return x


def add_noise(x, noise_paths, segment_len, snr_dB_level, prob):
    if np.random.binomial(1, prob) and noise_paths:
        noise, _ = load(noise_paths[np.random.randint(0, len(noise_paths))], segment_len)
        snr_dB = np.random.uniform(snr_dB_level[0], snr_dB_level[1])
        k = 10.0 ** (-snr_dB / 20.0)
        noise *= np.std(x) / np.std(noise)
        x = x + k * noise
    return x


def make_dataset(
    signal_paths, noise_paths, ir_list,
    timesteps, num_passes,
    p_reverb, p_clip, p_noise,
    snr_dB, clip_level,
    out_dir, overwrite=False,
):
    """Offline dataset generation: writes X/ (degraded) and Y/ (clean) wav pairs."""
    X_dir = os.path.join(out_dir, 'X')
    Y_dir = os.path.join(out_dir, 'Y')
    if not os.path.exists(out_dir):
        os.makedirs(X_dir)
        os.makedirs(Y_dir)
    elif not overwrite:
        print(f'Found existing dataset directory: {out_dir}')
        return

    for p in range(num_passes):
        print(f'Pass {p}')
        for i, signal in enumerate(tqdm(signal_paths, desc='Generating dataset')):
            try:
                y, sr = load(signal, timesteps)
            except sf.LibsndfileError as e:
                print(f'Skipping unreadable file {signal}: {e}')
                continue
            x = add_reverb(y, ir_list, p_reverb)
            x = add_distortion(x, clip_level, p_clip)
            x = add_noise(x, noise_paths, timesteps, snr_dB, p_noise)
            peak = max(np.max(np.abs(x)), np.max(np.abs(y)))
            if peak >= 1.0:
                x = x / peak
                y = y / peak
            scipy.io.wavfile.write(os.path.join(Y_dir, f'{i}_{p}.wav'), sr, np.int16(y * 32768))
            scipy.io.wavfile.write(os.path.join(X_dir, f'{i}_{p}.wav'), sr, np.int16(x * 32768))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def eval_scores(model_dict, batches, rate=16000):
    """Evaluate audio quality metrics across models. PESQ/STOI requires pysepm.

    `batches` is an iterable of (inputs, targets) numpy pairs. For full
    PESQ/STOI/SRMR metrics on a split, prefer `eval.py`.
    """
    try:
        import pysepm
    except ImportError:
        pysepm = None
        print('pysepm not available — PESQ/STOI will be skipped')

    import torch

    plt.figure(figsize=(10, 6))
    plt.subplot(121); plt.title('PESQ')
    plt.subplot(122); plt.title('STOI')

    scores = {}
    for model_name, (model, checkpoint) in model_dict.items():
        device = next(model.parameters()).device
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        model.eval()
        pesq, stoi = [], []
        for inputs, targets in batches:
            x = torch.from_numpy(inputs).float().to(device)
            with torch.no_grad():
                preds = model(x).cpu().numpy()
            if pysepm is not None:
                for k in range(len(targets)):
                    pesq.append(pysepm.pesq(targets[k], preds[k], rate))
                    stoi.append(pysepm.stoi(targets[k], preds[k], rate))
        scores[model_name] = {'pesq': float(np.mean(pesq)) if pesq else None,
                              'stoi': float(np.mean(stoi)) if stoi else None}
        plt.subplot(121); plt.plot(pesq, label=model_name)
        plt.subplot(122); plt.plot(stoi, label=model_name)
    plt.legend()
    plt.show()
    return scores


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def prepare_all(ver='prd', overwrite=False, seed=42,
                out_dir='./datasets/dereverb', dataset_dir='.'):
    """Download all raw data and generate train/val/test wav pairs for `ver`."""
    random.seed(seed)
    np.random.seed(seed)
    frame_step = 128
    timesteps_train = frame_step * 128
    timesteps_test = frame_step * 640

    ir_pkl = prepare_impulse_responses(os.path.join(dataset_dir, 'IR'))
    speech_dir = prepare_librispeech(ver, dataset_dir)
    noise_dir = prepare_esc50(dataset_dir)

    with open(ir_pkl, 'rb') as f:
        ir_list = pickle.load(f)

    signal_paths = sorted(glob.glob(speech_dir + '/*/*/*.flac'))
    noise_paths = sorted(glob.glob(noise_dir + '/*/*.wav'))
    print(f'Found {len(signal_paths)} speech examples and {len(noise_paths)} noise examples')

    train_signals, val_signals = split_dataset(signal_paths)
    train_noises, val_noises = split_dataset(noise_paths) if noise_paths else ([], [])

    common = dict(
        p_reverb=1.0, p_clip=0.0, p_noise=0.0,
        snr_dB=[6, 15], clip_level=[0.01, 0.05],
        overwrite=overwrite,
    )

    print('Train dataset:')
    make_dataset(train_signals, train_noises, ir_list, timesteps_train, 1,
                 out_dir=f'{out_dir}/train-{ver}', **common)
    print('Validation dataset:')
    make_dataset(val_signals, val_noises, ir_list, timesteps_train, 1,
                 out_dir=f'{out_dir}/val-{ver}', **common)
    print('Test dataset:')
    make_dataset(val_signals, val_noises, ir_list, timesteps_test, 1,
                 out_dir=f'{out_dir}/test-{ver}', **common)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--ver', choices=['dev', 'prd'], default='prd')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out-dir', default='./data',
                   help='base output directory for generated wav pairs')
    p.add_argument('--dataset-dir', default='.',
                   help='base directory for downloaded raw data (IR, LibriSpeech, ESC-50)')
    args = p.parse_args()
    prepare_all(args.ver, args.overwrite, args.seed, args.out_dir, args.dataset_dir)
