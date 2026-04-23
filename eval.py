"""Evaluate a trained dereverb model: reports loss, SRMR, PESQ, STOI, and C50."""
import argparse

import keras
import numpy as np
import tensorflow as tf
from pesq import pesq as pesq_score
from pystoi import stoi as stoi_score
from srmrpy import srmr as srmr_score
from tqdm import tqdm

from train import (FRAME_LENGTH, FRAME_STEP, dataset, dereverb_model,
                   log_l1_loss)


def c50_dB(reference, signal, sr=16000, reg=1e-3):
    """Intrusive C50 (clarity at 50 ms), in dB.

    Estimates the residual impulse response that maps `reference` (clean) to
    `signal` (reverberant or dereverberated) via Wiener deconvolution, then
    returns the early-to-late energy ratio with the boundary at 50 ms after
    the direct-path peak. Higher = clearer; a perfectly dereverberated signal
    (signal ≈ reference) gives an IR ≈ δ and C50 → ∞.
    """
    n = len(signal)
    n_fft = 1 << int(np.ceil(np.log2(2 * n)))  # zero-pad to avoid circular wrap
    R = np.fft.rfft(reference, n=n_fft)
    S = np.fft.rfft(signal, n=n_fft)
    eps = reg * float(np.mean(np.abs(R) ** 2))
    H = S * np.conj(R) / (np.abs(R) ** 2 + eps)
    h = np.fft.irfft(H, n=n_fft)[:n]

    peak = int(np.argmax(np.abs(h)))
    h = h[peak:]
    n50 = int(0.050 * sr)
    if len(h) <= n50:
        return float('nan')
    early = float(np.sum(h[:n50] ** 2))
    late = float(np.sum(h[n50:] ** 2))
    if late <= 0:
        return float('inf')
    return 10.0 * np.log10(early / late)


def evaluate(ckpt, ver='prd', split='test', batch_size=16, data_dir='./data'):
    split_dir = f'{data_dir}/{split}-{ver}'
    paths = tf.io.gfile.glob(f'{split_dir}/X/*.wav')
    if not paths:
        raise SystemExit(f'No wav pairs found under {split_dir}/')
    ds = dataset(batch_size, paths)

    model = dereverb_model((None,), FRAME_LENGTH, FRAME_STEP)
    print(f'Loading checkpoint "{ckpt}"')
    model.load_weights(ckpt)

    loss_fn = log_l1_loss(FRAME_LENGTH, FRAME_STEP)

    SR = 16000
    losses = []
    srmrs_in, srmrs_out = [], []
    pesqs_in, pesqs_out = [], []
    stois_in, stois_out = [], []
    c50s_in, c50s_out = [], []

    for i in tqdm(range(len(ds)), desc=f'{split}-{ver}'):
        x, y = ds[i]
        pred = model(x, training=False)
        losses.append(float(loss_fn(y, pred)))
        pred_np = keras.ops.convert_to_numpy(pred)
        for k in range(len(y)):
            pesqs_in.append(pesq_score(SR, y[k], x[k], 'wb'))
            pesqs_out.append(pesq_score(SR, y[k], pred_np[k], 'wb'))
            stois_in.append(stoi_score(y[k], x[k], SR, extended=False))
            stois_out.append(stoi_score(y[k], pred_np[k], SR, extended=False))
            srmrs_in.append(srmr_score(x[k], SR)[0])
            srmrs_out.append(srmr_score(pred_np[k], SR)[0])
            c50s_in.append(c50_dB(y[k], x[k], SR))
            c50s_out.append(c50_dB(y[k], pred_np[k], SR))

    results = {
        'loss': float(np.mean(losses)),
        'srmr_in': float(np.mean(srmrs_in)),
        'srmr_out': float(np.mean(srmrs_out)),
        'pesq_in': float(np.mean(pesqs_in)),
        'pesq_out': float(np.mean(pesqs_out)),
        'stoi_in': float(np.mean(stois_in)),
        'stoi_out': float(np.mean(stois_out)),
        'c50_in': float(np.nanmean(c50s_in)),
        'c50_out': float(np.nanmean(c50s_out)),
    }

    print()
    print(f'dataset        : {split}-{ver}  ({len(ds) * batch_size} examples)')
    print(f'loss           : {results["loss"]:.4f}')
    print(f'SRMR (input)   : {results["srmr_in"]:.3f}')
    print(f'SRMR (output)  : {results["srmr_out"]:.3f}')
    print(f'SRMR gain      : {results["srmr_out"] - results["srmr_in"]:+.3f}')
    print(f'PESQ (input)   : {results["pesq_in"]:.3f}')
    print(f'PESQ (output)  : {results["pesq_out"]:.3f}')
    print(f'PESQ gain      : {results["pesq_out"] - results["pesq_in"]:+.3f}')
    print(f'STOI (input)   : {results["stoi_in"]:.4f}')
    print(f'STOI (output)  : {results["stoi_out"]:.4f}')
    print(f'STOI gain      : {results["stoi_out"] - results["stoi_in"]:+.4f}')
    print(f'C50  (input)   : {results["c50_in"]:.3f} dB')
    print(f'C50  (output)  : {results["c50_out"]:.3f} dB')
    print(f'C50  gain      : {results["c50_out"] - results["c50_in"]:+.3f} dB')

    return results


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--ver', choices=['dev', 'prd'], default='prd')
    p.add_argument('--split', choices=['train', 'val', 'test'], default='test')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--ckpt-dir', required=True,
                   help='Path to the checkpoint (.weights.h5) to evaluate')
    p.add_argument('--data-dir', default='./data',
                   help='base directory holding {split}-{ver}/ wav pairs (default: ./data)')
    args = p.parse_args()
    evaluate(args.ckpt_dir, args.ver, args.split, args.batch_size, args.data_dir)
