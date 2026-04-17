"""Evaluate a trained dereverb model: reports loss, SI-SNR, PESQ, and STOI."""
import argparse

import keras
import numpy as np
import tensorflow as tf
from pesq import pesq as pesq_score
from pystoi import stoi as stoi_score
from tqdm import tqdm

from train import (FRAME_LENGTH, FRAME_STEP, checkpoint_path, dataset,
                   dereverb_model, log_l2_loss)


def si_snr(target, estimate, eps=1e-8):
    """Scale-invariant SNR in dB, batched. Inputs: (B, T) numpy arrays."""
    target = target - target.mean(axis=-1, keepdims=True)
    estimate = estimate - estimate.mean(axis=-1, keepdims=True)
    dot = np.sum(estimate * target, axis=-1, keepdims=True)
    s_target = dot * target / (np.sum(target ** 2, axis=-1, keepdims=True) + eps)
    e_noise = estimate - s_target
    ratio = (np.sum(s_target ** 2, axis=-1) + eps) / (np.sum(e_noise ** 2, axis=-1) + eps)
    return 10.0 * np.log10(ratio)


def evaluate(ver='prd', split='test', batch_size=16, loss='l1'):
    paths = tf.io.gfile.glob(f'./datasets/dereverb/{split}-{ver}/X/*.wav')
    if not paths:
        raise SystemExit(f'No wav pairs found under ./datasets/dereverb/{split}-{ver}/')
    ds = dataset(batch_size, paths)

    model = dereverb_model((None,), FRAME_LENGTH, FRAME_STEP)
    ckpt = checkpoint_path(ver, loss)
    print(f'Loading checkpoint "{ckpt}"')
    model.load_weights(ckpt)

    loss_fn = log_l2_loss(FRAME_LENGTH, FRAME_STEP)

    SR = 16000
    losses = []
    si_snrs_in, si_snrs_out = [], []
    pesqs_in, pesqs_out = [], []
    stois_in, stois_out = [], []

    for i in tqdm(range(len(ds)), desc=f'{split}-{ver}'):
        x, y = ds[i]
        pred = model(x, training=False)
        losses.append(float(loss_fn(y, pred)))
        pred_np = keras.ops.convert_to_numpy(pred)
        si_snrs_in.extend(si_snr(y, x).tolist())
        si_snrs_out.extend(si_snr(y, pred_np).tolist())
        for k in range(len(y)):
            pesqs_in.append(pesq_score(SR, y[k], x[k], 'wb'))
            pesqs_out.append(pesq_score(SR, y[k], pred_np[k], 'wb'))
            stois_in.append(stoi_score(y[k], x[k], SR, extended=False))
            stois_out.append(stoi_score(y[k], pred_np[k], SR, extended=False))

    results = {
        'loss': float(np.mean(losses)),
        'si_snr_in': float(np.mean(si_snrs_in)),
        'si_snr_out': float(np.mean(si_snrs_out)),
        'pesq_in': float(np.mean(pesqs_in)),
        'pesq_out': float(np.mean(pesqs_out)),
        'stoi_in': float(np.mean(stois_in)),
        'stoi_out': float(np.mean(stois_out)),
    }

    print()
    print(f'dataset        : {split}-{ver}  ({len(ds) * batch_size} examples)')
    print(f'loss           : {results["loss"]:.4f}')
    print(f'SI-SNR (input) : {results["si_snr_in"]:+.2f} dB')
    print(f'SI-SNR (output): {results["si_snr_out"]:+.2f} dB')
    print(f'SI-SNR gain    : {results["si_snr_out"] - results["si_snr_in"]:+.2f} dB')
    print(f'PESQ (input)   : {results["pesq_in"]:.3f}')
    print(f'PESQ (output)  : {results["pesq_out"]:.3f}')
    print(f'PESQ gain      : {results["pesq_out"] - results["pesq_in"]:+.3f}')
    print(f'STOI (input)   : {results["stoi_in"]:.4f}')
    print(f'STOI (output)  : {results["stoi_out"]:.4f}')
    print(f'STOI gain      : {results["stoi_out"] - results["stoi_in"]:+.4f}')

    return results


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--ver', choices=['dev', 'prd'], default='prd')
    p.add_argument('--split', choices=['train', 'val', 'test'], default='test')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--loss', default='l1',
                   help='Loss name used in checkpoint filename (default: l1)')
    args = p.parse_args()
    evaluate(args.ver, args.split, args.batch_size, args.loss)
