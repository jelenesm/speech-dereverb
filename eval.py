"""Evaluate a trained dereverb model: reports loss and SI-SNR on a dataset."""
import argparse

import keras
import numpy as np
import tensorflow as tf
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


def evaluate(ver='prd', split='test', batch_size=16):
    paths = tf.io.gfile.glob(f'./datasets/dereverb/{split}-{ver}/X/*.wav')
    if not paths:
        raise SystemExit(f'No wav pairs found under ./datasets/dereverb/{split}-{ver}/')
    ds = dataset(batch_size, paths)

    model = dereverb_model((None,), FRAME_LENGTH, FRAME_STEP)
    ckpt = checkpoint_path(ver)
    print(f'Loading checkpoint "{ckpt}"')
    model.load_weights(ckpt)

    loss_fn = log_l2_loss(FRAME_LENGTH, FRAME_STEP)

    losses, si_snrs_in, si_snrs_out = [], [], []
    for i in tqdm(range(len(ds)), desc=f'{split}-{ver}'):
        x, y = ds[i]
        pred = model(x, training=False)
        losses.append(float(loss_fn(y, pred)))
        pred_np = keras.ops.convert_to_numpy(pred)
        si_snrs_in.extend(si_snr(y, x).tolist())
        si_snrs_out.extend(si_snr(y, pred_np).tolist())

    mean_loss = float(np.mean(losses))
    mean_in = float(np.mean(si_snrs_in))
    mean_out = float(np.mean(si_snrs_out))

    print()
    print(f'dataset        : {split}-{ver}  ({len(ds) * batch_size} examples)')
    print(f'loss           : {mean_loss:.4f}')
    print(f'SI-SNR (input) : {mean_in:+.2f} dB')
    print(f'SI-SNR (output): {mean_out:+.2f} dB')
    print(f'SI-SNR gain    : {mean_out - mean_in:+.2f} dB')

    return {'loss': mean_loss, 'si_snr_in': mean_in, 'si_snr_out': mean_out}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--ver', choices=['dev', 'prd'], default='prd')
    p.add_argument('--split', choices=['train', 'val', 'test'], default='test')
    p.add_argument('--batch-size', type=int, default=16)
    args = p.parse_args()
    evaluate(args.ver, args.split, args.batch_size)
