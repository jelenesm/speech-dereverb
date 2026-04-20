"""Render example inputs/predictions (wavs only) from a trained model."""
import argparse
import glob
import os
import time

import numpy as np
import soundfile as sf
import torch

from train import (FRAME_LENGTH, FRAME_STEP, DereverbModel, load_batch,
                   select_device)


def make_overlapping_sequences(x, block_size, overlap):
    idx = 0
    sequences = []
    for _ in range(len(x) // block_size):
        sequences.append(x[idx:idx + block_size])
        idx += (block_size - overlap)
    return np.array(sequences)


def mix_overlapping_sequences(sequences, overlap):
    """Linear cross-fade across overlapping fixed-size sequences."""
    block_size = sequences[0].shape[0]
    for seq in sequences:
        assert seq.shape[0] == block_size
    if overlap <= 1 or overlap >= block_size:
        raise ValueError('overlap must be in (1, block_size)')

    K = len(sequences)
    H = block_size - overlap
    N = (K - 1) * H + block_size
    output = np.zeros(N)

    w = np.ones(block_size)
    w[:overlap] = np.linspace(0, 1, overlap)
    w[-overlap:] = np.linspace(1, 0, overlap)

    for k, s_k in enumerate(sequences):
        start = k * H
        output[start:start + block_size] += w * s_k
    return output


def block_inference(model, x, block_size, overlap):
    start = time.time()
    inputs = make_overlapping_sequences(x, block_size, overlap)
    device = next(model.parameters()).device
    inputs_t = torch.from_numpy(inputs).float().to(device)
    with torch.no_grad():
        preds = model(inputs_t).cpu().numpy()
    y = mix_overlapping_sequences(preds, overlap)
    print(f'inference time: {time.time() - start:.3f}s')
    return y


def test_models(model_dict, inputs, targets, num_examples, out_dir, batch_idx, device):
    """Write input/target/prediction wavs for a given batch across models."""
    def save(x, path):
        sf.write(path + '.wav', x, samplerate=16000)

    os.makedirs(out_dir, exist_ok=True)
    print(f'batch index: {batch_idx}')

    example_dirs = []
    for k in range(num_examples):
        d = os.path.join(out_dir, f'example_b{batch_idx}_{k}')
        os.makedirs(d, exist_ok=True)
        example_dirs.append(d)
        save(inputs[k], os.path.join(d, 'input'))
        save(targets[k], os.path.join(d, 'target'))

    for model_name, (model, checkpoint) in model_dict.items():
        model.to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        model.eval()
        preds = [block_inference(model, inp, block_size=16384, overlap=128)
                 for inp in inputs]
        for k, d in enumerate(example_dirs):
            save(preds[k], os.path.join(d, model_name))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--ver', choices=['dev', 'prd'], default='prd')
    p.add_argument('--num-examples', type=int, default=8)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--batch-idx', type=int, default=None,
                   help='Test batch index to render. Default: random.')
    p.add_argument('--ckpt', required=True,
                   help='Path to the checkpoint (.pt) to evaluate')
    p.add_argument('--out-dir', default='./demo')
    p.add_argument('--data-dir', default='./data',
                   help='base directory holding {split}-{ver}/ wav pairs (default: ./data)')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    np.random.seed(args.seed)

    test_paths = sorted(glob.glob(f'{args.data_dir}/test-{args.ver}/X/*.wav'))
    if not test_paths:
        raise SystemExit(f'No test wav pairs under {args.data_dir}/test-{args.ver}/X/')

    num_batches = len(test_paths) // args.batch_size
    batch_idx = (args.batch_idx if args.batch_idx is not None
                 else int(np.random.randint(0, num_batches)))
    inputs, targets = load_batch(test_paths, args.batch_size, batch_idx)

    device = select_device()
    model = DereverbModel(FRAME_LENGTH, FRAME_STEP)
    model_dict = {'unet': (model, args.ckpt)}

    test_models(model_dict, inputs, targets, args.num_examples, args.out_dir,
                batch_idx, device)
