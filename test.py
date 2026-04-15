"""Render example inputs/predictions (spectrograms + wavs) from a trained model."""
import argparse
import os
import time

import numpy as np
import soundfile as sf
import tensorflow as tf
from matplotlib import pyplot as plt

from train import FRAME_LENGTH, FRAME_STEP, STFT, checkpoint_path, dataset, dereverb_model


def plot_spectrogram(x, title, frame_length, frame_step, vmin=-40, vmax=40):
    X = STFT(frame_length, frame_step)(x[tf.newaxis, :])
    X = 10 * np.log10(np.sum(np.square(X[0]+1e-5), axis=-1))
    im = plt.imshow(np.transpose(X), vmin=vmin, vmax=vmax, origin='lower', aspect='auto')
    plt.xlabel('time index')
    plt.ylabel('freq index')
    plt.title(title)
    plt.colorbar(im, label='dB')


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
    preds = model(inputs)
    y = mix_overlapping_sequences(preds, overlap)
    print(f'inference time: {time.time() - start:.3f}s')
    return y


def test_models(model_dict, dataset, num_examples, frame_length, frame_step, out_dir):
    """Generate spectrogram pngs + wavs for a random test batch across models."""
    def save(x, path, title):
        plt.figure(figsize=(21, 7))
        plot_spectrogram(x, title, frame_length, frame_step)
        plt.tight_layout()
        plt.savefig(path + '.png', metadata={'Title': title}, bbox_inches='tight')
        plt.close()
        sf.write(path + '.wav', x, samplerate=16000)

    os.makedirs(out_dir, exist_ok=True)
    batch_idx = np.random.randint(0, len(dataset))
    print(f'batch index: {batch_idx}')
    inputs, targets = dataset[batch_idx]

    for k in range(num_examples):
        d = os.path.join(out_dir, f'example_{k}')
        os.makedirs(d, exist_ok=True)
        save(inputs[k], os.path.join(d, 'input'), f'input (batch {batch_idx})')
        save(targets[k], os.path.join(d, 'target'), f'target (batch {batch_idx})')

    for model_name, (model, checkpoint) in model_dict.items():
        model.load_weights(checkpoint)
        preds = [block_inference(model, inp, block_size=16384, overlap=128) for inp in inputs]
        for k in range(num_examples):
            d = os.path.join(out_dir, f'example_{k}')
            save(preds[k], os.path.join(d, model_name), f'{model_name} (batch {batch_idx})')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--ver', choices=['dev', 'prd'], default='prd')
    p.add_argument('--num-examples', type=int, default=8)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--out-dir', default='./demo_dereverb')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    np.random.seed(args.seed)

    test_paths = tf.io.gfile.glob(f'./datasets/dereverb/test-{args.ver}/X/*.wav')
    test_ds = dataset(args.batch_size, test_paths)

    model = dereverb_model((None,), FRAME_LENGTH, FRAME_STEP)
    ckpt = checkpoint_path(args.ver)
    model_dict = {'unet': (model, ckpt)}

    test_models(model_dict, test_ds, args.num_examples, FRAME_LENGTH, FRAME_STEP, args.out_dir)
