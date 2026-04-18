"""Side-by-side perceptual comparison of checkpoints on a test batch.

Pre-computes predictions and spectrograms for every checkpoint, then lets
you browse examples (Prev/Next example) and checkpoints (Prev/Next ckpt)
with input, target, and output spectrograms displayed side by side.
"""
import argparse
import glob
import os
import tempfile

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import soundfile as sf
import tensorflow as tf

from train import FRAME_LENGTH, FRAME_STEP, dataset, dereverb_model
from test import block_inference
from listen import spectrogram_db, Player


VMIN, VMAX = -40, 20
SR = 16000


def find_checkpoints(ckpt_dir):
    paths = sorted(glob.glob(os.path.join(ckpt_dir, '*.weights.h5')))
    return [(os.path.basename(p), p) for p in paths]


def precompute(model, checkpoints, inputs):
    predictions = {}
    for name, path in checkpoints:
        print(f'Checkpoint: {name}')
        model.load_weights(path)
        preds = []
        for x in inputs:
            y = block_inference(model, x, block_size=16384, overlap=128)
            preds.append(y[:len(x)])
        predictions[name] = preds
    return predictions


class Comparer:
    def __init__(self, inputs, targets, checkpoints, predictions):
        self.inputs = inputs
        self.targets = targets
        self.ckpt_names = [name for name, _ in checkpoints]
        self.predictions = predictions
        self.ex_idx = 0
        self.ckpt_idx = 0
        self.player = Player()
        self.tmpdir = tempfile.mkdtemp(prefix='compare_')

        self.input_specs = [spectrogram_db(x) for x in inputs]
        self.target_specs = [spectrogram_db(x) for x in targets]
        self.output_specs = {
            name: [spectrogram_db(p) for p in preds]
            for name, preds in predictions.items()
        }

        self.fig, (self.ax_in, self.ax_tgt, self.ax_out) = plt.subplots(
            1, 3, figsize=(18, 6))
        self.fig.subplots_adjust(bottom=0.22, wspace=0.25)

        spec = self.input_specs[0]
        nfreq, nframes = spec.shape
        extent = [0, nframes, 0, nfreq]
        imshow_kw = dict(vmin=VMIN, vmax=VMAX, origin='lower', aspect='auto',
                         extent=extent)

        self.ax_in.set_xlabel('time frame')
        self.ax_in.set_ylabel('freq bin')
        self.im_in = self.ax_in.imshow(spec, **imshow_kw)

        self.ax_tgt.set_xlabel('time frame')
        self.ax_tgt.set_ylabel('freq bin')
        self.im_tgt = self.ax_tgt.imshow(self.target_specs[0], **imshow_kw)

        self.ax_out.set_xlabel('time frame')
        self.ax_out.set_ylabel('freq bin')
        self.im_out = self.ax_out.imshow(
            self.output_specs[self.ckpt_names[0]][0], **imshow_kw)

        self.fig.colorbar(self.im_in, ax=[self.ax_in, self.ax_tgt, self.ax_out],
                          label='dB', shrink=0.85, pad=0.02)

        btn_w, btn_h = 0.10, 0.045
        gap = 0.02
        y_top, y_bot = 0.10, 0.03

        def row(n, y):
            total = n * btn_w + (n - 1) * gap
            x0 = 0.5 - total / 2
            return [self.fig.add_axes([x0 + i * (btn_w + gap), y, btn_w, btn_h])
                    for i in range(n)]

        top = row(4, y_top)
        bot = row(3, y_bot)

        self.btn_prev_ex = Button(top[0], '\u25C0 Prev example')
        self.btn_next_ex = Button(top[1], 'Next example \u25B6')
        self.btn_prev_ck = Button(top[2], '\u25C0 Prev ckpt')
        self.btn_next_ck = Button(top[3], 'Next ckpt \u25B6')
        self.btn_play_in = Button(bot[0], '\u25B6 Play input')
        self.btn_play_tgt = Button(bot[1], '\u25B6 Play target')
        self.btn_play_out = Button(bot[2], '\u25B6 Play output')

        self.btn_prev_ex.on_clicked(lambda _: self.step_example(-1))
        self.btn_next_ex.on_clicked(lambda _: self.step_example(+1))
        self.btn_prev_ck.on_clicked(lambda _: self.step_ckpt(-1))
        self.btn_next_ck.on_clicked(lambda _: self.step_ckpt(+1))
        self.btn_play_in.on_clicked(lambda _: self.play('input'))
        self.btn_play_tgt.on_clicked(lambda _: self.play('target'))
        self.btn_play_out.on_clicked(lambda _: self.play('output'))

        self._update_titles()
        self.fig.canvas.draw_idle()

    def _wav_path(self, tag):
        return os.path.join(self.tmpdir, f'{tag}.wav')

    def _update_titles(self):
        name = self.ckpt_names[self.ckpt_idx]
        self.ax_in.set_title(
            f'Input  (example {self.ex_idx + 1}/{len(self.inputs)})')
        self.ax_tgt.set_title('Target (clean)')
        self.ax_out.set_title(
            f'Output: {name}  '
            f'(ckpt {self.ckpt_idx + 1}/{len(self.ckpt_names)})')

    def refresh(self):
        name = self.ckpt_names[self.ckpt_idx]
        self.im_in.set_data(self.input_specs[self.ex_idx])
        self.im_tgt.set_data(self.target_specs[self.ex_idx])
        self.im_out.set_data(self.output_specs[name][self.ex_idx])
        self._update_titles()
        self.fig.canvas.draw_idle()

    def play(self, tag):
        name = self.ckpt_names[self.ckpt_idx]
        audios = {'input': self.inputs[self.ex_idx],
                  'target': self.targets[self.ex_idx],
                  'output': self.predictions[name][self.ex_idx]}
        wav = self._wav_path(tag)
        sf.write(wav, audios[tag], SR)
        self.player.play(wav, audios[tag], SR)

    def step_example(self, delta):
        self.player.stop()
        self.ex_idx = (self.ex_idx + delta) % len(self.inputs)
        self.refresh()

    def step_ckpt(self, delta):
        self.player.stop()
        self.ckpt_idx = (self.ckpt_idx + delta) % len(self.ckpt_names)
        self.refresh()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Compare checkpoints side-by-side')
    p.add_argument('--ver', choices=['dev', 'prd'], default='prd')
    p.add_argument('--split', choices=['train', 'val', 'test'], default='test')
    p.add_argument('--ckpt-dir', default='./checkpoints')
    p.add_argument('--batch-idx', type=int, default=None,
                   help='Batch index (default: random)')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--data-dir', default='./data',
                   help='base directory holding {split}-{ver}/ wav pairs (default: ./data)')
    args = p.parse_args()

    np.random.seed(args.seed)

    split_dir = f'{args.data_dir}/{args.split}-{args.ver}'
    paths = sorted(tf.io.gfile.glob(f'{split_dir}/X/*.wav'))
    if not paths:
        raise SystemExit(f'No wav pairs found under {split_dir}/')

    ds = dataset(args.batch_size, paths)
    batch_idx = (args.batch_idx if args.batch_idx is not None
                 else np.random.randint(0, len(ds)))
    print(f'Batch {batch_idx} ({args.batch_size} examples)')
    inputs, targets = ds[batch_idx]

    checkpoints = find_checkpoints(args.ckpt_dir)
    if not checkpoints:
        raise SystemExit(f'No *.weights.h5 files found in {args.ckpt_dir}/')
    print(f'Found {len(checkpoints)} checkpoint(s): '
          f'{[n for n, _ in checkpoints]}')

    model = dereverb_model((None,), FRAME_LENGTH, FRAME_STEP)
    predictions = precompute(model, checkpoints, inputs)

    Comparer(list(inputs), list(targets), checkpoints, predictions)
    plt.show()
