"""Evaluate a trained dereverb model: reports loss, SRMR, PESQ, and STOI."""
import argparse
import glob

import numpy as np
import torch
from pesq import pesq as pesq_score
from pystoi import stoi as stoi_score
from srmrpy import srmr as srmr_score
from tqdm import tqdm

from train import (FRAME_LENGTH, FRAME_STEP, DereverbModel, LogL1Loss,
                   load_batch, select_device)


def evaluate(ckpt, ver='prd', split='test', batch_size=16, data_dir='./data'):
    split_dir = f'{data_dir}/{split}-{ver}'
    paths = sorted(glob.glob(f'{split_dir}/X/*.wav'))
    if not paths:
        raise SystemExit(f'No wav pairs found under {split_dir}/')

    device = select_device()
    print(f'Device: {device}')

    model = DereverbModel(FRAME_LENGTH, FRAME_STEP).to(device)
    print(f'Loading checkpoint "{ckpt}"')
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    loss_fn = LogL1Loss(FRAME_LENGTH, FRAME_STEP).to(device)

    SR = 16000
    losses = []
    srmrs_in, srmrs_out = [], []
    pesqs_in, pesqs_out = [], []
    stois_in, stois_out = [], []

    num_batches = len(paths) // batch_size
    for i in tqdm(range(num_batches), desc=f'{split}-{ver}'):
        x, y = load_batch(paths, batch_size, i)
        x_t = torch.from_numpy(x).float().to(device)
        y_t = torch.from_numpy(y).float().to(device)
        with torch.no_grad():
            pred_t = model(x_t)
            losses.append(loss_fn(pred_t, y_t).item())
        pred_np = pred_t.cpu().numpy()
        for k in range(len(y)):
            pesqs_in.append(pesq_score(SR, y[k], x[k], 'wb'))
            pesqs_out.append(pesq_score(SR, y[k], pred_np[k], 'wb'))
            stois_in.append(stoi_score(y[k], x[k], SR, extended=False))
            stois_out.append(stoi_score(y[k], pred_np[k], SR, extended=False))
            srmrs_in.append(srmr_score(x[k], SR)[0])
            srmrs_out.append(srmr_score(pred_np[k], SR)[0])

    results = {
        'loss': float(np.mean(losses)),
        'srmr_in': float(np.mean(srmrs_in)),
        'srmr_out': float(np.mean(srmrs_out)),
        'pesq_in': float(np.mean(pesqs_in)),
        'pesq_out': float(np.mean(pesqs_out)),
        'stoi_in': float(np.mean(stois_in)),
        'stoi_out': float(np.mean(stois_out)),
    }

    print()
    print(f'dataset        : {split}-{ver}  ({num_batches * batch_size} examples)')
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

    return results


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--ver', choices=['dev', 'prd'], default='prd')
    p.add_argument('--split', choices=['train', 'val', 'test'], default='test')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--ckpt', required=True,
                   help='Path to the checkpoint (.pt) to evaluate')
    p.add_argument('--data-dir', default='./data',
                   help='base directory holding {split}-{ver}/ wav pairs (default: ./data)')
    args = p.parse_args()
    evaluate(args.ckpt, args.ver, args.split, args.batch_size, args.data_dir)
