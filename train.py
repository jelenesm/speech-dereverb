"""U-Net speech dereverberation: custom PyTorch modules, model, losses, and training loop.

Run this directly after prepare.py has generated the X/Y wav pairs under
./data/{train,val}-{ver}/ (or whatever --data-dir points at).
"""
import argparse
import glob
import os
import random

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# STFT / ISTFT / polar conversions
# ---------------------------------------------------------------------------

class STFT(nn.Module):
    """Real/imag STFT. Input (B, samples); output (B, 2, T, F).

    Right-pads by frame_length - frame_step so that T = samples // frame_step,
    matches keras.ops.stft(center=False) with a periodic Hann window, then drops
    the DC bin so F = frame_length // 2 (a power of two for the U-Net).
    """
    def __init__(self, frame_length=1024, frame_step=256):
        super().__init__()
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.pad_len = frame_length - frame_step
        self.register_buffer('window', torch.hann_window(frame_length, periodic=True))

    def forward(self, x):
        x = F.pad(x, (0, self.pad_len))
        X = torch.stft(x, n_fft=self.frame_length, hop_length=self.frame_step,
                       win_length=self.frame_length, window=self.window,
                       center=False, return_complex=True)
        X = X[:, 1:, :]                         # drop DC → (B, F, T)
        real = X.real.transpose(1, 2)           # (B, T, F)
        imag = X.imag.transpose(1, 2)
        return torch.stack([real, imag], dim=1)  # (B, 2, T, F)


class ISTFT(nn.Module):
    """Inverse of STFT. Input (B, 2, T, F); output (B, samples).

    Implemented as irfft + synthesis window + overlap-add (F.fold) to match
    tf.signal.inverse_stft, which does NOT divide by the window OLA envelope.
    torch.istft does divide, and rejects periodic Hann + center=False because
    the envelope vanishes at the first sample. Skipping the division means the
    round-trip signal is scaled by the COLA constant (1.5 for Hann @ 75%
    overlap) in the interior and tapered at the edges — this is exactly what
    the upstream Keras/TF graph produced, so downstream behavior (including
    the loss) is preserved.
    """
    def __init__(self, frame_length=1024, frame_step=256):
        super().__init__()
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.pad_len = frame_length - frame_step
        self.register_buffer('window', torch.hann_window(frame_length, periodic=True))

    def forward(self, x):
        real = x[:, 0].transpose(1, 2)          # (B, F, T)
        imag = x[:, 1].transpose(1, 2)
        B, _, T = real.shape
        zero_dc = torch.zeros(B, 1, T, dtype=real.dtype, device=real.device)
        X = torch.complex(torch.cat([zero_dc, real], dim=1),
                          torch.cat([zero_dc, imag], dim=1))  # (B, F_full, T)

        frames = torch.fft.irfft(X, n=self.frame_length, dim=1)  # (B, n_fft, T)
        frames = frames * self.window.view(1, -1, 1)             # synthesis window

        signal_len = (T - 1) * self.frame_step + self.frame_length
        y = F.fold(frames,
                   output_size=(signal_len, 1),
                   kernel_size=(self.frame_length, 1),
                   stride=(self.frame_step, 1)).squeeze(-1).squeeze(1)  # (B, signal_len)

        return y[:, :-self.pad_len]


def cartesian_to_logpolar(cart):
    """(B, 2, T, F) → log_mag (B, 1, T, F), phi (B, T, F).

    Note: log of magnitude-squared (not log-magnitude); preserved from the
    original Keras implementation.
    """
    real = cart[:, 0]
    imag = cart[:, 1]
    magsq = real * real + imag * imag
    log_mag = torch.log(torch.abs(magsq) + 1e-5)
    phi = torch.atan2(imag, real)
    return log_mag.unsqueeze(1), phi


def logpolar_to_cartesian(log_mag, phi):
    """(log_mag (B, 1, T, F), phi (B, T, F)) → (B, 2, T, F)."""
    mag = torch.exp(log_mag.squeeze(1))
    re = torch.cos(phi) * mag
    im = torch.sin(phi) * mag
    return torch.stack([re, im], dim=1)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SameConv2d(nn.Module):
    """Conv2d(k=5, s=2) with TF 'same' asymmetric padding: (1, 2) on each spatial dim."""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 5, stride=2, padding=0)

    def forward(self, x):
        x = F.pad(x, (1, 2, 1, 2))
        return self.conv(x)


class SameConvTranspose2d(nn.Module):
    """ConvTranspose2d(k=5, s=2) with TF 'same' asymmetric crop: 1 from start, 2 from end."""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_c, out_c, 5, stride=2, padding=0)

    def forward(self, x):
        y = self.deconv(x)
        return y[..., 1:-2, 1:-2]


class DownBlock(nn.Module):
    def __init__(self, in_c, out_c, use_bn=True):
        super().__init__()
        self.conv = SameConv2d(in_c, out_c)
        self.bn = nn.BatchNorm2d(out_c, momentum=0.01) if use_bn else nn.Identity()
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class UpBlock(nn.Module):
    def __init__(self, in_c, out_c, dropout=False):
        super().__init__()
        self.deconv = SameConvTranspose2d(in_c, out_c)
        self.bn = nn.BatchNorm2d(out_c, momentum=0.01)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.5) if dropout else nn.Identity()

    def forward(self, x, skip):
        x = self.dropout(self.act(self.bn(self.deconv(x))))
        return torch.cat([x, skip], dim=1)


def _keras_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class DereverbModel(nn.Module):
    """End-to-end dereverb U-Net: raw audio in, raw audio out."""

    def __init__(self, frame_length=512, frame_step=128):
        super().__init__()
        self.stft = STFT(frame_length, frame_step)
        self.istft = ISTFT(frame_length, frame_step)

        self.d1 = DownBlock(1, 64, use_bn=False)
        self.d2 = DownBlock(64, 128)
        self.d3 = DownBlock(128, 256)
        self.d4 = DownBlock(256, 512)
        self.d5 = DownBlock(512, 512)
        self.d6 = DownBlock(512, 512)

        self.b_conv = SameConv2d(512, 512)
        self.b_bn = nn.BatchNorm2d(512, momentum=0.01)
        self.b_act = nn.ReLU()

        self.u1 = UpBlock(512, 512, dropout=True)
        self.u2 = UpBlock(1024, 512, dropout=True)
        self.u3 = UpBlock(1024, 512)
        self.u4 = UpBlock(1024, 256)
        self.u5 = UpBlock(512, 128)
        self.u6 = UpBlock(256, 64)

        self.out_conv = SameConvTranspose2d(128, 1)

        self.apply(_keras_init)

    def forward(self, x):
        cart = self.stft(x)
        log_mag, phi = cartesian_to_logpolar(cart)

        x1 = self.d1(log_mag)
        x2 = self.d2(x1)
        x3 = self.d3(x2)
        x4 = self.d4(x3)
        x5 = self.d5(x4)
        x6 = self.d6(x5)

        b = self.b_act(self.b_bn(self.b_conv(x6)))

        h = self.u1(b, x6)
        h = self.u2(h, x5)
        h = self.u3(h, x4)
        h = self.u4(h, x3)
        h = self.u5(h, x2)
        h = self.u6(h, x1)

        log_mag_out = self.out_conv(h)
        cart_out = logpolar_to_cartesian(log_mag_out, phi)
        return self.istft(cart_out)


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

class _LogSpectralLossBase(nn.Module):
    def __init__(self, frame_length, frame_step):
        super().__init__()
        self.stft = STFT(frame_length, frame_step)

    def _log_mags(self, y_pred, y_true):
        a, _ = cartesian_to_logpolar(self.stft(y_true))
        b, _ = cartesian_to_logpolar(self.stft(y_pred))
        return a, b


class LogL1Loss(_LogSpectralLossBase):
    def forward(self, y_pred, y_true):
        a, b = self._log_mags(y_pred, y_true)
        return torch.mean(torch.abs(a - b))


class LogL2Loss(_LogSpectralLossBase):
    def forward(self, y_pred, y_true):
        a, b = self._log_mags(y_pred, y_true)
        return torch.mean((a - b) ** 2)


MULTI_SCALE_STFT_CONFIGS = [
    (2048, 512),
    (1024, 256),
    (512, 128),
    (256, 64),
]


class MultiScaleSpectralLoss(nn.Module):
    def __init__(self, configs=MULTI_SCALE_STFT_CONFIGS):
        super().__init__()
        self.stfts = nn.ModuleList([STFT(fl, fs) for fl, fs in configs])

    def forward(self, y_pred, y_true):
        loss = 0.0
        for stft in self.stfts:
            a, _ = cartesian_to_logpolar(stft(y_true))
            b, _ = cartesian_to_logpolar(stft(y_pred))
            loss = loss + torch.mean(torch.abs(a - b))
        return loss / len(self.stfts)


def _gaussian_kernel_1d(size, sigma):
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    return g / g.sum()


class SSIMLoss(_LogSpectralLossBase):
    """1 - mean SSIM of log-magnitude spectrograms. 11×11 Gaussian window (sigma=1.5)."""
    def __init__(self, frame_length, frame_step, max_val=20.0,
                 window_size=11, sigma=1.5):
        super().__init__(frame_length, frame_step)
        self.max_val = max_val
        g1d = _gaussian_kernel_1d(window_size, sigma)
        window = (g1d.unsqueeze(0) * g1d.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
        self.register_buffer('window', window)

    def forward(self, y_pred, y_true):
        a, b = self._log_mags(y_pred, y_true)
        C1 = (0.01 * self.max_val) ** 2
        C2 = (0.03 * self.max_val) ** 2
        w = self.window
        mu_a = F.conv2d(a, w)
        mu_b = F.conv2d(b, w)
        mu_aa, mu_bb, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
        sig_aa = F.conv2d(a * a, w) - mu_aa
        sig_bb = F.conv2d(b * b, w) - mu_bb
        sig_ab = F.conv2d(a * b, w) - mu_ab
        num = (2 * mu_ab + C1) * (2 * sig_ab + C2)
        den = (mu_aa + mu_bb + C1) * (sig_aa + sig_bb + C2)
        return 1.0 - (num / den).mean()


LOSS_BUILDERS = {
    'l1': lambda fl, fs: LogL1Loss(fl, fs),
    'l2': lambda fl, fs: LogL2Loss(fl, fs),
    'multi_scale': lambda fl, fs: MultiScaleSpectralLoss(),
    'ssim': lambda fl, fs: SSIMLoss(fl, fs),
}


# ---------------------------------------------------------------------------
# Dataset (reads X/Y wav pairs written by prepare.py)
# ---------------------------------------------------------------------------

class DereverbDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        x_path = self.paths[i]
        basename = os.path.basename(x_path)
        y_path = os.path.join(os.path.dirname(os.path.dirname(x_path)), 'Y', basename)
        x, _ = sf.read(x_path)
        y, _ = sf.read(y_path)
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()


def load_batch(paths, batch_size, idx):
    """Load a fixed batch as (X, Y) numpy arrays without a DataLoader.

    Used by test.py / eval.py / compare_models.py where deterministic
    batch access by index is more convenient than iterating a DataLoader.
    """
    start = idx * batch_size
    batch_paths = paths[start:start + batch_size]
    X, Y = [], []
    for x_path in batch_paths:
        basename = os.path.basename(x_path)
        y_path = os.path.join(os.path.dirname(os.path.dirname(x_path)), 'Y', basename)
        x, _ = sf.read(x_path)
        y, _ = sf.read(y_path)
        X.append(x)
        Y.append(y)
    return np.array(X), np.array(Y)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

FRAME_LENGTH = 512
FRAME_STEP = 128


def checkpoint_path(ver, loss='l1'):
    return f'./checkpoints/unet-{loss}-{ver}.pt'


def select_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def count_parameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def train(ver='prd', epochs=200, batch_size=32, lr=1e-4, resume=False, seed=42,
          loss='l1', data_dir='./data', num_workers=4):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = select_device()
    print(f'Device: {device}')

    train_paths = sorted(glob.glob(f'{data_dir}/train-{ver}/X/*.wav'))
    val_paths = sorted(glob.glob(f'{data_dir}/val-{ver}/X/*.wav'))
    if not train_paths:
        raise SystemExit(f'No train wav pairs under {data_dir}/train-{ver}/X/')
    if not val_paths:
        raise SystemExit(f'No val wav pairs under {data_dir}/val-{ver}/X/')

    train_ds = DereverbDataset(train_paths)
    val_ds = DereverbDataset(val_paths)
    loader_kw = dict(batch_size=batch_size, num_workers=num_workers, drop_last=True,
                     pin_memory=(device.type == 'cuda'),
                     persistent_workers=num_workers > 0)
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)

    print(f'*** Loss: {loss} ***')
    loss_fn = LOSS_BUILDERS[loss](FRAME_LENGTH, FRAME_STEP).to(device)

    model = DereverbModel(FRAME_LENGTH, FRAME_STEP).to(device)
    print(f'Model parameters: {count_parameters(model):,}')

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-7)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)

    ckpt = checkpoint_path(ver, loss)
    os.makedirs(os.path.dirname(ckpt), exist_ok=True)
    if resume and os.path.exists(ckpt):
        print(f'*** Loading checkpoint "{ckpt}" ***')
        model.load_state_dict(torch.load(ckpt, map_location=device))
    else:
        print('*** Training from scratch ***')

    best_val = float('inf')
    patience_counter = 0
    early_stop_patience = 10

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad()
            pred = model(x)
            loss_val = loss_fn(pred, y)
            loss_val.backward()
            optimizer.step()
            train_losses.append(loss_val.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                val_losses.append(loss_fn(model(x), y).item())
        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        lr_now = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch}: train_loss={train_loss:.4f}  '
              f'val_loss={val_loss:.4f}  lr={lr_now:.2e}')

        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), ckpt)
            print(f'*** Saved checkpoint "{ckpt}" (val_loss={val_loss:.4f}) ***')
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f'*** Early stopping at epoch {epoch} ***')
                break

    return model


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--ver', choices=['dev', 'prd'], default='prd')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--loss', choices=list(LOSS_BUILDERS), default='l1',
                   help='Loss function (default: l1)')
    p.add_argument('--resume', action='store_true',
                   help='Resume from existing checkpoint instead of training from scratch')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--data-dir', default='./data',
                   help='base directory holding {train,val}-{ver}/ wav pairs (default: ./data)')
    p.add_argument('--num-workers', type=int, default=4)
    args = p.parse_args()
    train(args.ver, args.epochs, args.batch_size, args.lr, args.resume, args.seed,
          loss=args.loss, data_dir=args.data_dir, num_workers=args.num_workers)
