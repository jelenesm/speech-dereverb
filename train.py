"""U-Net speech dereverberation: custom Keras layers, model, loss, and training loop.

Run this directly after prepare.py has generated the X/Y wav pairs under
./data/{train,val}-{ver}/ (or whatever --data-dir points at).
"""
import os
import argparse
import time

import keras
import numpy as np
import soundfile as sf
import tensorflow as tf
from keras import layers
from pystoi import stoi as stoi_score


class TimeBudget(keras.callbacks.Callback):
    """Stop training once wall-clock fit time exceeds `seconds`."""
    def __init__(self, seconds):
        super().__init__()
        self.seconds = seconds

    def on_train_begin(self, logs=None):
        self._t0 = time.time()

    def on_train_batch_end(self, batch, logs=None):
        if time.time() - self._t0 >= self.seconds:
            self.model.stop_training = True


def si_sdr(target, est, eps=1e-8):
    """Scale-invariant SDR in dB for a single 1D signal pair."""
    target = target - target.mean()
    est = est - est.mean()
    alpha = np.dot(est, target) / (np.dot(target, target) + eps)
    proj = alpha * target
    noise = est - proj
    return 10.0 * np.log10(np.sum(proj ** 2) / (np.sum(noise ** 2) + eps) + eps)


class MetricsOnVal(keras.callbacks.Callback):
    """Compute ESTOI gain and SI-SDR-i on the full val set each epoch and inject into logs."""
    def __init__(self, val_paths, batch_size=16, sr=16000):
        super().__init__()
        xs, ys = [], []
        for x_path in val_paths:
            basename = os.path.basename(x_path)
            y_path = os.path.join(os.path.dirname(os.path.dirname(x_path)), 'Y', basename)
            xi, _ = sf.read(x_path)
            yi, _ = sf.read(y_path)
            xs.append(xi)
            ys.append(yi)
        self.x = np.array(xs)
        self.y = np.array(ys)
        self.sr = sr
        self.batch_size = batch_size
        self.sisdr_in = float(np.mean([si_sdr(yi, xi) for xi, yi in zip(self.x, self.y)]))
        self.estoi_in = float(np.mean([stoi_score(yi, xi, sr, extended=True)
                                       for xi, yi in zip(self.x, self.y)]))

    def on_epoch_end(self, epoch, logs=None):
        preds = []
        for i in range(0, len(self.x), self.batch_size):
            p = self.model(self.x[i:i + self.batch_size], training=False)
            preds.append(keras.ops.convert_to_numpy(p))
        pred_np = np.concatenate(preds, axis=0)
        sisdr_out = float(np.mean([si_sdr(yi, pi) for pi, yi in zip(self.y, pred_np)]))
        estoi_out = float(np.mean([stoi_score(yi, pi, self.sr, extended=True)
                                   for pi, yi in zip(pred_np, self.y)]))
        sisdr_i = sisdr_out - self.sisdr_in
        estoi_gain = estoi_out - self.estoi_in
        logs = logs if logs is not None else {}
        logs['estoi_gain'] = estoi_gain
        logs['sisdr_i'] = sisdr_i
        print(f' - estoi_gain: {estoi_gain:+.4f} (out {estoi_out:.4f}, in {self.estoi_in:.4f})'
              f' - sisdr_i: {sisdr_i:+.3f} dB')


# ---------------------------------------------------------------------------
# Custom Keras layers
# ---------------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class STFT(layers.Layer):
    def __init__(self, frame_length=1024, frame_step=256, **kwargs):
        super().__init__(**kwargs)
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.pad_len = frame_length - frame_step

    def call(self, inputs):
        x = keras.ops.pad(inputs, [[0, 0], [0, self.pad_len]])
        X = keras.ops.stft(x, self.frame_length, self.frame_step,
                           self.frame_length, center=False)
        X = keras.ops.stack(X, axis=-1)
        # drop the DC bin so the frequency dimension is a power of two
        return X[:, :, 1:, :]

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"frame_length": self.frame_length, "frame_step": self.frame_step})
        return cfg


@keras.saving.register_keras_serializable()
class ISTFT(layers.Layer):
    def __init__(self, frame_length=1024, frame_step=256, **kwargs):
        super().__init__(**kwargs)
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.pad_len = frame_length - frame_step

    def call(self, inputs):
        # Add zero DC bin back: F goes from frame_length//2 to frame_length//2+1
        x = keras.ops.pad(inputs, [[0, 0], [0, 0], [1, 0], [0, 0]])
        x = [x[:, :, :, 0], x[:, :, :, 1]]
        x = keras.ops.istft(x, self.frame_length, self.frame_step,
                            self.frame_length, center=False)
        return x[:, :-self.pad_len]

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"frame_length": self.frame_length, "frame_step": self.frame_step})
        return cfg


@keras.saving.register_keras_serializable()
class cartesian_to_logpolar(layers.Layer):
    def call(self, cart):
        magsq = keras.ops.sum(keras.ops.square(cart), axis=-1)
        log_mag = keras.ops.log(keras.ops.abs(magsq) + 1e-5)
        phi = keras.ops.arctan2(cart[:, :, :, 1], cart[:, :, :, 0])
        return log_mag[..., tf.newaxis], phi


@keras.saving.register_keras_serializable()
class logpolar_to_cartesian(layers.Layer):
    def call(self, log_mag, phi):
        x = keras.ops.squeeze(keras.ops.exp(log_mag), axis=-1)
        re = keras.ops.multiply(keras.ops.cos(phi), x)
        im = keras.ops.multiply(keras.ops.sin(phi), x)
        return keras.ops.stack((re, im), axis=-1)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def dereverb_model(input_shape=(None,), frame_length=512, frame_step=128):
    inputs = layers.Input(shape=input_shape)

    x = STFT(frame_length, frame_step)(inputs)
    x, phi = cartesian_to_logpolar()(x)

    def down(c, y):
        y = layers.Conv2D(c, 5, 2, 'same')(y)
        y = layers.BatchNormalization()(y)
        return layers.LeakyReLU(alpha=0.2)(y)

    x1 = layers.LeakyReLU(alpha=0.2)(layers.Conv2D(64, 5, 2, 'same')(x))   # 128,128
    x2 = down(128, x1)                                                      # 64,64
    x3 = down(256, x2)                                                      # 32,32
    x4 = down(512, x3)                                                      # 16,16
    x5 = down(512, x4)                                                      # 8,8
    x6 = down(512, x5)                                                      # 4,4

    b = layers.Conv2D(512, 5, 2, 'same')(x6)
    b = layers.BatchNormalization()(b)
    b = layers.ReLU()(b)                                                    # 1,1

    def up(c, y, skip, dropout=False):
        y = layers.Conv2DTranspose(c, 5, 2, padding='same')(y)
        y = layers.BatchNormalization()(y)
        y = layers.ReLU()(y)
        if dropout:
            y = layers.Dropout(0.5)(y)
        return layers.concatenate([y, skip])

    x = up(512, b,  x6, dropout=True)   # 4,4
    x = up(512, x,  x5, dropout=True)   # 8,8
    x = up(512, x,  x4)                 # 16,16
    x = up(256, x,  x3)                 # 32,32
    x = up(128, x,  x2)                 # 64,64
    x = up(64,  x,  x1)                 # 128,128
    x = layers.Conv2DTranspose(1, 5, 2, 'same')(x)  # 256,256

    x = logpolar_to_cartesian()(x, phi)
    outputs = ISTFT(frame_length, frame_step)(x)
    return keras.Model(inputs=inputs, outputs=outputs)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def log_l2_loss(frame_length, frame_step):
    stft = STFT(frame_length, frame_step)
    to_logmag = cartesian_to_logpolar()

    def apply(y_true, y_pred):
        a, _ = to_logmag(stft(y_true))
        b, _ = to_logmag(stft(y_pred))
        return keras.ops.mean(keras.ops.square(a - b))
    return apply

def log_l1_loss(frame_length, frame_step):
    stft = STFT(frame_length, frame_step)
    to_logmag = cartesian_to_logpolar()

    def apply(y_true, y_pred):
        a, _ = to_logmag(stft(y_true))
        b, _ = to_logmag(stft(y_pred))
        return keras.ops.mean(keras.ops.abs(a - b))
    return apply


MULTI_SCALE_STFT_CONFIGS = [
    (2048, 512),
    (1024, 256),
    (512, 128),
    (256, 64),
]

def multi_scale_spectral_loss(configs=MULTI_SCALE_STFT_CONFIGS):
    scales = [(STFT(fl, fs), cartesian_to_logpolar()) for fl, fs in configs]

    def apply(y_true, y_pred):
        loss = 0.0
        for stft, to_logmag in scales:
            a, _ = to_logmag(stft(y_true))
            b, _ = to_logmag(stft(y_pred))
            loss += keras.ops.mean(keras.ops.abs(a - b))
        return loss / len(scales)
    return apply


def ssim_loss(frame_length, frame_step, max_val=20.0):
    stft = STFT(frame_length, frame_step)
    to_logmag = cartesian_to_logpolar()

    def apply(y_true, y_pred):
        a, _ = to_logmag(stft(y_true))
        b, _ = to_logmag(stft(y_pred))
        return 1.0 - keras.ops.mean(tf.image.ssim(a, b, max_val=max_val))
    return apply


LOSS_BUILDERS = {
    'l1': lambda fl, fs: log_l1_loss(fl, fs),
    'l2': lambda fl, fs: log_l2_loss(fl, fs),
    'multi_scale': lambda fl, fs: multi_scale_spectral_loss(),
    'ssim': lambda fl, fs: ssim_loss(fl, fs),
}

# ---------------------------------------------------------------------------
# Dataset (reads X/Y wav pairs written by prepare.py)
# ---------------------------------------------------------------------------

class dataset(keras.utils.PyDataset):
    def __init__(self, batch_size, paths, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.paths = paths

    def __len__(self):
        return len(self.paths) // self.batch_size

    def __getitem__(self, idx):
        i = idx * self.batch_size
        batch = self.paths[i:i + self.batch_size]
        X, Y = [], []
        for x_path in batch:
            basename = os.path.basename(x_path)
            y_path = os.path.join(os.path.dirname(os.path.dirname(x_path)), 'Y', basename)
            x, _ = sf.read(x_path)
            y, _ = sf.read(y_path)
            X.append(x)
            Y.append(y)
        return np.array(X), np.array(Y)


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

FRAME_LENGTH = 512
FRAME_STEP = 128

def checkpoint_path(ver, loss='l1'):
    return f'./checkpoints/unet-{loss}-{ver}.weights.h5'


def train(ver='prd', epochs=200, batch_size=32, lr=1e-4, resume=False, seed=42,
          loss='l1', data_dir='./data', time_budget=None):
    keras.utils.set_random_seed(seed)
    train_ds_dir = f'{data_dir}/train-{ver}'
    val_ds_dir = f'{data_dir}/val-{ver}'
    train_paths = tf.io.gfile.glob(train_ds_dir + '/X/*.wav')
    val_paths = tf.io.gfile.glob(val_ds_dir + '/X/*.wav')

    train_ds = dataset(batch_size, train_paths)
    val_ds = dataset(batch_size, val_paths)

    loss_fn = LOSS_BUILDERS[loss](FRAME_LENGTH, FRAME_STEP)
    print(f'*** Loss: {loss} ***')

    model = dereverb_model((None,), FRAME_LENGTH, FRAME_STEP)
    model.summary()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=loss_fn,
        jit_compile=False,
    )

    ckpt = checkpoint_path(ver, loss)
    os.makedirs(os.path.dirname(ckpt), exist_ok=True)
    if resume and os.path.exists(ckpt):
        print(f'*** Loaded model checkpoint "{ckpt}" ***')
        model.load_weights(ckpt)
    else:
        print('*** Training from scratch ***')

    callbacks = [
        MetricsOnVal(val_paths),
        keras.callbacks.ModelCheckpoint(filepath=ckpt, save_weights_only=True,
                                        monitor='val_loss', mode='min', save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=10),
    ]
    if time_budget:
        callbacks.append(TimeBudget(time_budget))

    model.fit(x=train_ds, epochs=epochs, validation_data=val_ds, callbacks=callbacks)
    # Fallback if the budget ran out before any checkpoint was written.
    if not os.path.exists(ckpt):
        model.save_weights(ckpt)
    return model, ckpt


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
    p.add_argument('--time-budget', type=int, default=None,
                   help='wall-clock training cap in seconds (default: disabled)')
    args = p.parse_args()
    _, ckpt = train(args.ver, args.epochs, args.batch_size, args.lr, args.resume, args.seed,
                    loss=args.loss, data_dir=args.data_dir, time_budget=args.time_budget)

    from eval import evaluate
    evaluate(ckpt, ver=args.ver, split='test', batch_size=16, data_dir=args.data_dir)
