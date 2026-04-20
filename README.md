# speech-dereverb

U-Net speech dereverberation in PyTorch. A 7-down/7-up convolutional U-Net operates on the log-magnitude STFT of 16 kHz speech to remove room reverb (and optionally clipping and additive noise). The STFT / ISTFT are custom `nn.Module`s, so the model maps raw audio to raw audio end-to-end.

```
waveform -> STFT -> log-polar -> U-Net on log-magnitude -> recombine w/ phase -> ISTFT -> waveform
```

## Example

Reverberant input (top) and the dereverbed output of the trained U-Net (bottom). Click unmute and play to listen.

https://github.com/user-attachments/assets/8689200e-472d-4bfe-a05c-151f18ae4cc2

https://github.com/user-attachments/assets/31db8d29-60c8-45e4-bb69-eafa7a5b785c

## Install

Dependencies are pinned in [pyproject.toml](pyproject.toml) and [uv.lock](uv.lock). Create the environment with [uv](https://github.com/astral-sh/uv):

```sh
uv sync
```

Then either activate `.venv/` or prefix the commands below with `uv run`.

## Pipeline

The three scripts form a linear pipeline. `--ver prd` uses the full `train-clean-100` LibriSpeech split; `--ver dev` uses the small `dev-clean` split for quick iteration.

1. **Prepare data** — downloads MIT IR Survey impulse responses, LibriSpeech, and ESC-50 noise, then materializes degraded / clean wav pairs under `./data/{train,val,test}-{ver}/{X,Y}/` (override with `--out-dir`; raw downloads land in the CWD by default, override with `--dataset-dir`):
   ```sh
   python prepare.py --ver prd
   ```
2. **Train** — fits the U-Net and writes the best checkpoint to `./checkpoints/unet-{loss}-{ver}.pt`. Select a loss with `--loss {l1,l2,multi_scale,ssim}` (default `l1`). Training uses `ReduceLROnPlateau` with a manual early-stopping counter; pass `--resume` to continue from an existing checkpoint.
   ```sh
   python train.py --ver prd --loss l1
   ```
3. **Render examples** — picks a random test batch and writes input / target / prediction wavs under `./demo/example_*/`:
   ```sh
   python test.py --ver prd --ckpt ./checkpoints/unet-l1-prd.pt
   ```

After training, three extra tools are available:

- **`python eval.py --ver prd --split test --ckpt ./checkpoints/unet-l1-prd.pt`** — reports SRMR, PESQ, and STOI (input, output, gain) over a split.
- **`python listen.py`** — interactive matplotlib viewer over `./demo/`: toggles between input / output / target spectrograms and plays the wavs. On WSL, playback goes through `powershell.exe`'s `Media.SoundPlayer` so no Linux audio stack is needed.
- **`python compare_models.py --ver prd --ckpt-dir ./checkpoints`** — side-by-side comparison of all `*.pt` checkpoints in a directory. Pre-computes predictions for a test batch, then lets you browse examples and cycle through checkpoints with inline spectrogram display and audio playback.

All four scripts above accept `--data-dir` (default `./data`) to point at an alternative dataset layout.

## Repository layout

| File | Role |
| --- | --- |
| [train.py](train.py) | STFT / ISTFT modules, U-Net, loss classes (L1, L2, multi-scale, SSIM), `Dataset`, training loop |
| [prepare.py](prepare.py) | Downloads, resampling, and offline reverb / clipping / noise degradation |
| [test.py](test.py) | Overlap-add block inference and demo wav rendering |
| [eval.py](eval.py) | SRMR, PESQ, and STOI metrics on a split |
| [listen.py](listen.py) | Interactive viewer for rendered examples |
| [compare_models.py](compare_models.py) | Side-by-side checkpoint comparison with spectrogram display and audio playback |
| [analyze.ipynb](analyze.ipynb) | Ad-hoc exploration notebook |

Sample rate is hard-coded to **16 kHz** throughout. The STFT module drops the DC bin so the frequency dimension is a power of two (required by the 7-level U-Net). See [CLAUDE.md](CLAUDE.md) for architectural details and gotchas.

## Data sources

- **Speech** — [LibriSpeech](https://www.openslr.org/12/) (`dev-clean` or `train-clean-100`)
- **Impulse responses** — [MIT IR Survey](http://mcdermottlab.mit.edu/Reverb/IRMAudio/Audio.zip)
- **Noise** — [ESC-50](https://github.com/karoldvl/ESC-50)
