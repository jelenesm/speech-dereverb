# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Speech dereverberation experiments built on PyTorch. A U-Net operating on log-magnitude STFT frames is trained to remove room reverb (and optionally clipping/noise) from 16 kHz speech. The pipeline lives in flat Python files at the repo root — no package structure, no `models/` directory.

## Architecture

End-to-end the pipeline is **waveform → STFT → log-polar (mag, phase) → U-Net on log-magnitude → recombine with original phase → ISTFT → waveform**. The model takes and returns raw audio; the spectral transform is baked into the `nn.Module` graph, so a single `model(x)` call handles everything.

- [train.py](train.py) — modules, model, losses, dataset, and training loop:
  - `STFT` / `ISTFT` `nn.Module`s wrap `torch.stft` / `torch.istft` with `center=False` and a periodic Hann window. Both right-pad (or trim) by `frame_length - frame_step` and drop/add back the DC bin so the frequency dimension is a power of two (important because the U-Net halves it 7×).
  - `cartesian_to_logpolar` / `logpolar_to_cartesian` — plain tensor functions (no learnable params).
  - `DereverbModel` builds the 7-down/7-up U-Net with skip connections in NCHW layout. The network only processes the log-magnitude branch; phase (`phi`) is carried around the U-Net and reattached before ISTFT via `logpolar_to_cartesian`.
  - Loss classes: `LogL1Loss`, `LogL2Loss`, `MultiScaleSpectralLoss`, `SSIMLoss`. Each operates on the log-magnitude STFT of `(pred, target)` waveforms. `LOSS_BUILDERS` maps CLI names (`l1`, `l2`, `multi_scale`, `ssim`) to constructors.
  - `DereverbDataset` (`torch.utils.data.Dataset`) reads `X/` + `Y/` wav pairs written by `prepare.py`, returning `(x, y)` tensors per item. Training wraps it in a `DataLoader`.
  - `load_batch(paths, batch_size, idx)` — numpy-array batch accessor used by `test.py` / `eval.py` / `compare_models.py` when deterministic batch-by-index access is more convenient than iterating a `DataLoader`.
  - `checkpoint_path(ver, loss)` — single source of truth for the checkpoint filename. Imported by `test.py` and `eval.py`; don't hardcode the path elsewhere.
  - `train(ver, epochs, batch_size, lr, resume, seed, loss, data_dir, num_workers)` runs a manual training loop with `torch.optim.Adam`, `ReduceLROnPlateau`, best-val checkpoint saving (via `torch.save(state_dict)`), and a manual early-stopping counter. `--resume` loads from the existing checkpoint instead of training from scratch.
- [prepare.py](prepare.py) — dataset preparation (no torch; pure numpy/scipy/librosa):
  - `prepare_impulse_responses()`, `prepare_librispeech()`, `prepare_esc50()` download and resample the raw data to 16 kHz.
  - `make_dataset()` does **offline** degradation: loads clean speech, applies reverb (convolution with sampled IR), optional clipping, optional noise, writes `X/` (degraded) + `Y/` (clean) wav pairs. Skips work if the output dir already exists unless `overwrite=True`.
  - `prepare_all(ver, overwrite)` ties the three downloads and three `make_dataset()` calls together.
  - `eval_scores()` is a PESQ/STOI evaluator stub — requires `pysepm` to produce real numbers. For quantitative metrics, prefer `eval.py` (below), which doesn't depend on pysepm.
- [test.py](test.py) — inference/demo rendering:
  - `block_inference()` runs long inputs through the model in overlapping blocks and reconstructs with linear cross-fade (`mix_overlapping_sequences`). Handles the numpy ↔ torch conversion internally and uses `next(model.parameters()).device` to pick the right device.
  - `test_models()` takes pre-loaded `(inputs, targets)` numpy arrays and a `{name: (model, checkpoint)}` dict, runs each checkpoint, and writes `input.wav`, `target.wav`, and `<model_name>.wav` per example under `./demo/example_b{batch_idx}_{k}/`.
- [eval.py](eval.py) — quantitative metrics on a split. Loads `ckpt` into a `DereverbModel`, runs it under `torch.no_grad()` in eval mode, reports mean loss plus SRMR / PESQ / STOI (input, output, gain) over a `{train,val,test}` split.
- [listen.py](listen.py) — interactive matplotlib viewer over `./demo/example_*/` directories produced by `test.py`. Toggles between input/output/target spectrograms and plays back the wavs. On WSL it shells out to `powershell.exe`'s `Media.SoundPlayer` (no Linux audio stack needed); elsewhere it falls back to `sounddevice`. Pure numpy — no torch dependency.
- [compare_models.py](compare_models.py) — side-by-side perceptual comparison of checkpoints on a test batch. Finds `*.pt` under `--ckpt-dir`, precomputes predictions for each, and launches a matplotlib UI.
- [analyze.ipynb](analyze.ipynb) — ad-hoc exploration notebook. Not part of the pipeline.

### Important gotchas

- **Modules live only in `train.py`.** `test.py` / `eval.py` / `compare_models.py` import `STFT` / `DereverbModel` / `DereverbDataset` / loss classes / `load_batch` / `checkpoint_path` / `select_device` from it. Don't duplicate them.
- **STFT module drops the DC bin** so that the frequency dimension equals `frame_length // 2` (a power of two). The U-Net assumes this — changing `frame_length` requires it to remain a power-of-two multiple of `2**num_downsampling_blocks` (currently 7, so freq dim must be divisible by 128).
- **Time dimension must also be divisible by `2**num_blocks` after framing.** For arbitrary-length audio inference, crop the input so `len(x) // frame_step` is divisible by 128 before calling `model(x)` — `block_inference()` handles this implicitly by using a fixed `block_size=16384`.
- Sample rate is hard-coded to **16 kHz** throughout (preprocessing resamples everything to this).
- `FRAME_LENGTH = 512`, `FRAME_STEP = 128` are module-level constants in `train.py` and imported by `test.py` / `eval.py` / `compare_models.py` — keep them in sync if you change them.
- The `ver` switch (`'dev'` vs `'prd'`) selects between LibriSpeech `dev-clean` (small) and `train-clean-100` (full) and is woven into all output paths and checkpoint filenames. `checkpoint_path(ver, loss)` is the single formatter.
- **NCHW throughout.** After `STFT`, tensors have shape `(B, 2, T, F)` (real/imag as channels). The log-magnitude branch is `(B, 1, T, F)`; phase `phi` is `(B, T, F)`. Conv2D operates on the `(T, F)` spatial dims. This matches standard PyTorch conventions but is transposed from the original Keras NHWC layout.
- **PyTorch Conv2d uses symmetric padding** (`padding=2` for `kernel=5, stride=2`). TF's `'same'` padding is asymmetric for this case; output sizes still match, but edge activations differ slightly. For training from scratch this is fine; for porting a pretrained Keras checkpoint it would need attention.
- **Checkpoint format is `.pt` (PyTorch `state_dict`).** `torch.save(model.state_dict(), path)` / `model.load_state_dict(torch.load(path, map_location=device))`. Old `.weights.h5` Keras checkpoints are not compatible.

## Running

Environment is managed by [uv](https://github.com/astral-sh/uv); `pyproject.toml` + `uv.lock` pin the dependencies. Use `uv sync` to create/refresh `.venv/`, then either activate it or prefix commands with `uv run`.

There is no build system and no test suite. The three scripts form a linear pipeline:

1. **`python prepare.py --ver {dev|prd}`** — one-time download and dataset generation:
   - MIT IR Survey impulse responses → `./IR/impulse_responses.pkl`
   - LibriSpeech `dev-clean` (`ver=dev`) or `train-clean-100` (`ver=prd`) → `./SpeechDev/` or `./Speech/`
   - ESC-50 environmental noise → `./Noise/<class>/`
   - Degraded/clean wav pairs → `./data/{train,val,test}-{ver}/{X,Y}/`
2. **`python train.py --ver {dev|prd}`** — builds the U-Net, picks a loss, runs the manual train/val loop. Checkpoint → `./checkpoints/unet-{loss}-{ver}.pt`. Add `--resume` to continue from an existing checkpoint.
3. **`python test.py --ver {dev|prd} --ckpt <path>`** — renders example wavs under `./demo/example_*/`.

After training, `python eval.py --ver {dev|prd} --split {train|val|test} --ckpt <path>` reports mean loss plus SRMR / PESQ / STOI. `python listen.py` opens the interactive viewer over `./demo/`. `python compare_models.py --ckpt-dir ./checkpoints` launches the side-by-side checkpoint comparer.
