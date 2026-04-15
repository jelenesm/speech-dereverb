# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Speech dereverberation experiments built on Keras / TensorFlow. A U-Net operating on log-magnitude STFT frames is trained to remove room reverb (and optionally clipping/noise) from 16 kHz speech. The pipeline lives in flat Python files at the repo root — no package structure, no `models/` directory.

## Architecture

End-to-end the pipeline is **waveform → STFT → log-polar (mag, phase) → U-Net on log-magnitude → recombine with original phase → ISTFT → waveform**. The model takes and returns raw audio; the spectral transform is baked into the Keras graph as custom layers, so a single `model(x)` call handles everything.

- [train.py](train.py) — model, loss, dataset, and training loop:
  - Custom Keras layers: `STFT` / `ISTFT` (drop the DC bin so the freq dimension is a power of two — important because the U-Net halves it 7×), `cartesian_to_logpolar` / `logpolar_to_cartesian`.
  - `dereverb_model()` builds the 7-down/7-up U-Net with skip connections. The network only processes the log-magnitude branch; phase (`phi`) is carried around the U-Net and reattached before ISTFT via `logpolar_to_cartesian`.
  - `log_l2_loss(frame_length, frame_step)` — training loss: mean squared error on log-magnitude STFTs of (target, prediction).
  - `dataset` (Keras `PyDataset`) reads `X/` + `Y/` wav pairs written by `prepare.py`.
  - `checkpoint_path(ver)` — single source of truth for the checkpoint filename. Imported by `test.py` and `eval.py`; don't hardcode the path elsewhere.
  - `train(ver, epochs, batch_size, lr, resume, seed)` fits with `ModelCheckpoint` (best val only), `ReduceLROnPlateau`, and `EarlyStopping`. `--resume` loads from the existing checkpoint instead of training from scratch.
- [prepare.py](prepare.py) — dataset preparation:
  - `prepare_impulse_responses()`, `prepare_librispeech()`, `prepare_esc50()` download and resample the raw data to 16 kHz.
  - `make_dataset()` does **offline** degradation: loads clean speech, applies reverb (convolution with sampled IR), optional clipping, optional noise, writes `X/` (degraded) + `Y/` (clean) wav pairs. Skips work if the output dir already exists unless `overwrite=True`.
  - `prepare_all(ver, overwrite)` ties the three downloads and three `make_dataset()` calls together.
  - `eval_scores()` is a PESQ/STOI evaluator stub — requires `pysepm` to produce real numbers. For quantitative metrics, prefer `eval.py` (below), which doesn't depend on pysepm.
- [test.py](test.py) — inference/demo rendering:
  - `block_inference()` runs long inputs through the model in overlapping blocks and reconstructs with linear cross-fade (`mix_overlapping_sequences`).
  - `test_models()` picks a random test batch, runs it through every model in a `{name: (model, checkpoint)}` dict, and writes `input.{png,wav}`, `target.{png,wav}`, and `<model_name>.{png,wav}` per example under `./demo_dereverb/example_*/`. Each PNG has the batch index in its `Title` tEXt chunk — `listen.py` reads it back.
  - `plot_spectrogram` lives here (not in `train.py`, which does no plotting).
- [eval.py](eval.py) — quantitative metrics on a split. Loads `checkpoint_path(ver)` and reports mean loss plus SI-SNR (input, output, gain) over a `{train,val,test}` split. Uses the same `dataset` class from `train.py`.
- [listen.py](listen.py) — interactive matplotlib viewer over `./demo_dereverb/example_*/` directories produced by `test.py`. Toggles between input/output/target spectrograms and plays back the wavs. On WSL it shells out to `powershell.exe`'s `Media.SoundPlayer` (no Linux audio stack needed); elsewhere it falls back to `sounddevice`.
- [analyze.ipynb](analyze.ipynb) — ad-hoc exploration notebook. Not part of the pipeline.

### Important gotchas

- **Layers live only in `train.py`.** `test.py` and `eval.py` import `STFT` / `dereverb_model` / `dataset` / `log_l2_loss` / `checkpoint_path` from it. Don't duplicate them.
- **STFT layer drops the DC bin** so that the frequency dimension equals `frame_length // 2` (a power of two). The U-Net assumes this — changing `frame_length` requires it to remain a power-of-two multiple of `2**num_downsampling_blocks` (currently 7, so freq dim must be divisible by 128).
- **Time dimension must also be divisible by `2**num_blocks` after framing.** For arbitrary-length audio inference, crop the input so `len(x) // frame_step` is divisible by 128 before calling `model(x)` — `block_inference()` handles this implicitly by using a fixed `block_size=16384`.
- Sample rate is hard-coded to **16 kHz** throughout (preprocessing resamples everything to this).
- `FRAME_LENGTH = 512`, `FRAME_STEP = 128` are module-level constants in `train.py` and imported by `test.py` / `eval.py` — keep them in sync if you change them.
- The `ver` switch (`'dev'` vs `'prd'`) selects between LibriSpeech `dev-clean` (small) and `train-clean-100` (full) and is woven into all output paths and checkpoint filenames. `checkpoint_path(ver)` is the single formatter.

## Running

Environment is managed by [uv](https://github.com/astral-sh/uv); `pyproject.toml` + `uv.lock` pin the dependencies. Use `uv sync` to create/refresh `.venv/`, then either activate it or prefix commands with `uv run`.

There is no build system and no test suite. The three scripts form a linear pipeline:

1. **`python prepare.py --ver {dev|prd}`** — one-time download and dataset generation:
   - MIT IR Survey impulse responses → `./IR/impulse_responses.pkl`
   - LibriSpeech `dev-clean` (`ver=dev`) or `train-clean-100` (`ver=prd`) → `./SpeechDev/` or `./Speech/`
   - ESC-50 environmental noise → `./Noise/<class>/`
   - Degraded/clean wav pairs → `./datasets/dereverb/{train,val,test}-{ver}/{X,Y}/`
2. **`python train.py --ver {dev|prd}`** — builds the U-Net, compiles with `log_l2_loss`, fits. Checkpoint → `./checkpoints/dereverb-unet-{ver}.weights.h5`. Add `--resume` to continue from an existing checkpoint.
3. **`python test.py --ver {dev|prd}`** — renders example spectrograms + wavs under `./demo_dereverb/example_*/`.

After training, `python eval.py --ver {dev|prd} --split {train|val|test}` reports mean loss and SI-SNR. `python listen.py` opens the interactive viewer over `./demo_dereverb/`.
