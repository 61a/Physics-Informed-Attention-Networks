# Physics-Informed Attention Networks

We implement a dispersion-guided network for fast radio burst candidate classification. A cold-plasma alignment bank represents the input under multiple DM hypotheses, learned attention combines these representations, and a Conformer encoder processes the resulting temporal features. The experiment interface supports complete-model comparisons, encoder substitutions, and component ablations.

## Installation

Install Python 3.10 or later and a PyTorch build appropriate for your hardware, then install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Input data

Prepare each `.npy` file as a finite detected-intensity array with axes **channel, time, frequency**. Set `input_shape` to the actual dimensions of the prepared arrays. The published FAST-FREX products provide one intensity channel, so their direct representation uses `(1, T, F)`. Keep the full time axis when decoding the sub-integrations: the number of time samples is `NAXIS2 × NSBLK`, rather than `NAXIS2` alone. Record any subsequent cropping or time/frequency averaging in the preprocessing configuration. The loader checks every array against the supplied dimensions.

Create a CSV manifest with columns `path,label,split`, where label is `0` or `1` and split is `train`, `val`, or `test`. Paths can be relative to the manifest directory or absolute. The manifest fixes the sample assignments for all training seeds and comparison models. Include observing-session identifiers in the optional `group` column to retain each session in one partition; the loader checks file uniqueness and group assignments.

Copy `configs/protocol.json` to create the experiment configuration. Fill `input_shape`, `frequencies_ghz`, `sampling_ms`, and `dm_values` using the prepared arrays and their recorded coordinates. Read channel frequencies from `DAT_FREQ` and the original sampling interval from `TBIN`. Update the time interval and channel centres after averaging. For windows dedispersed at a known reference DM, use a residual-DM grid. Keep the frequency coordinates in the same order as the data channels.

Alignment follows the cold-plasma delay law in GHz and milliseconds, converted to samples using `sampling_ms`. A positive Fourier phase advances delayed low-frequency channels, and zero padding handles the finite-window boundaries. Choose a window length that covers the largest displacement in the trial grid. This layer processes detected-intensity dynamic spectra.

## Comparison settings

| Setting | Input pathway | Encoder |
|---|---|---|
| Raw-input comparison | Dynamic spectrum directly to the encoder; full model uses the learned front-end | CNN, ViT, or Conformer |
| Backbone comparison | Shared dispersion bank and learned DM attention | Alternative encoders |
| Variant A | Dynamic spectrum directly to the encoder | Conformer |
| Variant B | Dispersion bank with uniform fusion | Conformer |
| Full model | Dispersion bank with learned DM attention | Conformer |

The full-model versus Variant A comparison evaluates the combined front-end. Full-model versus Variant B evaluates learned attention relative to uniform fusion.

For ViT-Small, the input-channel projection follows `C` in the recorded shape. Right padding to multiples of 16 gives a `ceil(T/16) × ceil(F/16)` patch grid, followed by one classification token. Positional embeddings and encoder weights are initialized from scratch. Select the windowing and averaging before training and use the same representation for all comparisons. Record the resulting time resolution and pulse support as part of that choice.

## Experiments

Run from this directory. All model weights are initialized from scratch.

```powershell
python train.py --manifest data/manifest.csv --config configs/protocol.json --output runs/full --encoder conformer --front-end learned --seeds 42 123 456 789 1024
python train.py --manifest data/manifest.csv --config configs/protocol.json --output runs/variant_a --encoder conformer --front-end raw --seeds 42 123 456 789 1024
python train.py --manifest data/manifest.csv --config configs/protocol.json --output runs/variant_b --encoder conformer --front-end uniform --seeds 42 123 456 789 1024
```

For the raw-input baseline comparison, select `--front-end raw` and one of `resnet18`, `vit_small`, `efficientnet_b0`, `convnext_tiny`, or `mobilenet_v3_large`. For the shared-front-end encoder comparison, use the same encoder names with `--front-end learned`.

Validation F1 selects the checkpoint and decision threshold. The threshold is then applied to the test predictions. Every seed saves its checkpoint, training log, validation predictions, test predictions, exact confusion matrix, and configuration. Repeated runs use the same input manifest. Aggregate statistics use the sample standard deviation.

```powershell
python benchmark.py --checkpoint runs/full/seed_42/checkpoint.pt --output runs/full/seed_42/benchmark.json
python summarize.py --runs runs/full --output runs/full/summary.json
```

Timing covers the complete GPU-resident forward pass: dispersion/FFT operations, fusion, encoding, and classification. The protocol uses batch size one, FP32, 50 warm-up iterations, and 200 timed repetitions, with CUDA synchronization before and after each interval. This forward-pass measurement starts with the input tensor on the GPU; file reading, host preprocessing, and host-to-device transfer are outside its timing boundary. Parameter counts include all trainable modules in the instantiated model.

## Experiment outputs

Each seed produces `checkpoint.pt`, `training_log.json`, `validation_predictions.csv`, `test_predictions.csv`, and `result.json`. The prediction files retain sample identifiers, labels, probabilities, decisions, and the selected threshold. `result.json` records the confusion-matrix counts, classification metrics, model configuration, and SHA-256 identifiers for the source, manifest, input arrays, and checkpoint.

`summarize.py` checks the prediction records against the run metadata, recalculates the metrics, and reports the mean and sample standard deviation across seeds. Associate each comparison row with its run directory and repository commit when preparing the experimental tables.

## Checks

```powershell
python -m unittest discover -s tests -v
```

The checks cover dispersion units and alignment direction, window-boundary handling, confusion-matrix arithmetic, validation-threshold selection, physical-layer gradients, Conformer pathways, and the five baseline encoders. Use the data manifest for observational experiments and a CUDA device for the stated inference benchmark.
