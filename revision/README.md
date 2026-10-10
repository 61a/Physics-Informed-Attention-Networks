# Physics-Informed Attention Networks

Reference implementation for dispersion-guided classification of fast radio burst candidate windows. The network combines a cold-plasma alignment bank, learned fusion over dispersion hypotheses, and a Conformer encoder. Raw-input baselines, alternative encoders with the shared front-end, and two component ablations use the same evaluation interface.

## Installation

Install Python 3.10 or later and a PyTorch build appropriate for your hardware, then install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Input data

Each `.npy` file must contain a finite intensity array of shape `(2, 60, 1024)`, with axes **channel, time, frequency**. The loader preserves these axes and rejects files with a different shape. Data are supplied separately by the researcher.

Create a CSV manifest with columns `path,label,split`, where label is `0` or `1` and split is `train`, `val`, or `test`. Paths are relative to the manifest directory, or absolute. A sample may occur in only one split. Supply an existing observation-independent split when one is available; the optional `group` column records observing-session identifiers and is checked for cross-split overlap. A sample-level split establishes filename separation, rather than observing-session independence.

Create a JSON configuration from `configs/protocol.json`. Add the actual ordered channel frequencies in GHz, the time sampling interval in milliseconds, and the dispersion grid used for the input. These quantities must describe the supplied arrays after preprocessing. For windows already dedispersed at a known reference DM, the grid describes residual DM. Frequency reversal must also reverse the frequency coordinates.

The Fourier transform uses a positive phase to advance delayed low-frequency channels. Zero padding prevents circular wraparound. The implementation requires the maximum trial displacement to be shorter than the input window; provide windows with adequate support for the chosen grid. This operation acts on detected intensity, rather than telescope voltages.

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

Timing uses the complete GPU-resident forward pass, batch size one, FP32, 50 warm-up iterations and 200 timed repetitions. CUDA is synchronized before and after each timing interval. File I/O, host preprocessing, and transfer to the GPU are excluded. Parameter counts refer to the complete instantiated model.

## Results and versioning

The run outputs, rather than example percentages, are the source of manuscript tables. This source package does not include new observational performance results or trained checkpoints. Each run records SHA-256 hashes of the configuration, source files, manifest, input arrays, and checkpoint. An accompanying experiment index should associate each manuscript table row with the corresponding run directory and published version.

## Checks

```powershell
python -m unittest discover -s tests -v
```

Tests check the dispersion units and alignment sign, suppression of wraparound, integer metric counts, validation-threshold selection, physical-layer gradients, and the model output pathway. Observational training and GPU benchmarking require the supplied dataset and a compatible runtime.
