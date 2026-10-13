# Physics-Informed Attention Networks

We provide code for our dispersion-guided fast radio burst candidate classifier. The model combines a cold-plasma alignment bank, learned fusion over dispersion-measure hypotheses, and a Conformer encoder.

## Code and usage

The [implementation guide](revision/README.md) covers input preparation, training, baseline comparisons, encoder substitutions, component ablations, and evaluation. Start with the [configuration template](revision/configs/protocol.json), using the physical coordinates and dimensions of the prepared intensity spectra.

The [complete-model benchmark script](revision/benchmark.py) specifies GPU-resident forward timing and saves the device, precision, input shape, and individual measurements when executed with a checkpoint.

## Verification

The [implementation checks](revision/tests/test_reference.py) cover physical alignment, input validation, model pathways, and classification calculations. They can be run locally as described in the guide and through [GitHub Actions](.github/workflows/revision-tests.yml).
