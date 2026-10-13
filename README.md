# Physics-Informed Attention Networks

We provide code and experimental records for our dispersion-guided fast radio burst candidate classifier. The model combines a cold-plasma alignment bank, learned fusion over dispersion-measure hypotheses, and a Conformer encoder.

## Code and usage

The [implementation guide](revision/README.md) covers input preparation, training, baseline comparisons, encoder substitutions, component ablations, and evaluation. Start with the [configuration template](revision/configs/protocol.json), using the physical coordinates and dimensions of the prepared intensity spectra.

The [complete-model benchmark script](revision/benchmark.py) specifies GPU-resident forward timing and saves the device, precision, input shape, and individual measurements when executed with a checkpoint.

## Experimental records

We publish the [experimental records](experiment_records/20261012/README.md) separately from the manuscript:

- Main comparison: [run-level counts](experiment_records/20261012/main_run_counts.md) and [metric summaries](experiment_records/20261012/main_metric_summary.json).
- Additional development runs: [original results and training logs](experiment_records/20261012/archive/) and [source index](experiment_records/20261012/archive_results_manifest.csv).
- Computational accounting: [parameter counts for indexed development models](experiment_records/20261012/parameter_counts.csv).
- Version identification: [archived source snapshot](experiment_records/20261012/source_snapshots/paper_repro/README.md).

The main comparison records support verification of metric aggregation. Archived logs and parameter counts describe the separately indexed development runs. Use `revision/` for the current execution interface; the source snapshots identify the historical implementations.

## Verification

The [implementation checks](revision/tests/test_reference.py) cover physical alignment, input validation, model pathways, and classification calculations. They can be run locally as described in the guide and through [GitHub Actions](.github/workflows/revision-tests.yml).
