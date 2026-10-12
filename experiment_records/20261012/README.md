# Experimental records

We provide run-level numerical records, model-specific parameter counts, and source identifiers for the classification experiments.

## Main comparison

The [count table](main_run_counts.md) reports five runs for each of 13 configurations on a fixed test set of 90 positive and 150 negative examples. [Metric summaries](main_metric_summary.json) give the arithmetic mean and sample standard deviation of the run-level metrics. Precision, recall, and F1 are calculated separately for each run before aggregation.

## Additional development runs

The [record index](archive_results_manifest.csv) links the original JSON files and training logs to repository paths and SHA256 identifiers. The [numerical summary](archive_metric_summary.json) retains all 13 configurations in the seed-42 comparison and all three recorded seeds for MorphNet v1, v5, and v9. The [original records](archive/) retain each run's operating threshold and model identity.

We keep the main comparison and archived development series separate. The main-series files report integer classification counts and their summaries; the archive reports the additional runs identified in its index. These files provide numerical reporting records rather than a distribution of trained checkpoints or main-series sample-level predictions.

## Parameter counts

The [parameter table](parameter_counts.csv) reports the complete-model totals recorded for the indexed development runs. Counts include the encoder, attention, projections, parameterized positional embeddings, input adaptation, and classification head, together with any trainable dispersion parameters. Temporarily frozen weights remain in total counts; buffers are excluded. The archived implementation uses 32 learnable DM anchors, while the current `revision/` implementation uses a fixed hypothesis grid.

## Source versions

The [source snapshot](source_snapshots/paper_repro/README.md) preserves the archived training and model definitions. [Checksums](source_snapshot_checksums.json) identify its retained source files. The current public execution code is in [revision/](../../revision/README.md). Record-specific model and threshold definitions should be used when interpreting each series.

## Inference benchmark

The [benchmark script](../../revision/benchmark.py) implements complete GPU-resident forward timing: batch=1, FP32, TF32 disabled, 50 warm-up forwards, and 200 timed forwards, with CUDA synchronization around each timed forward. It includes dispersion/FFT, fusion, encoding, and classification, and excludes file I/O, host preprocessing, and host-to-device transfer. The output records individual times, mean and sample standard deviation, device/software information, input shape, parameter counts, and checkpoint hash.

This section specifies the measurement procedure. Measured latency is reported only in an output produced by running that procedure on a specified checkpoint and device.
