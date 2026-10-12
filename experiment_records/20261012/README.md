# Experimental records (2026-10-12)

This directory provides numerical records and source identifiers separately from the manuscript.

## Main comparison series

- [Run-level confusion matrices and summaries](main_run_counts.md): five runs for each of 13 comparison configurations.
- [Machine-readable mean and sample standard deviation](main_metric_summary.json): calculated from the run-level integer counts.

## Archived development experiments

- [Parameter counts](parameter_counts.csv): recorded complete-model counts for the 22 identified development runs, with their model, seed, and validation-selected threshold.
- [Source and log index](archive_results_manifest.csv): original source identities and repository-relative paths with SHA256 values.
- [Original JSON records and training logs](archive/): the complete 13-configuration seed-42 series and three seeds each for MorphNet v1, v5, and v9.
- [Source snapshot](source_snapshots/paper_repro/) and [checksums](source_snapshot_checksums.json): the historical paper_repro implementation.

The main five-run series and the historical development series are distinct records. They are not pooled, and the historical source and single-seed parameter counts are not substituted for the main-series configurations. The archived series includes all configurations in its complete comparison and all three seeds for each listed MorphNet variant. These files document experimental observations; publishing them is not a new independent reproduction run.

## Parameter-count convention

Total parameters are counted over the complete instantiated classifier, including attention, encoder, input adaptation, projections, parameterized positions, and the classification head. Temporarily frozen parameters are included in the total; trainable counts are separate. Buffers are excluded. The historical paper_repro dispersion layer has 32 learnable DM anchors. The current public revision implementation instead uses fixed DM hypotheses; the two implementations are identified separately.

## Inference benchmark

The current [benchmark script](../../revision/benchmark.py) and [experiment guide](../../revision/README.md) define complete GPU-resident forward timing: model.eval(), inference_mode(), batch=1, FP32, TF32 disabled, 50 warm-up forwards and 200 timed forwards, with CUDA synchronization before and after each timing interval. Dispersion/FFT, fusion, encoding, and classification are included. File I/O, host preprocessing, and host-to-device transfer are excluded. The script saves individual timings, their mean and sample standard deviation, GPU/software information, input dimensions, parameter counts, and checkpoint hash.

These settings specify a benchmark procedure, not a newly measured latency result. Training elapsed time and diagnostic-loop timings are not reported as per-candidate inference latency.

## Source identities

The public revision implementation at the time this package was prepared was commit b65d265bcaf3acf028dd266eb59afed0fb41a46a. The historical source is preserved independently, including its original operator expression and pretraining/adaptation choices; it is not relabeled as the corrected revision implementation. The published records do not supply missing checkpoints, physical data metadata, or executed configurations for the main five-run series.
