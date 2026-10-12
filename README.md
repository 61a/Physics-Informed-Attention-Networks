# Physics-Informed Attention Networks

We provide the implementation of our dispersion-guided network for fast radio burst candidate classification. The model combines a cold-plasma alignment bank, learned fusion over dispersion-measure (DM) hypotheses, and a Conformer encoder.

The [code and experiment guide](revision/README.md) covers model training, baseline and backbone comparisons, component ablations, classification evaluation, and complete-model inference benchmarking. The three comparison settings use explicit module definitions and a shared evaluation interface.

Start with the [installation and experiment commands](revision/README.md) and the [configuration template](revision/configs/protocol.json). Set the channel frequencies, sampling interval, and DM grid to match the dynamic spectra used in the experiment.

Each training run saves its configuration, checkpoint, validation and test predictions, and confusion-matrix counts. The supplied summary script calculates run-level metrics and statistics across training seeds. [Automated checks](.github/workflows/revision-tests.yml) cover physical alignment, model pathways, and evaluation calculations.

## Experimental records

The [experimental records](experiment_records/20261012/README.md) provide run-level summaries, recorded complete-model parameter counts, original archived results and training logs, and identified historical source snapshots. Detailed records are kept outside the manuscript. The [parameter-count table](experiment_records/20261012/parameter_counts.csv) and [complete-model benchmark](revision/benchmark.py) specify the resources discussed in the reviewer response.

The main comparison series and archived development series are identified separately; uploading records is not a new model-training or independent reproduction run.
