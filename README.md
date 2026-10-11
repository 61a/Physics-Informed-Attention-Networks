# Physics-Informed Attention Networks

We provide the implementation of our dispersion-guided network for fast radio burst candidate classification. The model combines a cold-plasma alignment bank, learned fusion over dispersion-measure (DM) hypotheses, and a Conformer encoder.

The [code and experiment guide](revision/README.md) covers model training, baseline and backbone comparisons, component ablations, classification evaluation, and complete-model inference benchmarking. The three comparison settings use explicit module definitions and a shared evaluation interface.

Start with the [installation and experiment commands](revision/README.md) and the [configuration template](revision/configs/protocol.json). Set the channel frequencies, sampling interval, and DM grid to match the dynamic spectra used in the experiment.

Each training run saves its configuration, checkpoint, validation and test predictions, and confusion-matrix counts. The supplied summary script calculates run-level metrics and statistics across training seeds. [Automated checks](.github/workflows/revision-tests.yml) cover physical alignment, model pathways, and evaluation calculations.
