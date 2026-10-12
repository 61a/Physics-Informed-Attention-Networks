# Main comparison: run-level counts and metric summaries

The main comparison series reports five evaluations per configuration on a fixed test set containing 90 positive and 150 negative examples. Run identifiers 1–5 identify the supplied experimental records; they are not substituted for random-seed identifiers. Precision, recall, and F1 are computed separately for each run, followed by the arithmetic mean and sample standard deviation (n−1).

This record series is kept separate from the historical development runs in archive/. The archived paper_repro source is not asserted to identify the missing checkpoint-to-configuration provenance of these five main-series runs. Publication of these records is not a new training or inference run.

| 配置 | 召回率% | 精确率% | F1 |
|---|---:|---:|---:|
| raw_resnet18 | 73.56 \pm 0.93 | 80.93 \pm 0.53 | 0.771 \pm 0.006 |
| raw_vit_small | 79.78 \pm 0.93 | 75.58 \pm 0.52 | 0.776 \pm 0.004 |
| raw_efficientnet_b0 | 77.78 \pm 0.79 | 79.37 \pm 0.43 | 0.786 \pm 0.005 |
| raw_convnext_tiny | 83.56 \pm 0.93 | 81.39 \pm 0.45 | 0.825 \pm 0.006 |
| raw_mobilenet_v3_large | 72.89 \pm 0.61 | 78.10 \pm 0.43 | 0.754 \pm 0.003 |
| raw_conformer | 83.56 \pm 0.93 | 87.85 \pm 0.53 | 0.856 \pm 0.005 |
| learned_resnet18 | 88.00 \pm 0.93 | 89.19 \pm 0.48 | 0.886 \pm 0.004 |
| learned_vit_small | 82.67 \pm 0.61 | 86.52 \pm 0.50 | 0.845 \pm 0.003 |
| learned_efficientnet_b0 | 82.89 \pm 0.61 | 83.63 \pm 0.51 | 0.833 \pm 0.004 |
| learned_convnext_tiny | 90.67 \pm 0.61 | 87.93 \pm 0.40 | 0.893 \pm 0.003 |
| learned_mobilenet_v3_large | 79.11 \pm 0.93 | 80.36 \pm 0.35 | 0.797 \pm 0.004 |
| uniform_conformer | 92.00 \pm 0.93 | 88.84 \pm 0.56 | 0.904 \pm 0.006 |
| learned_conformer | 96.67 \pm 0.79 | 93.95 \pm 0.56 | 0.953 \pm 0.005 |

## 逐次实验计数
| 配置 | 重复编号 | TP | FN | FP | TN |
|---|---:|---:|---:|---:|---:|
| raw_resnet18 | 1 | 65 | 25 | 15 | 135 |
| raw_resnet18 | 2 | 66 | 24 | 16 | 134 |
| raw_resnet18 | 3 | 67 | 23 | 15 | 135 |
| raw_resnet18 | 4 | 66 | 24 | 16 | 134 |
| raw_resnet18 | 5 | 67 | 23 | 16 | 134 |
| raw_vit_small | 1 | 71 | 19 | 22 | 128 |
| raw_vit_small | 2 | 72 | 18 | 23 | 127 |
| raw_vit_small | 3 | 73 | 17 | 24 | 126 |
| raw_vit_small | 4 | 71 | 19 | 23 | 127 |
| raw_vit_small | 5 | 72 | 18 | 24 | 126 |
| raw_efficientnet_b0 | 1 | 69 | 21 | 18 | 132 |
| raw_efficientnet_b0 | 2 | 70 | 20 | 19 | 131 |
| raw_efficientnet_b0 | 3 | 71 | 19 | 18 | 132 |
| raw_efficientnet_b0 | 4 | 70 | 20 | 18 | 132 |
| raw_efficientnet_b0 | 5 | 70 | 20 | 18 | 132 |
| raw_convnext_tiny | 1 | 74 | 16 | 17 | 133 |
| raw_convnext_tiny | 2 | 75 | 15 | 18 | 132 |
| raw_convnext_tiny | 3 | 76 | 14 | 17 | 133 |
| raw_convnext_tiny | 4 | 75 | 15 | 17 | 133 |
| raw_convnext_tiny | 5 | 76 | 14 | 17 | 133 |
| raw_mobilenet_v3_large | 1 | 65 | 25 | 18 | 132 |
| raw_mobilenet_v3_large | 2 | 66 | 24 | 19 | 131 |
| raw_mobilenet_v3_large | 3 | 65 | 25 | 18 | 132 |
| raw_mobilenet_v3_large | 4 | 66 | 24 | 19 | 131 |
| raw_mobilenet_v3_large | 5 | 66 | 24 | 18 | 132 |
| raw_conformer | 1 | 75 | 15 | 10 | 140 |
| raw_conformer | 2 | 76 | 14 | 11 | 139 |
| raw_conformer | 3 | 74 | 16 | 10 | 140 |
| raw_conformer | 4 | 75 | 15 | 11 | 139 |
| raw_conformer | 5 | 76 | 14 | 10 | 140 |
| learned_resnet18 | 1 | 79 | 11 | 9 | 141 |
| learned_resnet18 | 2 | 80 | 10 | 10 | 140 |
| learned_resnet18 | 3 | 78 | 12 | 9 | 141 |
| learned_resnet18 | 4 | 80 | 10 | 10 | 140 |
| learned_resnet18 | 5 | 79 | 11 | 10 | 140 |
| learned_vit_small | 1 | 74 | 16 | 11 | 139 |
| learned_vit_small | 2 | 75 | 15 | 12 | 138 |
| learned_vit_small | 3 | 74 | 16 | 11 | 139 |
| learned_vit_small | 4 | 75 | 15 | 12 | 138 |
| learned_vit_small | 5 | 74 | 16 | 12 | 138 |
| learned_efficientnet_b0 | 1 | 74 | 16 | 14 | 136 |
| learned_efficientnet_b0 | 2 | 75 | 15 | 15 | 135 |
| learned_efficientnet_b0 | 3 | 74 | 16 | 15 | 135 |
| learned_efficientnet_b0 | 4 | 75 | 15 | 14 | 136 |
| learned_efficientnet_b0 | 5 | 75 | 15 | 15 | 135 |
| learned_convnext_tiny | 1 | 81 | 9 | 11 | 139 |
| learned_convnext_tiny | 2 | 82 | 8 | 12 | 138 |
| learned_convnext_tiny | 3 | 81 | 9 | 11 | 139 |
| learned_convnext_tiny | 4 | 82 | 8 | 11 | 139 |
| learned_convnext_tiny | 5 | 82 | 8 | 11 | 139 |
| learned_mobilenet_v3_large | 1 | 71 | 19 | 17 | 133 |
| learned_mobilenet_v3_large | 2 | 72 | 18 | 18 | 132 |
| learned_mobilenet_v3_large | 3 | 70 | 20 | 17 | 133 |
| learned_mobilenet_v3_large | 4 | 72 | 18 | 18 | 132 |
| learned_mobilenet_v3_large | 5 | 71 | 19 | 17 | 133 |
| uniform_conformer | 1 | 82 | 8 | 10 | 140 |
| uniform_conformer | 2 | 83 | 7 | 11 | 139 |
| uniform_conformer | 3 | 84 | 6 | 10 | 140 |
| uniform_conformer | 4 | 82 | 8 | 11 | 139 |
| uniform_conformer | 5 | 83 | 7 | 10 | 140 |
| learned_conformer | 1 | 86 | 4 | 6 | 144 |
| learned_conformer | 2 | 87 | 3 | 5 | 145 |
| learned_conformer | 3 | 88 | 2 | 6 | 144 |
| learned_conformer | 4 | 87 | 3 | 5 | 145 |
| learned_conformer | 5 | 87 | 3 | 6 | 144 |

